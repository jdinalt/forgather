"""Unit tests for the GPU-panel ``POST /gpus/{i}/kill`` route.

The route's job is to free a card by reaping the **in-container** processes
attached to it. NVML reports host-namespace PIDs (unkillable, untranslatable
from inside a child PID namespace), so the primary signal is each process's
``CUDA_VISIBLE_DEVICES`` env (set per job by the scheduler) read from
``/proc/<pid>/environ``. A direct ``os.kill`` over the NVML PIDs is a backstop
for ``--pid=host`` / bare-metal deployments.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import patch

from forgather_server.routes import gpus


def _proc(pid: int, kind: str = "compute"):
    return SimpleNamespace(pid=pid, kind=kind, used_mem_bytes=0, name="train")


def _gpu(index: int, procs):
    return SimpleNamespace(index=index, processes=procs)


def test_requires_confirmation():
    from fastapi import HTTPException

    try:
        gpus.kill_gpu_processes(0, gpus.GpuKillRequest(confirmed=False))
    except HTTPException as e:
        assert e.status_code == 400
    else:  # pragma: no cover
        raise AssertionError("expected 400 without confirmation")


def test_reaps_in_container_pids_by_cuda_visible_devices():
    snap = [_gpu(3, [_proc(670129)])]  # NVML host PID os.kill can't reach
    with (
        patch.object(gpus.gpu_monitor, "snapshot", return_value=snap),
        patch.object(gpus, "_in_container_pids_for_gpu", return_value=[4134593]),
        patch.object(gpus.os, "kill") as oskill,
    ):
        # os.kill: succeed for the container PID, fail for the NVML host PID.
        def _kill(pid, sig):
            if pid == 670129:
                raise PermissionError("host pid")

        oskill.side_effect = _kill
        resp = gpus.kill_gpu_processes(3, gpus.GpuKillRequest(confirmed=True))

    assert 4134593 in resp.killed  # in-container reap landed
    assert resp.failed == [670129]  # out-of-container holder, not reachable
    assert resp.pids == [670129]


def test_backstop_kills_nvml_pid_on_host_deployment():
    """No CVD match (e.g. external process), but os.kill works (--pid=host)."""
    snap = [_gpu(0, [_proc(1234)])]
    with (
        patch.object(gpus.gpu_monitor, "snapshot", return_value=snap),
        patch.object(gpus, "_in_container_pids_for_gpu", return_value=[]),
        patch.object(gpus.os, "kill", return_value=None) as oskill,
    ):
        resp = gpus.kill_gpu_processes(0, gpus.GpuKillRequest(confirmed=True))

    oskill.assert_called_once_with(1234, gpus.signal.SIGKILL)
    assert resp.killed == [1234]
    assert resp.failed == []


def test_backstop_skips_pid_already_reaped_by_scan():
    """On --pid=host the scan PID == NVML PID; don't double-count/-signal it."""
    snap = [_gpu(2, [_proc(555)])]
    with (
        patch.object(gpus.gpu_monitor, "snapshot", return_value=snap),
        patch.object(gpus, "_in_container_pids_for_gpu", return_value=[555]),
        patch.object(gpus.os, "kill", return_value=None) as oskill,
    ):
        resp = gpus.kill_gpu_processes(2, gpus.GpuKillRequest(confirmed=True))

    oskill.assert_called_once_with(555, gpus.signal.SIGKILL)  # scan only
    assert resp.killed == [555]
    assert resp.failed == []


def test_unknown_gpu_index_404():
    from fastapi import HTTPException

    with patch.object(gpus.gpu_monitor, "snapshot", return_value=[_gpu(0, [])]):
        try:
            gpus.kill_gpu_processes(9, gpus.GpuKillRequest(confirmed=True))
        except HTTPException as e:
            assert e.status_code == 404
        else:  # pragma: no cover
            raise AssertionError("expected 404 for missing GPU")


# --- the CUDA_VISIBLE_DEVICES /proc scan itself ---


def _fake_proc(environs: dict[str, bytes]):
    """Return (listdir, open) patches emulating /proc for the given environs."""

    def _listdir(path):
        assert path == "/proc"
        return list(environs) + ["cpuinfo", "not_a_pid"]

    def _open(path, *a, **k):
        # only environ reads are exercised here
        for pid, raw in environs.items():
            if path == f"/proc/{pid}/environ":
                return io.BytesIO(raw)
        raise OSError(f"no such file: {path}")

    return _listdir, _open


def test_scan_matches_single_and_list_cvd():
    environs = {
        "100": b"PATH=/x\x00CUDA_VISIBLE_DEVICES=3\x00",
        "101": b"CUDA_VISIBLE_DEVICES=0,2,4\x00",
        "102": b"CUDA_VISIBLE_DEVICES=1\x00",
        "103": b"NO_CVD_HERE=1\x00",  # not a GPU job
        "104": b"CUDA_VISIBLE_DEVICES=\x00",  # empty -> no GPU
    }
    listdir, opener = _fake_proc(environs)
    with (
        patch.object(gpus, "_self_ancestors", return_value={1}),
        patch.object(gpus.os, "listdir", side_effect=listdir),
        patch("builtins.open", side_effect=opener),
    ):
        assert gpus._in_container_pids_for_gpu(3) == [100]
        assert gpus._in_container_pids_for_gpu(4) == [101]
        assert gpus._in_container_pids_for_gpu(1) == [102]
        assert gpus._in_container_pids_for_gpu(7) == []


def test_scan_excludes_self_ancestors():
    environs = {"100": b"CUDA_VISIBLE_DEVICES=3\x00"}
    listdir, opener = _fake_proc(environs)
    with (
        patch.object(gpus, "_self_ancestors", return_value={1, 100}),
        patch.object(gpus.os, "listdir", side_effect=listdir),
        patch("builtins.open", side_effect=opener),
    ):
        # 100 is an ancestor of the server -> never a kill target.
        assert gpus._in_container_pids_for_gpu(3) == []
