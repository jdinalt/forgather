"""Unit tests for the `submit` DiLoCo opt-in (unification of `diloco worker`)."""

import argparse

import pytest

from forgather.cli import diloco as diloco_mod
from forgather.cli import submit as submit_mod
from forgather.cli import submit_orch


def _submit_args(**over):
    base = dict(
        project_dir="/abs/project",
        config_template="diloco.yaml",
        _dynamic_args={},
        run_global=False,
        # DiLoCo opt-in
        diloco=False,
        diloco_server=None,
        resume_workers=False,
        count=1,
        worker_id=None,
        heartbeat_interval=30.0,
        dry_run=False,
        # shared
        via_server=None,
        dataset=None,
        priority=0,
        requested_gpus=None,
        foreground=False,
        local_only=False,
        local_fallback=False,
        member=[],
        rdzv_host=None,
        rdzv_port=None,
        allow_version_mismatch=False,
        json=False,
        wait=False,
        wait_timeout=3600,
        poll_interval=10,
        remainder=[],
    )
    base.update(over)
    return argparse.Namespace(**base)


def _worker_capture(called):
    def _fake(args):
        called["args"] = args
        return 0

    return _fake


def test_diloco_server_dispatches_to_worker(monkeypatch):
    called = {}
    monkeypatch.setattr(diloco_mod, "_worker_cmd", _worker_capture(called))
    rc = submit_mod.submit_cmd(_submit_args(diloco_server="localhost:8512"))
    assert rc == 0
    assert called["args"].diloco_server == "localhost:8512"


def test_resume_workers_dispatches_to_worker(monkeypatch):
    called = {}
    monkeypatch.setattr(diloco_mod, "_worker_cmd", _worker_capture(called))
    rc = submit_mod.submit_cmd(_submit_args(resume_workers=True))
    assert rc == 0
    assert called["args"].resume_workers is True


def test_global_plus_diloco_server_composes(monkeypatch):
    """``--global --diloco-server`` is the multi-node DiLoCo composition:
    one bundle = one DiLoCo worker group. The submit should not error
    and must take the ``--global`` fan-out path (not the worker path)."""

    def _fail_worker(args):
        raise AssertionError("compose path must not dispatch to _worker_cmd")

    monkeypatch.setattr(diloco_mod, "_worker_cmd", _fail_worker)
    # Capture the cluster submit call so we don't need a live server.
    called = {}

    def _capture_submit_global(client, args, **kwargs):
        called["kwargs"] = kwargs
        called["args"] = args
        return 0

    monkeypatch.setattr(submit_orch, "submit_global", _capture_submit_global)
    from forgather.cli import server_client

    monkeypatch.setattr(
        server_client.ServerClient, "__init__", lambda self, base=None: None
    )
    monkeypatch.setattr(submit_orch, "resolve_dataset_source", lambda c, a: None)
    monkeypatch.setattr(submit_orch, "collect_dynamic_args", lambda a: {})

    rc = submit_mod.submit_cmd(
        _submit_args(diloco_server="X", run_global=True)
    )
    assert rc == 0
    assert called["args"].diloco_server == "X"


def test_global_plus_bare_diloco_still_errors(monkeypatch):
    """``--global --diloco`` (no --diloco-server) is still a category
    error — the bare ``--diloco`` is the independent-replicas mode,
    which doesn't compose with ``--global`` (one rendezvous = one
    group). Force the operator to pin a server explicitly."""
    monkeypatch.setattr(
        diloco_mod,
        "_worker_cmd",
        lambda args: (_ for _ in ()).throw(AssertionError),
    )
    rc = submit_mod.submit_cmd(_submit_args(diloco=True, run_global=True))
    assert rc == 1


def test_global_plus_resume_workers_errors(monkeypatch):
    """``--resume-workers`` is the independent-replica re-launch flow
    and doesn't compose with ``--global`` (which is one rendezvous
    group)."""
    monkeypatch.setattr(
        diloco_mod,
        "_worker_cmd",
        lambda args: (_ for _ in ()).throw(AssertionError),
    )
    rc = submit_mod.submit_cmd(
        _submit_args(
            diloco_server="X", run_global=True, resume_workers=True
        )
    )
    assert rc == 1


def test_global_plus_worker_count_errors(monkeypatch):
    """K>1 multi-node DiLoCo groups is a follow-up (PR-C). Today: a
    single bundle = a single group, so ``--diloco-worker-count > 1``
    with ``--global`` is rejected."""
    monkeypatch.setattr(
        diloco_mod,
        "_worker_cmd",
        lambda args: (_ for _ in ()).throw(AssertionError),
    )
    rc = submit_mod.submit_cmd(
        _submit_args(diloco_server="X", run_global=True, count=3)
    )
    assert rc == 1


def test_plain_submit_does_not_dispatch_to_diloco(monkeypatch):
    # No --diloco-server / --resume-workers → single-node path (train_cmd).
    from forgather.cli import train as train_mod

    seen = {}
    monkeypatch.setattr(
        train_mod, "train_cmd", lambda args: seen.setdefault("train", True)
    )
    monkeypatch.setattr(
        diloco_mod, "_worker_cmd", lambda args: (_ for _ in ()).throw(AssertionError)
    )
    rc = submit_mod.submit_cmd(_submit_args())
    assert rc == 0
    assert seen.get("train") is True


# --- fail-loud mode-flag validation -----------------------------------------


def test_diloco_knob_without_diloco_mode_errors():
    rc = submit_mod.submit_cmd(_submit_args(count=4))
    assert rc == 1


def test_global_knob_without_global_errors():
    rc = submit_mod.submit_cmd(_submit_args(member=["h1:2"]))
    assert rc == 1


def test_requested_gpus_accepted_in_diloco_mode(monkeypatch):
    # --requested-gpus is the unified per-worker GPU knob now (no error). Mock
    # the worker launch so the test never contacts a live server.
    called = {}
    monkeypatch.setattr(diloco_mod, "_worker_cmd", _worker_capture(called))
    rc = submit_mod.submit_cmd(_submit_args(diloco_server="X", requested_gpus=4))
    assert rc == 0
    assert called["args"].requested_gpus == 4


def test_requested_gpus_rejected_in_global_mode():
    rc = submit_mod.submit_cmd(_submit_args(run_global=True, requested_gpus=4))
    assert rc == 1


def test_diloco_flag_alone_triggers_diloco_mode(monkeypatch):
    called = {}
    monkeypatch.setattr(diloco_mod, "_worker_cmd", _worker_capture(called))
    rc = submit_mod.submit_cmd(_submit_args(diloco=True))
    assert rc == 0
    assert "args" in called  # dispatched to the worker path


# --- dynamic-args forwarding (the submit-partition regression) --------------


def test_dynamic_args_forwarded_on_submit_path():
    """`submit --diloco-server` must forward the config's dynamic args.

    Regression guard: submit declares dynamic args on its own subparser, so
    main.py partitions them into args._dynamic_args; the worker path must read
    them from there (not just from raw namespace attributes). Exercised via the
    real CLI (--local-only --dry-run prints the spawned `forgather train` cmd).
    """
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    project = str(repo / "examples" / "base_lm_project")
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "forgather.cli",
            "-p",
            project,
            "-t",
            "diloco.yaml",
            "submit",
            "--diloco-server",
            "localhost:8512",
            "--local-only",
            "--dry-run",
            "--max-steps",
            "7",
        ],
        capture_output=True,
        text=True,
    )
    assert "--max-steps 7" in out.stdout, out.stdout + out.stderr
