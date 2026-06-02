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
        server=None,
        resume_workers=False,
        count=1,
        worker_id=None,
        heartbeat_interval=30.0,
        gpus_per_worker=1,
        devices=None,
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
    rc = submit_mod.submit_cmd(_submit_args(server="localhost:8512"))
    assert rc == 0
    assert called["args"].server == "localhost:8512"


def test_resume_workers_dispatches_to_worker(monkeypatch):
    called = {}
    monkeypatch.setattr(diloco_mod, "_worker_cmd", _worker_capture(called))
    rc = submit_mod.submit_cmd(_submit_args(resume_workers=True))
    assert rc == 0
    assert called["args"].resume_workers is True


def test_global_and_diloco_are_mutually_exclusive(monkeypatch):
    # Should error before dispatching to either path.
    monkeypatch.setattr(
        diloco_mod, "_worker_cmd", lambda args: (_ for _ in ()).throw(AssertionError)
    )
    rc = submit_mod.submit_cmd(_submit_args(server="X", run_global=True))
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


def test_diloco_worker_prints_deprecation(monkeypatch, capsys):
    monkeypatch.setattr(diloco_mod, "_worker_cmd", lambda args: 0)
    diloco_mod.diloco_cmd(argparse.Namespace(diloco_subcommand="worker"))
    err = capsys.readouterr().err
    assert "deprecated" in err and "submit --diloco-server" in err


# --- fail-loud mode-flag validation -----------------------------------------


def test_diloco_knob_without_diloco_mode_errors():
    rc = submit_mod.submit_cmd(_submit_args(count=4))
    assert rc == 1


def test_global_knob_without_global_errors():
    rc = submit_mod.submit_cmd(_submit_args(member=["h1:2"]))
    assert rc == 1


def test_requested_gpus_in_diloco_mode_errors():
    rc = submit_mod.submit_cmd(_submit_args(server="X", requested_gpus=4))
    assert rc == 1


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
