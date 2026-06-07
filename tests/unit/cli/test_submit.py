"""Unit tests for `forgather submit` routing and the cluster-submit delegation.

The submit verb is a thin router: single-node delegates to `train --schedule`
(train_cmd), `--global` delegates to submit_orch.submit_global. The deprecated
`cluster submit` also routes through submit_global. All network-touching calls
are monkeypatched.
"""

import argparse

import pytest

from forgather.cli import cluster as cluster_mod
from forgather.cli import submit as submit_mod
from forgather.cli import submit_orch
from forgather.cli import train as train_mod


def _submit_args(**over):
    base = dict(
        project_dir="/abs/project",
        config_template="ddp.yaml",
        _dynamic_args={},
        run_global=False,
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
        backend="http",
        replicate=1,
        count=1,
        remainder=[],
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_member_spec_parsing():
    assert submit_orch.parse_member_spec("h1:2") == ("h1", 2, None)
    assert submit_orch.parse_member_spec("h1:4:eth0") == ("h1", 4, "eth0")
    with pytest.raises(RuntimeError):
        submit_orch.parse_member_spec("h1")
    with pytest.raises(RuntimeError):
        submit_orch.parse_member_spec("h1:0")


def test_single_node_delegates_to_train_schedule(monkeypatch):
    captured = {}

    def fake_train_cmd(args):
        captured["schedule"] = args.schedule
        captured["enqueue"] = args.enqueue
        captured["config"] = args.config_template

    monkeypatch.setattr(train_mod, "train_cmd", fake_train_cmd)
    rc = submit_mod.submit_cmd(_submit_args())
    assert rc == 0
    assert captured["schedule"] is True
    assert captured["enqueue"] is False
    assert captured["config"] == "ddp.yaml"


def test_global_delegates_to_submit_global(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        submit_orch, "resolve_dataset_source", lambda c, a: {"kind": "auto"}
    )
    monkeypatch.setattr(submit_orch, "collect_dynamic_args", lambda a: {"x": 1})

    def fake_submit_global(client, args, **kw):
        captured.update(kw)
        return 0

    monkeypatch.setattr(submit_orch, "submit_global", fake_submit_global)
    rc = submit_mod.submit_cmd(_submit_args(run_global=True))
    assert rc == 0
    assert captured["project_dir"] == "/abs/project"
    assert captured["config"] == "ddp.yaml"
    assert captured["dynamic_args"] == {"x": 1}
    assert captured["dataset_source"] == {"kind": "auto"}


def test_backend_requires_diloco(monkeypatch, capsys):
    # --backend is a DiLoCo-worker knob; without --diloco it's a misuse.
    rc = submit_mod.submit_cmd(_submit_args(backend="shared_memory"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "--backend" in err and "--diloco" in err


def test_shared_memory_backend_rejects_global(monkeypatch, capsys):
    # The shared-memory backend is single-host; it can't span a --global fan-out.
    # Use the composition form (--global + --diloco-server) so the run reaches the
    # mode-flag validation rather than the earlier --global/--diloco gate.
    rc = submit_mod.submit_cmd(
        _submit_args(run_global=True, diloco_server="srv1", backend="shared_memory")
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "shared_memory" in err and "single-host" in err


def test_collective_routes_to_launch_collective(monkeypatch):
    # --backend collective is one torchrun job (not N workers): it routes to
    # launch_collective, not the worker path, carrying the replicate degree.
    from forgather.cli import diloco_orch

    captured = {}
    monkeypatch.setattr(submit_orch, "collect_dynamic_args", lambda a: {"x": 1})

    def fake_launch_collective(args, dynamic_args):
        captured["replicate"] = args.replicate
        captured["backend"] = args.backend
        captured["dyn"] = dynamic_args
        return 0

    monkeypatch.setattr(diloco_orch, "launch_collective", fake_launch_collective)
    rc = submit_mod.submit_cmd(
        _submit_args(backend="collective", diloco_server="srv1", replicate=4)
    )
    assert rc == 0
    assert captured["replicate"] == 4 and captured["backend"] == "collective"
    assert captured["dyn"] == {"x": 1}


def test_collective_rejects_global(capsys):
    # Collective is single-host — not compatible with the multi-node fan-out.
    rc = submit_mod.submit_cmd(
        _submit_args(backend="collective", run_global=True, diloco_server="srv1")
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "collective" in err and "single-host" in err


def test_replicate_requires_collective(capsys):
    # --diloco-replicate is the collective one-job knob; meaningless elsewhere.
    rc = submit_mod.submit_cmd(_submit_args(replicate=2, diloco=True))
    assert rc == 1
    err = capsys.readouterr().err
    assert "--diloco-replicate" in err and "collective" in err


def test_worker_count_incompatible_with_collective(capsys):
    # Collective is one job; --diloco-worker-count (N jobs) is a different model.
    rc = submit_mod.submit_cmd(
        _submit_args(backend="collective", diloco_server="srv1", count=2)
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "--diloco-worker-count" in err


def test_no_config_errors(monkeypatch):
    monkeypatch.setattr(submit_orch, "resolve_default_config", lambda a: None)
    rc = submit_mod.submit_cmd(_submit_args(config_template=None))
    assert rc == 1


def test_cluster_submit_delegates_with_deprecation(monkeypatch, capsys):
    captured = {}

    def fake_submit_global(client, args, **kw):
        captured.update(kw)
        return 0

    monkeypatch.setattr(submit_orch, "submit_global", fake_submit_global)

    cluster_args = argparse.Namespace(
        project_dir="/abs/project",
        config_template="ddp.yaml",
        dynamic_arg=["lr=0.1", "warmup=true"],
        dataset_source="auto",
        member=[],
        rdzv_host=None,
        rdzv_port=None,
        priority=0,
        allow_version_mismatch=False,
        json=False,
        wait=False,
        wait_timeout=3600,
        poll_interval=10,
    )
    rc = cluster_mod._cmd_submit(object(), cluster_args)
    assert rc == 0
    err = capsys.readouterr().err
    assert "deprecated" in err and "submit --global" in err
    # legacy KEY=VAL dynamic args are parsed into a dict
    assert captured["dynamic_args"] == {"lr": 0.1, "warmup": True}
    assert captured["dataset_source"] == {"kind": "auto"}
