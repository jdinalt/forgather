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
