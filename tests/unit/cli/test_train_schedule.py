"""Unit tests for ``forgather train`` scheduler routing (--schedule/--foreground).

The foreground torchrun path is unchanged and exercised elsewhere (dry-run
smoke); these focus on the new --schedule routing: the deprecated --enqueue
alias, the --devices guard, and that the enqueue path calls submit_orch with
the values materialized from the config. The network-touching submit_orch
helpers are monkeypatched, so the test still loads a real project/config (no
torch needed) but never contacts a server.
"""

import argparse
from pathlib import Path

import pytest

from forgather.cli import submit_orch
from forgather.cli.train import train_cmd

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT = str(REPO_ROOT / "examples" / "base_lm_project")


def _train_args(**over):
    base = dict(
        project_dir=PROJECT,
        config_template="ddp.yaml",
        no_dyn=True,
        _dynamic_args={},
        devices=None,
        nproc=None,
        dry_run=False,
        schedule=False,
        foreground=False,
        enqueue=False,
        dataset=None,
        priority=0,
        requested_gpus=None,
        via_server=None,
        local_fallback=False,
        local_only=False,
        remainder=[],
    )
    base.update(over)
    return argparse.Namespace(**base)


class _FakeClient:
    pass


def test_schedule_devices_are_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        train_cmd(_train_args(schedule=True, devices="0"))
    assert exc.value.code == 1


def test_enqueue_is_deprecated_alias(capsys):
    # --enqueue normalizes to --schedule and warns; pairing it with --devices
    # trips the same guard, so we exit before any project/server work.
    with pytest.raises(SystemExit):
        train_cmd(_train_args(enqueue=True, devices="0"))
    err = capsys.readouterr().err
    assert "--enqueue is deprecated" in err


def test_schedule_enqueues_with_materialized_values(monkeypatch, capsys):
    captured = {}

    def fake_use_orchestrator(args):
        return _FakeClient()

    def fake_resolve_dataset_source(client, args):
        return None

    def fake_submit_single(client, **kwargs):
        captured.update(kwargs)
        return {
            "queue_id": "q-123",
            "priority": kwargs["priority"],
            "requested_gpus": kwargs["requested_gpus"],
        }

    monkeypatch.setattr(submit_orch, "use_orchestrator", fake_use_orchestrator)
    monkeypatch.setattr(
        submit_orch, "resolve_dataset_source", fake_resolve_dataset_source
    )
    monkeypatch.setattr(submit_orch, "submit_single", fake_submit_single)

    train_cmd(_train_args(schedule=True, priority=5))

    out = capsys.readouterr().out
    assert "queued: q-123" in out
    assert captured["config"] == "ddp.yaml"
    assert captured["priority"] == 5
    # ddp.yaml materializes nproc_per_node == 2 → default requested_gpus.
    assert captured["requested_gpus"] == 2
    assert captured["dataset_source"] is None


def test_foreground_attaches(monkeypatch):
    attached = {}

    monkeypatch.setattr(submit_orch, "use_orchestrator", lambda args: _FakeClient())
    monkeypatch.setattr(
        submit_orch, "resolve_dataset_source", lambda client, args: None
    )
    monkeypatch.setattr(
        submit_orch,
        "submit_single",
        lambda client, **kw: {"queue_id": "q-9", "priority": 0, "requested_gpus": 2},
    )
    monkeypatch.setattr(
        submit_orch,
        "attach_submitted",
        lambda client, queue_id, **kw: attached.update(queue_id=queue_id),
    )

    train_cmd(_train_args(schedule=True, foreground=True))
    assert attached["queue_id"] == "q-9"


def test_bad_dataset_value_errors_cleanly(monkeypatch):
    # An explicit bad --dataset value raises ValueError in resolve_dataset_source
    # (before any client call); train_cmd must turn it into a clean exit, not a
    # traceback. resolve_dataset_source is intentionally NOT mocked here.
    monkeypatch.setattr(submit_orch, "use_orchestrator", lambda args: _FakeClient())
    with pytest.raises(SystemExit) as exc:
        train_cmd(_train_args(schedule=True, dataset="bogus"))
    assert exc.value.code == 1


def test_local_only_falls_through_to_foreground(monkeypatch, capsys):
    # --local-only → use_orchestrator returns None → foreground torchrun path.
    # --dry-run keeps it from actually launching; it should print the command.
    monkeypatch.setattr(submit_orch, "use_orchestrator", lambda args: None)
    train_cmd(_train_args(schedule=True, local_only=True, dry_run=True))
    out = capsys.readouterr().out
    assert "torchrun" in out
    assert "ddp.yaml" in out
