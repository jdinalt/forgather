"""Unit tests for `forgather mkdocs` scheduler-by-default routing.

The docs server is long-running, so it submits to the scheduler by default;
--local-only runs `mkdocs serve` in the foreground. submit_orch.use_orchestrator
and the foreground subprocess are monkeypatched.
"""

import argparse

import pytest

from forgather.cli import mkdocs as mk
from forgather.cli import submit_orch


class _FakeClient:
    def __init__(self):
        self.enqueued = None

    def enqueue_job(self, **kw):
        self.enqueued = kw
        return {"queue_id": "qd-1", "priority": kw["priority"]}


def _args(**over):
    base = dict(
        config_file="mkdocs.yml",
        port=8000,
        host="127.0.0.1",
        strict=False,
        livereload=True,
        dirty=False,
        watch=[],
        enqueue=False,
        priority=0,
        via_server=None,
        local_only=False,
        local_fallback=False,
        dry_run=False,
        project_dir=".",
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_default_enqueues(monkeypatch, capsys):
    fake = _FakeClient()
    monkeypatch.setattr(submit_orch, "use_orchestrator", lambda args: fake)
    rc = mk.mkdocs_cmd(_args())
    assert rc == 0
    assert fake.enqueued["job_type"] == "mkdocs"
    assert "queued: qd-1" in capsys.readouterr().out


def test_local_only_foreground_dry_run(monkeypatch, capsys):
    # --local-only skips use_orchestrator entirely; --dry-run prints the command.
    called = {"orch": False}

    def _orch(args):
        called["orch"] = True
        return None

    monkeypatch.setattr(submit_orch, "use_orchestrator", _orch)
    rc = mk.mkdocs_cmd(_args(local_only=True, dry_run=True))
    assert rc == 0
    assert called["orch"] is False
    out = capsys.readouterr().out
    assert "mkdocs serve" in out


def test_enqueue_alias_warns(monkeypatch, capsys):
    monkeypatch.setattr(submit_orch, "use_orchestrator", lambda args: _FakeClient())
    mk.mkdocs_cmd(_args(enqueue=True))
    assert "--enqueue is deprecated" in capsys.readouterr().err
