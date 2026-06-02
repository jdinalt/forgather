"""Unit tests for the merged ``forgather job`` (jobs + queue + scheduler) and
the deprecated ``forgather sched`` alias.

``ServerClient.from_args`` is monkeypatched to a fake so these never touch a
real server; they assert the verb dispatch and the sched->job remapping.
"""

import argparse

import pytest

from forgather.cli import job as job_mod
from forgather.cli import sched as sched_mod
from forgather.cli import server_client


class _FakeClient:
    def __init__(self):
        self.calls = []

    # scheduler / queue
    def get_scheduler(self):
        self.calls.append(("get_scheduler",))
        return {"enabled": True, "last_tick_at": None}

    def set_scheduler(self, enabled):
        self.calls.append(("set_scheduler", enabled))
        return {"enabled": enabled, "last_tick_at": None}

    def list_queue(self):
        self.calls.append(("list_queue",))
        return []

    def list_jobs(self, include_dead=False):
        self.calls.append(("list_jobs",))
        return []

    def cancel(self, queue_id):
        self.calls.append(("cancel", queue_id))
        return {"aborted": queue_id}

    def cleanup_jobs(self):
        self.calls.append(("cleanup_jobs",))
        return {"count": 3}

    def gc_jobs(self):
        self.calls.append(("gc_jobs",))
        return {"swept": 2}

    # per-job control
    def job_control(self, job_id, action):
        self.calls.append(("job_control", job_id, action))
        return {"success": True, "message": action}


@pytest.fixture
def fake_client(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(
        server_client.ServerClient, "from_args", classmethod(lambda cls, args: client)
    )
    return client


def _job(**over):
    base = dict(server=None, job_subcommand=None)
    base.update(over)
    return argparse.Namespace(**base)


def test_job_list_queries_queue_and_jobs(fake_client, capsys):
    job_mod.job_cmd(_job(job_subcommand="list"))
    names = [c[0] for c in fake_client.calls]
    assert "list_queue" in names and "list_jobs" in names
    assert "(empty)" in capsys.readouterr().out


def test_job_scheduler_status(fake_client, capsys):
    job_mod.job_cmd(_job(job_subcommand="scheduler", scheduler_action="status"))
    assert ("get_scheduler",) in fake_client.calls
    assert "enabled=True" in capsys.readouterr().out


def test_job_scheduler_pause(fake_client):
    job_mod.job_cmd(_job(job_subcommand="scheduler", scheduler_action="pause"))
    assert ("set_scheduler", False) in fake_client.calls


def test_job_cancel(fake_client, capsys):
    job_mod.job_cmd(_job(job_subcommand="cancel", queue_id="q-7"))
    assert ("cancel", "q-7") in fake_client.calls
    assert "aborted: q-7" in capsys.readouterr().out


def test_job_gc(fake_client, capsys):
    job_mod.job_cmd(_job(job_subcommand="gc"))
    assert ("gc_jobs",) in fake_client.calls
    assert "swept 2" in capsys.readouterr().out


def test_job_stop_control(fake_client, capsys):
    job_mod.job_cmd(_job(job_subcommand="stop", job_id="j-1"))
    assert ("job_control", "j-1", "stop") in fake_client.calls
    assert "OK: stop" in capsys.readouterr().out


def test_job_no_subcommand_errors(fake_client):
    with pytest.raises(SystemExit) as exc:
        job_mod.job_cmd(_job(job_subcommand=None))
    assert exc.value.code == 1


# --- sched deprecation shim ------------------------------------------------


def test_sched_status_remaps_to_scheduler(fake_client, capsys):
    sched_mod.sched_cmd(argparse.Namespace(server=None, sched_subcommand="status"))
    err = capsys.readouterr().err
    assert "deprecated" in err and "job scheduler status" in err
    assert ("get_scheduler",) in fake_client.calls


def test_sched_list_remaps_to_job_list(fake_client, capsys):
    sched_mod.sched_cmd(argparse.Namespace(server=None, sched_subcommand="list"))
    out = capsys.readouterr()
    assert "deprecated" in out.err and "forgather job list" in out.err
    assert ("list_queue",) in fake_client.calls


def test_sched_cancel_remaps(fake_client, capsys):
    sched_mod.sched_cmd(
        argparse.Namespace(server=None, sched_subcommand="cancel", queue_id="q-9")
    )
    assert ("cancel", "q-9") in fake_client.calls
