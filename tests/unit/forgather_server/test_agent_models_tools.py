"""Tier-1 results-visibility tools: models / runs / checkpoints / job_status."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from forgather_server import job_records, models_catalog
from forgather_server.agent import tools_models
from forgather_server.agent.registry import EXTENDED, READ, ToolRegistry


@dataclass
class _Fake:
    a: int
    b: str


def _reg():
    reg = ToolRegistry()
    tools_models.register_all(reg)
    return reg


def test_registration_risk_and_tier():
    by = {s.name: s for s in _reg().specs()}
    # Common results tools are core (visible even in deferred mode).
    for name in ("list_models", "run_summary", "list_checkpoints", "job_status",
                 "list_runs", "list_evaluations"):
        assert by[name].risk == READ and by[name].tier != EXTENDED
    # read_run_tty (older runs not in the job list) stays extended.
    assert by["read_run_tty"].risk == READ and by["read_run_tty"].tier == EXTENDED
    assert by["read_run_tty"].summary


def test_list_models_serializes_dataclasses(monkeypatch):
    monkeypatch.setattr(
        models_catalog, "get_project_models", lambda p: [_Fake(1, "x"), _Fake(2, "y")]
    )
    out = tools_models._list_models({"project_dir": "/proj"})
    assert out["models"] == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


def test_list_checkpoints_passthrough(monkeypatch):
    monkeypatch.setattr(models_catalog, "list_run_checkpoints", lambda o: [_Fake(3, "z")])
    assert tools_models._list_checkpoints({"output_dir": "/o"})["checkpoints"] == [
        {"a": 3, "b": "z"}
    ]


def test_run_summary_is_dict_passthrough(monkeypatch):
    payload = {"summary": {"best_loss": 1.23}, "log_path": "/o/run/tty.log"}
    monkeypatch.setattr(models_catalog, "get_run_summary", lambda r: payload)
    assert tools_models._run_summary({"run_dir": "/o/run"}) == payload


def test_read_run_tty_tails_and_caps(monkeypatch):
    captured = {}

    def fake_read(run_dir, max_bytes):
        captured["max_bytes"] = max_bytes
        return "\n".join(f"line{i}" for i in range(500))

    monkeypatch.setattr(models_catalog, "read_run_tty", fake_read)
    out = tools_models._read_run_tty({"run_dir": "/o/run", "tail_lines": 10})
    assert captured["max_bytes"] == tools_models._RUN_TTY_MAX_BYTES  # small cap
    tail = out["tail"].splitlines()
    assert len(tail) == 10 and tail[-1] == "line499"


def test_read_run_tty_missing_log_raises(monkeypatch):
    def boom(run_dir, max_bytes):
        raise FileNotFoundError

    monkeypatch.setattr(models_catalog, "read_run_tty", boom)
    with pytest.raises(ValueError):
        tools_models._read_run_tty({"run_dir": "/nope"})


def test_job_status_unknown_queue_id(monkeypatch):
    monkeypatch.setattr(job_records, "get_record", lambda q: None)
    with pytest.raises(ValueError):
        tools_models._job_status({"queue_id": "missing"})


def test_job_status_not_yet_correlated(monkeypatch):
    monkeypatch.setattr(
        job_records, "get_record",
        lambda q: SimpleNamespace(job_id=None, status="starting"),
    )
    out = tools_models._job_status({"queue_id": "q1"})
    assert out["trainer"] is None and out["status"] == "starting" and "note" in out


def test_job_status_live(monkeypatch):
    monkeypatch.setattr(
        job_records, "get_record",
        lambda q: SimpleNamespace(job_id="jid-1", status="running"),
    )
    import forgather.trainer_control as tc

    monkeypatch.setattr(tc, "get_job_status", lambda jid: {"step": 42, "loss": 1.5})
    out = tools_models._job_status({"queue_id": "q1"})
    assert out["trainer"]["step"] == 42 and out["status"] == "running"


def test_job_status_trainer_unreachable(monkeypatch):
    monkeypatch.setattr(
        job_records, "get_record",
        lambda q: SimpleNamespace(job_id="jid-1", status="running"),
    )
    import forgather.trainer_control as tc

    def boom(jid):
        raise ConnectionError("refused")

    monkeypatch.setattr(tc, "get_job_status", boom)
    out = tools_models._job_status({"queue_id": "q1"})
    assert "error" in out and out["status"] == "running"
