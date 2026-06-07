"""Service-lifecycle tools: list_services, start_service, stop_service."""

from __future__ import annotations

import pytest

from forgather_server import queue_store, scheduler, services
from forgather_server.agent import tools_services
from forgather_server.agent.registry import CONFIRM, EXTENDED, READ, Proposal, ToolRegistry


def _reg():
    reg = ToolRegistry()
    tools_services.register_all(reg)
    return reg


def test_registration_risk_and_tier():
    by = {s.name: s for s in _reg().specs()}
    assert by["list_services"].risk == READ and by["list_services"].tier != EXTENDED
    assert by["start_service"].risk == CONFIRM and by["start_service"].tier != EXTENDED
    assert by["stop_service"].risk == CONFIRM and by["stop_service"].tier == EXTENDED


def test_list_services_shape(monkeypatch):
    svc = services.Service(type="dataset", name="d1", enabled=True, args={"port": 8766})
    st = services.ServiceStatus(service=svc, running=True, queue_id="q1", status="running")
    monkeypatch.setattr(services, "list_services", lambda: [svc])
    monkeypatch.setattr(services, "status_for_each", lambda lst: [st])
    out = tools_services._list_services({})["services"][0]
    assert out["type"] == "dataset" and out["running"] is True and out["queue_id"] == "q1"


def test_start_service_unknown_type():
    with pytest.raises(ValueError):
        tools_services._start_service({"type": "bogus"})


def test_start_service_inference_requires_model():
    with pytest.raises(ValueError):
        tools_services._start_service({"type": "inference", "args": {"port": 8137}})


def test_start_service_diloco_requires_output_and_workers():
    with pytest.raises(ValueError):
        tools_services._start_service({"type": "diloco", "args": {"output_dir": "/m"}})


def test_start_service_preview_then_commit_enqueues(monkeypatch):
    monkeypatch.setattr(services, "active_signatures", lambda: {})
    upserts, added = [], []
    monkeypatch.setattr(services, "upsert_service", lambda *a, **k: upserts.append((a, k)))
    monkeypatch.setattr(queue_store, "add_item", lambda item: added.append(item) or item)

    prop = tools_services._start_service({"type": "dataset"})
    assert isinstance(prop, Proposal)
    assert prop.extra["job_type"] == "dataset_server"
    assert upserts == [] and added == []  # preview: no side effect

    msg = prop.commit()
    assert upserts and added  # committed: persisted + enqueued
    assert "started dataset service" in msg


def test_start_service_dedup_does_not_respawn(monkeypatch):
    # A signature-identical instance is already running.
    sig = services.Service(type="dataset", name="agent-default", enabled=True, args={}).signature()
    monkeypatch.setattr(services, "active_signatures", lambda: {sig: ("qExisting", "running")})
    added = []
    monkeypatch.setattr(services, "upsert_service", lambda *a, **k: None)
    monkeypatch.setattr(queue_store, "add_item", lambda item: added.append(item))

    prop = tools_services._start_service({"type": "dataset"})
    assert prop.extra["already_active"] is True
    msg = prop.commit()
    assert added == [] and "already active" in msg  # not re-spawned


def test_stop_service_unknown(monkeypatch):
    monkeypatch.setattr(services, "get_service", lambda t, n: None)
    with pytest.raises(ValueError):
        tools_services._stop_service({"type": "dataset", "name": "ghost"})


def test_stop_service_disables_and_aborts(monkeypatch):
    svc = services.Service(type="dataset", name="d1", enabled=True, args={"port": 8766})
    sig = svc.signature()
    monkeypatch.setattr(services, "get_service", lambda t, n: svc)
    monkeypatch.setattr(services, "active_signatures", lambda: {sig: ("q7", "running")})
    disabled, aborted = [], []
    monkeypatch.setattr(services, "set_enabled", lambda t, n, e: disabled.append((t, n, e)))
    monkeypatch.setattr(scheduler, "abort_or_cancel", lambda q: aborted.append(q) or True)

    prop = tools_services._stop_service({"type": "dataset", "name": "d1"})
    assert prop.extra["running"] is True
    assert disabled == [] and aborted == []  # preview only
    msg = prop.commit()
    assert disabled == [("dataset", "d1", False)] and aborted == ["q7"]
    assert "stopped dataset:d1" in msg
