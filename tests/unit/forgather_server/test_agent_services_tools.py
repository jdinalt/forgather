"""Service tools: list_services, per-type start tools, stop_service."""

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
    # The three important start tools are core; tb/mkdocs are extended.
    for name in ("start_dataset_server", "start_inference_server", "start_diloco_server"):
        assert by[name].risk == CONFIRM and by[name].tier != EXTENDED
    for name in ("start_tensorboard", "start_mkdocs"):
        assert by[name].risk == CONFIRM and by[name].tier == EXTENDED
    assert by["stop_service"].risk == CONFIRM and by["stop_service"].tier == EXTENDED
    # The generic guess-the-args tool is gone.
    assert "start_service" not in by


def test_list_services_shape(monkeypatch):
    svc = services.Service(type="dataset", name="d1", enabled=True, args={"port": 8766})
    st = services.ServiceStatus(service=svc, running=True, queue_id="q1", status="running")
    monkeypatch.setattr(services, "list_services", lambda: [svc])
    monkeypatch.setattr(services, "status_for_each", lambda lst: [st])
    out = tools_services._list_services({})["services"][0]
    assert out["type"] == "dataset" and out["running"] is True and out["queue_id"] == "q1"


# ---- start: validation -----------------------------------------------------


def test_start_inference_requires_a_model():
    with pytest.raises(ValueError, match="model_path"):
        tools_services._start_inference_server({"port": 8137})


def test_start_inference_rejects_both_models():
    with pytest.raises(ValueError, match="exactly one"):
        tools_services._start_inference_server(
            {"port": 8137, "model_path": "/m", "models": [{"name": "a", "path": "/a"}]}
        )


def test_start_inference_requires_port():
    with pytest.raises(ValueError, match="port"):
        tools_services._start_inference_server({"model_path": "/m"})


def test_start_diloco_requires_output_and_workers():
    with pytest.raises(ValueError, match="output_dir"):
        tools_services._start_diloco_server({"output_dir": "/m"})  # no num_workers


def test_start_tensorboard_requires_logdir_and_port():
    with pytest.raises(ValueError):
        tools_services._start_tensorboard({"logdir": "/runs"})  # no port


# ---- start: preview + commit ----------------------------------------------


def test_start_dataset_server_defaults_enqueue(monkeypatch):
    monkeypatch.setattr(services, "active_signatures", lambda: {})
    upserts, added = [], []
    monkeypatch.setattr(services, "upsert_service", lambda *a, **k: upserts.append((a, k)))
    monkeypatch.setattr(queue_store, "add_item", lambda item: added.append(item) or item)

    prop = tools_services._start_dataset_server({})  # no args -> default server
    assert isinstance(prop, Proposal) and prop.extra["job_type"] == "dataset_server"
    assert prop.extra["args"] == {}  # defaults applied downstream
    assert upserts == [] and added == []  # preview: no side effect

    msg = prop.commit()
    assert upserts and added and "started dataset service" in msg


def test_start_inference_collects_args(monkeypatch):
    monkeypatch.setattr(services, "active_signatures", lambda: {})
    monkeypatch.setattr(services, "upsert_service", lambda *a, **k: None)
    monkeypatch.setattr(queue_store, "add_item", lambda item: item)
    prop = tools_services._start_inference_server(
        {"model_path": "/models/m", "port": 8137, "dtype": "bfloat16",
         "keep_on_gpu": True, "requested_gpus": 2}
    )
    a = prop.extra["args"]
    assert a["model_path"] == "/models/m" and a["port"] == 8137
    assert a["dtype"] == "bfloat16" and a["keep_on_gpu"] is True and a["requested_gpus"] == 2
    assert prop.extra["job_type"] == "inference"


def test_start_diloco_advanced_passthrough(monkeypatch):
    monkeypatch.setattr(services, "active_signatures", lambda: {})
    monkeypatch.setattr(services, "upsert_service", lambda *a, **k: None)
    monkeypatch.setattr(queue_store, "add_item", lambda item: item)
    prop = tools_services._start_diloco_server(
        {"output_dir": "/m", "num_workers": 2, "sync_every": 250,
         "advanced": {"bf16_comm": True, "dylu": True}}
    )
    a = prop.extra["args"]
    assert a["output_dir"] == "/m" and a["num_workers"] == 2 and a["sync_every"] == 250
    assert a["bf16_comm"] is True and a["dylu"] is True  # merged from advanced


def test_start_dedup_does_not_respawn(monkeypatch):
    sig = services.Service(type="dataset", name="agent-default", enabled=True, args={}).signature()
    monkeypatch.setattr(services, "active_signatures", lambda: {sig: ("qExisting", "running")})
    added = []
    monkeypatch.setattr(services, "upsert_service", lambda *a, **k: None)
    monkeypatch.setattr(queue_store, "add_item", lambda item: added.append(item))

    prop = tools_services._start_dataset_server({})
    assert prop.extra["already_active"] is True
    msg = prop.commit()
    assert added == [] and "already active" in msg


# ---- stop ------------------------------------------------------------------


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


def test_start_inference_description_mentions_from_checkpoint():
    by = {s.name: s for s in _reg().specs()}
    desc = by["start_inference_server"].description
    assert "from_checkpoint" in desc and "QUEUE" in desc
