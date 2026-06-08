"""Tier-3 tools: inference list/query, cluster_status, config overrides."""

from __future__ import annotations

import asyncio

import pytest

from forgather_server import overrides_store
from forgather_server.agent import _inference, runtime, tools_advanced
from forgather_server.agent.registry import CONFIRM, EXTENDED, READ, Proposal, ToolRegistry


def _reg():
    reg = ToolRegistry()
    tools_advanced.register_all(reg)
    return reg


def test_registration_risk_and_tier():
    by = {s.name: s for s in _reg().specs()}
    assert by["list_inference_servers"].risk == READ and by["list_inference_servers"].tier == EXTENDED
    assert by["query_model"].risk == CONFIRM and by["query_model"].tier == EXTENDED
    assert by["cluster_status"].risk == READ and by["cluster_status"].tier == EXTENDED
    assert by["get_config_overrides"].risk == READ and by["get_config_overrides"].tier == EXTENDED
    assert by["set_config_overrides"].risk == CONFIRM and by["set_config_overrides"].tier == EXTENDED


# ---- inference -------------------------------------------------------------


def test_list_inference_servers(monkeypatch):
    monkeypatch.setattr(
        _inference, "list_servers",
        lambda: [{"id": "i1", "base_url": "http://h:8137", "models": ["m"], "reachable": True}],
    )
    out = asyncio.run(tools_advanced._list_inference_servers({}))
    assert out["servers"][0]["id"] == "i1"


def test_query_model_requires_prompt_or_messages():
    with pytest.raises(ValueError):
        tools_advanced._query_model({})


def test_query_model_preview_then_commit(monkeypatch):
    calls = []

    def fake_chat(server_id, messages, *, model=None, max_tokens=256, temperature=None):
        calls.append((server_id, messages, max_tokens))
        return {
            "server": {"id": "i1", "base_url": "http://h"},
            "model": "m", "message": {"role": "assistant", "content": "hello!"},
            "finish_reason": "stop", "usage": {"completion_tokens": 2},
        }

    monkeypatch.setattr(_inference, "chat", fake_chat)
    prop = tools_advanced._query_model({"prompt": "hi", "server_id": "i1"})
    assert isinstance(prop, Proposal) and calls == []  # preview only
    msg = asyncio.run(prop.commit())
    assert calls and calls[0][1] == [{"role": "user", "content": "hi"}]
    assert "hello!" in msg


# ---- cluster ---------------------------------------------------------------


def test_cluster_status_inactive(monkeypatch):
    import forgather_server.cluster as cluster

    monkeypatch.setattr(cluster, "is_active", lambda: False)
    out = tools_advanced._cluster_status({})
    assert out["active"] is False


def test_cluster_status_active(monkeypatch):
    import forgather_server.cluster as cluster
    from types import SimpleNamespace

    monkeypatch.setattr(cluster, "is_active", lambda: True)
    monkeypatch.setattr(
        cluster, "self_identity",
        lambda: SimpleNamespace(cluster_name="lab", node_id="n1"),
    )
    member = SimpleNamespace(
        node_id="n1", hostname="h", address="10.0.0.1", port=8765,
        reachable=True, last_source="peer_pull", tls=True,
    )
    monkeypatch.setattr(cluster, "members", lambda: [member])
    monkeypatch.setattr(cluster, "master_node_id", lambda: "n1")
    monkeypatch.setattr(cluster, "is_self_master", lambda: True)
    out = tools_advanced._cluster_status({})
    assert out["active"] and out["cluster_name"] == "lab"
    assert out["members"][0]["node_id"] == "n1" and "probe" not in out["members"][0]


# ---- overrides -------------------------------------------------------------


def test_get_config_overrides(monkeypatch):
    monkeypatch.setattr(
        overrides_store, "get_overrides_payload",
        lambda p, c: {"values": {"lr": 0.1}, "requested_gpus": 2},
    )
    out = tools_advanced._get_config_overrides({"project_dir": "/p", "config": "c.yaml"})
    assert out["values"]["lr"] == 0.1


def test_set_config_overrides_preview_then_commit(monkeypatch):
    saved = []
    monkeypatch.setattr(
        overrides_store, "set_overrides",
        lambda p, c, v, requested_gpus=None, **k: saved.append((p, c, v, requested_gpus)),
    )
    prop = tools_advanced._set_config_overrides(
        {"project_dir": "/p", "config": "c.yaml", "values": {"lr": 0.2}, "requested_gpus": 4}
    )
    assert isinstance(prop, Proposal) and saved == []  # preview only
    msg = prop.commit()
    assert saved == [("/p", "c.yaml", {"lr": 0.2}, 4)] and "saved overrides" in msg


def test_set_config_overrides_rejects_non_dict_values():
    with pytest.raises(ValueError):
        tools_advanced._set_config_overrides(
            {"project_dir": "/p", "config": "c.yaml", "values": "nope"}
        )


# ---- system prompt mentions the new capabilities ---------------------------


def test_system_prompt_mentions_new_sections():
    sp = runtime.SYSTEM_PROMPT
    for token in ("run_summary", "run_eval", "start_service(type=\"dataset\")",
                  "diloco_status", "query_model"):
        assert token in sp
