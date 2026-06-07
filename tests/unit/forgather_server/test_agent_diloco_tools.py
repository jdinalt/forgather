"""DiLoCo monitor/control tools + the _diloco discovery/query helper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from forgather_server.agent import _diloco, tools_diloco
from forgather_server.agent.registry import CONFIRM, EXTENDED, READ, Proposal, ToolRegistry


def _reg():
    reg = ToolRegistry()
    tools_diloco.register_all(reg)
    return reg


class _FakeClient:
    last = {}

    def __init__(self, server_addr, token=None, verify_tls=True, timeout=600, max_retries=3):
        _FakeClient.last = {"addr": server_addr, "token": token}

    def get_status(self):
        return {"round": 3, "synced_workers": 2}

    def get_known_workers(self):
        return {"workers": [{"worker_id": "w0", "running": True}, {"worker_id": "w1", "running": False}]}

    def save_state(self):
        return {"status": "saved"}

    def shutdown(self):
        return {"status": "stopping"}

    def relay_command(self, command, worker_id=None):
        return {"status": "queued", "command": command, "worker_id": worker_id}


@pytest.fixture
def fake_one(monkeypatch):
    """One discoverable diloco server, backed by _FakeClient."""
    monkeypatch.setattr(_diloco, "DiLoCoClient", _FakeClient)
    entry = {
        "id": "abc123", "label": "diloco:8512", "base_url": "http://h:8512",
        "source": "local", "token": "tok", "verify_tls": True, "healthy": True,
    }
    monkeypatch.setattr(_diloco, "_discover", lambda: [entry])
    return entry


# ---- registration ----------------------------------------------------------


def test_registration_risk_and_tier():
    by = {s.name: s for s in _reg().specs()}
    assert by["list_diloco_servers"].risk == READ and by["list_diloco_servers"].tier == EXTENDED
    assert by["diloco_status"].risk == READ and by["diloco_status"].tier == EXTENDED
    assert by["diloco_control"].risk == CONFIRM and by["diloco_control"].tier == EXTENDED


# ---- helper ----------------------------------------------------------------


def test_discover_merges_and_reachability(monkeypatch):
    import forgather_server.cluster_diloco_inventory as cdi

    local = SimpleNamespace(
        server_id="s1", label="d:8512", base_url="http://h:8512",
        source="local", auth_token="t", verify_tls=True,
    )
    monkeypatch.setattr(cdi, "local_servers", lambda: [local])
    monkeypatch.setattr(cdi.master_inventory, "servers_snapshot", lambda: [])
    monkeypatch.setattr(_diloco, "DiLoCoClient", _FakeClient)
    servers = _diloco.list_servers()
    assert servers[0]["id"] == "s1" and servers[0]["reachable"] is True
    assert "token" not in servers[0]  # token projected out


def test_pick_no_server_raises(monkeypatch):
    monkeypatch.setattr(_diloco, "_discover", lambda: [])
    with pytest.raises(ValueError):
        _diloco.status()


def test_status_combines_status_and_workers(fake_one):
    out = _diloco.status(server_id="abc123")
    assert out["status"]["round"] == 3
    assert {w["worker_id"] for w in out["workers"]} == {"w0", "w1"}
    assert out["server"]["id"] == "abc123"


def test_control_save_state(fake_one):
    out = _diloco.control("abc123", "save_state")
    assert out["result"]["status"] == "saved"


def test_control_relay_requires_command(fake_one):
    with pytest.raises(ValueError):
        _diloco.control("abc123", "relay")


def test_control_relay_dispatches(fake_one):
    out = _diloco.control("abc123", "relay", command="abort", worker_id="w0")
    assert out["result"]["command"] == "abort" and out["result"]["worker_id"] == "w0"


# ---- tool wrappers ---------------------------------------------------------


def test_diloco_control_unknown_action():
    with pytest.raises(ValueError):
        tools_diloco._diloco_control({"action": "explode"})


def test_diloco_control_relay_needs_command():
    with pytest.raises(ValueError):
        tools_diloco._diloco_control({"action": "relay"})


def test_diloco_control_preview_then_commit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        _diloco, "control",
        lambda sid, action, command=None, worker_id=None: calls.append((sid, action, command))
        or {"server": {"id": sid or "x"}, "result": {"ok": True}},
    )
    prop = tools_diloco._diloco_control({"server_id": "abc", "action": "shutdown"})
    assert isinstance(prop, Proposal) and calls == []  # preview: no side effect
    msg = prop.commit()
    assert calls == [("abc", "shutdown", None)] and "shutdown" in msg
