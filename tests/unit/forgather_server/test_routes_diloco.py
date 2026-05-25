"""Tests for routes/diloco.py — server discovery + status proxy + registry."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from forgather_server import diloco_server_registry as reg
from forgather_server.routes import diloco as diloco_routes


def _patch_async_client(monkeypatch, handler):
    """Make ``httpx.AsyncClient(...)`` use a MockTransport in this module.

    The route uses ``async with httpx.AsyncClient(...) as client`` so we
    need a real AsyncClient (with a MockTransport) — not a MagicMock
    that breaks ``__aenter__``/``__aexit__``.
    """
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        # Drop kwargs the real AsyncClient understands so we can swap in
        # the MockTransport without colliding with verify / timeout.
        kwargs.pop("verify", None)
        kwargs.pop("timeout", None)
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(diloco_routes.httpx, "AsyncClient", factory)


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A FastAPI app with the diloco router mounted and a tmp registry."""
    target = tmp_path / "diloco_server_registry.json"
    monkeypatch.setattr(reg, "diloco_server_registry_file", lambda: target)
    a = FastAPI()
    a.include_router(diloco_routes.router, prefix="/api")
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def no_local_servers(monkeypatch):
    """Stub the JobRecord scan so tests don't see real local instances."""
    monkeypatch.setattr(diloco_routes, "_local_servers", lambda: [])


# ---------------------------------------------------------------------------
# Registry endpoints
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_list_empty(self, client, no_local_servers):
        r = client.get("/api/diloco/registry")
        assert r.status_code == 200
        assert r.json() == []

    def test_add_and_list(self, client, no_local_servers):
        r = client.post(
            "/api/diloco/registry",
            json={"label": "WAN", "base_url": "http://10.0.0.1:8512"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["label"] == "WAN"
        assert body["base_url"] == "http://10.0.0.1:8512"
        assert body["has_auth_token"] is False
        assert body["verify_tls"] is True
        assert len(body["id"]) == 8

        listed = client.get("/api/diloco/registry").json()
        assert len(listed) == 1
        assert listed[0]["id"] == body["id"]

    def test_add_rejects_missing_url(self, client):
        r = client.post("/api/diloco/registry", json={"label": "x", "base_url": ""})
        assert r.status_code == 400

    def test_add_rejects_bad_scheme(self, client):
        r = client.post(
            "/api/diloco/registry",
            json={"label": "x", "base_url": "ftp://x:1"},
        )
        assert r.status_code == 400

    def test_delete(self, client, no_local_servers):
        added = client.post(
            "/api/diloco/registry",
            json={"label": "x", "base_url": "http://x:1"},
        ).json()
        r = client.delete(f"/api/diloco/registry/{added['id']}")
        assert r.status_code == 200
        assert client.get("/api/diloco/registry").json() == []

    def test_delete_missing_returns_404(self, client):
        r = client.delete("/api/diloco/registry/nope")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Unified servers list
# ---------------------------------------------------------------------------


class TestServersList:
    def test_lists_registered(self, client, no_local_servers):
        client.post(
            "/api/diloco/registry",
            json={"label": "WAN", "base_url": "http://10.0.0.1:8512"},
        )
        r = client.get("/api/diloco/servers")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 1
        assert body[0]["source"] == "registered"
        assert body[0]["base_url"] == "http://10.0.0.1:8512"
        assert body[0]["id"].startswith("registered:")

    def test_lists_local(self, client, monkeypatch):
        """Stub a fake JobRecord so the local-servers helper has work to do."""

        class FakeRec:
            queue_id = "q1"
            job_type = "diloco_server"
            config = "diloco:8512"
            status = "running"
            job_params = {"host": "127.0.0.1", "port": 8512}

        monkeypatch.setattr(
            diloco_routes.job_records, "list_records", lambda: [FakeRec()]
        )
        r = client.get("/api/diloco/servers")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 1
        assert body[0]["source"] == "local"
        assert body[0]["queue_id"] == "q1"
        assert body[0]["base_url"] == "http://127.0.0.1:8512"
        assert body[0]["alive"] is True

    def test_host_0000_is_remapped(self, client, monkeypatch):
        class FakeRec:
            queue_id = "q1"
            job_type = "diloco_server"
            config = "diloco:8512"
            status = "running"
            job_params = {"host": "0.0.0.0", "port": 8512}

        monkeypatch.setattr(
            diloco_routes.job_records, "list_records", lambda: [FakeRec()]
        )
        body = client.get("/api/diloco/servers").json()
        assert body[0]["base_url"] == "http://localhost:8512"


# ---------------------------------------------------------------------------
# Status / info / control proxy
# ---------------------------------------------------------------------------


class TestProxy:
    def test_status_loopback_passes_through(
        self, client, no_local_servers, monkeypatch
    ):
        """Loopback is always allowed (no need to register)."""
        upstream_payload = {"status": "running", "mode": "sync"}

        def fake_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/status"
            return httpx.Response(200, json=upstream_payload)

        _patch_async_client(monkeypatch, fake_handler)
        r = client.get(
            "/api/diloco/server-status",
            params={"base": "http://127.0.0.1:8512"},
        )
        assert r.status_code == 200
        assert r.json() == upstream_payload

    def test_status_refused_for_unknown_remote(self, client, no_local_servers):
        r = client.get(
            "/api/diloco/server-status",
            params={"base": "http://10.0.0.99:8512"},
        )
        assert r.status_code == 403
        assert "registry" in r.json()["detail"].lower()

    def test_status_allowed_after_registering(
        self, client, no_local_servers, monkeypatch
    ):
        client.post(
            "/api/diloco/registry",
            json={"label": "WAN", "base_url": "http://10.0.0.99:8512"},
        )

        def fake_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "running"})

        _patch_async_client(monkeypatch, fake_handler)
        r = client.get(
            "/api/diloco/server-status",
            params={"base": "http://10.0.0.99:8512"},
        )
        assert r.status_code == 200

    def test_info_proxies(self, client, no_local_servers, monkeypatch):
        upstream = {
            "output_dir": "/tmp/m",
            "num_parameters": 1234,
            "expected_client_settings": {"sync_every": None},
        }

        def fake_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/info"
            return httpx.Response(200, json=upstream)

        _patch_async_client(monkeypatch, fake_handler)
        r = client.get(
            "/api/diloco/server-info",
            params={"base": "http://127.0.0.1:8512"},
        )
        assert r.status_code == 200
        assert r.json() == upstream

    def test_control_rejects_unknown_action(self, client, no_local_servers):
        r = client.post(
            "/api/diloco/server-control/eat_my_homework",
            params={"base": "http://127.0.0.1:8512"},
            json={},
        )
        assert r.status_code == 400

    def test_control_proxies_known_action(self, client, no_local_servers, monkeypatch):
        captured = {}

        def fake_handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["method"] = request.method
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"status": "ok"})

        _patch_async_client(monkeypatch, fake_handler)
        r = client.post(
            "/api/diloco/server-control/kick_worker",
            params={"base": "http://127.0.0.1:8512"},
            json={"worker_id": "w1"},
        )
        assert r.status_code == 200
        assert captured["path"] == "/control/kick_worker"
        assert captured["method"] == "POST"
        assert captured["body"] == {"worker_id": "w1"}

    def test_upstream_unreachable_returns_502(
        self, client, no_local_servers, monkeypatch
    ):
        def fake_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        _patch_async_client(monkeypatch, fake_handler)
        r = client.get(
            "/api/diloco/server-status",
            params={"base": "http://127.0.0.1:8512"},
        )
        assert r.status_code == 502
