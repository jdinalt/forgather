"""Bearer + verify_tls attachment tests for the DiLoCo proxy (issue #90).

Validates that ``routes/diloco.py`` correctly:

* Reads the override ``X-Diloco-Auth-Token`` header from the caller and
  forwards it as ``Authorization: Bearer …``.
* Falls back to ``_token_for_local`` (JobRecord auto-lookup) when no
  override is provided.
* Falls back to the registry's ``find_token`` for user-added remotes.
* Skips upstream cert chain validation when the registry entry has
  ``verify_tls=False`` — the SSH-tunneled-remote opt-out.
* Sends *no* Authorization header when none of the sources have one,
  matching ``--no-auth`` servers.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from forgather_server import diloco_server_registry as reg
from forgather_server.routes import diloco as diloco_routes


@pytest.fixture
def app(tmp_path, monkeypatch):
    target = tmp_path / "diloco_server_registry.json"
    monkeypatch.setattr(reg, "diloco_server_registry_file", lambda: target)
    a = FastAPI()
    a.include_router(diloco_routes.router, prefix="/api")
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


class _Recorder:
    """Captures (path, headers, verify_kwarg) for each upstream request."""

    def __init__(self):
        self.requests = []
        self.verify_kwargs = []

    def install(self, monkeypatch):
        real = httpx.AsyncClient
        recorder = self

        def factory(*args, **kwargs):
            recorder.verify_kwargs.append(kwargs.get("verify"))
            kwargs.pop("verify", None)
            kwargs.pop("timeout", None)
            return real(
                transport=httpx.MockTransport(self._handler),
                **kwargs,
            )

        monkeypatch.setattr(diloco_routes.httpx, "AsyncClient", factory)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((str(request.url), dict(request.headers)))
        return httpx.Response(200, json={"status": "ok"})


# ---------------------------------------------------------------------------
# Header precedence
# ---------------------------------------------------------------------------


def test_override_header_wins(client, monkeypatch):
    """An incoming ``X-Diloco-Auth-Token`` flips into ``Authorization``
    and bypasses every other lookup source."""
    rec = _Recorder()
    rec.install(monkeypatch)
    monkeypatch.setattr(diloco_routes, "_local_servers", lambda: [])

    # Pre-register the URL so SSRF allowlists it.
    reg.add_entry(
        label="r", base_url="http://10.0.0.5:8512", auth_token="from-registry"
    )

    resp = client.get(
        "/api/diloco/server-status",
        params={"base": "http://10.0.0.5:8512"},
        headers={"X-Diloco-Auth-Token": "from-override"},
    )
    assert resp.status_code == 200
    # The override token must be the one that landed on the upstream.
    _, headers = rec.requests[-1]
    assert headers["authorization"] == "Bearer from-override"


def test_registry_token_used_for_remote(client, monkeypatch):
    """When no override is present and no local job matches, the proxy
    falls back to the registry's stored token."""
    rec = _Recorder()
    rec.install(monkeypatch)
    monkeypatch.setattr(diloco_routes, "_local_servers", lambda: [])
    monkeypatch.setattr(diloco_routes, "_token_for_local", lambda _b: None)

    reg.add_entry(
        label="r", base_url="http://10.0.0.6:8512", auth_token="from-registry"
    )

    resp = client.get(
        "/api/diloco/server-status",
        params={"base": "http://10.0.0.6:8512"},
    )
    assert resp.status_code == 200
    _, headers = rec.requests[-1]
    assert headers["authorization"] == "Bearer from-registry"


def test_no_auth_sends_no_authorization_header(client, monkeypatch):
    """When no source has a token, no Authorization header is sent —
    required for --no-auth upstreams that don't tolerate the field."""
    rec = _Recorder()
    rec.install(monkeypatch)
    monkeypatch.setattr(diloco_routes, "_local_servers", lambda: [])
    monkeypatch.setattr(diloco_routes, "_token_for_local", lambda _b: None)

    reg.add_entry(label="r", base_url="http://10.0.0.7:8512", auth_token="")

    resp = client.get(
        "/api/diloco/server-status",
        params={"base": "http://10.0.0.7:8512"},
    )
    assert resp.status_code == 200
    _, headers = rec.requests[-1]
    assert "authorization" not in headers


# ---------------------------------------------------------------------------
# verify_tls opt-out
# ---------------------------------------------------------------------------


def test_verify_tls_false_disables_chain_validation(client, monkeypatch):
    """A registry entry with verify_tls=False propagates verify=False to
    the httpx client — the SSH-tunneled-remote opt-out."""
    rec = _Recorder()
    rec.install(monkeypatch)
    monkeypatch.setattr(diloco_routes, "_local_servers", lambda: [])

    reg.add_entry(
        label="ssh-tunneled",
        base_url="https://10.0.0.8:8512",
        verify_tls=False,
    )

    resp = client.get(
        "/api/diloco/server-status",
        params={"base": "https://10.0.0.8:8512"},
    )
    assert resp.status_code == 200
    assert rec.verify_kwargs[-1] is False


def test_verify_tls_true_uses_httpx_verify(client, monkeypatch):
    """Default verify_tls=True path produces a non-False verify value
    (which may be True or an SSLContext depending on the cluster's
    TLS provisioning — both are 'verification on')."""
    rec = _Recorder()
    rec.install(monkeypatch)
    monkeypatch.setattr(diloco_routes, "_local_servers", lambda: [])

    reg.add_entry(
        label="real-remote",
        base_url="https://10.0.0.9:8512",
        verify_tls=True,
    )

    resp = client.get(
        "/api/diloco/server-status",
        params={"base": "https://10.0.0.9:8512"},
    )
    assert resp.status_code == 200
    # Verify is *not* False — it's either True or an SSLContext.
    assert rec.verify_kwargs[-1] is not False


# ---------------------------------------------------------------------------
# Control endpoint also attaches credentials
# ---------------------------------------------------------------------------


def test_control_endpoint_attaches_bearer(client, monkeypatch):
    """The control POST path follows the same precedence rules as GETs."""
    rec = _Recorder()
    rec.install(monkeypatch)
    monkeypatch.setattr(diloco_routes, "_local_servers", lambda: [])
    monkeypatch.setattr(diloco_routes, "_token_for_local", lambda _b: None)

    reg.add_entry(
        label="r",
        base_url="http://10.0.0.10:8512",
        auth_token="ctl-bearer",
    )
    resp = client.post(
        "/api/diloco/server-control/save_state",
        params={"base": "http://10.0.0.10:8512"},
        json={},
    )
    assert resp.status_code == 200
    _, headers = rec.requests[-1]
    assert headers["authorization"] == "Bearer ctl-bearer"
