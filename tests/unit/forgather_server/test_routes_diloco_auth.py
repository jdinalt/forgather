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


def test_upstream_401_carries_auth_failed_header(client, monkeypatch):
    """Closes #94 — when the upstream DiLoCo server returns 401, the
    proxy tags the response with ``X-Upstream-Auth-Failed: 1`` so the
    webui's fetch wrapper (auth.ts) doesn't bounce the operator to
    the login screen. Without this tag, every upstream 401 looks like
    a session-expired event."""
    real = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs.pop("verify", None)
        kwargs.pop("timeout", None)
        return real(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(401, json={"error": "unauthorized"})
            ),
            **kwargs,
        )

    monkeypatch.setattr(diloco_routes.httpx, "AsyncClient", _factory)
    monkeypatch.setattr(diloco_routes, "_local_servers", lambda: [])

    reg.add_entry(label="r", base_url="http://10.0.0.11:8512", auth_token="wrong")

    resp = client.get(
        "/api/diloco/server-status",
        params={"base": "http://10.0.0.11:8512"},
    )
    assert resp.status_code == 401
    assert resp.headers.get("x-upstream-auth-failed") == "1"


def test_upstream_200_does_not_carry_auth_failed_header(client, monkeypatch):
    """The opposite case — a clean 200 must NOT carry the header."""
    rec = _Recorder()
    rec.install(monkeypatch)
    monkeypatch.setattr(diloco_routes, "_local_servers", lambda: [])
    monkeypatch.setattr(diloco_routes, "_token_for_local", lambda _b: "tok")

    reg.add_entry(label="r", base_url="http://10.0.0.12:8512", auth_token="tok")

    resp = client.get(
        "/api/diloco/server-status",
        params={"base": "http://10.0.0.12:8512"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-upstream-auth-failed") is None


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


# ---------------------------------------------------------------------------
# _token_for_local matches LAN-routable URLs against local JobRecords
# ---------------------------------------------------------------------------


def _fake_job(host, port, token, routable_host=None, status="running"):
    """Build a JobRecord stand-in good enough for _token_for_local's filters."""
    return type(
        "FakeRec",
        (),
        {
            "queue_id": f"q-{port}",
            "job_type": "diloco_server",
            "status": status,
            "auth_token": token,
            "job_params": {
                "host": host,
                "port": port,
                **({"routable_host": routable_host} if routable_host else {}),
            },
        },
    )()


def test_token_for_local_matches_loopback_url_against_loopback_bind(monkeypatch):
    """Original case: loopback URL, loopback bind."""
    monkeypatch.setattr(
        diloco_routes.job_records,
        "list_records",
        lambda: [_fake_job("127.0.0.1", 8512, "tok-loop")],
    )
    assert diloco_routes._token_for_local("http://localhost:8512") == "tok-loop"
    assert diloco_routes._token_for_local("http://127.0.0.1:8512") == "tok-loop"


def test_token_for_local_matches_loopback_url_against_0000_bind(monkeypatch):
    """A 0.0.0.0-bound server is reachable via loopback — the local
    operator browsing on the same host should still get the token."""
    monkeypatch.setattr(
        diloco_routes.job_records,
        "list_records",
        lambda: [_fake_job("0.0.0.0", 8512, "tok-any")],
    )
    assert diloco_routes._token_for_local("http://localhost:8512") == "tok-any"


def test_token_for_local_matches_lan_url_against_routable_host(monkeypatch):
    """Regression test for the cross-host LAN browse case (the bug
    the user hit): server binds 0.0.0.0, scheduler stamps
    ``routable_host=192.168.9.43`` on the JobRecord, webui synthesizes
    the URL ``https://192.168.9.43:8512`` for the Job card, and the
    proxy needs to tie that back to the record to attach the
    bearer."""
    monkeypatch.setattr(
        diloco_routes.job_records,
        "list_records",
        lambda: [
            _fake_job(
                "0.0.0.0",
                8512,
                "tok-routable",
                routable_host="192.168.9.43",
            )
        ],
    )
    assert diloco_routes._token_for_local("https://192.168.9.43:8512") == "tok-routable"


def test_token_for_local_matches_lan_url_against_explicit_bind(monkeypatch):
    """The operator who explicitly typed ``--host 10.0.0.5`` should
    get the same match — no routable_host needed."""
    monkeypatch.setattr(
        diloco_routes.job_records,
        "list_records",
        lambda: [_fake_job("10.0.0.5", 8512, "tok-explicit")],
    )
    assert diloco_routes._token_for_local("https://10.0.0.5:8512") == "tok-explicit"


def test_token_for_local_returns_none_for_unrelated_lan_host(monkeypatch):
    """A LAN URL that doesn't match any record's host *or*
    routable_host returns None — the proxy falls through to the
    registry path."""
    monkeypatch.setattr(
        diloco_routes.job_records,
        "list_records",
        lambda: [
            _fake_job("0.0.0.0", 8512, "tok", routable_host="192.168.9.43"),
        ],
    )
    # Different host on the LAN — doesn't match.
    assert diloco_routes._token_for_local("https://192.168.9.99:8512") is None
    # Right host, different port — doesn't match.
    assert diloco_routes._token_for_local("https://192.168.9.43:9999") is None


def test_token_for_local_ignores_terminated_records(monkeypatch):
    """A just-died JobRecord shouldn't keep handing out its token —
    we filter on ``status in {starting, running}``."""
    monkeypatch.setattr(
        diloco_routes.job_records,
        "list_records",
        lambda: [
            _fake_job(
                "0.0.0.0",
                8512,
                "tok",
                routable_host="192.168.9.43",
                status="done",
            )
        ],
    )
    assert diloco_routes._token_for_local("https://192.168.9.43:8512") is None
