"""Tests for the bearer-token auth on the DiLoCo HTTP wire (issue #90).

Cover the request-time verifier in
:mod:`forgather.ml.diloco.auth` and its integration with
:class:`forgather.ml.diloco.server.DiLoCoServer`:

* No-token / no-bearer-header → 401 + ``WWW-Authenticate``.
* Wrong token → 401.
* Matching token → 200.
* ``/health`` is open even when auth is enabled (liveness probes).
* ``auth_token=None`` keeps the legacy "auth disabled" semantics so
  pre-#90 callers (in-process workers, older tests) still work.
* The per-port token file round-trip mirrors dataset_server's pattern:
  the loopback discovery helper finds the token without env or argv.

The wire is HTTP (not HTTPS) — TLS coverage lives in ``test_tls.py``
in the shared package; this file is auth-only so the cases stay
isolated.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pytest
import torch

from forgather.ml.diloco.auth import (
    SERVICE_REALM,
    read_standalone_token,
    standalone_token_file,
    verify_bearer,
    write_standalone_token,
)
from forgather.ml.diloco.server import DiLoCoServer

from .conftest import make_initial_checkpoint


def _state_dict():
    torch.manual_seed(42)
    return {
        "embedding.weight": torch.randn(8, 4),
        "layer.weight": torch.randn(4, 4),
    }


@pytest.fixture
def server_with_token(tmp_path):
    """DiLoCoServer started with a fixed bearer token."""
    ckpt = make_initial_checkpoint(_state_dict(), tmp_path)
    s = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=1,
        port=0,
        heartbeat_timeout=0,
        auth_token="topsecret-test-token",
    )
    s.start()
    time.sleep(0.2)
    yield s
    s.stop()


@pytest.fixture
def server_no_auth(tmp_path):
    """DiLoCoServer with auth disabled (auth_token=None)."""
    ckpt = make_initial_checkpoint(_state_dict(), tmp_path)
    s = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=1,
        port=0,
        heartbeat_timeout=0,
    )
    s.start()
    time.sleep(0.2)
    yield s
    s.stop()


def _get(url, headers=None, timeout=5):
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    return urllib.request.urlopen(req, timeout=timeout)


# ---------------------------------------------------------------------------
# 401 paths
# ---------------------------------------------------------------------------


def test_missing_authorization_returns_401(server_with_token):
    url = f"http://localhost:{server_with_token.port}/status"
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _get(url)
    err = exc_info.value
    assert err.code == 401
    assert err.headers.get("WWW-Authenticate") == f'Bearer realm="{SERVICE_REALM}"'
    body = json.loads(err.read().decode("utf-8"))
    assert body["realm"] == SERVICE_REALM


def test_wrong_token_returns_401(server_with_token):
    url = f"http://localhost:{server_with_token.port}/status"
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _get(url, headers={"Authorization": "Bearer not-the-token"})
    assert exc_info.value.code == 401


def test_malformed_authorization_returns_401(server_with_token):
    """Header that isn't ``Bearer <token>`` (e.g. Basic auth) is rejected."""
    url = f"http://localhost:{server_with_token.port}/status"
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _get(url, headers={"Authorization": "Basic ZGVtbzpkZW1v"})
    assert exc_info.value.code == 401


def test_post_endpoint_also_gated(server_with_token):
    """The bearer check applies to POST endpoints, not just GET."""
    url = f"http://localhost:{server_with_token.port}/heartbeat"
    body = json.dumps({"worker_id": "alpha"}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    assert exc_info.value.code == 401


# ---------------------------------------------------------------------------
# 200 paths
# ---------------------------------------------------------------------------


def test_correct_bearer_returns_200(server_with_token):
    url = f"http://localhost:{server_with_token.port}/status"
    resp = _get(url, headers={"Authorization": "Bearer topsecret-test-token"})
    assert resp.status == 200
    body = json.loads(resp.read().decode("utf-8"))
    assert "status" in body


def test_health_endpoint_is_open(server_with_token):
    """``/health`` is intentionally exempt so liveness probes work
    without sharing the bearer token. Returns 200 even without a header."""
    url = f"http://localhost:{server_with_token.port}/health"
    resp = _get(url)
    assert resp.status == 200
    body = json.loads(resp.read().decode("utf-8"))
    assert body["status"] == "ok"


def test_no_auth_mode_accepts_unauth_requests(server_no_auth):
    """Legacy / trusted-LAN deployments with ``auth_token=None``
    still serve everything without a header — required for the
    backwards-compat surface."""
    url = f"http://localhost:{server_no_auth.port}/status"
    resp = _get(url)
    assert resp.status == 200


# ---------------------------------------------------------------------------
# verify_bearer unit (no network)
# ---------------------------------------------------------------------------


class _FakeHandler:
    """Minimal BaseHTTPRequestHandler stand-in for unit testing verify_bearer."""

    def __init__(self, auth_header=None):
        self.headers = {"Authorization": auth_header} if auth_header else {}
        self._sent_status = None
        self._sent_headers = {}
        self.wfile = _Capture()

    def send_response(self, code):
        self._sent_status = code

    def send_header(self, k, v):
        self._sent_headers[k] = v

    def end_headers(self):
        pass


class _Capture:
    def __init__(self):
        self.data = b""

    def write(self, b):
        self.data += b


def test_verify_bearer_with_none_token_passes():
    """auth_token=None / "" → verifier always returns True; no 401."""
    h = _FakeHandler(auth_header=None)
    assert verify_bearer(h, None) is True
    assert h._sent_status is None

    h = _FakeHandler(auth_header=None)
    assert verify_bearer(h, "") is True
    assert h._sent_status is None


def test_verify_bearer_constant_time_compare():
    """A near-miss token still 401s (no early-return on first mismatch)."""
    expected = "a" * 64
    near_miss = "a" * 63 + "b"
    h = _FakeHandler(auth_header=f"Bearer {near_miss}")
    assert verify_bearer(h, expected) is False
    assert h._sent_status == 401


def test_verify_bearer_case_insensitive_scheme():
    """'bearer ' vs 'Bearer ' both accepted (some HTTP clients lowercase)."""
    expected = "abcd"
    h = _FakeHandler(auth_header="bearer abcd")
    assert verify_bearer(h, expected) is True


# ---------------------------------------------------------------------------
# Per-port token file
# ---------------------------------------------------------------------------


def test_token_file_round_trip(tmp_path, monkeypatch):
    """write_standalone_token → standalone_token_file → reads same value
    with mode 0600."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = write_standalone_token(9999, "abc123")
    assert path == standalone_token_file(9999)
    assert path.read_text() == "abc123"
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600


def test_token_file_overwrite_keeps_mode_0600(tmp_path, monkeypatch):
    """Rewriting an existing token file (e.g. --regen-token) preserves
    the 0600 mode so a careless rotation can't expose the token."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = write_standalone_token(8000, "first")
    write_standalone_token(8000, "second")
    assert path.read_text() == "second"
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600


def test_loopback_token_autodiscovery(tmp_path, monkeypatch):
    """Loopback URL → reads the matching per-port file."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    write_standalone_token(7000, "found-via-loopback")
    assert read_standalone_token("http://127.0.0.1:7000/status") == "found-via-loopback"
    assert read_standalone_token("https://localhost:7000/info") == "found-via-loopback"


def test_non_loopback_url_does_not_autodiscover(tmp_path, monkeypatch):
    """Remote URLs MUST NOT pick up local token files; the token is
    bound to the loopback trust boundary."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    write_standalone_token(7000, "should-not-leak")
    assert read_standalone_token("http://10.0.0.5:7000/status") is None


def test_missing_token_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert read_standalone_token("http://127.0.0.1:55555/status") is None
