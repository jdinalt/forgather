"""mTLS coverage for DiLoCo (issue #90).

When the DiLoCo server is started with an SSL context that loads a
cluster CA bundle and ``verify_mode = ssl.CERT_OPTIONAL``, a client
that presents a CA-signed cert at the TLS handshake is treated as
authenticated *regardless* of whether it carries a bearer token.
This matches the ``_PEER_ALLOWED_PATHS`` pattern in
``tools/forgather_server/auth.py`` and means cluster peers don't need
to share the per-server token to talk to each other.

Tests:

* mTLS-only client (cluster cert, no bearer) → 200.
* No-cert + no-bearer client over TLS → 401.
* No-cert + matching-bearer client over TLS → 200 (bearer path still
  works on a TLS server).
* :func:`peer_cert_authenticated` unit-tests on a fake handler.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pytest
import torch

from forgather.ml.diloco.auth import (
    authenticate_request,
    peer_cert_authenticated,
)
from forgather.ml.diloco.server import DiLoCoServer
from forgather.tls import load_config
from forgather.tls.ca import (
    create_ca,
    install_server_cert,
    mint_server_cert,
    rebuild_bundle,
)
from forgather.tls.config import save_config
from forgather.tls.runtime import stdlib_ssl_context, urllib_ssl_context

from .conftest import make_initial_checkpoint


@pytest.fixture
def tls_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGATHER_TLS_DIR", str(tmp_path / "tls"))
    return tmp_path / "tls"


def _provisioned_cfg(tls_root):
    cfg = load_config()
    cfg.enabled = True
    cfg.san_hostnames = ["localhost"]
    cfg.san_ips = ["127.0.0.1"]
    create_ca(cfg, common_name="Test CA")
    minted = mint_server_cert(cfg, hostnames=["localhost"], ips=["127.0.0.1"])
    install_server_cert(cfg, minted)
    rebuild_bundle(cfg)
    save_config(cfg)
    return load_config()


def _state_dict():
    torch.manual_seed(0)
    return {"layer.weight": torch.randn(4, 4)}


@pytest.fixture
def tls_server(tmp_path, tls_root):
    """DiLoCo server with TLS + bearer + mTLS-capable handshake."""
    cfg = _provisioned_cfg(tls_root)
    ctx = stdlib_ssl_context()
    assert ctx is not None
    ckpt = make_initial_checkpoint(_state_dict(), tmp_path)
    s = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=1,
        port=0,
        heartbeat_timeout=0,
        auth_token="bearer-fallback",
        ssl_context=ctx,
    )
    s.start()
    time.sleep(0.2)
    yield s, cfg
    s.stop()


# ---------------------------------------------------------------------------
# End-to-end mTLS handshake
# ---------------------------------------------------------------------------


def test_mtls_client_skips_bearer(tls_server):
    """Client presenting the cluster cert at handshake is accepted
    without a bearer token. The skip-bearer path is what makes the
    cluster's inter-peer story tolerable — we don't have to share
    each server's per-port token."""
    server, _cfg = tls_server
    # urllib_ssl_context loads cfg.server_cert+cfg.server_key as a
    # client cert when provisioned. The server's CA bundle includes
    # the same cluster CA, so the cert chains and the handshake
    # succeeds.
    client_ctx = urllib_ssl_context()
    assert client_ctx is not None
    url = f"https://localhost:{server.port}/status"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, context=client_ctx, timeout=5) as resp:
        assert resp.status == 200


def test_no_cert_no_bearer_over_tls_returns_401(tls_server):
    """A client over TLS but with neither a client cert nor a bearer
    token is rejected. mTLS skip-bearer is an *opt-in* alternative,
    not a free pass."""
    server, cfg = tls_server
    # Build a verify-only context: trust the cluster CA but do NOT
    # load this node's cert+key as client identity.
    import ssl

    ctx = ssl.create_default_context(cafile=str(cfg.effective_bundle()))
    ctx.check_hostname = False
    # Note: cfg.is_provisioned() is True for the test cluster, so
    # urllib_ssl_context would normally load a cert. We bypass that
    # by building a verify-only context inline.
    url = f"https://localhost:{server.port}/status"
    req = urllib.request.Request(url, method="GET")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, context=ctx, timeout=5)
    assert exc_info.value.code == 401


def test_no_cert_with_matching_bearer_over_tls_returns_200(tls_server):
    """The bearer-token path is unchanged on a TLS server — clients
    that don't have a cluster cert can still authenticate with the
    per-port token."""
    server, cfg = tls_server
    import ssl

    ctx = ssl.create_default_context(cafile=str(cfg.effective_bundle()))
    ctx.check_hostname = False
    url = f"https://localhost:{server.port}/status"
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"Authorization": "Bearer bearer-fallback"},
    )
    with urllib.request.urlopen(req, context=ctx, timeout=5) as resp:
        assert resp.status == 200


# ---------------------------------------------------------------------------
# Unit-level: peer_cert_authenticated / authenticate_request
# ---------------------------------------------------------------------------


class _Capture:
    def __init__(self):
        self.data = b""

    def write(self, b):
        self.data += b


class _FakeConn:
    """Stand-in for an SSLSocket on a connection: returns a chosen
    ``getpeercert`` result."""

    def __init__(self, cert=None, raises=None):
        self._cert = cert
        self._raises = raises

    def getpeercert(self):
        if self._raises is not None:
            raise self._raises
        return self._cert


class _FakeHandler:
    def __init__(self, connection=None, auth_header=None):
        self.connection = connection
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


def test_mtls_control_plus_cleartext_bulk(tmp_path, tls_root):
    """Recommended production posture (per the design doc):
    control plane on TLS + mTLS, bulk plane on cleartext + no-auth.
    End-to-end: mTLS register, then /global_params via the cleartext
    bulk listener."""
    import socket

    cfg = _provisioned_cfg(tls_root)
    ctx = stdlib_ssl_context()
    assert ctx is not None
    # Pick a free port for the cleartext bulk listener.
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    bulk_port = s.getsockname()[1]
    s.close()

    ckpt = make_initial_checkpoint(_state_dict(), tmp_path)
    server = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=1,
        port=0,
        heartbeat_timeout=0,
        auth_token="bearer-fallback",
        ssl_context=ctx,
        bulk_port=bulk_port,
        bulk_ssl_context=None,  # cleartext bulk
        bulk_auth_enabled=False,  # no bearer on bulk
    )
    server.start()
    time.sleep(0.2)
    try:
        # Register over mTLS — uses the cluster client cert; no bearer.
        client_ctx = urllib_ssl_context()
        body = json.dumps(
            {
                "worker_id": "alpha",
                "hostname": "test",
                "param_shapes": {"layer.weight": [4, 4]},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"https://localhost:{server.port}/register",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, context=client_ctx, timeout=5) as resp:
            assert resp.status == 200
            advertised = resp.headers.get("X-Forgather-Bulk-Url")
            assert advertised == server.get_bulk_url()
            assert advertised.startswith("http://")  # cleartext

        # Fetch global params on the cleartext bulk listener — no auth.
        bulk_req = urllib.request.Request(
            f"http://localhost:{bulk_port}/global_params",
            method="GET",
        )
        with urllib.request.urlopen(bulk_req, timeout=5) as resp:
            assert resp.status == 200

        # And confirm the control port still requires auth — the
        # bearer fallback should reject anonymous callers even
        # though mTLS clients get in.
        import ssl as _ssl

        verify_only = _ssl.create_default_context(cafile=str(cfg.effective_bundle()))
        verify_only.check_hostname = False
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(
                f"https://localhost:{server.port}/status",
                context=verify_only,
                timeout=5,
            )
        assert exc_info.value.code == 401
    finally:
        server.stop()


def test_peer_cert_authenticated_no_connection():
    """No connection attribute → not authenticated (defensive default)."""
    h = _FakeHandler(connection=None)
    assert peer_cert_authenticated(h) is False


def test_peer_cert_authenticated_cleartext_socket():
    """Plain socket (no getpeercert) → not authenticated."""

    class _PlainSocket:
        pass

    h = _FakeHandler(connection=_PlainSocket())
    assert peer_cert_authenticated(h) is False


def test_peer_cert_authenticated_no_cert_presented():
    """SSLSocket with empty cert dict → not authenticated."""
    h = _FakeHandler(connection=_FakeConn(cert={}))
    assert peer_cert_authenticated(h) is False


def test_peer_cert_authenticated_with_cert():
    """Populated cert dict (CA-signed at handshake) → authenticated."""
    h = _FakeHandler(connection=_FakeConn(cert={"subject": (("CN", "peer"),)}))
    assert peer_cert_authenticated(h) is True


def test_peer_cert_authenticated_swallows_oserror():
    """getpeercert raising OSError (socket torn down) returns False
    rather than propagating — auth path should fail closed gracefully."""
    h = _FakeHandler(connection=_FakeConn(raises=OSError("socket closed")))
    assert peer_cert_authenticated(h) is False


def test_authenticate_request_prefers_peer_cert():
    """Peer cert → authenticated; bearer header is irrelevant.

    This is the case where a cluster peer talks to us with mTLS but
    didn't bother to send a bearer. We accept it."""
    h = _FakeHandler(
        connection=_FakeConn(cert={"subject": (("CN", "peer"),)}),
        auth_header=None,
    )
    assert authenticate_request(h, "expected-token") is True
    assert h._sent_status is None  # no 401 sent


def test_authenticate_request_falls_back_to_bearer():
    """No peer cert + matching bearer → authenticated."""
    h = _FakeHandler(
        connection=_FakeConn(cert={}),
        auth_header="Bearer expected-token",
    )
    assert authenticate_request(h, "expected-token") is True
    assert h._sent_status is None


def test_authenticate_request_no_cert_no_bearer_401():
    """No peer cert + no bearer → 401."""
    h = _FakeHandler(connection=_FakeConn(cert={}), auth_header=None)
    assert authenticate_request(h, "expected-token") is False
    assert h._sent_status == 401
