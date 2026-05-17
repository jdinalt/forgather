"""Tests for the mTLS-aware uvicorn HTTP protocol shim (issue #31).

The protocol subclasses must:

* capture the peer's DER-encoded client cert in ``connection_made`` via
  ``transport.get_extra_info("ssl_object").getpeercert(binary_form=True)``;
* inject it into ``scope["extensions"]["forgather.tls"]`` whenever the parent
  parser assigns a fresh request scope (via our property setter).

These tests stub the transport with a minimal object that returns a
controllable ``ssl_object``. We don't drive uvicorn end-to-end here —
that's covered by the multi-node smoke test in ``scripts/`` — but we
do verify the scope-injection by exercising the property setter
directly with a freshly-built scope dict, mirroring what the parser
does at request time.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _event_loop():
    """uvicorn's HTTP protocol bases call asyncio.get_event_loop() in __init__.

    On Python 3.12 that raises when no loop is running (the implicit-loop
    creation behaviour was removed). Provide a fresh loop for each test.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        asyncio.set_event_loop(None)
        loop.close()


class _FakeSSL:
    def __init__(self, der: bytes | None):
        self._der = der

    def getpeercert(self, binary_form: bool = False):
        return self._der


class _FakeTransport:
    """Minimal asyncio.Transport stand-in covering get_extra_info()."""

    def __init__(
        self,
        ssl_object=None,
        sockname=("127.0.0.1", 8765),
        peername=("10.0.0.5", 54321),
    ):
        self._info = {
            "ssl_object": ssl_object,
            "sslcontext": object() if ssl_object is not None else None,
            "sockname": sockname,
            "peername": peername,
            "socket": MagicMock(),
        }

    def get_extra_info(self, key, default=None):
        return self._info.get(key, default)

    # Methods uvicorn's protocols may touch on the transport.
    def is_closing(self):  # pragma: no cover
        return False

    def write(self, data):  # pragma: no cover
        pass

    def close(self):  # pragma: no cover
        pass


def _make_protocol():
    """Build a ForgatherProtocol with a no-op uvicorn config."""
    from forgather_server.asgi_tls_protocol import ForgatherProtocol
    from uvicorn.config import Config
    from uvicorn.server import ServerState

    async def _stub_app(scope, receive, send):
        return

    cfg = Config(app=_stub_app, loop="asyncio", lifespan="off")
    cfg.load()
    state = ServerState()
    return ForgatherProtocol(config=cfg, server_state=state, app_state={})


class TestPeerCertCapture:
    def test_captures_peer_cert_in_connection_made(self):
        proto = _make_protocol()
        der = b"\x30\x82\x01\x00fake-cert-der-bytes"
        transport = _FakeTransport(ssl_object=_FakeSSL(der))

        proto.connection_made(transport)

        assert proto._peer_cert_der == der

    def test_no_peer_cert_when_handshake_returns_none(self):
        proto = _make_protocol()
        transport = _FakeTransport(ssl_object=_FakeSSL(None))

        proto.connection_made(transport)

        assert proto._peer_cert_der is None

    def test_no_peer_cert_on_plain_http(self):
        proto = _make_protocol()
        transport = _FakeTransport(ssl_object=None)

        proto.connection_made(transport)

        assert proto._peer_cert_der is None

    def test_getpeercert_raising_is_swallowed(self):
        proto = _make_protocol()

        class _RaisingSSL:
            def getpeercert(self, binary_form=False):
                raise ValueError("handshake not complete")

        transport = _FakeTransport(ssl_object=_RaisingSSL())
        proto.connection_made(transport)

        assert proto._peer_cert_der is None


class TestScopeInjection:
    """Verify the scope-setter injects TLS extension when cert is present."""

    def test_scope_setter_injects_tls_extension(self):
        proto = _make_protocol()
        der = b"some-cert-der"
        proto._peer_cert_der = der

        # Emulate what the parser does — assign a fresh scope dict.
        proto.scope = {"type": "http", "method": "GET", "path": "/x"}

        assert (
            proto.scope["extensions"]["forgather.tls"]["client_cert_verified"] is True
        )
        assert proto.scope["extensions"]["forgather.tls"]["client_cert_chain_der"] == [
            der
        ]

    def test_scope_setter_skips_injection_when_no_cert(self):
        proto = _make_protocol()
        # No cert captured -> no extension injected.
        assert proto._peer_cert_der is None

        proto.scope = {"type": "http", "method": "GET", "path": "/x"}

        assert (
            "extensions" not in proto.scope
            or "forgather.tls" not in proto.scope.get("extensions", {})
        )

    def test_scope_setter_preserves_other_extensions(self):
        """If the parser ever adds other extensions (e.g. http.response.push),
        our injection must merge into the existing dict, not clobber it."""
        proto = _make_protocol()
        proto._peer_cert_der = b"der"

        proto.scope = {
            "type": "http",
            "method": "GET",
            "path": "/x",
            "extensions": {"some.other": {"foo": "bar"}},
        }

        assert proto.scope["extensions"]["some.other"] == {"foo": "bar"}
        assert (
            proto.scope["extensions"]["forgather.tls"]["client_cert_verified"] is True
        )

    def test_scope_none_assignment_is_passthrough(self):
        """Parsers reset scope to None between requests; setter must allow that."""
        proto = _make_protocol()
        proto._peer_cert_der = b"der"

        proto.scope = None
        assert proto.scope is None
