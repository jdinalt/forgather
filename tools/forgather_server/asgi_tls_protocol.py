"""Custom uvicorn HTTP protocols that surface the peer's client cert.

Uvicorn (0.46) does not populate any TLS extension on the ASGI HTTP
scope, so application middleware can't tell whether the peer presented
a client certificate. We need that information to implement mutual-TLS
authentication for inter-node cluster calls (see issue #31).

This module ships thin subclasses of uvicorn's HTTP protocol classes
that:

* capture ``ssl_object.getpeercert(binary_form=True)`` in
  ``connection_made``, after the TLS handshake has completed;
* inject ``scope["extensions"]["forgather.tls"] = {...}`` whenever the
  parent class assigns a new request scope, via a property setter.
  The namespaced key avoids colliding with the in-flight ASGI ``tls``
  extension spec — when uvicorn ships native support, both can
  coexist and the middleware can prefer either.

Both ``H11Protocol`` and ``HttpToolsProtocol`` build the scope inline
in their request-parsing path; intercepting via a ``scope`` property
keeps us decoupled from the (different) parser internals.

The listener must be configured with ``ssl_cert_reqs=CERT_OPTIONAL``
and ``ssl_ca_certs=<bundle>`` (see :func:`forgather.tls.uvicorn_ssl_kwargs`).
With that posture, any cert that reaches the protocol has already
passed chain validation against the cluster CA — so the middleware
only needs to check *presence*, not validity.

``pick_protocol_class()`` returns the right subclass based on what's
installed in the current venv (``httptools`` if available, else
``h11``). Pass the result via ``uvicorn.run(..., http=<class>)``.
"""

from __future__ import annotations

import logging
from typing import Optional, Type

from uvicorn.protocols.http.h11_impl import H11Protocol

log = logging.getLogger("forgather_server.asgi_tls")


def _capture_peer_cert(transport) -> Optional[bytes]:
    """Pull the peer cert DER from the asyncio SSL transport, if any.

    Returns None for plain HTTP and for any failure path. Asyncio's SSL
    transport guarantees the handshake is complete before this is
    called, but ``getpeercert`` is defensive against being invoked
    before ready.
    """
    ssl_object = transport.get_extra_info("ssl_object")
    if ssl_object is None:
        return None
    try:
        der = ssl_object.getpeercert(binary_form=True)
    except Exception:
        return None
    return der or None


class _TLSScopeMixin:
    """Inject ``scope['extensions']['tls']`` whenever the parent sets a scope.

    Implemented via a ``scope`` property so we don't have to override
    the parser-specific request-handling method (which differs between
    h11 and httptools and is verbose to copy). Every time the parent
    does ``self.scope = {...}`` to build a new request scope, our
    setter routes through and adds the TLS extension if we captured a
    peer cert in :meth:`connection_made`.
    """

    # Initialized to None in __init__ so the property setter has
    # something to read on early scope assignments (which happen during
    # super().__init__ for some uvicorn versions).
    _peer_cert_der: Optional[bytes] = None

    def __init__(self, *args, **kwargs):
        # Set BEFORE super().__init__ so any scope assignment during
        # init (uvicorn currently does `self.scope: HTTPScope = None`
        # in __init__) finds the attribute populated.
        self._peer_cert_der = None
        self._scope = None
        super().__init__(*args, **kwargs)

    def connection_made(self, transport) -> None:  # type: ignore[override]
        super().connection_made(transport)
        self._peer_cert_der = _capture_peer_cert(transport)

    @property
    def scope(self):
        return self._scope

    @scope.setter
    def scope(self, value) -> None:
        if value is not None and self._peer_cert_der is not None:
            extensions = value.setdefault("extensions", {})
            # CERT_OPTIONAL + ssl_ca_certs means: if a cert was
            # presented, it has already passed chain validation against
            # the cluster CA. Middleware can treat presence as proof.
            # Namespaced under "forgather.tls" so we don't collide
            # with the official ASGI ``tls`` extension if uvicorn ever
            # ships native support.
            extensions["forgather.tls"] = {
                "client_cert_chain_der": [self._peer_cert_der],
                "client_cert_verified": True,
            }
        self._scope = value


class ForgatherH11Protocol(_TLSScopeMixin, H11Protocol):
    """h11-backed protocol with TLS-extension injection."""


def pick_protocol_class() -> Type:
    """Return the right protocol subclass for this venv.

    Prefers httptools (faster, what ``uvicorn[standard]`` ships) when
    available; falls back to h11 (uvicorn's pure-python dep).
    """
    try:
        from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol
    except ImportError:
        return ForgatherH11Protocol

    class ForgatherHttpToolsProtocol(_TLSScopeMixin, HttpToolsProtocol):
        """httptools-backed protocol with TLS-extension injection."""

    return ForgatherHttpToolsProtocol


# Convenience alias for callers that just want "the right class".
ForgatherProtocol = pick_protocol_class()
