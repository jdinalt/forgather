"""Shared TLS support for Forgather servers.

A single per-host config under ``~/.config/forgather/tls/`` powers TLS
for all three servers (``forgather server``, ``dataset_server``,
``inference_server``). Operators run ``forgather tls init`` once;
servers and CLI clients then auto-pick up the same CA, cert, and
trust-bundle.

Public API (see :mod:`forgather.tls.config`, :mod:`forgather.tls.ca`,
:mod:`forgather.tls.policy`, :mod:`forgather.tls.discovery` for details):

* :func:`load_config` — read+resolve the shared config.
* :func:`is_enabled` — whether TLS should apply on this host.
* :func:`uvicorn_ssl_kwargs` — kwargs to pass to ``uvicorn.run``.
* :func:`httpx_verify` — value for httpx ``verify=`` (CA bundle path).
* :func:`client_scheme` — ``"https"`` or ``"http"`` for client URL builders.
* :func:`enforce_non_loopback_policy` — refuse non-loopback bind w/o TLS.
* :func:`tls_dir` — root config directory path.
* :class:`TLSRequiredError`, :class:`TLSConfigError`.
"""

from __future__ import annotations

from .config import (
    TLSConfig,
    TLSConfigError,
    load_config,
    tls_dir,
)
from .policy import (
    TLSRequiredError,
    enforce_non_loopback_policy,
    host_is_loopback,
)
from .runtime import (
    client_scheme,
    httpx_client_cert,
    httpx_verify,
    httpx_verify_for_url,
    is_enabled,
    uvicorn_ssl_kwargs,
)

__all__ = [
    "TLSConfig",
    "TLSConfigError",
    "TLSRequiredError",
    "client_scheme",
    "enforce_non_loopback_policy",
    "host_is_loopback",
    "httpx_client_cert",
    "httpx_verify",
    "httpx_verify_for_url",
    "is_enabled",
    "load_config",
    "tls_dir",
    "uvicorn_ssl_kwargs",
]
