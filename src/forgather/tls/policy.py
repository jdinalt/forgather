"""Bind-address policy: refuse non-loopback HTTP unless explicitly insecure."""

from __future__ import annotations

import ipaddress
import logging
from typing import Optional

log = logging.getLogger("forgather.tls")


LOOPBACK_HOSTNAMES = frozenset({"localhost", "ip6-localhost", "ip6-loopback"})


class TLSRequiredError(RuntimeError):
    """Raised when policy demands TLS but it isn't configured/enabled."""


def host_is_loopback(host: str) -> bool:
    """Classify a host string as loopback / non-loopback.

    Empty string and ``0.0.0.0`` / ``::`` count as non-loopback (they
    bind every interface). A hostname like ``localhost`` is loopback.
    Unresolvable hostnames fall back to non-loopback (fail safe).
    """
    if not host:
        return False
    h = host.strip().lower()
    if h in LOOPBACK_HOSTNAMES:
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    if ip.is_unspecified:
        return False
    return ip.is_loopback


def enforce_non_loopback_policy(
    host: str,
    *,
    tls_enabled: bool,
    insecure: bool,
    service: str = "server",
) -> None:
    """Refuse to start an HTTP server on a non-loopback bind without ``insecure``.

    No-op when:
      * host is loopback, or
      * TLS is enabled (the actual SSL kwargs are wired separately), or
      * the operator passed ``--insecure``.
    """
    if tls_enabled:
        return
    if host_is_loopback(host):
        return
    if insecure:
        log.warning(
            "%s: --insecure: serving plaintext HTTP on non-loopback host %s",
            service,
            host,
        )
        return
    raise TLSRequiredError(
        f"{service}: refusing to bind non-loopback host {host!r} without TLS.\n"
        "  Run 'forgather tls init' to provision a local CA + server cert,\n"
        "  or pass --insecure to opt out (cleartext bearer tokens on the wire)."
    )
