"""Bind-address policy: refuse non-loopback HTTP unless explicitly insecure."""

from __future__ import annotations

import ipaddress
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .config import TLSConfig

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
    cfg: "Optional[TLSConfig]" = None,
) -> None:
    """Refuse to start an HTTP server on a non-loopback bind without ``insecure``.

    No-op when:
      * host is loopback, or
      * TLS is enabled (the actual SSL kwargs are wired separately), or
      * the operator passed ``--insecure``.

    ``cfg`` (optional) lets us tailor the failure message to the
    actual state — if a cert is already on disk but ``enabled: false``,
    the operator needs ``--tls`` or ``tls enable``, not another
    ``tls init`` that would mint a redundant cert.
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

    # Try to give a state-aware error. Importing here (not at module
    # load) avoids a circular import with runtime.py.
    advice: list[str]
    if cfg is None:
        try:
            from .config import load_config

            cfg = load_config()
        except Exception:
            cfg = None  # type: ignore[assignment]
    insecure_line = (
        "  (Or pass --insecure to opt out — cleartext bearer tokens on the wire.)"
    )
    if cfg is not None and cfg.is_provisioned() and not cfg.enabled:
        advice = [
            "  A server cert is already provisioned but TLS is disabled in",
            f"  {cfg.config_file}.",
            "  Re-enable it for all servers:  forgather tls enable",
            "  Or just this invocation:       pass --tls",
            "  (Last resort, plaintext):      pass --insecure",
        ]
    elif cfg is not None and cfg.has_ca_authority() and not cfg.is_provisioned():
        advice = [
            "  A CA is present but no server cert. Re-issue one:",
            "    forgather tls renew --server",
            "  Or, if this host is a peer, install one from the CA holder:",
            "    forgather tls install --cert ... --key ... --ca ...",
            insecure_line,
        ]
    elif cfg is not None and not cfg.has_ca_authority():
        advice = [
            "  This host has no CA. If you're the cluster head:",
            "    forgather tls init",
            "  If you're a peer, get a cert from the CA holder and:",
            "    forgather tls install --cert server.crt --key server.key --ca ca.crt",
            insecure_line,
        ]
    else:
        advice = [
            "  Run 'forgather tls init' to provision a local CA + server cert.",
            insecure_line,
        ]
    raise TLSRequiredError(
        f"{service}: refusing to bind non-loopback host {host!r} without TLS.\n"
        + "\n".join(advice)
    )
