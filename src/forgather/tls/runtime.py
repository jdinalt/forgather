"""Runtime helpers: uvicorn ssl kwargs, httpx verify, client scheme picker."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from .config import TLSConfig, load_config
from .policy import host_is_loopback

log = logging.getLogger("forgather.tls")


def is_enabled(cfg: Optional[TLSConfig] = None) -> bool:
    """Whether TLS should apply by default on this host.

    True when ``config.yaml`` exists, ``enabled: true``, AND server
    cert + key files are present. Servers can still override per-invocation
    with ``--tls`` / ``--no-tls``.
    """
    cfg = cfg or load_config()
    return cfg.enabled and cfg.is_provisioned()


def _resolve_state(
    args: Optional[argparse.Namespace], cfg: Optional[TLSConfig]
) -> tuple[TLSConfig, bool, Optional[str], Optional[str]]:
    """Combine CLI flags with shared config.

    Returns ``(cfg, tls_on, cert_path, key_path)``. CLI ``--tls`` /
    ``--no-tls`` win over config; ``--tls-cert`` / ``--tls-key`` override
    paths.
    """
    cfg = cfg or load_config()
    tls_flag: Optional[bool] = None
    cert_override: Optional[str] = None
    key_override: Optional[str] = None
    if args is not None:
        tls_flag = getattr(args, "tls", None)
        if tls_flag is None and getattr(args, "no_tls", False):
            tls_flag = False
        cert_override = getattr(args, "tls_cert", None)
        key_override = getattr(args, "tls_key", None)

    if tls_flag is True:
        on = True
    elif tls_flag is False:
        on = False
    else:
        on = cfg.enabled

    cert = cert_override or (str(cfg.server_cert) if on else None)
    key = key_override or (str(cfg.server_key) if on else None)
    return cfg, on, cert, key


def uvicorn_ssl_kwargs(
    args: Optional[argparse.Namespace] = None, cfg: Optional[TLSConfig] = None
) -> dict:
    """Return ``{"ssl_keyfile": …, "ssl_certfile": …}`` or ``{}``.

    Empty dict ⇒ caller skips TLS for this invocation. Raises
    :class:`FileNotFoundError` if TLS is requested but cert/key are
    missing from disk.
    """
    cfg, on, cert, key = _resolve_state(args, cfg)
    if not on:
        return {}
    if not cert or not key:
        raise FileNotFoundError(
            "TLS enabled but server cert/key not configured "
            "(run 'forgather tls init' or pass --tls-cert/--tls-key)"
        )
    if not Path(cert).is_file():
        raise FileNotFoundError(f"TLS cert not found: {cert}")
    if not Path(key).is_file():
        raise FileNotFoundError(f"TLS key not found: {key}")
    return {"ssl_keyfile": key, "ssl_certfile": cert}


def is_tls_active(
    args: Optional[argparse.Namespace] = None, cfg: Optional[TLSConfig] = None
) -> bool:
    """Whether the server will actually serve TLS for this invocation."""
    _, on, cert, key = _resolve_state(args, cfg)
    if not on:
        return False
    return bool(cert and key and Path(cert).is_file() and Path(key).is_file())


def httpx_verify(cfg: Optional[TLSConfig] = None) -> object:
    """Value to pass to httpx's ``verify=`` for talking to forgather peers.

    Returns an ``ssl.SSLContext`` carrying the shared CA bundle when
    one exists. By default the context is built with
    ``check_hostname = False`` because forgather's threat model on a
    LAN is "trust any leaf signed by my CA" — DHCP-issued IPs and
    ephemeral hostnames make RFC-6125 hostname verification mostly
    theatre, and an attacker who can mint a CA-signed cert can also
    pick whatever hostname/IP they like. Flip
    ``verify_hostname: true`` in ``config.yaml`` for strict
    hostname-SAN matching (e.g. public-DNS clusters).

    Returns ``True`` (system trust) only when no bundle exists. The
    forgather peer will then fail-closed against the system trust
    store, which is the correct failure mode — we never silently
    disable verification.
    """
    import ssl

    cfg = cfg or load_config()
    bundle = cfg.effective_bundle()
    if bundle is None:
        if cfg.enabled:
            log.warning(
                "TLS enabled in %s but no CA bundle present at %s — "
                "forgather peer connections will fall back to the "
                "system trust store and almost certainly fail to verify. "
                "Run 'forgather tls init' or import a peer CA via "
                "'forgather tls import-ca'.",
                cfg.config_file,
                cfg.ca_bundle,
            )
        return True
    ctx = ssl.create_default_context(cafile=str(bundle))
    if not cfg.verify_hostname:
        # Chain validation stays on (CERT_REQUIRED is the default for
        # create_default_context); we just don't insist that the SAN
        # matches the URL hostname. The chain check is the actual
        # security boundary on a private CA.
        ctx.check_hostname = False
    return ctx


def httpx_verify_for_url(
    url: str, cfg: Optional[TLSConfig] = None
) -> object:
    """Same as :func:`httpx_verify` but returns ``False`` for plain ``http://``.

    Saves an unnecessary file-system check when the URL won't be using TLS.
    """
    if not url.lower().startswith("https"):
        return True
    return httpx_verify(cfg)


def client_scheme(
    host: str = "127.0.0.1", cfg: Optional[TLSConfig] = None
) -> str:
    """Default URL scheme for clients connecting to ``host``.

    On the local host, picks ``https`` iff TLS is locally provisioned
    and enabled. For non-local hosts, the caller usually has explicit
    URLs — but we still return ``https`` when TLS is provisioned, to
    let helpers like the webui-banner emit the right scheme.
    """
    cfg = cfg or load_config()
    if is_enabled(cfg):
        return "https"
    return "http"


def add_server_tls_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard ``--tls`` / ``--no-tls`` / ``--insecure`` flags.

    Shared by all three servers so the operator-facing surface is
    identical. ``--tls-cert`` / ``--tls-key`` are escape hatches for
    BYOC scenarios; they don't touch shared config.
    """
    group = parser.add_mutually_exclusive_group()
    # Resolve the platform-correct TLS root for the help text so
    # macOS / Windows / non-XDG-Linux installs see the actual path
    # rather than the Linux-only ~/.config/forgather/tls.
    from .config import tls_dir as _tls_dir_fn

    _tls_root_help = str(_tls_dir_fn())
    group.add_argument(
        "--tls",
        dest="tls",
        action="store_const",
        const=True,
        default=None,
        help=(
            "Force TLS on (overrides shared config). "
            f"Cert/key resolved from {_tls_root_help}/ unless "
            "--tls-cert/--tls-key are given."
        ),
    )
    group.add_argument(
        "--no-tls",
        dest="no_tls",
        action="store_true",
        help="Force TLS off (overrides shared config).",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Allow binding a non-loopback host without TLS. Bearer "
        "tokens traverse the network in cleartext.",
    )
    parser.add_argument(
        "--tls-cert",
        default=None,
        type=os.path.expanduser,
        help="Override the server certificate path.",
    )
    parser.add_argument(
        "--tls-key",
        default=None,
        type=os.path.expanduser,
        help="Override the server private-key path.",
    )
