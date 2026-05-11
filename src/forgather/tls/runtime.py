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

    Returns a path string when a local CA bundle exists. When TLS is
    enabled on this host but no bundle is present, we intentionally
    *still* fall back to ``True`` (system trust) so the connection
    fails closed against forgather peers (system trust will not
    contain the self-signed cluster CA). Returning ``False`` here
    would silently disable verification — never what we want.

    Falling back to ``True`` rather than raising lets callers in
    mixed environments (e.g. talking to a public LLM endpoint *and*
    a local forgather peer with the same httpx client) keep working
    against system-trusted hosts; the forgather peers will simply
    fail to verify, which is the correct failure mode.
    """
    cfg = cfg or load_config()
    bundle = cfg.effective_bundle()
    if bundle is not None:
        return str(bundle)
    if cfg.enabled:
        log.warning(
            "TLS enabled in %s but no CA bundle present at %s — "
            "forgather peer connections will fall back to the system "
            "trust store and almost certainly fail to verify. Run "
            "'forgather tls init' or import a peer CA via "
            "'forgather tls import-ca'.",
            cfg.config_file,
            cfg.ca_bundle,
        )
    return True


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
    group.add_argument(
        "--tls",
        dest="tls",
        action="store_const",
        const=True,
        default=None,
        help="Force TLS on (overrides shared config). "
        "Cert/key resolved from ~/.config/forgather/tls/ unless "
        "--tls-cert/--tls-key are given.",
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
