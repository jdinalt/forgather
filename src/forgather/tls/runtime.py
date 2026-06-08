"""Runtime helpers: uvicorn ssl kwargs, httpx verify, client scheme picker."""

from __future__ import annotations

import argparse
import logging
import os
import ssl
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


def server_tls_files(
    args: Optional[argparse.Namespace] = None, cfg: Optional[TLSConfig] = None
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve the server's ``(cert, key, ca_bundle)`` file paths, or
    ``(None, None, None)`` when TLS is off for this invocation.

    The path analogue of :func:`stdlib_ssl_context` (which builds an
    ``ssl.SSLContext`` from the same files): a transport that needs the raw PEM
    material rather than a Python ``SSLContext`` — e.g. gRPC's
    ``ssl_server_credentials`` — resolves the identical cert/key/CA the control
    plane uses. The CA bundle is ``cfg.effective_bundle()`` (the cluster CA used
    to validate presented client certs for mTLS); ``None`` when no bundle exists.
    """
    cfg, on, cert, key = _resolve_state(args, cfg)
    if not on:
        return None, None, None
    bundle = cfg.effective_bundle()
    return cert, key, (str(bundle) if bundle is not None else None)


def uvicorn_ssl_kwargs(
    args: Optional[argparse.Namespace] = None, cfg: Optional[TLSConfig] = None
) -> dict:
    """Return uvicorn TLS kwargs, or ``{}`` when TLS is off for this invocation.

    When TLS is on and a CA bundle exists, also requests (but does not
    require) a client cert from the connecting peer:
    ``ssl_cert_reqs=ssl.CERT_OPTIONAL`` + ``ssl_ca_certs=<bundle>``.
    The TLS handshake validates any presented cert against the cluster
    CA; the ASGI auth gate then decides whether cert-presence is
    sufficient for the request's path. Browser/CLI bearer clients that
    don't present a cert are unaffected.

    Raises :class:`FileNotFoundError` if TLS is requested but cert/key
    are missing from disk.
    """
    import ssl

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
    kwargs: dict = {"ssl_keyfile": key, "ssl_certfile": cert}
    bundle = cfg.effective_bundle()
    if bundle is not None:
        kwargs["ssl_cert_reqs"] = ssl.CERT_OPTIONAL
        kwargs["ssl_ca_certs"] = str(bundle)
    return kwargs


def stdlib_ssl_context(
    args: Optional[argparse.Namespace] = None, cfg: Optional[TLSConfig] = None
) -> Optional[ssl.SSLContext]:
    """Return a server-side :class:`ssl.SSLContext` for ``http.server`` use.

    The stdlib analogue of :func:`uvicorn_ssl_kwargs`: returns a context
    that can be applied to a listening socket via
    ``ctx.wrap_socket(sock, server_side=True)``. Returns ``None`` when
    TLS is off for this invocation.

    When a cluster CA bundle is present, the context is configured with
    ``verify_mode = ssl.CERT_OPTIONAL`` and the bundle loaded as the CA,
    so the TLS handshake validates any client cert that *is* presented
    against the cluster CA. The application layer can then read
    ``conn.getpeercert()`` to learn whether a valid peer cert was
    supplied (mTLS), or fall back to a bearer-token check otherwise.

    Raises :class:`FileNotFoundError` if TLS is requested but cert/key
    files are missing.
    """
    cfg, on, cert, key = _resolve_state(args, cfg)
    if not on:
        return None
    if not cert or not key:
        raise FileNotFoundError(
            "TLS enabled but server cert/key not configured "
            "(run 'forgather tls init' or pass --tls-cert/--tls-key)"
        )
    if not Path(cert).is_file():
        raise FileNotFoundError(f"TLS cert not found: {cert}")
    if not Path(key).is_file():
        raise FileNotFoundError(f"TLS key not found: {key}")
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(certfile=cert, keyfile=key)
    bundle = cfg.effective_bundle()
    if bundle is not None:
        ctx.load_verify_locations(cafile=str(bundle))
        ctx.verify_mode = ssl.CERT_OPTIONAL
    return ctx


def urllib_ssl_context(
    cfg: Optional[TLSConfig] = None, verify: bool = True
) -> Optional[ssl.SSLContext]:
    """Return a client-side :class:`ssl.SSLContext` for ``urllib.request`` use.

    The stdlib analogue of :func:`httpx_peer_kwargs`: builds a single
    context that carries (a) the cluster CA bundle for verifying the
    *peer's* cert, and (b) when this node is provisioned, its own
    cert+key for presenting identity via mTLS. Pass to
    ``urllib.request.urlopen(..., context=...)``.

    ``verify=False`` returns an unverified context — opt-in escape
    hatch for SSH-tunneled remotes and similar cases where the operator
    has accepted the trust boundary externally.

    Returns ``None`` when no bundle exists and ``verify`` is True; the
    caller should then either fall back to system trust (urllib's
    default when no ``context=`` is passed) or refuse outright.
    """
    cfg = cfg or load_config()
    if not verify:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        # ``verify=False`` disables verifying the *peer's* cert (the
        # SSH-tunnel escape hatch) — it must NOT also drop *our* client
        # cert, or the mTLS skip-bearer path silently stops working and
        # a token-less worker gets a hard 401. Present our identity even
        # here; the server still validates it against the cluster CA.
        if cfg.is_provisioned():
            try:
                ctx.load_cert_chain(str(cfg.server_cert), str(cfg.server_key))
            except (OSError, ssl.SSLError) as e:
                log.debug("urllib_ssl_context: could not load client cert: %s", e)
        return ctx
    bundle = cfg.effective_bundle()
    if bundle is None:
        return None
    ctx = ssl.create_default_context(cafile=str(bundle))
    if not cfg.verify_hostname:
        ctx.check_hostname = False
    if cfg.is_provisioned():
        # Load this node's cert+key so we can present mTLS identity
        # when the peer asks for it. Silent no-op for peers that
        # don't request a client cert.
        ctx.load_cert_chain(str(cfg.server_cert), str(cfg.server_key))
    return ctx


def httpx_client_cert(
    cfg: Optional[TLSConfig] = None,
) -> Optional[tuple[str, str]]:
    """Return ``(cert_path, key_path)`` for httpx ``cert=`` or ``None``.

    Prefer :func:`httpx_peer_kwargs` for new code — httpx's ``cert=``
    parameter is deprecated and will be removed; the SSLContext-based
    approach (cert chain loaded into the same context as the CA
    bundle) keeps working across that transition.
    """
    cfg = cfg or load_config()
    if not cfg.is_provisioned():
        return None
    return (str(cfg.server_cert), str(cfg.server_key))


def httpx_peer_kwargs(cfg: Optional[TLSConfig] = None) -> dict:
    """Return kwargs for an inter-node ``httpx.AsyncClient(...)``.

    Builds a single ``ssl.SSLContext`` that carries both the cluster
    CA bundle (for verifying the *peer's* cert) and this node's
    cert+key (for presenting *our* identity via mTLS). The context
    is passed via ``verify=``; the deprecated ``cert=`` kwarg is not
    used.

    Failure modes:

    * No CA bundle present → ``{"verify": True}`` (system trust; will
      fail closed against forgather peers, which is the right outcome
      — we never silently disable verification).
    * CA bundle present but cert+key missing → ``{"verify": <ctx>}``
      with the CA loaded; the call will reach the peer but can't
      authenticate via mTLS, and the peer will 401 if it's the
      inter-node path. Bearer-token clients pick this branch.
    """
    cfg = cfg or load_config()
    verify = httpx_verify(cfg)
    cert_pair = httpx_client_cert(cfg)
    if cert_pair is not None and isinstance(verify, ssl.SSLContext):
        cert_path, key_path = cert_pair
        verify.load_cert_chain(cert_path, key_path)
    return {"verify": verify}


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


def httpx_verify_for_url(url: str, cfg: Optional[TLSConfig] = None) -> object:
    """Same as :func:`httpx_verify` but returns ``False`` for plain ``http://``.

    Saves an unnecessary file-system check when the URL won't be using TLS.
    """
    if not url.lower().startswith("https"):
        return True
    return httpx_verify(cfg)


def client_scheme(host: str = "127.0.0.1", cfg: Optional[TLSConfig] = None) -> str:
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
