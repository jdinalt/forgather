#!/usr/bin/env python3
"""Forgather web server entry point.

Usage (typically via the CLI shim):
    forgather server -H 127.0.0.1 -p 8765
    python tools/forgather_server/server.py -H 127.0.0.1 -p 8765

The server is intended to be a single-user prototype. By default it
binds to ``127.0.0.1`` and gates every ``/api/`` request behind a
bearer token persisted under ``~/.config/forgather/server/auth_token``. CLI
clients pick the token up automatically; the webui prompts for it on
first connect (jupyter-style ``?token=…`` URL is printed at startup).

Pass ``--no-auth`` to skip the gate (useful for local dev only — any
other user on the same host can then talk to the server). Pass
``--regen-token`` to rotate the bearer token at startup.
"""

import argparse
import logging
import sys
from argparse import RawTextHelpFormatter
from pathlib import Path

# Support both `python -m forgather_server.server` and standalone execution.
if __name__ == "__main__" and __package__ is None:
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from forgather_server import auth, cluster, paths, server_config
    from forgather_server.app import create_app
else:
    from . import auth, cluster, paths, server_config
    from .app import create_app

import uvicorn

from forgather.tls import (
    TLSRequiredError,
    enforce_non_loopback_policy,
    is_enabled as tls_is_enabled,
    load_config as tls_load_config,
    uvicorn_ssl_kwargs as tls_uvicorn_ssl_kwargs,
)
from forgather.tls.runtime import (
    add_server_tls_args,
    is_tls_active as tls_is_active,
)


def main():
    # First pass: pull out --config so we can read defaults from disk
    # before adding the rest of the arguments. ``parse_known_args`` so
    # the rest of the command line is left for the real parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    try:
        cfg_path, cfg_data = server_config.load(pre_args.config)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    cfg_arg_defaults = server_config.args_defaults(cfg_data)

    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Forgather web server (prototype)",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            "Path to a YAML config file. Defaults to "
            f"{server_config.default_config_path()} (created with a "
            "commented template if absent). Keys under 'args:' override "
            "the CLI argument defaults; values passed on the command line "
            "still win."
        ),
    )
    parser.add_argument("-H", "--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("-p", "--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument(
        "-l",
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload (development)",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable token/password authentication (any local user can connect)",
    )
    parser.add_argument(
        "--regen-token",
        action="store_true",
        help="Generate a fresh auth token at startup (invalidates existing CLIs)",
    )
    parser.add_argument(
        "--persist-sessions",
        action="store_true",
        help=(
            "Persist browser sessions to disk so the webui doesn't "
            "force a re-login on every server restart. Sessions still "
            "obey the 30-day TTL and can be revoked via /api/auth/logout. "
            "Convenience for rapid dev / restart cycles; remove the "
            "file at <config>/server/sessions.json to drop all "
            "persisted sessions."
        ),
    )
    parser.add_argument(
        "--cluster",
        default=None,
        metavar="NAME",
        help=(
            "Join the named cluster (multi-node mode). Without this "
            "flag, the server runs in single-node / standalone mode "
            "and does not advertise on the LAN. The cluster name is "
            "the unit of scoping for mDNS discovery — only servers "
            "with the same --cluster value will see each other."
        ),
    )
    add_server_tls_args(parser)
    parser.add_argument(
        "--lock-inference-proxy",
        action="store_true",
        help=(
            "Restrict the inference proxy to localhost upstreams. Default "
            "is to allow any URL the operator types into the panel — the "
            "research-tool norm, since the operator can already submit "
            "arbitrary training jobs and so SSRF adds no capability. Pass "
            "this flag for a stricter posture where forgather runs in an "
            "environment with non-operator-controlled clients."
        ),
    )
    parser.add_argument(
        "--cluster-address",
        action="append",
        default=[],
        metavar="IP",
        help=(
            "Address(es) to advertise to cluster peers, overriding "
            "auto-detection. Repeatable. Use this when the server "
            "runs inside a container whose network namespace hides "
            "the host's real interfaces — psutil sees only loopback "
            "or the container bridge, the auto-detector falls back "
            "to 127.0.0.1, and peers on other hosts can't reach you. "
            "Example: --cluster-address 192.168.1.27"
        ),
    )
    if cfg_arg_defaults:
        # ``set_defaults`` only overrides destinations parser actually
        # knows about; quietly ignore unknown keys (with a warning) so a
        # forward-compat config that names a future arg doesn't crash.
        known = {a.dest for a in parser._actions}
        applied = {k: v for k, v in cfg_arg_defaults.items() if k in known}
        unknown = set(cfg_arg_defaults) - known
        if unknown:
            logging.getLogger("forgather_server").warning(
                "ignoring unknown server-config args: %s", sorted(unknown)
            )
        parser.set_defaults(**applied)

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger("forgather_server").setLevel(log_level)

    # Run before _configure_auth so the auth-token file lands on a
    # tightened directory and any legacy 0644 state files get fixed up
    # before we read or rewrite them.
    paths.tighten_existing_state_perms()

    try:
        ssl_kwargs = tls_uvicorn_ssl_kwargs(args)
    except FileNotFoundError as exc:
        print(f"TLS config error: {exc}", file=sys.stderr)
        sys.exit(2)
    tls_on = bool(ssl_kwargs)

    try:
        enforce_non_loopback_policy(
            args.host,
            tls_enabled=tls_on,
            insecure=args.insecure,
            service="forgather server",
            cfg=tls_load_config(),
        )
    except TLSRequiredError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    _configure_auth(args, tls_on=tls_on)

    if args.cluster:
        _activate_cluster(args, tls_on=tls_on)

    # The inference proxy reads this flag at request time. Set it via
    # module attribute (rather than env var) so a future `forgather server
    # --lock-inference-proxy=on/off` at runtime would also work cleanly.
    if args.lock_inference_proxy:
        from .routes import inference_proxy as _inf_proxy

        _inf_proxy.LOCK_TO_LOCALHOST = True
        logging.info("inference proxy: locked to localhost upstreams")

    app = create_app()

    scheme = "https" if tls_on else "http"
    logging.info(f"Starting Forgather server on {scheme}://{args.host}:{args.port}")
    logging.info(f"Server config: {cfg_path}")
    # Pin the HTTP protocol to our subclass so peer client certs are
    # surfaced on the ASGI scope for mTLS-aware auth (issue #31).
    # Falls back to the default protocol when TLS is off — no point
    # paying the import / hook cost. The helper picks httptools when
    # available, h11 otherwise (matching uvicorn's own selection).
    extra: dict = {}
    if tls_on:
        if __package__ is None:
            from forgather_server.asgi_tls_protocol import ForgatherProtocol
        else:
            from .asgi_tls_protocol import ForgatherProtocol
        extra["http"] = ForgatherProtocol
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=True,
        reload=args.reload,
        **ssl_kwargs,
        **extra,
    )


def _configure_auth(args, *, tls_on: bool = False) -> None:
    """Print the jupyter-style banner and set up auth state.

    Always touches the token file (so the CLI can find it) even when
    auth is disabled — that way a later ``--no-auth``-less restart
    still works without forcing the user to log in fresh.
    """
    if args.regen_token:
        token = auth.regenerate_token()
    else:
        token = auth.load_token()

    if args.persist_sessions:
        auth.enable_session_persistence()

    on_loopback = args.host in ("127.0.0.1", "::1", "localhost")
    scheme = "https" if tls_on else "http"

    print()
    if args.no_auth:
        auth.disable_auth()
        print("    !! Forgather server is running with --no-auth !!")
        print(f"    !! Any other local user on this host can read/control jobs.")
        print(f"        {scheme}://{args.host}:{args.port}/")
        print()
        return

    print("    Forgather server is running at:")
    print(f"        {scheme}://{args.host}:{args.port}/?token={token}")
    if on_loopback and args.host != "localhost":
        print(f"        {scheme}://localhost:{args.port}/?token={token}")
    print()
    print(f"    CLI auth: token in {paths.auth_token_file()} (mode 0600)")
    if not auth.has_password():
        print(
            "    First successful token login will prompt to set a "
            "password for future browser logins."
        )
    if not on_loopback and not tls_on:
        print()
        print(
            f"    !! Bound to non-loopback host {args.host} without TLS — "
            f"the bearer token traverses the network in cleartext."
        )
        print(
            "    !! Run 'forgather tls init' to enable TLS, or pass "
            "--insecure to suppress this check."
        )
    elif tls_on:
        print()
        print(f"    TLS: serving HTTPS from {tls_load_config().server_cert}")
    print()


def _activate_cluster(args, *, tls_on: bool = False) -> None:
    """Stamp the cluster identity and print a banner.

    The discovery + membership tasks are started later by the FastAPI
    lifespan handler — they need a running event loop. This function
    only activates the cluster module so the rest of the server can
    see it as "active" before the loop comes up.
    """
    advertise = tuple(args.cluster_address or ())
    ident = cluster.activate(
        args.cluster, port=args.port, advertise_addresses=advertise, tls=tls_on
    )
    print()
    print(
        f"    Cluster mode: name={ident.cluster_name!r} "
        f"node_id={ident.node_id} hostname={ident.hostname}"
    )
    if advertise:
        print(f"    Advertising operator-supplied addresses: {list(advertise)}")
    else:
        print(
            "    Address auto-detection enabled; check the startup log "
            "for the chosen interface(s)."
        )
    print(
        "    Inter-node API is unauthenticated on the assumption of a " "trusted LAN."
    )
    if tls_on:
        print("    Cluster peer transport: HTTPS (CA bundle from forgather.tls).")
    else:
        print("    Cluster peer transport: HTTP (cleartext).")
    print()


if __name__ == "__main__":
    main()
