#!/usr/bin/env python3
"""Forgather web server entry point.

Usage (typically via the CLI shim):
    forgather server -H 127.0.0.1 -p 8765
    python tools/forgather_server/server.py -H 127.0.0.1 -p 8765

The server is intended to be a single-user prototype. By default it
binds to ``127.0.0.1`` and gates every ``/api/`` request behind a
bearer token persisted under ``~/.forgather/server/auth_token``. CLI
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
    from forgather_server import auth, cluster, paths
    from forgather_server.app import create_app
else:
    from . import auth, cluster, paths
    from .app import create_app

import uvicorn


def main():
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Forgather web server (prototype)",
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

    _configure_auth(args)

    if args.cluster:
        _activate_cluster(args)

    app = create_app()

    logging.info(f"Starting Forgather server on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=True,
        reload=args.reload,
    )


def _configure_auth(args) -> None:
    """Print the jupyter-style banner and set up auth state.

    Always touches the token file (so the CLI can find it) even when
    auth is disabled — that way a later ``--no-auth``-less restart
    still works without forcing the user to log in fresh.
    """
    if args.regen_token:
        token = auth.regenerate_token()
    else:
        token = auth.load_token()

    on_loopback = args.host in ("127.0.0.1", "::1", "localhost")

    print()
    if args.no_auth:
        auth.disable_auth()
        print("    !! Forgather server is running with --no-auth !!")
        print(f"    !! Any other local user on this host can read/control jobs.")
        print(f"        http://{args.host}:{args.port}/")
        print()
        return

    print("    Forgather server is running at:")
    print(f"        http://{args.host}:{args.port}/?token={token}")
    if on_loopback and args.host != "localhost":
        print(f"        http://localhost:{args.port}/?token={token}")
    print()
    print(f"    CLI auth: token in {paths.auth_token_file()} (mode 0600)")
    if not auth.has_password():
        print(
            "    First successful token login will prompt to set a "
            "password for future browser logins."
        )
    if not on_loopback:
        print()
        print(
            f"    !! Bound to non-loopback host {args.host} without TLS — "
            f"the bearer token traverses the network in cleartext."
        )
        print(
            "    !! Run behind an SSH tunnel or a TLS-terminating "
            "reverse proxy for LAN access."
        )
    print()


def _activate_cluster(args) -> None:
    """Stamp the cluster identity and print a banner.

    The discovery + membership tasks are started later by the FastAPI
    lifespan handler — they need a running event loop. This function
    only activates the cluster module so the rest of the server can
    see it as "active" before the loop comes up.
    """
    advertise = tuple(args.cluster_address or ())
    ident = cluster.activate(
        args.cluster, port=args.port, advertise_addresses=advertise
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
        "    Inter-node API is unauthenticated on the assumption of a "
        "trusted LAN."
    )
    print()


if __name__ == "__main__":
    main()
