"""Argument parser for `forgather dataset-server` (multi-action wrapper).

Subcommands:

- ``start [server flags…]`` — launches the dataset server. Flags are
  forwarded verbatim to ``tools/dataset_server/server.py``. Supports
  the full ``--help`` of the wrapped script.
- ``status [--server URL]`` — prints health + auth + policy info.
- ``list [--server URL] [--json]`` — list loaded handles.
- ``cache [--server URL] [--json]`` — list HF cache contents on the
  server's host.
- ``local [--server URL] [--json]`` — list registered local mappings.
"""

from __future__ import annotations

import argparse
from argparse import RawTextHelpFormatter

DEFAULT_SERVER_URL = "http://127.0.0.1:8766"
SERVER_URL_ENV = "FORGATHER_DATASET_SERVER"


def _add_client_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--server",
        default=None,
        metavar="URL",
        help=(
            f"Dataset server base URL (default: ${SERVER_URL_ENV} or "
            f"{DEFAULT_SERVER_URL})"
        ),
    )
    p.add_argument(
        "--token",
        default=None,
        metavar="TOKEN",
        help=(
            "Bearer token. If omitted, falls back to "
            "$FORGATHER_DATASET_SERVER_TOKEN, then to the per-port "
            "token file under <forgather_config_dir>/dataset_server/ for "
            "localhost URLs."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable output.",
    )
    p.add_argument(
        "--insecure",
        action="store_true",
        help=(
            "Skip TLS chain + hostname validation for the upstream "
            "HTTPS request. Use when the channel is secured by some "
            "other means (SSH tunnel, VPN, air-gapped LAN) so the "
            "remote's cert won't validate against your local CA. "
            "You are explicitly opting out of cert validation — "
            "responses are no longer authenticated by TLS."
        ),
    )


def create_dataset_server_parser(global_args):
    """Create parser for `forgather dataset-server`."""
    parser = argparse.ArgumentParser(
        prog="forgather dataset-server",
        description=(
            "Run or query the forgather dataset server.\n\n"
            "When a client process sets the FORGATHER_DATASET_SERVER\n"
            "environment variable to this server's URL, calls to\n"
            "fast_load_iterable_dataset(...) are transparently routed\n"
            "through the server instead of loading data locally — useful\n"
            "for multi-node training where downloading the same dataset\n"
            "on every node is impractical.\n\n"
            "Subcommands:\n"
            "  start    Launch the server (forwards flags to the script).\n"
            "  status   Show server health, auth, and policy.\n"
            "  list     List loaded handles.\n"
            "  cache    List HF cache contents on the server's host.\n"
            "  local    List configured local mappings.\n\n"
            "Typical workflow:\n\n"
            "  # Terminal 1\n"
            "  $ forgather dataset-server start --local stories=/data/tinystories\n\n"
            "  # Terminal 2\n"
            "  $ export FORGATHER_DATASET_SERVER=http://localhost:8766\n"
            "  $ forgather -t fast-iter.yaml dataset \\\n"
            "        --target train_dataset_split -n 3\n"
        ),
        formatter_class=RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="ds_subcommand", metavar="<action>", required=False
    )

    # `start` — by default enqueues a scheduled ``dataset_server`` job through
    # the forgather server (background, cluster-registered, auto-provisioned
    # auth/TLS), exactly like ``forgather diloco server``. ``--local-only``
    # runs it in the foreground instead. The server flags below mirror the
    # subset the scheduler honors (== the webui's "Start dataset server"
    # modal); any extra flags are captured and forwarded only on the
    # foreground path.
    start_parser = subparsers.add_parser(
        "start",
        help="Launch the dataset server (scheduled job; --local-only for foreground).",
        formatter_class=RawTextHelpFormatter,
    )
    # Locality: the forgather server is the default, required path.
    start_parser.add_argument(
        "--local-fallback",
        action="store_true",
        help=(
            "If the forgather server isn't reachable, fall back to running\n"
            "the dataset server in the foreground instead of erroring."
        ),
    )
    start_parser.add_argument(
        "--local-only",
        action="store_true",
        help=(
            "Never contact the forgather server; run the dataset server in\n"
            "the foreground (the pre-scheduler behavior). Extra flags after\n"
            "the known ones are forwarded verbatim to the server script."
        ),
    )
    start_parser.add_argument(
        "--via-server",
        metavar="URL",
        default=None,
        help=(
            "forgather-server base URL to enqueue through (default:\n"
            "$FORGATHER_SERVER_URL or http://127.0.0.1:8765)."
        ),
    )
    start_parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Scheduler priority for the enqueued job (default: 0).",
    )
    # Server flags — the managed subset (mirrors dataset_server_ops /
    # the webui modal). Auth tokens + TLS are provisioned by the
    # scheduler on the server path; on --local-only the server script
    # auto-generates/persists a per-port token as before.
    start_parser.add_argument("-H", "--host", default="127.0.0.1", help="Bind address")
    start_parser.add_argument("-p", "--port", type=int, default=8766, help="Bind port")
    start_parser.add_argument(
        "-l",
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    start_parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Disable HF cache loading; only --local datasets are servable.",
    )
    start_parser.add_argument(
        "--allow-paths",
        action="store_true",
        help="Allow clients to request loading by absolute filesystem path.",
    )
    start_parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow HF downloads when the cache is missing a dataset.",
    )
    start_parser.add_argument(
        "--local",
        action="append",
        default=[],
        dest="local_maps",
        metavar="NAME=PATH",
        help="Register a local dataset as 'local/NAME'. Repeatable.",
    )
    start_parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable bearer-token auth (only on an already-trusted network).",
    )
    start_parser.add_argument(
        "--regen-token",
        action="store_true",
        help="Rotate the persisted per-port auth token at startup.",
    )
    start_parser.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help="Optional YAML config file (see the dataset server README).",
    )

    # Diagnostic subcommands.
    status_parser = subparsers.add_parser(
        "status",
        help="Show server health, auth status, and loading policy.",
        formatter_class=RawTextHelpFormatter,
    )
    _add_client_args(status_parser)

    list_parser = subparsers.add_parser(
        "list",
        help="List currently loaded dataset handles.",
        formatter_class=RawTextHelpFormatter,
    )
    _add_client_args(list_parser)

    cache_parser = subparsers.add_parser(
        "cache",
        help="List HF datasets currently cached on the server's host.",
        formatter_class=RawTextHelpFormatter,
    )
    _add_client_args(cache_parser)

    local_parser = subparsers.add_parser(
        "local",
        help="List local dataset mappings registered on the server.",
        formatter_class=RawTextHelpFormatter,
    )
    _add_client_args(local_parser)

    return parser
