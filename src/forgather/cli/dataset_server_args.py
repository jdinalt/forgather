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

    # `start` — REMAINDER passthrough; --help goes to the wrapped script.
    start_parser = subparsers.add_parser(
        "start",
        help="Launch the dataset server (forwards flags to the script).",
        formatter_class=RawTextHelpFormatter,
        add_help=False,
    )
    start_parser.add_argument("dummy", nargs="?", default="", help=argparse.SUPPRESS)
    start_parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments to forward to tools/dataset_server/server.py",
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
