"""Argument parser for `forgather dataset-server` (wrapper)."""

from __future__ import annotations

import argparse
from argparse import RawTextHelpFormatter


def create_dataset_server_parser(global_args):
    """Create parser for the dataset-server command."""
    parser = argparse.ArgumentParser(
        prog="forgather dataset-server",
        description=(
            "Run the forgather dataset server (proof of concept).\n\n"
            "When a client process sets the FORGATHER_DATASET_SERVER\n"
            "environment variable to this server's URL, calls to\n"
            "fast_load_iterable_dataset(...) are transparently routed\n"
            "through the server instead of loading data locally — useful\n"
            "for multi-node training where downloading the same dataset\n"
            "on every node is impractical.\n\n"
            "All arguments are forwarded to `python -m tools.dataset_server`.\n"
            "Common options:\n"
            "  --host HOST          Bind address (default: 127.0.0.1)\n"
            "  --port PORT          Port to listen on (default: 8766;\n"
            "                       NOT 8765 — that's the forgather\n"
            "                       orchestration server's port)\n"
            "  --allow-load         Enable POST /v1/load so clients can\n"
            "                       request the server to lazily load\n"
            "                       HuggingFace datasets on demand.\n"
            "                       Required for transparent\n"
            "                       FORGATHER_DATASET_SERVER routing.\n"
            "  --log-level LEVEL    DEBUG / INFO / WARNING / ERROR\n\n"
            "Typical workflow:\n\n"
            "  # Terminal 1\n"
            "  $ forgather dataset-server --allow-load\n\n"
            "  # Terminal 2\n"
            "  $ export FORGATHER_DATASET_SERVER=http://localhost:8766\n"
            "  $ forgather -t en.yaml dataset --target validation_dataset_split -n 3\n"
        ),
        formatter_class=RawTextHelpFormatter,
        add_help=False,  # Forward --help to the wrapped script.
    )
    parser.add_argument(
        "dummy",
        nargs="?",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments to forward to python -m tools.dataset_server",
    )
    return parser
