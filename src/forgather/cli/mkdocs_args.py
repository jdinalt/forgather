"""Argument parser for the `mkdocs` subcommand."""

import argparse
from argparse import RawTextHelpFormatter

from .submit_orch import add_locality_args


def create_mkdocs_parser(_global_args):
    parser = argparse.ArgumentParser(
        prog="forgather mkdocs",
        description="Run the docs server (mkdocs serve) via the scheduler or locally",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("-f", "--config-file", required=True, help="Path to mkdocs.yml")
    parser.add_argument("-p", "--port", type=int, default=8000)
    parser.add_argument("-H", "--host", default="127.0.0.1")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--livereload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable live-reload (default: on)",
    )
    parser.add_argument(
        "--dirty", action="store_true", help="Only rebuild changed files"
    )
    parser.add_argument(
        "-w",
        "--watch",
        action="append",
        default=[],
        help="Extra path to watch (repeatable)",
    )
    # The docs server is long-running, so it submits to the scheduler
    # (background) by default; --local-only runs it in the foreground.
    parser.add_argument("--enqueue", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument(
        "--server",
        "--via-server",
        dest="server",
        default=None,
        help="forgather-server URL (or $FORGATHER_SERVER_URL)",
    )
    add_locality_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the local command without executing (foreground only)",
    )
    return parser
