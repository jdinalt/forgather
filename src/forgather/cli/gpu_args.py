"""Argument parser for gpu command."""

import argparse
from argparse import RawTextHelpFormatter


def create_gpu_parser(global_args):
    parser = argparse.ArgumentParser(
        prog="forgather gpu",
        description="Inspect and configure GPUs via the forgather-server",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        metavar="URL",
        help="forgather-server base URL (default: $FORGATHER_SERVER_URL or http://127.0.0.1:8765)",
    )
    subparsers = parser.add_subparsers(dest="gpu_subcommand", help="GPU subcommands")

    subparsers.add_parser(
        "status",
        help="Show GPU status table",
        formatter_class=RawTextHelpFormatter,
    )

    disable_parser = subparsers.add_parser(
        "disable",
        help="Disable a GPU (scheduler will not assign jobs to it)",
        formatter_class=RawTextHelpFormatter,
    )
    disable_parser.add_argument("idx", type=int, help="GPU index")

    enable_parser = subparsers.add_parser(
        "enable",
        help="Enable a GPU",
        formatter_class=RawTextHelpFormatter,
    )
    enable_parser.add_argument("idx", type=int, help="GPU index")

    priority_parser = subparsers.add_parser(
        "priority",
        help="Set minimum job priority for a GPU",
        formatter_class=RawTextHelpFormatter,
    )
    priority_parser.add_argument("idx", type=int, help="GPU index")
    priority_parser.add_argument("level", type=int, help="Minimum priority level")

    kill_parser = subparsers.add_parser(
        "kill",
        help="Kill all compute processes on a GPU (requires --yes)",
        formatter_class=RawTextHelpFormatter,
    )
    kill_parser.add_argument("idx", type=int, help="GPU index")
    kill_parser.add_argument(
        "--yes", action="store_true", help="Required: confirm kill"
    )

    return parser
