"""Argument parser for sched command."""

import argparse
from argparse import RawTextHelpFormatter


def create_sched_parser(global_args):
    parser = argparse.ArgumentParser(
        prog="forgather sched",
        description="Manage the forgather-server job queue and scheduler",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        metavar="URL",
        help="forgather-server base URL (default: $FORGATHER_SERVER_URL or http://127.0.0.1:8765)",
    )
    subparsers = parser.add_subparsers(
        dest="sched_subcommand", help="Scheduler subcommands"
    )

    subparsers.add_parser(
        "status",
        help="Show scheduler status and queue/job counts",
        formatter_class=RawTextHelpFormatter,
    )

    subparsers.add_parser(
        "list",
        help="List queued and active jobs",
        formatter_class=RawTextHelpFormatter,
    )

    subparsers.add_parser(
        "pause",
        help="Pause the scheduler (stop dispatching new jobs)",
        formatter_class=RawTextHelpFormatter,
    )

    subparsers.add_parser(
        "resume",
        help="Resume the scheduler",
        formatter_class=RawTextHelpFormatter,
    )

    cancel_parser = subparsers.add_parser(
        "cancel",
        help="Cancel a queued or running job",
        formatter_class=RawTextHelpFormatter,
    )
    cancel_parser.add_argument("queue_id", help="Queue ID to cancel")

    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Remove terminal job records (all, or a specific job)",
        formatter_class=RawTextHelpFormatter,
    )
    cleanup_parser.add_argument(
        "job_id",
        nargs="?",
        default=None,
        help="Job ID to remove (omit to remove all terminal records)",
    )

    subparsers.add_parser(
        "gc",
        help=(
            "Sweep orphan TTY logs from ~/.forgather/server/jobs/ "
            "(files not referenced by any record, older than the TTL)"
        ),
        formatter_class=RawTextHelpFormatter,
    )

    return parser
