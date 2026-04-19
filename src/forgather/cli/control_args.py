"""Argument parser for control command."""

import argparse
from argparse import RawTextHelpFormatter


def create_control_parser(global_args):
    """Create parser for control command."""
    parser = argparse.ArgumentParser(
        prog="forgather control",
        description="Control running training jobs",
        formatter_class=RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="control_subcommand", help="Control subcommands"
    )

    # list subcommand
    list_parser = subparsers.add_parser(
        "list",
        help="List discoverable training jobs",
        formatter_class=RawTextHelpFormatter,
    )
    list_parser.add_argument(
        "--remote",
        type=str,
        metavar="HOST:PORT",
        help="Query remote host for jobs (e.g., compute-node-01:8947)",
    )

    # status subcommand
    status_parser = subparsers.add_parser(
        "status",
        help="Get status of a training job",
        formatter_class=RawTextHelpFormatter,
    )
    status_parser.add_argument("job_id", help="Job ID to query")

    # stop subcommand
    stop_parser = subparsers.add_parser(
        "stop",
        help="Send graceful stop command to a training job (saves final checkpoint)",
        formatter_class=RawTextHelpFormatter,
    )
    stop_parser.add_argument("job_id", help="Job ID to stop")

    # abort subcommand
    abort_parser = subparsers.add_parser(
        "abort",
        help="Abort training job WITHOUT saving checkpoint",
        formatter_class=RawTextHelpFormatter,
    )
    abort_parser.add_argument("job_id", help="Job ID to abort")
    abort_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )

    # save subcommand
    save_parser = subparsers.add_parser(
        "save",
        help="Trigger checkpoint save in a training job",
        formatter_class=RawTextHelpFormatter,
    )
    save_parser.add_argument("job_id", help="Job ID to save")

    # save-stop subcommand
    save_stop_parser = subparsers.add_parser(
        "save-stop",
        help="Save checkpoint and stop training job",
        formatter_class=RawTextHelpFormatter,
    )
    save_stop_parser.add_argument("job_id", help="Job ID to save and stop")

    # cleanup subcommand
    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Remove endpoint files for dead training jobs",
        formatter_class=RawTextHelpFormatter,
    )
    cleanup_parser.add_argument(
        "--force", action="store_true", help="Remove all job files without confirmation"
    )

    return parser
