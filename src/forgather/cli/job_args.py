"""Argument parser for job command."""

import argparse
from argparse import RawTextHelpFormatter


def create_job_parser(global_args):
    parser = argparse.ArgumentParser(
        prog="forgather job",
        description="Control and inspect server-managed training jobs",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        metavar="URL",
        help="forgather-server base URL (default: $FORGATHER_SERVER_URL or http://127.0.0.1:8765)",
    )
    subparsers = parser.add_subparsers(dest="job_subcommand", help="Job subcommands")

    for name, help_text in [
        ("status", "Print trainer status for a running job"),
        ("save", "Trigger a checkpoint save"),
        ("stop", "Send a graceful stop (saves final checkpoint)"),
        ("save-stop", "Save a checkpoint then stop"),
        ("abort", "Abort immediately without saving"),
        ("kill", "Send SIGTERM to the job process group"),
        ("tail", "Stream live TTY output until job ends or Ctrl-C"),
        ("dump", "Write full captured TTY log to stdout"),
        ("logs", "Alias for dump"),
    ]:
        p = subparsers.add_parser(
            name, help=help_text, formatter_class=RawTextHelpFormatter
        )
        p.add_argument("job_id", help="Job ID (queue_id or job_id)")

    fk = subparsers.add_parser(
        "force-kill",
        help="Send SIGKILL to the job process group (destructive)",
        formatter_class=RawTextHelpFormatter,
    )
    fk.add_argument("job_id", help="Job ID")
    fk.add_argument("--yes", action="store_true", help="Required: confirm SIGKILL")

    return parser
