"""Argument parser for job command (jobs + queue + scheduler)."""

import argparse
from argparse import RawTextHelpFormatter


def create_job_parser(global_args):
    parser = argparse.ArgumentParser(
        prog="forgather job",
        description="Inspect and control server-managed jobs and the scheduler queue",
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

    # Per-job control / inspection — each takes a job id (queue_id or job_id).
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

    # Queue-level verbs (formerly `forgather sched`).
    subparsers.add_parser(
        "list",
        help="List queued and active jobs",
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
            "Sweep orphan TTY logs from ~/.config/forgather/server/jobs/ "
            "(files not referenced by any record, older than the TTL)"
        ),
        formatter_class=RawTextHelpFormatter,
    )

    # Scheduler control (formerly `forgather sched status/pause/resume`).
    sched_parser = subparsers.add_parser(
        "scheduler",
        help="Show or control the scheduler (status / pause / resume)",
        formatter_class=RawTextHelpFormatter,
    )
    sched_sub = sched_parser.add_subparsers(
        dest="scheduler_action", help="Scheduler actions"
    )
    sched_sub.add_parser(
        "status",
        help="Show scheduler status and queue/job counts",
        formatter_class=RawTextHelpFormatter,
    )
    sched_sub.add_parser(
        "pause",
        help="Pause the scheduler (stop dispatching new jobs)",
        formatter_class=RawTextHelpFormatter,
    )
    sched_sub.add_parser(
        "resume",
        help="Resume the scheduler",
        formatter_class=RawTextHelpFormatter,
    )

    return parser
