"""Deprecated alias of `forgather job` for scheduler/queue verbs.

`forgather sched <verb>` historically managed the queue and scheduler. Those
verbs now live under `forgather job` (one coherent surface for jobs + queue +
scheduler). This shim remaps the old namespace onto `job_cmd` and prints a
one-line deprecation note, so existing scripts keep working.
"""

import sys


def sched_cmd(args):
    from .job import job_cmd

    sub = getattr(args, "sched_subcommand", None)

    # Map the old sched verbs onto the new job verbs.
    if sub in ("status", "pause", "resume"):
        new = f"job scheduler {sub}"
        args.job_subcommand = "scheduler"
        args.scheduler_action = sub
    elif sub in ("list", "cancel", "cleanup", "gc"):
        new = f"job {sub}"
        args.job_subcommand = sub
    elif sub is None:
        print(
            "error: specify a subcommand (status, list, pause, resume, cancel, "
            "cleanup, gc). Note: 'forgather sched' is deprecated; use "
            "'forgather job'.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print(f"error: unknown subcommand: {sub}", file=sys.stderr)
        sys.exit(1)

    print(
        f"note: 'forgather sched {sub}' is deprecated; use 'forgather {new}'.",
        file=sys.stderr,
    )
    job_cmd(args)
