"""Shared submit/orchestrator surface for scheduler-backed launches.

``train``, ``eval``, ``submit`` and ``cluster`` all enqueue work through the
forgather server. The locality decision (server-required vs ``--local-fallback``
vs ``--local-only``), dataset-source routing, and config-name resolution already
exist in :mod:`forgather.cli.diloco_orch` (the orchestrator core, first grown
for DiLoCo). This module re-exports those primitives so a non-DiLoCo caller
imports one coherent surface, and adds the generic enqueue/validation helpers
plus the shared locality CLI flags and a job-tail helper.

The orchestrator primitives are imported *from* ``diloco_orch`` (rather than the
reverse) on purpose: the DiLoCo test-suite monkeypatches ``_orchestrator`` /
``use_orchestrator`` / ``_resolve_config_name`` on that module, so the canonical
definitions must stay there for those seams to work. This module is the lean,
ML-free facade — importing it must not pull in torch.
"""

import sys

# Orchestrator primitives — canonical definitions live in diloco_orch.
from .diloco_orch import _resolve_config_name as resolve_default_config
from .diloco_orch import (  # noqa: F401  (re-exported)
    parse_dataset_source,
    resolve_dataset_source,
    use_orchestrator,
)

# ---------------------------------------------------------------------------
# Shared CLI flags
# ---------------------------------------------------------------------------


def add_locality_args(parser):
    """Add the ``--local-fallback`` / ``--local-only`` locality flags.

    The forgather server is the default, required path. ``--local-fallback``
    degrades to a direct/foreground action only when the server is unreachable;
    ``--local-only`` skips the server entirely. Without either, an unreachable
    server is an error (no silent local degrade). Shared by submit / train
    ``--schedule`` / eval / diloco / dataset-server.
    """
    parser.add_argument(
        "--local-fallback",
        action="store_true",
        help=(
            "If the forgather server isn't reachable, fall back to a\n"
            "direct/foreground action instead of erroring."
        ),
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Never contact the forgather server; act directly/foreground.",
    )


def add_via_server_arg(parser, help=None):
    """Add the ``--via-server URL`` flag (forgather-server base URL override)."""
    parser.add_argument(
        "--via-server",
        type=str,
        default=None,
        metavar="URL",
        help=help
        or (
            "forgather-server base URL to enqueue on "
            "(default: env / http://127.0.0.1:8765)."
        ),
    )


# ---------------------------------------------------------------------------
# Dynamic-args validation (shared by every submitter)
# ---------------------------------------------------------------------------


def collect_dynamic_args(args):
    """Collect + validate a config's dynamic args from the parsed namespace.

    Mirrors the validation ``forgather train`` has always done: enforce
    ``required: true`` schema entries and numeric bounds here (rather than via
    argparse) so the non-action paths (pp / ls / code) don't trip on
    placeholder defaults. Exits(1) with a clear message on a missing-required
    or out-of-bounds value. Returns the ``dynamic_args`` dict.
    """
    from .dynamic_args import (
        get_dynamic_args,
        required_dynamic_arg_dests,
        validate_dynamic_arg_bounds,
    )

    dynamic_args = get_dynamic_args(args)
    required = required_dynamic_arg_dests(args.project_dir, args.config_template)
    missing = [d for d in required if d not in dynamic_args]
    if missing:
        flags = ", ".join(f"--{d.replace('_', '-')}" for d in missing)
        print(f"error: required dynamic arg(s) missing: {flags}", file=sys.stderr)
        raise SystemExit(1)
    bound_errors = validate_dynamic_arg_bounds(
        args.project_dir, args.config_template, dynamic_args
    )
    if bound_errors:
        print(
            "error: dynamic arg constraint violated: " + "; ".join(bound_errors),
            file=sys.stderr,
        )
        raise SystemExit(1)
    return dynamic_args


# ---------------------------------------------------------------------------
# Enqueue
# ---------------------------------------------------------------------------


def submit_single(
    client,
    *,
    project_dir,
    config,
    dynamic_args,
    requested_gpus,
    priority,
    dataset_source=None,
    job_type="training",
    job_params=None,
):
    """Enqueue one single-node job through the orchestrator.

    Returns the created queue item dict (``{queue_id, priority,
    requested_gpus, ...}``). Raises ``ServerUnreachable`` / ``RuntimeError``
    from the client on failure (the caller decides how to report).
    """
    return client.enqueue_job(
        project_dir=project_dir,
        config=config,
        job_type=job_type,
        job_params=job_params or {},
        requested_gpus=requested_gpus,
        priority=priority,
        dynamic_args=dynamic_args or None,
        dataset_source=dataset_source,
    )


# ---------------------------------------------------------------------------
# Attach to a job's captured TTY
# ---------------------------------------------------------------------------


def tail_job(client, job_id):
    """Stream a job's captured TTY to stdout until it exits or Ctrl-C.

    Used by ``--foreground`` submit paths to attach to the job they just
    enqueued, and by ``forgather job tail``.
    """
    import asyncio

    async def _run():
        try:
            async for kind, data in client.stream_tty(job_id, follow=True):
                if kind == "bytes":
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                else:
                    print(f"error: {data}", file=sys.stderr)
        except KeyboardInterrupt:
            pass

    asyncio.run(_run())
