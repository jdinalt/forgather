"""`forgather submit` — submit the current project's config to the scheduler.

The single entry point for launching a run, mirroring the webui submit modal:

- default: single-node training (same path as `train --schedule`).
- `--global`: multi-node rendezvous fan-out (one torchrun world across nodes).
- `--diloco-server <id>` / `--resume-workers`: DiLoCo worker(s) joining a
  param-server (N independent local-SGD replicas).

`--global` and the DiLoCo opt-in are different parallelism axes and are mutually
exclusive.
"""

import os
import sys


def submit_cmd(args):
    from . import submit_orch

    diloco_mode = (
        getattr(args, "diloco", False)
        or bool(getattr(args, "diloco_server", None))
        or getattr(args, "resume_workers", False)
    )
    run_global = getattr(args, "run_global", False)

    if diloco_mode and run_global:
        print(
            "error: --global and --diloco can't be combined — they're different "
            "parallelism models (--global is one rendezvous across nodes; "
            "DiLoCo is independent local-SGD replicas). Pick one.",
            file=sys.stderr,
        )
        return 1

    # Fail loud on flags that belong to a different mode (no silent no-op).
    rc = _check_mode_flags(args, diloco_mode, run_global)
    if rc:
        return rc

    # Resolve the config: explicit -t, else the project's default_config.
    config = getattr(
        args, "config_template", None
    ) or submit_orch.resolve_default_config(args)
    if not config:
        print(
            "error: no config to submit. Pass -t <config> (a global forgather "
            "flag, before 'submit') or set the project's default_config.",
            file=sys.stderr,
        )
        return 1
    args.config_template = config

    if diloco_mode:
        # DiLoCo worker(s) via the shared worker-launch impl. --diloco selects
        # the mode; --diloco-server (dest=diloco_server) pins a param-server;
        # per-worker GPUs come from --requested-gpus.
        from .diloco import _worker_cmd

        return _worker_cmd(args) or 0
    if run_global:
        return _submit_global(args, submit_orch, config)
    return _submit_single(args, submit_orch, config)


def _check_mode_flags(args, diloco_mode, run_global):
    """Reject flags that belong to a different submit mode (fail loud).

    The submit parser carries flags for all three modes (single-node, --global,
    DiLoCo); a flag from the wrong mode would otherwise be parsed and silently
    ignored. Returns a non-zero exit code on a misuse, else None.
    """

    def _err(flags, msg):
        print(f"error: {', '.join(flags)} {msg}", file=sys.stderr)
        return 1

    # DiLoCo-worker knobs require --diloco.
    if not diloco_mode:
        misused = []
        if getattr(args, "count", 1) != 1:
            misused.append("--diloco-worker-count")
        if getattr(args, "worker_id", None):
            misused.append("--worker-id")
        if getattr(args, "heartbeat_interval", 30.0) != 30.0:
            misused.append("--heartbeat-interval")
        if misused:
            return _err(misused, "— DiLoCo-worker flag(s); pass --diloco.")

    # Multi-node knobs require --global.
    if not run_global:
        misused = []
        if getattr(args, "member", None):
            misused.append("--member")
        if getattr(args, "rdzv_host", None):
            misused.append("--rdzv-host")
        if getattr(args, "rdzv_port", None) is not None:
            misused.append("--rdzv-port")
        if getattr(args, "allow_version_mismatch", False):
            misused.append("--allow-version-mismatch")
        if getattr(args, "wait", False):
            misused.append("--wait")
        if misused:
            return _err(misused, "— multi-node flag(s); pass --global.")

    # --requested-gpus sizes a single-node job or a DiLoCo worker; --global
    # sizes nodes via --member instead.
    if run_global and getattr(args, "requested_gpus", None) is not None:
        return _err(
            ["--requested-gpus"],
            "doesn't apply to --global; size each node with --member HOST:GPUS.",
        )

    return None


def _submit_single(args, submit_orch, config):
    """Single-node submit: identical to `forgather train --schedule`."""
    from .train import train_cmd

    # Drive train's scheduled path. submit has no foreground torchrun fallback
    # of its own — --local-only / --local-fallback resolve through train_cmd.
    # --dry-run flows through (train's schedule path prints the request).
    args.schedule = True
    args.enqueue = False
    args.devices = None
    args.nproc = None
    train_cmd(args)
    return 0


def _submit_global(args, submit_orch, config):
    """Multi-node fan-out across the cluster."""
    from .server_client import ServerClient, ServerUnreachable

    client = ServerClient(getattr(args, "server", None) or None)
    try:
        dataset_source = submit_orch.resolve_dataset_source(client, args)
        dynamic_args = submit_orch.collect_dynamic_args(args)
        # Multi-node needs an absolute project path (every peer resolves it);
        # submit_global validates and reports.
        return submit_orch.submit_global(
            client,
            args,
            project_dir=getattr(args, "project_dir", None),
            config=config,
            dynamic_args=dynamic_args,
            dataset_source=dataset_source,
        )
    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as e:
        # ValueError: a bad --dataset value (parse_dataset_source).
        print(str(e), file=sys.stderr)
        return 1
