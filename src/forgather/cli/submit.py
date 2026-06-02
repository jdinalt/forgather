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

    diloco_mode = bool(getattr(args, "server", None)) or getattr(
        args, "resume_workers", False
    )
    run_global = getattr(args, "run_global", False)

    if diloco_mode and run_global:
        print(
            "error: --global and the DiLoCo opt-in (--diloco-server / "
            "--resume-workers) can't be combined — they're different "
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
        # DiLoCo worker(s): the shared worker-launch impl (also reached by the
        # deprecated `forgather diloco worker`). --diloco-server maps to its
        # param-server arg (dest="server").
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

    # DiLoCo-worker knobs require the DiLoCo opt-in.
    if not diloco_mode:
        misused = []
        if getattr(args, "count", 1) != 1:
            misused.append("--count")
        if getattr(args, "worker_id", None):
            misused.append("--worker-id")
        if getattr(args, "heartbeat_interval", 30.0) != 30.0:
            misused.append("--heartbeat-interval")
        if getattr(args, "gpus_per_worker", 1) != 1:
            misused.append("--gpus-per-worker")
        if getattr(args, "devices", None):
            misused.append("--devices")
        if getattr(args, "dry_run", False):
            misused.append("--dry-run")
        if misused:
            return _err(
                misused,
                "apply only to DiLoCo workers; pass --diloco-server <id> "
                "(or --resume-workers).",
            )

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
            return _err(misused, "apply only to multi-node submit; pass --global.")

    # --requested-gpus is the single-node GPU knob; the other modes have their
    # own (DiLoCo: --gpus-per-worker; --global: per-member counts).
    if (diloco_mode or run_global) and getattr(
        args, "requested_gpus", None
    ) is not None:
        alt = "--gpus-per-worker" if diloco_mode else "the per-member GPU count"
        return _err(["--requested-gpus"], f"is single-node only; use {alt}.")

    return None


def _submit_single(args, submit_orch, config):
    """Single-node submit: identical to `forgather train --schedule`."""
    from .train import train_cmd

    # Drive train's scheduled path. submit has no foreground torchrun fallback
    # of its own — --local-only / --local-fallback resolve through train_cmd.
    args.schedule = True
    args.enqueue = False
    args.devices = None
    args.nproc = None
    args.dry_run = False
    train_cmd(args)
    return 0


def _submit_global(args, submit_orch, config):
    """Multi-node fan-out across the cluster."""
    from .server_client import ServerClient, ServerUnreachable

    client = ServerClient(getattr(args, "via_server", None) or None)
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
