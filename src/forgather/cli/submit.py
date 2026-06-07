"""`forgather submit` — submit the current project's config to the scheduler.

The single entry point for launching a run, mirroring the webui submit modal:

- default: single-node training (same path as `train --schedule`).
- `--global`: multi-node rendezvous fan-out (one torchrun world across nodes).
- `--diloco-server <id>` / `--resume-workers`: DiLoCo worker(s) joining a
  param-server (N independent local-SGD replicas).
- `--global --diloco-server <id>`: composition — the multi-node bundle is one
  logical DiLoCo worker group (e.g. multi-node Pipeline Parallel averaged with
  another such group via DiLoCo). All ranks share one base ``worker_id``; the
  PP callback registers the group with the param-server.

Without ``--diloco-server`` the two axes (``--global`` and ``--diloco``) are
different parallelism models and remain mutually exclusive.
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
    # Composition: --global + --diloco-server is a multi-node bundle whose
    # ranks join one DiLoCo group. --resume-workers and --diloco-worker-count
    # are independent-replica modes that don't compose with --global (one
    # rendezvous = one group; K independent groups is PR-C territory).
    diloco_compose = run_global and bool(getattr(args, "diloco_server", None))
    if diloco_compose:
        if getattr(args, "resume_workers", False):
            print(
                "error: --resume-workers can't be combined with --global "
                "(resume targets the independent-replica DiLoCo flow).",
                file=sys.stderr,
            )
            return 1
        if (getattr(args, "count", 1) or 1) != 1:
            print(
                "error: --diloco-worker-count > 1 doesn't compose with --global "
                "yet (K independent multi-node DiLoCo groups — coming in a "
                "follow-up). Submit one group at a time.",
                file=sys.stderr,
            )
            return 1
    elif diloco_mode and run_global:
        print(
            "error: --global only composes with --diloco-server (a multi-node "
            "bundle joining one DiLoCo group). --diloco / --resume-workers on "
            "their own are independent-replica modes — pick one or pin a "
            "param-server with --diloco-server.",
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

    # Composition takes priority over the bare-DiLoCo dispatch: when both
    # --global and --diloco-server are set, the multi-node bundle path
    # builds the DiLoCo block and the fan-out enqueues it as one logical
    # worker group. Falling through to _worker_cmd here would re-dispatch
    # as N independent single-node workers.
    if diloco_compose:
        return _submit_global(args, submit_orch, config)
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
        if getattr(args, "backend", "http") != "http":
            misused.append("--backend")
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

    # The shared-memory backend has the co-located workers share a CPU master
    # region on one host; it can't span a multi-node fan-out.
    if run_global and getattr(args, "backend", "http") == "shared_memory":
        return _err(
            ["--backend shared_memory"],
            "is single-host; not compatible with --global.",
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
