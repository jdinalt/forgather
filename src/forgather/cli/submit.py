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

    # The collective backend runs N replicas as ONE torchrun job bound to a
    # coordinator — its own dispatch path, distinct from the N-worker-job flow.
    # It implies DiLoCo mode (it needs the coordinator for /info + dataset
    # shards). Collective is a launch *topology*, selected by the replica count
    # (--diloco-replicate N), not by --backend — the sync backend itself is
    # server-authoritative and derived at launch (issue #154).
    collective_mode = int(getattr(args, "replicate", 1) or 1) > 1
    diloco_mode = (
        getattr(args, "diloco", False)
        or bool(getattr(args, "diloco_server", None))
        or getattr(args, "resume_workers", False)
        or collective_mode
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
    if collective_mode:
        # Collective backend: one torchrun job of N replicas bound to a
        # coordinator (not N worker jobs). Intercepted before the worker path.
        from .diloco_orch import launch_collective

        dynamic_args = submit_orch.collect_dynamic_args(args)
        return launch_collective(args, dynamic_args) or 0
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
        if (getattr(args, "backend", None) or "http").strip().lower() != "http":
            misused.append("--backend")
        if getattr(args, "worker_env", None):
            misused.append("--env")
        if misused:
            return _err(misused, "— DiLoCo-worker flag(s); pass --diloco.")

    # --backend is server-authoritative in the orchestrated path: the sync
    # backend is declared once on the param server and derived at launch from
    # /info (issue #154). It's accepted only on the --local-only dev path, where
    # torchrun is invoked directly and there's no server to query. 'collective'
    # is a launch topology, selected with --diloco-replicate, never via --backend.
    backend = (getattr(args, "backend", None) or "").strip().lower()
    if backend == "collective":
        return _err(
            ["--backend collective"],
            "isn't a submit option: collective is a launch topology, selected "
            "with --diloco-replicate N.",
        )
    if backend and not getattr(args, "local_only", False):
        return _err(
            ["--backend"],
            "is only honored with --local-only (dev/debug); the orchestrated "
            "path derives the backend from the param server (set it on the "
            "'diloco server').",
        )

    # --env is threaded into job_params.extra_env only on the plain DiLoCo-worker
    # path (launch_workers / launch_resume → _enqueue_worker_jobs). The collective
    # topology (launch_collective) and the --global + --diloco-server compose
    # bundle (_submit_global) build their own job_params and don't carry it, so
    # reject --env there rather than silently dropping it (fail loud).
    if getattr(args, "worker_env", None):
        compose = run_global and bool(getattr(args, "diloco_server", None))
        collective = int(getattr(args, "replicate", 1) or 1) > 1
        if compose or collective:
            return _err(
                ["--env"],
                "isn't threaded on the collective (--diloco-replicate) or "
                "--global compose path; it's honored only on plain --diloco "
                "worker submits.",
            )

    # --local-fallback degrades to a local foreground launch when the server is
    # down — but then there's no server to derive the backend from, and the
    # backend isn't passed (orchestrated). Reject it for DiLoCo submits: run with
    # the server up (orchestrated), or commit to --local-only --backend <kind>.
    if diloco_mode and getattr(args, "local_fallback", False):
        return _err(
            ["--local-fallback"],
            "can't derive a backend when it degrades to local. Run with the "
            "param server up (orchestrated), or use --local-only --backend "
            "<kind>.",
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
            return _err(misused, "— multi-node flag(s); pass --global.")

    # --requested-gpus sizes a single-node job or a DiLoCo worker; --global
    # sizes nodes via --member instead.
    if run_global and getattr(args, "requested_gpus", None) is not None:
        return _err(
            ["--requested-gpus"],
            "doesn't apply to --global; size each node with --member HOST:GPUS.",
        )

    # Collective (--diloco-replicate N) is one single-host torchrun world; it
    # can't span a multi-node fan-out, and --diloco-worker-count (N separate
    # jobs) is a different model that doesn't compose with it.
    replicate = int(getattr(args, "replicate", 1) or 1)
    if replicate > 1 and run_global:
        return _err(
            ["--diloco-replicate"],
            "selects the single-host collective topology; not compatible with "
            "--global.",
        )
    if replicate > 1 and getattr(args, "count", 1) != 1:
        return _err(
            ["--diloco-worker-count"],
            "is the N-separate-jobs model; collective is one job — size it with "
            "--diloco-replicate alone.",
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
