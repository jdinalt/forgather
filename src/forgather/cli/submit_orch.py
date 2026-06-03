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


def validate_dynamic_args(project_dir, config_template, dynamic_args):
    """Enforce a config's ``required: true`` dynamic args and numeric bounds.

    Done here (rather than via argparse) so the non-action paths (pp / ls /
    code) don't trip on placeholder defaults. Exits(1) with a clear message on
    a missing-required or out-of-bounds value. Shared by ``collect_dynamic_args``
    and the DiLoCo worker path (which collects its dynamic args differently but
    must enforce the same constraints).
    """
    from .dynamic_args import required_dynamic_arg_dests, validate_dynamic_arg_bounds

    required = required_dynamic_arg_dests(project_dir, config_template)
    missing = [d for d in required if d not in dynamic_args]
    if missing:
        flags = ", ".join(f"--{d.replace('_', '-')}" for d in missing)
        print(f"error: required dynamic arg(s) missing: {flags}", file=sys.stderr)
        raise SystemExit(1)
    bound_errors = validate_dynamic_arg_bounds(
        project_dir, config_template, dynamic_args
    )
    if bound_errors:
        print(
            "error: dynamic arg constraint violated: " + "; ".join(bound_errors),
            file=sys.stderr,
        )
        raise SystemExit(1)


def collect_dynamic_args(args):
    """Collect + validate a config's dynamic args from the parsed namespace.

    Mirrors the validation ``forgather train`` has always done. Returns the
    ``dynamic_args`` dict.
    """
    from .dynamic_args import get_dynamic_args

    dynamic_args = get_dynamic_args(args)
    validate_dynamic_args(args.project_dir, args.config_template, dynamic_args)
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


# ---------------------------------------------------------------------------
# Multi-node fan-out (--global / the old `cluster submit`)
# ---------------------------------------------------------------------------


def parse_member_spec(spec):
    """Parse ``HOST:GPUS[:IFACE]`` into ``(host, gpus:int, iface|None)``."""
    parts = spec.split(":")
    if len(parts) == 2:
        host, gpus = parts
        iface = None
    elif len(parts) == 3:
        host, gpus, iface = parts
    else:
        raise RuntimeError(
            f"invalid --member spec {spec!r}: expected HOST:GPUS[:IFACE]"
        )
    try:
        n = int(gpus)
    except ValueError:
        raise RuntimeError(f"invalid GPU count in --member {spec!r}: {gpus!r}")
    if n < 1:
        raise RuntimeError(f"GPU count must be >= 1 in --member {spec!r}")
    return host, n, (iface or None)


def resolve_host_to_node_id(members, host):
    """Look up a hostname in the membership table.

    Match priority: exact hostname > address. Returns the matching member dict
    or raises a clear RuntimeError listing available hostnames.
    """
    for m in members:
        if m.get("hostname") == host:
            return m
    for m in members:
        if m.get("address") == host:
            return m
    available = ", ".join(m.get("hostname", "?") for m in members) or "(none)"
    raise RuntimeError(
        f"no cluster member matches hostname {host!r}; available: {available}"
    )


def submit_global(client, args, *, project_dir, config, dynamic_args, dataset_source):
    """Submit a multi-node training fan-out (the `submit --global` core).

    Shared by `forgather submit --global` and the deprecated `forgather cluster
    submit`. The caller resolves ``dynamic_args`` (submit uses the config
    schema; cluster's legacy path uses ad-hoc KEY=VAL) and ``dataset_source``;
    everything else (member resolution, rdzv, the cluster_jobs_submit call,
    rendering, and --wait polling) lives here. Reads the multi-node knobs off
    ``args``: member, rdzv_host, rdzv_port, priority, allow_version_mismatch,
    json, wait, wait_timeout, poll_interval. Returns a process exit code.
    """
    import json as _json
    import os
    import time

    # A multi-node fan-out resolves the project path on every peer, so a
    # relative path (or the default ".") resolves against each peer's own
    # cwd — almost never the same files. Require an absolute path, like the
    # webui config picker.
    if not project_dir or project_dir in (".", ""):
        print(
            "error: multi-node submit needs an absolute project path. Pass "
            "`-p <abs-path>` (a global forgather flag, before the subcommand).",
            file=sys.stderr,
        )
        return 1
    if not os.path.isabs(project_dir):
        print(
            f"error: multi-node submit needs an absolute project path; got "
            f"relative {project_dir!r}. Resolve with `readlink -f` so every "
            "peer sees the same files.",
            file=sys.stderr,
        )
        return 1
    if not config:
        print(
            "error: multi-node submit needs a config (`-t <config>`, a global "
            "forgather flag, before the subcommand).",
            file=sys.stderr,
        )
        return 1

    members_payload = client.cluster_members()
    members = members_payload.get("members") or []
    if not members:
        print(
            "no cluster members visible; is the server running?",
            file=sys.stderr,
        )
        return 1

    # --member specs (comma- or repeat-separated) resolve hostnames; with none,
    # default to "every reachable peer with all its available GPUs".
    specs = [s for entry in (args.member or []) for s in entry.split(",") if s]
    if specs:
        spec_members = []
        for spec in specs:
            host, gpus, iface = parse_member_spec(spec)
            m = resolve_host_to_node_id(members, host)
            if not m.get("reachable"):
                print(
                    f"warning: member {host} is currently unreachable; submit will fail",
                    file=sys.stderr,
                )
            spec_members.append(
                {
                    "node_id": m["node_id"],
                    "nproc_per_node": gpus,
                    "nccl_socket_ifname": iface,
                }
            )
    else:
        try:
            gpu_payload = client._get("/cluster/gpus").json()
        except Exception as e:
            print(
                f"could not enumerate cluster GPUs to default --member: {e}; "
                "pass --member explicitly",
                file=sys.stderr,
            )
            return 1
        nodes = gpu_payload.get("nodes") or []
        spec_members = []
        for node in nodes:
            if not node.get("reachable"):
                continue
            count = sum(
                1
                for g in (node.get("gpus") or [])
                if not g.get("disabled") and not g.get("excluded")
            )
            if count < 1:
                continue
            spec_members.append(
                {
                    "node_id": node["node_id"],
                    "nproc_per_node": count,
                    "nccl_socket_ifname": None,
                }
            )
        if not spec_members:
            print(
                "no reachable cluster member has any usable GPUs; "
                "pass --member explicitly to override",
                file=sys.stderr,
            )
            return 1

    rdzv_node_id = None
    if args.rdzv_host:
        rdzv_node_id = resolve_host_to_node_id(members, args.rdzv_host)["node_id"]

    if getattr(args, "dry_run", False):
        print(f"[dry-run] would submit multi-node bundle: config={config}")
        print(f"  project={project_dir}")
        for m in spec_members:
            iface = m.get("nccl_socket_ifname")
            suffix = f" iface={iface}" if iface else ""
            print(f"  node {m['node_id']} x{m['nproc_per_node']}{suffix}")
        if dynamic_args:
            print(f"  dynamic_args={dynamic_args}")
        return 0

    resp = client.cluster_jobs_submit(
        project_dir=project_dir,
        config=config,
        members=spec_members,
        dynamic_args=dynamic_args,
        priority=args.priority,
        rdzv_node_id=rdzv_node_id,
        rdzv_port=args.rdzv_port,
        allow_version_mismatch=args.allow_version_mismatch,
        dataset_source=dataset_source,
    )

    bundle = resp.get("cluster_job") or {}
    warnings = resp.get("warnings") or []

    if args.json:
        print(_json.dumps(resp, indent=2))
    else:
        bid = bundle.get("cluster_job_id")
        rdzv = bundle.get("rdzv_endpoint")
        print(f"submitted: {bid}")
        print(f"rdzv:      {rdzv}")
        print("members:")
        for m in bundle.get("members") or []:
            print(
                f"  rank {m.get('node_rank')}: "
                f"{m.get('hostname')} x{m.get('nproc_per_node')}  "
                f"queue_id={m.get('queue_id')}"
            )
        for w in warnings:
            print(f"warning: {w}", file=sys.stderr)

    if not args.wait:
        return 0

    # --wait: poll until terminal, exit non-zero on failure.
    bundle_id = bundle.get("cluster_job_id")
    if not bundle_id:
        print("no bundle id in response; cannot wait", file=sys.stderr)
        return 1

    deadline = time.monotonic() + max(0, args.wait_timeout)
    last_status = None
    while time.monotonic() < deadline:
        time.sleep(args.poll_interval)
        try:
            cur = client.cluster_job_get(bundle_id) or {}
        except Exception as e:
            print(f"poll error (continuing): {e}", file=sys.stderr)
            continue
        rolled = cur.get("rolled_up_status") or cur.get("status")
        if rolled != last_status:
            print(f"status: {rolled}")
            last_status = rolled
        if rolled in ("done", "failed", "cancelled"):
            return 0 if rolled == "done" else 2
    print(
        f"timed out after {args.wait_timeout}s waiting for terminal status",
        file=sys.stderr,
    )
    return 3


def attach_submitted(client, queue_id, *, poll_interval=1.0):
    """Wait for an enqueued job to be dispatched, then stream its TTY.

    Backs the ``--foreground`` submit paths: a job is queued first and only
    gets a captured TTY once the scheduler dispatches it (it appears in
    ``list_jobs`` at that point). Poll the queue→jobs transition, then tail.
    Ctrl-C detaches the terminal without stopping the job (use
    ``forgather job stop`` to actually stop it).
    """
    import time

    from .server_client import ServerUnreachable

    print(
        f"queued: {queue_id}; waiting for the scheduler to start it "
        "(Ctrl-C detaches without stopping the job)...",
        file=sys.stderr,
    )
    try:
        while True:
            try:
                jobs = client.list_jobs(include_dead=True)
            except (ServerUnreachable, RuntimeError) as e:
                print(f"error: {e}", file=sys.stderr)
                return
            if any(j.get("queue_id") == queue_id for j in jobs):
                break  # dispatched — a TTY now exists to stream
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print(
            f"\ndetached; {queue_id} is still queued "
            "('forgather job cancel' to remove it).",
            file=sys.stderr,
        )
        return
    tail_job(client, queue_id)
