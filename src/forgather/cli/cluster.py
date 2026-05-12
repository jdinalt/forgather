"""Cluster subcommands for the forgather-server CLI."""

import json
import sys
import time


def _format_delta(ts_or_none):
    if ts_or_none is None:
        return "never"
    import datetime

    try:
        epoch = float(ts_or_none)
    except (ValueError, TypeError):
        return str(ts_or_none)
    ts = datetime.datetime.fromtimestamp(epoch, tz=datetime.timezone.utc)
    now = datetime.datetime.now(datetime.timezone.utc)
    delta = int((now - ts).total_seconds())
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    return f"{delta // 3600}h ago"


def _short_uuid(node_id):
    """First 8 hex chars of a UUID, for compact log output."""
    return (node_id or "")[:8]


def _resolve_host_to_node_id(members, host):
    """Look up a hostname in the membership table.

    Match priority: exact hostname > address. Returns the matching
    member dict or raises a clear RuntimeError. Surface a hint at
    available hostnames when nothing matches — typo detection beats
    uuid-soup error messages.
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


def _gpus_available_on(member):
    """Best-effort GPU count for a member from its probe.

    The cluster /api/cluster/members payload doesn't include the
    full GPU list (that's a separate endpoint), but the probe's
    ``cpu`` summary doesn't either. Return ``None`` here; callers
    that need a GPU count fall back to ``/api/cluster/gpus``
    explicitly.
    """
    return None


def _cmd_nodes(client, args):
    payload = client.cluster_members()
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    cluster = payload.get("cluster_name") or "(standalone)"
    self_id = payload.get("self_node_id")
    master_id = payload.get("master_node_id")
    members = payload.get("members") or []

    if not members:
        print(f"cluster: {cluster}  (no members visible — server in standalone mode?)")
        return 0

    print(f"cluster: {cluster}")
    print(f"master:  {_short_uuid(master_id)}")
    print(f"self:    {_short_uuid(self_id)}")
    print()

    # Pull GPU info too — without it the table is much less useful.
    try:
        gpu_payload = client._get("/cluster/gpus").json()
        gpus_by_node = {n["node_id"]: n.get("gpus", []) for n in gpu_payload.get("nodes", [])}
    except Exception:
        gpus_by_node = {}

    col_w = {"role": 8, "host": 18, "addr": 22, "uuid": 10, "reach": 10, "gpus": 12, "ver": 16}

    def header():
        return (
            f"{'Role':<{col_w['role']}}  "
            f"{'Hostname':<{col_w['host']}}  "
            f"{'Address':<{col_w['addr']}}  "
            f"{'UUID':<{col_w['uuid']}}  "
            f"{'Reachable':<{col_w['reach']}}  "
            f"{'GPUs':<{col_w['gpus']}}  "
            f"{'Version':<{col_w['ver']}}"
        )

    print(header())
    print("-" * (sum(col_w.values()) + 2 * (len(col_w) - 1)))
    for m in members:
        nid = m.get("node_id", "")
        role_bits = []
        if nid == master_id:
            role_bits.append("master")
        if nid == self_id:
            role_bits.append("self")
        role = ",".join(role_bits) if role_bits else "-"
        gpus = gpus_by_node.get(nid, [])
        gpu_str = (
            f"{len(gpus)} ({sum(1 for g in gpus if not g.get('disabled') and not g.get('excluded'))} idle)"
            if gpus
            else "?"
        )
        ver = (m.get("probe") or {}).get("versions", {}).get("forgather") or "?"
        print(
            f"{role:<{col_w['role']}}  "
            f"{(m.get('hostname') or '?'):<{col_w['host']}}  "
            f"{(m.get('address') or '?') + ':' + str(m.get('port') or '?'):<{col_w['addr']}}  "
            f"{_short_uuid(nid):<{col_w['uuid']}}  "
            f"{('yes' if m.get('reachable') else 'NO'):<{col_w['reach']}}  "
            f"{gpu_str:<{col_w['gpus']}}  "
            f"{ver:<{col_w['ver']}}"
        )
    return 0


def _cmd_jobs(client, args):
    if args.cluster_job_id:
        bundle = client.cluster_job_get(args.cluster_job_id)
        if bundle is None:
            print(f"no cluster job with id {args.cluster_job_id}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(bundle, indent=2))
            return 0
        _print_bundle_detail(bundle)
        return 0

    bundles = client.cluster_jobs_list()
    if args.json:
        print(json.dumps(bundles, indent=2))
        return 0

    if not bundles:
        print("(no cluster jobs)")
        return 0

    col_w = {"id": 22, "status": 10, "rdzv": 22, "cfg": 40, "members": 16, "time": 12}

    def header():
        return (
            f"{'Bundle':<{col_w['id']}}  "
            f"{'Status':<{col_w['status']}}  "
            f"{'rdzv':<{col_w['rdzv']}}  "
            f"{'Project/Config':<{col_w['cfg']}}  "
            f"{'Members':<{col_w['members']}}  "
            f"{'Submitted':<{col_w['time']}}"
        )

    print(header())
    print("-" * (sum(col_w.values()) + 2 * (len(col_w) - 1)))
    for b in bundles:
        bid = b.get("cluster_job_id", "")
        status = b.get("rolled_up_status") or b.get("status") or "?"
        rdzv = b.get("rdzv_endpoint") or "?"
        proj = (b.get("project_dir") or "").rsplit("/", 1)[-1]
        cfg = f"{proj}/{b.get('config') or '?'}"
        if len(cfg) > col_w["cfg"]:
            cfg = "..." + cfg[-(col_w["cfg"] - 3) :]
        member_count = len(b.get("members") or [])
        member_str = f"{member_count} ranks"
        time_str = _format_delta(b.get("submitted_at"))
        print(
            f"{bid:<{col_w['id']}}  "
            f"{status:<{col_w['status']}}  "
            f"{rdzv:<{col_w['rdzv']}}  "
            f"{cfg:<{col_w['cfg']}}  "
            f"{member_str:<{col_w['members']}}  "
            f"{time_str:<{col_w['time']}}"
        )
    return 0


def _print_bundle_detail(bundle):
    print(f"bundle:           {bundle.get('cluster_job_id')}")
    print(f"project / config: {bundle.get('project_dir')} / {bundle.get('config')}")
    print(f"rdzv endpoint:    {bundle.get('rdzv_endpoint')}")
    print(f"rdzv id:          {bundle.get('rdzv_id')}")
    print(f"status:           {bundle.get('status')}")
    print(f"rolled-up status: {bundle.get('rolled_up_status')}")
    print(f"submitted:        {_format_delta(bundle.get('submitted_at'))}")
    if bundle.get("cancelled_at"):
        print(f"cancelled:        {_format_delta(bundle.get('cancelled_at'))}")
    print()
    print("Members:")
    for m in bundle.get("members") or []:
        rank = m.get("node_rank")
        host = m.get("hostname")
        nproc = m.get("nproc_per_node")
        cur = m.get("current_status") or "?"
        qid = m.get("queue_id")
        print(f"  rank {rank}: {host} x{nproc}  status={cur}  queue_id={qid}")
        err = m.get("error")
        if err:
            print(f"           error: {err}")


def _parse_member_spec(spec):
    """Parse ``HOST:GPUS[:IFACE]`` into a tuple."""
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


def _parse_dataset_source(spec):
    """Translate the CLI ``--dataset-source`` value to the dict shape
    the forgather_server expects in the submit body.

    Accepted values:
      ``None`` / ``"local"`` / ``""``: omit (training script default).
      ``"auto"``: cluster auto-routing.
      ``"server:<server_id>"``: pin to a specific known server (run
        ``forgather cluster datasets`` for ids).
    """
    if spec is None:
        return None
    spec = spec.strip()
    if spec in ("", "local"):
        return None
    if spec == "auto":
        return {"kind": "auto"}
    if spec.startswith("server:"):
        sid = spec[len("server:") :].strip()
        if not sid:
            raise RuntimeError(
                "--dataset-source server:<id> requires a non-empty id"
            )
        return {"kind": "server", "server_id": sid}
    raise RuntimeError(
        f"unknown --dataset-source value: {spec!r} "
        "(expected 'auto', 'local', or 'server:<id>')"
    )


def _parse_dynamic_args(specs):
    out = {}
    for s in specs:
        if "=" not in s:
            raise RuntimeError(
                f"invalid --dynamic-arg {s!r}: expected KEY=VAL"
            )
        k, v = s.split("=", 1)
        # Cast integers and bools where obvious; otherwise pass through
        # as a string. Server-side preprocessor coerces from there.
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def _cmd_submit(client, args):
    import os

    # ``-p`` / ``-t`` are GLOBAL forgather flags (consumed before the
    # subcommand parser). Validate here so the operator gets a clear
    # message rather than a downstream "project_dir = None" crash.
    project_dir = getattr(args, "project_dir", None)
    config = getattr(args, "config_template", None)
    if not project_dir or project_dir in (".", ""):
        # Default project_dir is "." which is meaningless for a remote
        # peer fanout — every peer would resolve "." to its own cwd.
        # Refuse the submit unless the operator explicitly set -p.
        print(
            "error: cluster submit needs an absolute project path. Pass "
            "``-p <abs-path>`` (a global forgather flag, before "
            "``cluster submit``).",
            file=sys.stderr,
        )
        return 1
    # Reject relative paths even if the operator passed something
    # other than ".". A relative ``-p`` like ``./foo`` resolves
    # against each peer's cwd, which is per-container and almost
    # never the same on every node — silently produces a confusing
    # "no such project" error from one or more peers, hours into
    # debugging. The webui already enforces absolute paths via the
    # config picker; mirror that here.
    if not os.path.isabs(project_dir):
        print(
            f"error: cluster submit needs an absolute project path; got "
            f"relative {project_dir!r}. Resolve with ``readlink -f`` and "
            "pass the canonical path so every peer sees the same files.",
            file=sys.stderr,
        )
        return 1
    if not config:
        print(
            "error: cluster submit needs ``-t <config>`` (a global "
            "forgather flag, before ``cluster submit``).",
            file=sys.stderr,
        )
        return 1

    members_payload = client.cluster_members()
    members = members_payload.get("members") or []
    if not members:
        print(
            "no cluster members visible; is the server running with --cluster?",
            file=sys.stderr,
        )
        return 1

    # If --member specs were passed, resolve hostnames; otherwise default to
    # "every reachable peer with all its GPUs".
    if args.member:
        spec_members = []
        for spec in args.member:
            host, gpus, iface = _parse_member_spec(spec)
            m = _resolve_host_to_node_id(members, host)
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
        rdzv_node_id = _resolve_host_to_node_id(members, args.rdzv_host)["node_id"]

    dynamic_args = _parse_dynamic_args(args.dynamic_arg)
    dataset_source = _parse_dataset_source(getattr(args, "dataset_source", None))

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
        print(json.dumps(resp, indent=2))
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

    # --wait: poll until terminal, exit non-zero on failure
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
    print(f"timed out after {args.wait_timeout}s waiting for terminal status", file=sys.stderr)
    return 3


def _cmd_cancel(client, args):
    resp = client.cluster_job_cancel(args.cluster_job_id)
    cancelled = resp.get("cancelled")
    print(f"cancelled: {cancelled}")
    for entry in resp.get("per_member") or []:
        result = entry.get("result")
        if isinstance(result, dict):
            print(
                f"  {_short_uuid(entry.get('node_id'))} "
                f"queue={entry.get('queue_id')}  "
                f"cancelled={result.get('cancelled')}  "
                f"detail={result.get('detail') or ''}"
            )
        else:
            print(
                f"  {_short_uuid(entry.get('node_id'))} "
                f"queue={entry.get('queue_id')}  "
                f"result={result}"
            )
    return 0 if cancelled else 1


def _cmd_datasets(client, args):
    """Cluster-wide dataset-server inventory: which servers the master
    has aggregated, which datasets are routable, where each one lives.

    Useful as a smoke test before launching cluster auto-routing
    training — if a dataset doesn't show up here, the router can't
    serve it to a training client.
    """
    import json as _json

    inv = client.cluster_dataset_inventory()
    if args.json:
        print(_json.dumps(inv, indent=2))
        return 0

    is_master = inv.get("is_master")
    warm_ts = inv.get("last_dataset_pass_ts")
    print(
        f"is_master={is_master} "
        f"servers={len(inv.get('servers', []))} "
        f"datasets={len(inv.get('datasets', []))} "
        f"last_pass={_format_delta(warm_ts)}"
    )
    if not is_master:
        print(
            "Note: this node is not master. Values were proxied from the "
            "master if reachable; an empty payload means the master is "
            "unreachable.",
            file=sys.stderr,
        )

    servers = inv.get("servers", [])
    if servers:
        print()
        print("SERVERS")
        for s in servers:
            health = "ok" if s.get("healthy") else "DOWN"
            err = s.get("last_health_error") or ""
            err_suffix = f"  err={err}" if err else ""
            print(
                f"  {s.get('server_id', '?')[:14]:<14} "
                f"{s.get('base_url', '?'):<45} "
                f"{s.get('source', '?'):<6} "
                f"peer={(s.get('peer_node_id') or '-')[:8]:<8} "
                f"{health}{err_suffix}"
            )
    datasets = inv.get("datasets", [])
    if datasets:
        print()
        print("DATASETS")
        server_id_to_url = {s["server_id"]: s["base_url"] for s in servers}
        for d in datasets:
            ids = d.get("server_ids", [])
            hosts = [
                server_id_to_url.get(sid, sid)[:30] for sid in ids
            ]
            length = d.get("length")
            length_str = str(length) if length is not None else "?"
            print(
                f"  {d.get('dataset_id', '?'):<28} "
                f"{d.get('source', '?'):<6} "
                f"len={length_str:<10} "
                f"hosts={', '.join(hosts)}"
            )
    return 0


def cluster_cmd(args):
    from .server_client import ServerClient, ServerUnreachable

    client = ServerClient.from_args(args)
    sub = getattr(args, "cluster_subcommand", None)
    if sub is None:
        print(
            "error: specify a subcommand (nodes, jobs, submit, cancel, datasets)",
            file=sys.stderr,
        )
        return 1

    try:
        if sub == "nodes":
            return _cmd_nodes(client, args)
        if sub == "jobs":
            return _cmd_jobs(client, args)
        if sub == "submit":
            return _cmd_submit(client, args)
        if sub == "cancel":
            return _cmd_cancel(client, args)
        if sub == "datasets":
            return _cmd_datasets(client, args)
        print(f"error: unknown subcommand {sub!r}", file=sys.stderr)
        return 1
    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
