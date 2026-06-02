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
    # 0 is the dataclass default for "never set yet" — render as
    # ``never`` rather than "55 years ago", which would falsely
    # suggest a stuck loop.
    if epoch <= 0:
        return "never"
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
        gpus_by_node = {
            n["node_id"]: n.get("gpus", []) for n in gpu_payload.get("nodes", [])
        }
    except Exception:
        gpus_by_node = {}

    col_w = {
        "role": 8,
        "host": 18,
        "addr": 22,
        "uuid": 10,
        "reach": 10,
        "gpus": 12,
        "ver": 16,
    }

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
    ds = bundle.get("dataset_source")
    if ds:
        # Render the kind plus a compact key=val tail when relevant
        # (server_id for the pinned-server case). "local" / None is
        # the default loader — same convention as the submit modal.
        kind = ds.get("kind", "?")
        extra = ""
        if kind == "server" and ds.get("server_id"):
            extra = f"  server_id={ds['server_id']}"
        print(f"dataset source:   {kind}{extra}")
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


def _parse_dynamic_args(specs):
    out = {}
    for s in specs:
        if "=" not in s:
            raise RuntimeError(f"invalid --dynamic-arg {s!r}: expected KEY=VAL")
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
    """Deprecated alias of `forgather submit --global`.

    The multi-node fan-out core now lives in submit_orch.submit_global, shared
    with the `submit` verb. This keeps the legacy ad-hoc `--dynamic-arg KEY=VAL`
    parsing (the submit verb uses the richer config schema instead) and prints
    a one-line deprecation note.
    """
    from . import submit_orch

    project_dir = getattr(args, "project_dir", None)
    config = getattr(args, "config_template", None)
    dynamic_args = _parse_dynamic_args(args.dynamic_arg)
    dataset_source = submit_orch.parse_dataset_source(
        getattr(args, "dataset_source", None)
    )

    print(
        "note: 'forgather cluster submit' is deprecated; "
        "use 'forgather submit --global'.",
        file=sys.stderr,
    )
    return submit_orch.submit_global(
        client,
        args,
        project_dir=project_dir,
        config=config,
        dynamic_args=dynamic_args,
        dataset_source=dataset_source,
    )


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


def _cmd_resolve(client, args):
    """Dry-run the cluster dataset router for a given path.

    Useful smoke test before launching training — if this returns 503
    or 410, training would have the same outcome (modulo retry).
    """
    import json as _json

    try:
        # CLI still accepts ``args.path`` for user-facing simplicity
        # ("the dataset path you'd use in fast_load_iterable_dataset");
        # the wire param it maps onto is ``dataset_id``.
        resp = client.cluster_dataset_resolve(args.path)
    except RuntimeError as e:
        # 503 / 410 / other server errors come back here. Print the
        # detail and exit non-zero so scripts can branch on it.
        print(f"resolve failed: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(_json.dumps(resp, indent=2))
        return 0
    print(f"server_id: {resp.get('server_id', '?')}")
    print(f"base_url:  {resp.get('base_url', '?')}")
    has_token = bool(resp.get("auth_token"))
    print(f"token:     {'(set)' if has_token else '(no-auth)'}")
    return 0


def _cmd_server(client, args):
    """Cluster-proxied diagnostics for a single dataset_server.

    Mirrors the per-server ``forgather dataset-server status|list|cache|local``
    CLI but routes through the master so the operator only needs the
    cluster bearer, not each peer's dataset_server bearer.
    """
    import json as _json

    op = args.op
    if op == "status":
        try:
            health = client.cluster_server_proxy_get(args.server_id, "health")
            auth_status = client.cluster_server_proxy_get(args.server_id, "auth-status")
        except RuntimeError as e:
            print(f"status failed: {e}", file=sys.stderr)
            return 1
        if args.json:
            print(_json.dumps({"health": health, "auth_status": auth_status}, indent=2))
            return 0
        print(
            f"service: {health.get('service', '?')}  version: {health.get('version', '?')}"
        )
        print(f"status:  {health.get('status', '?')}")
        policy = health.get("policy") or {}
        if policy:
            print("policy:")
            for k, v in policy.items():
                print(f"  {k:<20}{v}")
        print(f"auth_required: {auth_status.get('auth_required')}")
        return 0

    # list / cache / local: fetch + render.
    upstream_op = {"list": "datasets", "cache": "cache", "local": "local"}[op]
    try:
        body = client.cluster_server_proxy_get(args.server_id, upstream_op)
    except RuntimeError as e:
        print(f"{op} failed: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(_json.dumps(body, indent=2))
        return 0

    if op == "list":
        handles = body.get("handles") if isinstance(body, dict) else body
        if not handles:
            print("(no datasets loaded on this server)")
            return 0
        print(f"{len(handles)} dataset handle(s):")
        for h in handles:
            la = h.get("load_args") or {}
            print(
                f"  {h.get('handle', '?')[:16]:<17} "
                f"src={h.get('source', '?'):<6} "
                f"len={h.get('length', '?'):<10} "
                f"args={la}"
            )
        return 0

    if op == "cache":
        datasets = body.get("datasets") if isinstance(body, dict) else None
        if not datasets:
            print("(HF cache empty or unreadable)")
            return 0
        print(f"cache_root: {body.get('cache_root', '?')}")
        print(f"{len(datasets)} cached repo(s):")
        for d in datasets:
            cfgs = d.get("configs") or []
            size_mb = (d.get("size_bytes") or 0) / (1024 * 1024)
            print(f"  {d.get('repo', '?')}  ({size_mb:.1f} MB, {len(cfgs)} config(s))")
            for c in cfgs:
                splits = c.get("splits") or []
                split_names = ", ".join(s.get("name", "?") for s in splits)
                print(f"    {c.get('config', '?')}: splits=[{split_names}]")
        return 0

    if op == "local":
        entries = body.get("local") if isinstance(body, dict) else body
        if not entries:
            print("(no local/<name> mappings registered)")
            return 0
        if isinstance(entries, dict):
            entries = [
                {"name": k, **(v if isinstance(v, dict) else {})}
                for k, v in entries.items()
            ]
        print(f"{len(entries)} local mapping(s):")
        for e in entries:
            print(
                f"  local/{e.get('name', '?'):<20} "
                f"path={e.get('path', '?')} "
                f"len={e.get('length', '?')}"
            )
        return 0

    return 1  # unreachable — argparse already restricts choices


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
    if args.verbose:
        m = inv.get("metrics") or {}
        # Master age (seconds since this node became master), and
        # the cumulative poll/failure counts across all servers —
        # quickest "is the inventory keeping up" signal from one
        # glance.
        master_age = m.get("master_age_seconds")
        master_age_str = (
            f"{int(master_age)}s" if isinstance(master_age, (int, float)) else "n/a"
        )
        print(
            f"  healthy={m.get('healthy_servers', 0)}/{m.get('total_servers', 0)}  "
            f"polls(health)={m.get('total_health_failures', 0)}/{m.get('total_health_polls', 0)} failed  "
            f"polls(datasets)={m.get('total_dataset_failures', 0)}/{m.get('total_dataset_polls', 0)} failed  "
            f"master_age={master_age_str}"
        )

    verbose = getattr(args, "verbose", False)

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
            if verbose:
                # Show per-server health + refresh ages so an
                # operator can spot a stuck loop. Errors carry through
                # untruncated so the failure mode is obvious.
                print(
                    f"      last_health_check:    {_format_delta(s.get('last_health_check'))}"
                )
                print(
                    f"      last_dataset_refresh: {_format_delta(s.get('last_dataset_refresh'))}"
                )
                if s.get("last_dataset_error"):
                    print(f"      last_dataset_error:   {s['last_dataset_error']}")
                # Poll counts. ``consecutive_*`` is the "currently
                # stuck?" signal independent of total counts.
                h_total = s.get("total_health_polls", 0) or 0
                h_fail = s.get("health_failures", 0) or 0
                h_streak = s.get("consecutive_health_failures", 0) or 0
                d_total = s.get("total_dataset_polls", 0) or 0
                d_fail = s.get("dataset_failures", 0) or 0
                d_streak = s.get("consecutive_dataset_failures", 0) or 0
                streak = ""
                if h_streak > 0 or d_streak > 0:
                    streak = f" (streak: health={h_streak} datasets={d_streak})"
                print(
                    f"      polls: health={h_fail}/{h_total} failed, "
                    f"datasets={d_fail}/{d_total} failed{streak}"
                )
    datasets = inv.get("datasets", [])
    if datasets:
        print()
        print("DATASETS")
        server_id_to_url = {s["server_id"]: s["base_url"] for s in servers}
        for d in datasets:
            ids = d.get("server_ids", [])
            hosts = [server_id_to_url.get(sid, sid)[:30] for sid in ids]
            length = d.get("length")
            length_str = str(length) if length is not None else "?"
            print(
                f"  {d.get('dataset_id', '?'):<28} "
                f"{d.get('source', '?'):<6} "
                f"len={length_str:<10} "
                f"hosts={', '.join(hosts)}"
            )
            if verbose:
                # load_args is the resolved descriptor on /v1/datasets
                # — exactly what the server saw at load time. Useful
                # for distinguishing handles that hash the same path
                # but differ on revision / split / data_files.
                la = d.get("load_args")
                if la:
                    print(f"      load_args: {la}")
                cols = d.get("column_names")
                if cols:
                    print(f"      columns:   {cols}")
    return 0


def cluster_cmd(args):
    from .server_client import ServerClient, ServerUnreachable

    client = ServerClient.from_args(args)
    sub = getattr(args, "cluster_subcommand", None)
    if sub is None:
        print(
            "error: specify a subcommand "
            "(nodes, jobs, submit, cancel, datasets, resolve, server)",
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
        if sub == "resolve":
            return _cmd_resolve(client, args)
        if sub == "server":
            return _cmd_server(client, args)
        print(f"error: unknown subcommand {sub!r}", file=sys.stderr)
        return 1
    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as e:
        # ValueError: a bad --dataset-source value (parse_dataset_source).
        print(str(e), file=sys.stderr)
        return 1
