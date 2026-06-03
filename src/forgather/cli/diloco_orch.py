"""Orchestrator-backed DiLoCo CLI handlers.

These talk to the forgather-server REST API (via :class:`ServerClient`)
rather than directly to a parameter server. The orchestrator already
resolves each upstream param-server's bearer token + TLS verification on
our behalf, so these commands only need the orchestrator's own auth
(``~/.config/forgather/server/auth_token`` + cluster CA) — the same
trust path the webui uses.

The read/diagnostic handlers here are written to be scriptable: every one
takes ``--json`` and emits a single JSON object/array on stdout so agents
and shell pipelines can consume them.
"""

import json
import os
import re
import sys


def _orchestrator(args):
    """Build a ServerClient for the orchestrator.

    ``--via-server URL`` overrides the default discovery (env
    ``FORGATHER_SERVER_URL`` → ``{scheme}://127.0.0.1:8765``).
    """
    from .server_client import ServerClient

    return ServerClient(getattr(args, "via_server", None) or None)


def _normalize_target(target):
    """Strip a trailing slash; return ``None`` for an empty target."""
    if not target:
        return None
    return target.rstrip("/")


_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", ""}


def _split_host_port(netloc):
    """Split ``host:port`` (and ``[::1]:port``) into ``(host, port_str)``."""
    netloc = netloc.strip()
    if netloc.startswith("["):  # bracketed IPv6 literal
        host, _, rest = netloc[1:].partition("]")
        return host, rest.lstrip(":")
    if ":" in netloc:
        host, _, port = netloc.rpartition(":")
        return host, port
    return netloc, ""


def _host_port(value):
    """``(host, port_str)`` from a full URL or a bare ``host:port``."""
    if "://" in value:
        from urllib.parse import urlparse

        u = urlparse(value)
        return (u.hostname or ""), (str(u.port) if u.port else "")
    return _split_host_port(value)


def _hosts_equiv(a, b):
    """Host equality treating loopback aliases (localhost / 127.0.0.1 /
    ::1) as the same — they resolve to the same machine."""
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    if a in _LOOPBACK_HOSTS and b in _LOOPBACK_HOSTS:
        return True
    try:
        from forgather.tls.policy import host_is_loopback

        if host_is_loopback(a) and host_is_loopback(b):
            return True
    except Exception:
        pass
    return False


def match_server(servers, target):
    """Resolve a ``--server`` value to a known server's ``base_url``.

    Accepts a server ``id`` (e.g. ``local:<qid>`` / ``registered:<id>``),
    a ``label``, a full ``base_url``, or a bare ``host:port``. Host:port
    matching is scheme-agnostic and treats loopback aliases (``localhost``
    / ``127.0.0.1`` / ``::1``) as equivalent, so a server the orchestrator
    lists as ``https://127.0.0.1:8512`` matches ``--server localhost:8512``.
    Returns the ``base_url`` or ``None`` when no server matches.
    """
    t = _normalize_target(target)
    if t is None:
        return None
    th, tp = _host_port(t)
    for s in servers:
        if t == s.get("id") or t == s.get("label"):
            return s.get("base_url")
        base = (s.get("base_url") or "").rstrip("/")
        if not base:
            continue
        if t == base:
            return base
        bh, bp = _host_port(base)
        if tp and bp and tp == bp and _hosts_equiv(th, bh):
            return base
    return None


#: Loopback default for the direct path when no server is specified and the
#: forgather server can't be consulted for discovery. (The DiLoCo parameter
#: server's default port — distinct from the orchestrator's 8765.)
DEFAULT_DIRECT_SERVER = "localhost:8512"


def resolve_one(servers, explicit):
    """Resolve the DiLoCo param-server target, with implicit single-server
    selection.

    * ``explicit`` set → match it against ``servers`` (``None`` if unknown).
    * ``explicit`` unset → the common case is exactly one server: return its
      ``base_url``. Zero → ``None`` (caller decides the fallback). More than
      one → raise (ambiguous; the operator must pick).
    """
    if explicit:
        return match_server(servers, explicit)
    if len(servers) == 1:
        return servers[0].get("base_url")
    if len(servers) > 1:
        from .server_client import ServerUnreachable

        choices = ", ".join((s.get("id") or s.get("base_url") or "?") for s in servers)
        raise ServerUnreachable(
            f"{len(servers)} DiLoCo servers are running — pass --server to "
            f"pick one (e.g. {choices})."
        )
    return None


def _local_only(args):
    return getattr(args, "local_only", False)


def _local_fallback(args):
    return getattr(args, "local_fallback", False)


def _server_required_error(base):
    """The forgather server is the default, required path — fail loud rather
    than silently degrade to a local action that a server-coordinated
    workflow didn't ask for."""
    from .server_client import ServerUnreachable

    return ServerUnreachable(
        f"the forgather server at {base} isn't reachable. Start it "
        f"('forgather server'), or pass --local-fallback to fall back to a "
        f"direct/foreground action, or --local-only to skip the server."
    )


def use_orchestrator(args):
    """Launch-command locality decision.

    Returns a :class:`ServerClient` to enqueue through, or ``None`` to act
    locally (foreground). ``--local-only`` → local; ``--local-fallback`` →
    server when up else local; default → server **required** (raises
    ServerUnreachable when it's down).
    """
    if _local_only(args):
        return None
    client = _orchestrator(args)
    if client.ping():
        return client
    if _local_fallback(args):
        return None
    raise _server_required_error(client.base)


def resolve_orchestrator_base(args):
    """status/control/shutdown locality decision.

    Returns ``(client, base_url)`` to route through the server, or
    ``(None, None)`` to act directly on the parameter server. ``--local-only``
    → direct; ``--local-fallback`` → direct when the server is down or
    doesn't know the target; default → server **required** (raises
    ServerUnreachable when down, or when up but the target is unknown — so a
    server-coordinated workflow doesn't silently bypass it).
    """
    from .server_client import AuthRequired, ServerUnreachable

    if _local_only(args):
        return None, None
    client = _orchestrator(args)
    if not client.ping():
        if _local_fallback(args):
            return None, None
        raise _server_required_error(client.base)
    try:
        servers = client.list_diloco_servers()
    except (AuthRequired, ServerUnreachable, RuntimeError):
        if _local_fallback(args):
            return None, None
        raise
    explicit = getattr(args, "server", None)
    base = resolve_one(servers, explicit)  # may raise on ambiguity
    if base is None:
        if _local_fallback(args):
            return None, None
        if explicit:
            raise ServerUnreachable(
                f"the forgather server doesn't know '{explicit}'. Register it "
                f"('forgather diloco register <url>'), or pass --local-fallback "
                f"/ --local-only to use a direct connection."
            )
        raise ServerUnreachable(
            "no DiLoCo servers are running — start one "
            "('forgather diloco server …') or pass --server."
        )
    return client, base


# ---------------------------------------------------------------------------
# Rich status assembly + rendering (shared by the direct and orchestrator
# paths — both supply the same four getter callables).
# ---------------------------------------------------------------------------


def assemble_status(
    *,
    get_status,
    get_info,
    get_known_workers,
    get_work_queues,
    get_work_queue_detail=None,
):
    """Merge the DiLoCo read endpoints into one dict.

    Each getter is a zero-arg callable; any that raises is recorded as
    ``None`` so a partial server (e.g. one with no work queues yet) still
    produces useful output rather than failing the whole command.

    ``get_work_queue_detail`` is the only two-arg getter — ``(dataset_id,
    shuffle_seed) -> detail dict``. When provided, each queue summary is
    enriched in place with its ``by_worker`` dispatch breakdown (the
    per-unit bitmaps are intentionally dropped — they don't render in a
    terminal). A failed detail fetch leaves that queue's summary untouched.
    """

    def _safe(fn):
        try:
            return fn()
        except Exception:
            return None

    queues = _safe(get_work_queues) if get_work_queues else None
    if queues and get_work_queue_detail is not None:
        for q in queues:
            if not isinstance(q, dict):
                continue
            try:
                detail = get_work_queue_detail(q["dataset_id"], q["shuffle_seed"])
            except Exception:
                continue
            if isinstance(detail, dict) and "by_worker" in detail:
                q["by_worker"] = detail["by_worker"]

    return {
        "status": _safe(get_status),
        "info": _safe(get_info),
        "known_workers": _safe(get_known_workers),
        "work_queues": queues,
    }


def _render_aggregate_stats(agg):
    """Render the server's unified aggregate training stats block.

    Skips lines whose metric hasn't been reported yet so a fresh server (or an
    older one without the ``aggregate_stats`` field) doesn't print a wall of
    zeros. Counters are lifetime totals; gauges are summed over the workers
    currently reporting.
    """
    lines = []
    tt = agg.get("total_tokens")
    if tt:
        lines.append(f"  Total tokens:  {tt:,}")
    ts = agg.get("total_steps")
    if ts:
        lines.append(f"  Total steps:   {ts:,}")
    tf = agg.get("total_flos")
    if tf:
        lines.append(f"  Total FLOPs:   {tf:.3e}")
    tps = agg.get("tok_per_sec")
    if tps:
        lines.append(f"  Throughput:    {tps:,.0f} tok/s")
    mfu = agg.get("mfu")
    if mfu:
        lines.append(f"  MFU:           {mfu * 100:.1f}%")
    pm = agg.get("peak_memory")
    if pm:
        lines.append(f"  Peak memory:   {pm / 1e9:.2f} GB")
    gn = agg.get("grad_norm")
    if gn is not None:
        lines.append(f"  Grad norm:     {gn:.3f}")
    tl = agg.get("train_loss")
    if tl is not None:
        lines.append(f"  Train loss:    {tl:.4f}")
    el = agg.get("eval_loss")
    if el is not None:
        es = agg.get("eval_step")
        suffix = f" (@ step {es:,})" if es is not None else ""
        lines.append(f"  Eval loss:     {el:.4f}{suffix}")
    if lines:
        print()
        nr = agg.get("num_reporting")
        if nr:
            print(f"Training stats (aggregate of {nr} reporting):")
        else:
            print("Training stats (aggregate):")
        for line in lines:
            print(line)


def render_status(merged, *, want_queues):
    """Human-readable rendering of an :func:`assemble_status` result."""
    import datetime

    status = merged.get("status") or {}
    info = merged.get("info") or {}

    print("DiLoCo Server Status")
    print("=" * 50)
    print(f"  Status:        {status.get('status', 'unknown')}")
    print(f"  Mode:          {status.get('mode', 'sync')}")
    print(f"  Sync round:    {status.get('sync_round', 0)}")
    print(
        f"  Workers:       {status.get('num_registered', 0)}/"
        f"{status.get('num_workers', '?')}"
    )
    if status.get("uptime_seconds"):
        uptime = status["uptime_seconds"]
        print(f"  Uptime:        {int(uptime // 3600)}h {int((uptime % 3600) // 60)}m")
    params = status.get("model_params") or info.get("num_parameters")
    size_mb = status.get("model_size_mb") or info.get("model_size_mb")
    if params:
        line = f"  Parameters:    {params:,}"
        if size_mb:
            line += f" ({size_mb:.1f} MB)"
        print(line)

    # Outer-optimizer config (the core DiLoCo hyperparameters) and the
    # checkpoint/output dir — both shown in the webui dashboard but absent
    # from the CLI until now. Prefer the server's full one-line description
    # (class + every hyperparameter, incl. nesterov; generalizes beyond SGD);
    # fall back to reconstructing it from lr/momentum for older servers.
    outer_opt = status.get("outer_optimizer")
    outer_lr = status.get("outer_lr")
    outer_momentum = status.get("outer_momentum")
    if outer_opt:
        print(f"  Outer opt:     {outer_opt}")
    elif outer_lr is not None or outer_momentum is not None:
        print(
            f"  Outer opt:     SGD(lr={outer_lr if outer_lr is not None else '?'}, "
            f"momentum={outer_momentum if outer_momentum is not None else '?'})"
        )
    save_dir = status.get("save_dir") or info.get("output_dir")
    if save_dir:
        print(f"  Save dir:      {save_dir}")

    if status.get("mode") == "async":
        print(f"  Submissions:   {status.get('total_submissions', 0)}")
        dn_buf = status.get("dn_buffer_size", 0)
        if dn_buf > 0:
            print(f"  DN buffer:     {status.get('dn_buffered', 0)}/{dn_buf}")
        if status.get("dylu_enabled"):
            print(f"  DyLU base H:   {status.get('dylu_base_sync_every', '?')}")

    deaths = status.get("total_worker_deaths", 0)
    if deaths:
        print(f"  Worker deaths: {deaths}")
    frag = status.get("fragment_submissions")
    if frag:
        print(f"  Frag submits:  {frag}")
    hb_timeout = status.get("heartbeat_timeout", 0)
    if hb_timeout:
        line = f"  HB timeout:    {hb_timeout}s"
        min_workers = status.get("min_workers")
        if min_workers is not None:
            line += f" (min workers: {min_workers})"
        print(line)
    pending = status.get("pending_submissions", [])
    if pending:
        print(f"  Pending sync:  {', '.join(pending)}")

    _render_aggregate_stats(status.get("aggregate_stats") or {})

    workers = status.get("workers", {})
    if workers:
        print()
        print("Workers (registered):")
        print(
            f"  {'ID':<26} {'Host':<12} {'Round':<6} {'Steps/s':<8} "
            f"{'Progress':<28} {'Last HB'}"
        )
        print("  " + "-" * 94)
        for wid, w in workers.items():
            last_hb = "—"
            if w.get("last_heartbeat"):
                last_hb = datetime.datetime.fromtimestamp(w["last_heartbeat"]).strftime(
                    "%H:%M:%S"
                )
            print(
                f"  {str(wid):<26} "
                f"{str(w.get('hostname', '?')):<12} "
                f"{w.get('sync_round', 0):<6} "
                f"{(w.get('steps_per_second') or 0):<8.2f} "
                f"{_worker_progress(w.get('stats')):<28} "
                f"{last_hb}"
            )

    known = (merged.get("known_workers") or {}).get("workers")
    if known:
        running = sum(1 for w in known if w.get("running"))
        print()
        print(f"Known workers: {len(known)} ({running} running)")
        # List the not-running (resumable) names so an operator can pick a
        # worker_id to relaunch under (`--resume-workers` / a `--worker-id`
        # that resumes that worker's checkpoint). Running names can't be
        # relaunched, so they stay out of this roster.
        resumable = [w for w in known if not w.get("running")]
        if resumable:
            print("  Resumable (not running):")
            for w in sorted(resumable, key=lambda x: str(x.get("worker_id", ""))):
                wid = str(w.get("worker_id", "?"))
                last = w.get("last_registered")
                when = (
                    datetime.datetime.fromtimestamp(last).strftime("%Y-%m-%d %H:%M")
                    if last
                    else "—"
                )
                print(f"    {wid:<30} last seen {when}")

    if want_queues:
        queues = merged.get("work_queues")
        print()
        if not queues:
            print("Work-unit dispatch: none")
        else:
            print("Work-unit dispatch:")
            for q in queues:
                label = _queue_label(q) or str(q.get("dataset_id", "?"))
                seed = q.get("shuffle_seed", "?")
                issued = q.get("issued_count", 0)
                done = q.get("completed_count", 0)
                total = q.get("total_units", 0) or 0
                rows = (q.get("hint") or {}).get("length")
                pct = f" ({100 * issued / total:.0f}% issued)" if total else ""
                head = f"  {label}@{seed}: {issued}/{total} issued{pct}"
                if done:
                    head += f", {done} confirmed"
                if rows:
                    head += f" — {rows:,} rows"
                print(head)
                # When the label isn't the raw hash, keep the opaque
                # dataset_id visible (it keys the queue) but secondary.
                if _queue_label(q):
                    print(f"    dataset_id: {q.get('dataset_id', '?')}")
                by_worker = q.get("by_worker") or {}
                if by_worker:
                    print(f"    {'worker':<30} {'issued':>8} {'completed':>10}")
                    for wid, c in sorted(by_worker.items()):
                        print(
                            f"    {str(wid):<30} "
                            f"{c.get('units_issued', 0):>8} "
                            f"{c.get('units_completed', 0):>10}"
                        )
    return 0


def _queue_label(q):
    """Human-readable dataset label from a queue summary's ``hint`` fields
    (``path:name@split``), or ``None`` when the worker shipped no hint
    (legacy server) so the caller falls back to the raw ``dataset_id`` hash.
    Mirrors the webui's ``formatQueueLabel``."""
    hint = (q or {}).get("hint") or {}
    path = hint.get("path")
    if not path:
        return None
    label = str(path)
    if hint.get("name"):
        label += f":{hint['name']}"
    if hint.get("split"):
        label += f"@{hint['split']}"
    return label


def _worker_progress(stats, bar_width=8):
    """A compact per-worker progress cell for the status table, from the
    worker's reported ``step_total`` / ``max_steps`` (DiLoCoCallback stats).

    Returns ``"[####----]  58% 4,672/8,030"`` when both are known, the bare
    step count when only ``step_total`` is reported (older worker / no target),
    or ``"—"`` when neither is available.
    """
    if not isinstance(stats, dict):
        return "—"
    step = stats.get("step_total")
    mx = stats.get("max_steps")
    if step is None:
        return "—"
    step = int(step)
    if not mx or mx <= 0:
        return f"{step:,}"
    mx = int(mx)
    frac = max(0.0, min(1.0, step / mx))
    filled = int(round(bar_width * frac))
    bar = "#" * filled + "-" * (bar_width - filled)
    return f"[{bar}] {frac * 100:3.0f}% {step:,}/{mx:,}"


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def servers_cmd(args):
    """List DiLoCo servers the orchestrator knows about (local + registered)."""
    from .server_client import AuthRequired, ServerUnreachable

    client = _orchestrator(args)
    try:
        servers = client.list_diloco_servers()
    except (ServerUnreachable, AuthRequired, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(json.dumps(servers, indent=2))
        return 0

    if not servers:
        print("No DiLoCo servers known to the forgather server.")
        print("Start one with 'forgather diloco server …' or register an")
        print("external one with 'forgather diloco register <url>'.")
        return 0

    print(f"{'ID':<26} {'SOURCE':<11} {'STATE':<18} BASE_URL")
    print("-" * 90)
    for s in servers:
        src = s.get("source", "?")
        if src == "local":
            state = "alive" if s.get("alive") else "stopped"
        else:
            bits = []
            bits.append("auth" if s.get("has_auth_token") else "no-auth")
            bits.append("verify" if s.get("verify_tls") else "no-verify")
            state = ", ".join(bits)
        print(
            f"{str(s.get('id', '?')):<26} {src:<11} {state:<18} "
            f"{s.get('base_url', '?')}"
        )
    return 0


def logs_cmd(args):
    """Dump or follow the captured TTY log of a DiLoCo worker/server job.

    ``JOB`` may be a queue_id, a local DiLoCo server id/label, or a
    worker_id — resolved to the underlying job via the orchestrator.
    """
    from .server_client import AuthRequired, ServerUnreachable

    client = _orchestrator(args)
    try:
        job_id = _resolve_job_id(client, args.job)
    except (ServerUnreachable, AuthRequired, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1

    if getattr(args, "path", False):
        # Print the on-disk TTY path (on the forgather server's host) and
        # exit — for piping into the user's own tail/cat tooling. Resolved
        # server-side via the same get_record path as the dump/stream
        # endpoints, so it works for any job they can read (not just ones
        # surfaced by /api/jobs).
        try:
            tty_path = client.job_tty_path(job_id)
        except (ServerUnreachable, AuthRequired, RuntimeError) as e:
            print(str(e), file=sys.stderr)
            return 1
        if not tty_path:
            print(
                f"no captured TTY path for '{args.job}' (job {job_id}).",
                file=sys.stderr,
            )
            return 1
        print(tty_path)
        return 0

    if getattr(args, "follow", False):
        return _follow_tty(client, job_id)
    try:
        data = client.job_dump(job_id)
    except (ServerUnreachable, AuthRequired, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1
    sys.stdout.buffer.write(data)
    sys.stdout.flush()
    return 0


def _resolve_job_id(client, token):
    """Map a user-supplied JOB token to an orchestrator job id (queue_id).

    Tries, in order: a local DiLoCo server's id/label/queue_id; a job's
    id/queue_id; a DiLoCo worker_id stamped on a job's params. Falls back
    to the token verbatim (the server returns a clean 404 if it's wrong).
    """
    try:
        servers = client.list_diloco_servers()
    except Exception:
        servers = []
    for s in servers:
        if s.get("source") == "local" and token in (
            s.get("id"),
            s.get("label"),
            s.get("queue_id"),
        ):
            if s.get("queue_id"):
                return s["queue_id"]

    try:
        jobs = client.list_jobs(include_dead=True)
    except Exception:
        jobs = []
    for j in jobs:
        if token in (j.get("id"), j.get("queue_id")):
            return j.get("id") or j.get("queue_id")
    for j in jobs:
        diloco = (j.get("job_params") or {}).get("diloco") or {}
        if diloco.get("worker_id") == token:
            return j.get("id") or j.get("queue_id")
    return token


def register_cmd(args):
    """Register an external DiLoCo server with the forgather server."""
    from .server_client import AuthRequired, ServerUnreachable

    client = _orchestrator(args)
    try:
        entry = client.add_diloco_registry(
            base_url=args.url,
            label=getattr(args, "label", None),
            auth_token=getattr(args, "auth_token", None),
            verify_tls=not getattr(args, "no_verify_tls", False),
        )
    except (ServerUnreachable, AuthRequired, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(entry, indent=2))
        return 0
    # Show the "registered:<id>" form so it's copy-pasteable straight into
    # `diloco servers` output / `diloco unregister`.
    print(f"Registered '{entry.get('label')}' as registered:{entry.get('id')}")
    print(
        f"  {entry.get('base_url')}  "
        f"(auth={'yes' if entry.get('has_auth_token') else 'no'}, "
        f"verify_tls={entry.get('verify_tls')})"
    )
    return 0


def unregister_cmd(args):
    """Remove a previously-registered external DiLoCo server."""
    from .server_client import AuthRequired, ServerUnreachable

    client = _orchestrator(args)
    rid = args.entry_id
    # Accept both the "registered:<id>" form printed by `diloco servers`
    # and the bare registry id.
    if rid.startswith("registered:"):
        rid = rid[len("registered:") :]
    try:
        resp = client.delete_diloco_registry(rid)
    except (ServerUnreachable, AuthRequired, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"Unregistered {resp.get('deleted', rid)}")
    return 0


# ---------------------------------------------------------------------------
# Launch as scheduled jobs (orchestrator-first; the direct/foreground path
# lives in diloco.py). The decision here is simply "is the forgather server
# up?" — when it is, server/worker launches become scheduled jobs.
# ---------------------------------------------------------------------------


def _server_job_params(args):
    """Build the ``diloco_server`` job_params from the `server` subparser
    args — the same shape the webui's DiLoCoServerModal submits.

    Paths are absolutized against the CLI's CWD before they go into the job:
    the scheduler launches the job from the forgather repo root, so a
    relative ``--output-dir`` typed in the user's shell would otherwise
    resolve against the wrong directory."""
    p = {
        "output_dir": os.path.abspath(args.output_dir),
        "port": args.port,
        "num_workers": args.num_workers,
        "host": args.host,
        "async_mode": getattr(args, "async_mode", False),
        "dylu": getattr(args, "dylu", False),
        "save_every": args.save_every,
        "save_total_limit": args.save_total_limit,
        "outer_lr": args.outer_lr,
        "outer_momentum": args.outer_momentum,
        "no_nesterov": args.no_nesterov,
        "heartbeat_timeout": args.heartbeat_timeout,
        "min_workers": args.min_workers,
        "sync_every": args.sync_every,
        "num_fragments": args.num_fragments,
        "bf16_comm": args.bf16_comm,
        "no_auth": getattr(args, "no_auth", False),
        "bulk_cleartext": getattr(args, "bulk_cleartext", False),
    }
    if getattr(args, "dn_buffer_size", 0):
        p["dn_buffer_size"] = args.dn_buffer_size
    if p["dylu"]:
        p["dylu_base_sync_every"] = args.dylu_base_sync_every
    if getattr(args, "from_checkpoint", None):
        p["from_checkpoint"] = os.path.abspath(args.from_checkpoint)
    if not p["no_auth"] and getattr(args, "regen_token", False):
        p["regen_token"] = True
    return p


def launch_server(args):
    """Enqueue a diloco_server job; the scheduler starts it on idle GPUs (0)."""
    from .server_client import AuthRequired, ServerUnreachable

    client = _orchestrator(args)
    try:
        item = client.enqueue_job(
            project_dir=os.path.abspath(args.output_dir),
            config=f"diloco:{args.port}",
            job_type="diloco_server",
            job_params=_server_job_params(args),
            requested_gpus=0,
            priority=getattr(args, "priority", 0),
        )
    except (ServerUnreachable, AuthRequired, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(item, indent=2))
        return 0
    qid = item.get("queue_id")
    print(f"Enqueued DiLoCo server job {qid} (the scheduler will start it).")
    print(f"  status:  forgather diloco servers")
    print(f"  logs:    forgather diloco logs {qid} -f")
    return 0


def parse_dataset_source(spec):
    """``--dataset`` value → an EnqueueRequest.dataset_source dict (or None).

    ``None``/``local``/``""`` → None (in-process loader). ``auto`` → cluster
    auto-routing. ``server:<id>`` → a specific registered/local server.
    Raises ValueError on anything else (including an empty ``server:`` id). The
    single parser for this wire format — shared by train/eval/submit and the
    ``cluster submit`` / ``submit --global`` paths.
    """
    if spec is not None:
        spec = spec.strip()
    if not spec or spec == "local":
        return None
    if spec == "auto":
        return {"kind": "auto"}
    if spec.startswith("server:"):
        server_id = spec[len("server:") :].strip()
        if not server_id:
            raise ValueError("--dataset server:<id> requires a non-empty id")
        return {"kind": "server", "server_id": server_id}
    raise ValueError(
        f"invalid --dataset {spec!r}; expected 'auto', 'local', or 'server:<id>'"
    )


def resolve_dataset_source(client, args):
    """Resolve ``--dataset`` into an ``EnqueueRequest.dataset_source`` dict.

    An explicit ``--dataset`` value always wins (including ``local``, the
    in-process loader). When ``--dataset`` is **unset**, the default is
    mode-aware, matching the webui Submit modal: ``auto`` (cluster routing)
    when the forgather server is in cluster mode, otherwise local (``None``).
    The cluster probe is best-effort — a failed/empty probe falls back to
    local — and the chosen default is logged so the picked source is visible
    to scripts. Raises ``ValueError`` on an invalid explicit value.
    """
    spec = getattr(args, "dataset", None)
    if spec:
        # Explicit value wins verbatim (local / auto / server:<id>).
        return parse_dataset_source(spec)
    # Unset → mode-aware default. cluster_self() is null in standalone mode,
    # an identity object in cluster mode (same gate the webui uses).
    try:
        in_cluster = client.cluster_self() is not None
    except Exception:
        in_cluster = False
    if in_cluster:
        # Informational only → stderr, so it never pollutes the --json stdout
        # the launch handlers emit for agent/script consumption.
        print(
            "dataset source: auto (cluster routing) — server is in cluster "
            "mode; pass --dataset local to override",
            file=sys.stderr,
        )
        return {"kind": "auto"}
    return None


def _resolve_worker_server(client, args):
    """Resolve the DiLoCo server the worker(s) connect to (a routable
    base_url when the forgather server knows it, else the verbatim
    ``--server``). Auto-picks the single running server when ``--server`` is
    omitted. Returns the server string, or ``None`` after printing an error
    (ambiguous / none running)."""
    from .server_client import AuthRequired, ServerUnreachable

    explicit = getattr(args, "server", None)
    try:
        servers = client.list_diloco_servers()
    except (ServerUnreachable, AuthRequired, RuntimeError):
        servers = []
    try:
        resolved = resolve_one(servers, explicit)  # may raise on ambiguity
    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        return None
    server = resolved or explicit
    if not server:
        print(
            "error: no DiLoCo server is running to connect the worker(s) to; "
            "start one ('forgather diloco server …') or pass --server.",
            file=sys.stderr,
        )
        return None
    return server


def _resolve_config_name(args):
    """The config name to launch workers with: the explicit ``-t``, or the
    project's ``default_config`` from meta.yaml (like ``forgather train`` and
    the other subcommands). Returns ``None`` when neither is available (no
    project / no default), so the caller can error clearly."""
    explicit = getattr(args, "config_template", None)
    if explicit:
        return explicit
    try:
        from forgather import MetaConfig

        project_dir = MetaConfig.find_project_dir(getattr(args, "project_dir", "."))
        return MetaConfig(project_dir).default_config() or None
    except Exception:
        return None


def _enqueue_worker_jobs(client, names, server, args, dynamic_args, dataset_source):
    """Enqueue one training job per worker name (shared by launch + resume).

    All jobs share the resolved ``server`` address, the config/project, the
    dynamic args, the dataset source, and the per-worker GPU/priority knobs.
    """
    from .server_client import AuthRequired, ServerUnreachable

    config = _resolve_config_name(args)
    hb = getattr(args, "heartbeat_interval", None)
    # Per-worker GPUs: --requested-gpus (the unified knob on `submit`) wins;
    # fall back to the deprecated `diloco worker --gpus-per-worker`, else 1.
    gpus = (
        getattr(args, "requested_gpus", None)
        or getattr(args, "gpus_per_worker", None)
        or 1
    )
    priority = getattr(args, "priority", 0)
    project_dir = getattr(args, "project_dir", ".")

    if getattr(args, "dry_run", False):
        print(
            f"[dry-run] would enqueue {len(names)} DiLoCo worker(s) against "
            f"{server}: config={config} gpus/worker={gpus} priority={priority}"
        )
        for name in names:
            print(f"  {name}")
        if dynamic_args:
            print(f"  dynamic_args={dynamic_args}")
        return 0

    results = []
    for name in names:
        diloco = {"server_addr": server, "worker_id": name}
        if hb is not None:
            diloco["heartbeat_interval"] = hb
        try:
            item = client.enqueue_job(
                project_dir=project_dir,
                config=config,
                job_type="training",
                job_params={"diloco": diloco},
                requested_gpus=gpus,
                priority=priority,
                dynamic_args=dynamic_args or None,
                dataset_source=dataset_source,
            )
        except (ServerUnreachable, AuthRequired, RuntimeError) as e:
            # Report what already enqueued so the operator can cancel/retry.
            for r in results:
                print(f"  enqueued {r['worker_id']} as {r['queue_id']}")
            print(f"error: failed to enqueue worker '{name}': {e}", file=sys.stderr)
            return 1
        results.append({"worker_id": name, "queue_id": item.get("queue_id")})

    if getattr(args, "json", False):
        print(json.dumps(results, indent=2))
        return 0
    print(f"Enqueued {len(results)} worker(s) against {server}:")
    for r in results:
        print(f"  {r['worker_id']:<28} {r['queue_id']}")
    print("The scheduler will start them on idle GPUs.")
    return 0


def launch_workers(args, dynamic_args):
    """Enqueue N DiLoCo worker (training) jobs with auto-named workers.

    Mirrors the webui SubmitModal worker-pool path: resolve the server,
    generate unique worker names (unless a single explicit --worker-id is
    given), and enqueue one training job per worker with the shared dynamic
    args + dataset source.
    """
    if not _resolve_config_name(args):
        print(
            "error: launching a worker needs a config — none given (-t) and "
            "no default_config in the project's meta.yaml. Pass -t <config> "
            "(e.g. forgather -p <project> -t <config> diloco worker …).",
            file=sys.stderr,
        )
        return 1
    from .server_client import AuthRequired, ServerUnreachable

    client = _orchestrator(args)
    try:
        dataset_source = resolve_dataset_source(client, args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    server = _resolve_worker_server(client, args)
    if server is None:
        return 1

    count = getattr(args, "count", 1) or 1
    explicit_wid = (getattr(args, "worker_id", None) or "").strip() or None
    if explicit_wid and count == 1:
        names = [explicit_wid]
    else:
        # Server-side generation guarantees uniqueness + exclusion (matches
        # the webui). Exclude an explicit id so it isn't regenerated.
        try:
            resp = client.generate_diloco_worker_names(
                count, exclude=[explicit_wid] if explicit_wid else []
            )
        except (ServerUnreachable, AuthRequired, RuntimeError) as e:
            print(str(e), file=sys.stderr)
            return 1
        names = resp.get("names", [])
        if not names:
            print("error: could not generate worker names", file=sys.stderr)
            return 1

    return _enqueue_worker_jobs(
        client, names, server, args, dynamic_args, dataset_source
    )


_PP_SUFFIX = re.compile(r"_pp\d+$")


def _stopped_base_workers(known):
    """Base worker-ids (pipeline ``_pp<N>`` suffix stripped, deduped) with no
    currently-running rank — the resumable set.

    Like the webui's resumable-worker derivation, but stricter: a base is
    resumable only when *every* rank under it is stopped. The webui treats a
    base as resumable if any rank is stopped; excluding bases with a live rank
    avoids duplicating a still-running rank during a bulk relaunch.
    """
    running_bases, all_bases, seen = set(), [], set()
    for w in known.get("workers", []) or []:
        wid = (w.get("worker_id") or "").strip()
        if not wid:
            continue
        base = _PP_SUFFIX.sub("", wid)
        if w.get("running"):
            running_bases.add(base)
        if base not in seen:
            seen.add(base)
            all_bases.append(base)
    return [b for b in all_bases if b not in running_bases]


def launch_resume(args, dynamic_args):
    """Re-enqueue every stopped known worker (reusing its id), so a worker
    set comes back after a server shutdown / manual stop with one command.

    The worker-id is reused verbatim, so each resumed worker lands on its
    existing per-worker output dir and resumes from its checkpoint.
    """
    from .server_client import AuthRequired, ServerUnreachable

    if not _resolve_config_name(args):
        print(
            "error: resuming workers needs a config — none given (-t) and no "
            "default_config in the project's meta.yaml. Pass -t <config>.",
            file=sys.stderr,
        )
        return 1
    client = _orchestrator(args)
    try:
        dataset_source = resolve_dataset_source(client, args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    # Resume needs a server the forgather server knows — that's where the
    # known-worker roster comes from.
    explicit = getattr(args, "server", None)
    try:
        servers = client.list_diloco_servers()
    except (ServerUnreachable, AuthRequired, RuntimeError):
        servers = []
    try:
        base = resolve_one(servers, explicit)
    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        return 1
    if not base:
        print(
            "error: no DiLoCo server found to resume workers from; start one "
            "or pass --server.",
            file=sys.stderr,
        )
        return 1

    try:
        known = client.diloco_known_workers(base)
    except (ServerUnreachable, AuthRequired, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1
    names = _stopped_base_workers(known)
    if not names:
        print(f"No stopped workers to resume on {base}.")
        return 0
    return _enqueue_worker_jobs(client, names, base, args, dynamic_args, dataset_source)


# ---------------------------------------------------------------------------
# Control surface — a uniform adapter over the direct DiLoCoClient and the
# orchestrator proxy, so the multi-step shutdown flow is written once.
# ---------------------------------------------------------------------------


class _DirectOps:
    """Talk straight to the parameter server via DiLoCoClient."""

    def __init__(self, client):
        self._c = client

    def relay(self, command, worker_id=None):
        return self._c.relay_command(command, worker_id=worker_id)

    def get_status(self):
        return self._c.get_status()

    def save_state(self):
        return self._c.save_state()

    def shutdown(self):
        return self._c.shutdown()


class _OrchestratorOps:
    """Route control actions through the forgather server's proxy (which
    resolves the upstream token + TLS for us)."""

    def __init__(self, client, base):
        self._c = client
        self._base = base

    def relay(self, command, worker_id=None):
        return self._c.diloco_server_control(
            "command", self._base, command=command, worker_id=worker_id
        )

    def get_status(self):
        return self._c.diloco_server_status(self._base)

    def save_state(self):
        return self._c.diloco_server_control("save_state", self._base)

    def shutdown(self):
        return self._c.diloco_server_control("shutdown", self._base)


def make_control_ops(args, *, timeout=30):
    """Build a control surface (orchestrator proxy or direct DiLoCoClient).

    Returns ``(ops, label)`` where ``label`` describes the target for human
    messages. Honors ``--local-only`` / ``--local-fallback`` / ``--via-server``
    via :func:`resolve_orchestrator_base` (which raises ServerUnreachable when
    the server is required but down — the caller catches and exits non-zero).
    """
    client, base = resolve_orchestrator_base(args)
    if base is not None:
        return _OrchestratorOps(client, base), f"{base} (via forgather server)"
    from forgather.ml.diloco.client import DiLoCoClient

    # Direct path (--local-only / --local-fallback when down): no discovery
    # is possible, so fall back to the loopback default when --server is
    # omitted.
    server = getattr(args, "server", None) or DEFAULT_DIRECT_SERVER
    c = DiLoCoClient(
        server,
        timeout=timeout,
        token=getattr(args, "auth_token", None),
        verify_tls=not getattr(args, "no_verify_tls", False),
    )
    return _DirectOps(c), server


def _follow_tty(client, job_id):
    """Stream a job's TTY to stdout until it ends or Ctrl-C. Mirrors the
    ``forgather job tail`` pattern."""
    import asyncio

    from .server_client import AuthRequired, ServerUnreachable

    async def _run():
        async for kind, data in client.stream_tty(job_id, follow=True):
            if kind == "error":
                print(f"\n[tty error] {data}", file=sys.stderr)
                continue
            sys.stdout.buffer.write(data if isinstance(data, bytes) else data.encode())
            sys.stdout.flush()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        return 0
    except (ServerUnreachable, AuthRequired, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0
