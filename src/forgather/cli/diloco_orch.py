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


def match_server(servers, target):
    """Resolve a ``--server`` value to a known server's ``base_url``.

    Accepts a server ``id`` (e.g. ``local:<qid>`` / ``registered:<id>``),
    a ``label``, a full ``base_url``, or a bare ``host:port`` matched
    against the netloc of a known ``base_url``. Returns the ``base_url``
    or ``None`` when no server matches.
    """
    t = _normalize_target(target)
    if t is None:
        return None
    for s in servers:
        if t == s.get("id") or t == s.get("label"):
            return s.get("base_url")
        base = (s.get("base_url") or "").rstrip("/")
        if not base:
            continue
        if t == base:
            return base
        # bare host:port → compare against the base_url's netloc
        if "://" not in t and base.split("://", 1)[-1] == t:
            return base
    return None


def resolve_orchestrator_base(args):
    """Auto-detect: return ``(client, base_url)`` when the target server is
    reachable through the orchestrator, else ``(None, None)``.

    Honors ``--direct`` (skip the orchestrator entirely) and falls back to
    ``(None, None)`` when the orchestrator is down or doesn't know the
    target — the caller then uses the direct-to-param-server path.
    """
    if getattr(args, "direct", False):
        return None, None
    from .server_client import AuthRequired, ServerUnreachable

    client = _orchestrator(args)
    if not client.ping():
        return None, None
    try:
        servers = client.list_diloco_servers()
    except (ServerUnreachable, AuthRequired, RuntimeError):
        return None, None
    base = match_server(servers, getattr(args, "server", None))
    if base is None:
        return None, None
    return client, base


# ---------------------------------------------------------------------------
# Rich status assembly + rendering (shared by the direct and orchestrator
# paths — both supply the same four getter callables).
# ---------------------------------------------------------------------------


def assemble_status(*, get_status, get_info, get_known_workers, get_work_queues):
    """Merge the four DiLoCo read endpoints into one dict.

    Each getter is a zero-arg callable; any that raises is recorded as
    ``None`` so a partial server (e.g. one with no work queues yet) still
    produces useful output rather than failing the whole command.
    """

    def _safe(fn):
        try:
            return fn()
        except Exception:
            return None

    return {
        "status": _safe(get_status),
        "info": _safe(get_info),
        "known_workers": _safe(get_known_workers),
        "work_queues": _safe(get_work_queues) if get_work_queues else None,
    }


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
    hb_timeout = status.get("heartbeat_timeout", 0)
    if hb_timeout:
        print(f"  HB timeout:    {hb_timeout}s")
    pending = status.get("pending_submissions", [])
    if pending:
        print(f"  Pending sync:  {', '.join(pending)}")

    workers = status.get("workers", {})
    if workers:
        print()
        print("Workers (registered):")
        print(f"  {'ID':<30} {'Host':<15} {'Round':<8} {'Steps/s':<10} {'Last HB'}")
        print("  " + "-" * 75)
        for wid, w in workers.items():
            last_hb = "—"
            if w.get("last_heartbeat"):
                last_hb = datetime.datetime.fromtimestamp(w["last_heartbeat"]).strftime(
                    "%H:%M:%S"
                )
            print(
                f"  {wid:<30} "
                f"{str(w.get('hostname', '?')):<15} "
                f"{w.get('sync_round', 0):<8} "
                f"{(w.get('steps_per_second') or 0):<10.2f} "
                f"{last_hb}"
            )

    known = (merged.get("known_workers") or {}).get("workers")
    if known:
        running = sum(1 for w in known if w.get("running"))
        print()
        print(f"Known workers: {len(known)} ({running} running)")

    if want_queues:
        queues = merged.get("work_queues")
        print()
        if not queues:
            print("Work-unit queues: none")
        else:
            print("Work-unit queues:")
            print(
                f"  {'dataset_id':<24} {'seed':<12} {'issued':>8} "
                f"{'done':>8} {'total':>8}"
            )
            print("  " + "-" * 64)
            for q in queues:
                ds = str(q.get("dataset_id", "?"))
                if len(ds) > 23:
                    ds = ds[:20] + "..."
                print(
                    f"  {ds:<24} {str(q.get('shuffle_seed', '?')):<12} "
                    f"{q.get('issued_count', 0):>8} "
                    f"{q.get('completed_count', 0):>8} "
                    f"{q.get('total_units', 0):>8}"
                )
    return 0


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
