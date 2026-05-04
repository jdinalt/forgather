"""Scheduler/queue subcommands for the forgather-server CLI."""

import sys


def _format_delta(ts_or_none):
    """Return a short relative-time string like '12s ago' or 'never'."""
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


def _status_line(sched, queue, jobs):
    enabled = sched.get("enabled", False)
    queued = len(queue)
    running = sum(1 for j in jobs if j.get("status") in ("starting", "running"))
    last_tick = _format_delta(sched.get("last_tick_at"))
    return (
        f"enabled={enabled}  queued={queued}  running={running}  last_tick={last_tick}"
    )


def _sched_status(client):
    sched = client.get_scheduler()
    queue = client.list_queue()
    jobs = client.list_jobs()
    print(_status_line(sched, queue, jobs))


def _sched_list(client):
    queue = client.list_queue()
    jobs = client.list_jobs()

    queue_ids_in_jobs = {j.get("queue_id") for j in jobs}
    pending = [q for q in queue if q.get("queue_id") not in queue_ids_in_jobs]
    pending_sorted = sorted(
        pending, key=lambda q: (-q.get("priority", 0), q.get("submitted_at", ""))
    )

    terminal = ("done", "failed", "aborted")
    running_jobs = [j for j in jobs if j.get("status") not in terminal]
    terminal_jobs = [j for j in jobs if j.get("status") in terminal]
    running_jobs.sort(key=lambda j: j.get("started_at") or "")
    terminal_jobs.sort(key=lambda j: j.get("finished_at") or "")

    col_w = {
        "status": 8,
        "id": 24,
        "type": 10,
        "pri": 4,
        "gpus": 4,
        "proj": 50,
        "time": 12,
    }

    def header():
        return (
            f"{'Status':<{col_w['status']}}  "
            f"{'ID':<{col_w['id']}}  "
            f"{'Type':<{col_w['type']}}  "
            f"{'Pri':>{col_w['pri']}}  "
            f"{'GPUs':>{col_w['gpus']}}  "
            f"{'Project/Config':<{col_w['proj']}}  "
            f"{'Time':<{col_w['time']}}"
        )

    def sep():
        return "-" * (
            col_w["status"]
            + col_w["id"]
            + col_w["type"]
            + col_w["pri"]
            + col_w["gpus"]
            + col_w["proj"]
            + col_w["time"]
            + 12
        )

    def proj_str(item):
        pd = item.get("project_dir") or ""
        cfg = item.get("config") or ""
        s = f"{pd}/{cfg}" if pd and cfg else (pd or cfg)
        if len(s) > col_w["proj"]:
            s = "..." + s[-(col_w["proj"] - 3) :]
        return s

    def row_queue(q):
        pri = q.get("priority", 0)
        gpus = q.get("requested_gpus", "?")
        jt = q.get("job_type") or "?"
        time_str = _format_delta(q.get("submitted_at"))
        return (
            f"{'queued':<{col_w['status']}}  "
            f"{str(q.get('queue_id','')):<{col_w['id']}}  "
            f"{jt:<{col_w['type']}}  "
            f"{pri:>{col_w['pri']}}  "
            f"{str(gpus):>{col_w['gpus']}}  "
            f"{proj_str(q):<{col_w['proj']}}  "
            f"{time_str:<{col_w['time']}}"
        )

    def row_job(j):
        status = j.get("status") or "?"
        jid = j.get("queue_id") or j.get("job_id") or "?"
        jt = j.get("job_type") or "?"
        pri = j.get("priority", 0)
        gpus = ",".join(str(g) for g in (j.get("gpu_indices") or [])) or str(
            j.get("requested_gpus", "?")
        )
        started = j.get("started_at")
        submitted = j.get("submitted_at")
        time_str = _format_delta(started or submitted)
        return (
            f"{status:<{col_w['status']}}  "
            f"{str(jid):<{col_w['id']}}  "
            f"{jt:<{col_w['type']}}  "
            f"{pri:>{col_w['pri']}}  "
            f"{str(gpus):>{col_w['gpus']}}  "
            f"{proj_str(j):<{col_w['proj']}}  "
            f"{time_str:<{col_w['time']}}"
        )

    print(header())
    print(sep())
    for q in pending_sorted:
        print(row_queue(q))
    for j in running_jobs:
        print(row_job(j))
    for j in terminal_jobs:
        print(row_job(j))

    if not pending_sorted and not running_jobs and not terminal_jobs:
        print("(empty)")


def sched_cmd(args):
    from .server_client import ServerClient, ServerUnreachable

    client = ServerClient.from_args(args)

    sub = getattr(args, "sched_subcommand", None)
    if sub is None:
        print(
            "error: specify a subcommand (status, list, pause, resume, cancel, cleanup, gc)",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        if sub == "status":
            _sched_status(client)

        elif sub == "list":
            _sched_list(client)

        elif sub == "pause":
            sched = client.set_scheduler(False)
            queue = client.list_queue()
            jobs = client.list_jobs()
            print(_status_line(sched, queue, jobs))

        elif sub == "resume":
            sched = client.set_scheduler(True)
            queue = client.list_queue()
            jobs = client.list_jobs()
            print(_status_line(sched, queue, jobs))

        elif sub == "cancel":
            result = client.cancel(args.queue_id)
            print(f"aborted: {result.get('aborted', args.queue_id)}")

        elif sub == "cleanup":
            if args.job_id:
                result = client.job_delete(args.job_id)
                print(f"removed: {result.get('removed', args.job_id)}")
            else:
                result = client.cleanup_jobs()
                count = result.get("count", len(result.get("removed", [])))
                print(f"removed {count} records")

        elif sub == "gc":
            result = client.gc_jobs()
            print(f"swept {result.get('swept', 0)} orphan tty file(s)")

        else:
            print(f"error: unknown subcommand: {sub}", file=sys.stderr)
            sys.exit(1)

    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
