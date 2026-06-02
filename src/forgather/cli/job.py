"""Job, queue, and scheduler subcommands for the forgather-server CLI.

This is the single coherent surface for the server's scheduler model: a
"job" is a scheduler queue item (``queue_id``) that becomes a running/terminal
job. Per-job control (status/save/stop/.../tail/logs) sits alongside the
queue-level verbs (list/cancel/cleanup/gc) and scheduler control
(``scheduler pause|resume|status``). ``forgather sched`` is a deprecated alias
that maps onto these verbs.
"""

import sys

# ---------------------------------------------------------------------------
# Scheduler / queue rendering
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Per-job control
# ---------------------------------------------------------------------------


def _do_control(client, job_id, action):
    result = client.job_control(job_id, action)
    if result.get("success"):
        print(f"OK: {result.get('message', action)}")
    else:
        print(f"FAIL: {result.get('message', action)}", file=sys.stderr)
        sys.exit(1)


def _scheduler_cmd(client, action):
    if action == "status" or action is None:
        _sched_status(client)
    elif action == "pause":
        sched = client.set_scheduler(False)
        print(_status_line(sched, client.list_queue(), client.list_jobs()))
    elif action == "resume":
        sched = client.set_scheduler(True)
        print(_status_line(sched, client.list_queue(), client.list_jobs()))
    else:
        print(f"error: unknown scheduler action: {action}", file=sys.stderr)
        sys.exit(1)


def job_cmd(args):
    from . import submit_orch
    from .server_client import ServerClient, ServerUnreachable

    client = ServerClient.from_args(args)

    sub = getattr(args, "job_subcommand", None)
    if sub is None:
        print(
            "error: specify a subcommand (status, save, stop, save-stop, abort, "
            "kill, force-kill, tail, dump, logs, list, cancel, cleanup, gc, "
            "scheduler)",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        if sub == "status":
            r = client.job_status(args.job_id)
            if r.status_code == 409:
                print("still starting; trainer endpoint not yet registered")
                return
            if not r.ok:
                try:
                    detail = r.json().get("detail", r.text)
                except Exception:
                    detail = r.text
                print(f"server: {detail}", file=sys.stderr)
                sys.exit(1)
            data = r.json()
            for key, value in data.items():
                print(f"{key}: {value}")

        elif sub in ("save", "stop", "save-stop", "abort", "kill"):
            _do_control(client, args.job_id, sub)

        elif sub == "force-kill":
            if not getattr(args, "yes", False):
                print(
                    "force-kill requires --yes (SIGKILL is destructive)",
                    file=sys.stderr,
                )
                sys.exit(1)
            _do_control(client, args.job_id, "force-kill")

        elif sub == "tail":
            submit_orch.tail_job(client, args.job_id)

        elif sub in ("dump", "logs"):
            data = client.job_dump(args.job_id)
            sys.stdout.buffer.write(data)

        elif sub == "list":
            _sched_list(client)

        elif sub == "cancel":
            result = client.cancel(args.queue_id)
            print(f"aborted: {result.get('aborted', args.queue_id)}")

        elif sub == "cleanup":
            if getattr(args, "job_id", None):
                result = client.job_delete(args.job_id)
                print(f"removed: {result.get('removed', args.job_id)}")
            else:
                result = client.cleanup_jobs()
                count = result.get("count", len(result.get("removed", [])))
                print(f"removed {count} records")

        elif sub == "gc":
            result = client.gc_jobs()
            print(f"swept {result.get('swept', 0)} orphan tty file(s)")

        elif sub == "scheduler":
            _scheduler_cmd(client, getattr(args, "scheduler_action", None))

        else:
            print(f"error: unknown subcommand: {sub}", file=sys.stderr)
            sys.exit(1)

    except ServerUnreachable as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(1)
