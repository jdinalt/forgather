"""Agent tools for the job scheduler and dataset workflow.

Three groups, all wrapping existing server machinery:

- Job visibility (READ): ``list_jobs`` and ``read_job_output`` let the agent
  watch a job it submitted — its status and a tail of its console/TTY output.
- Run-as-job (CONFIRM): ``run_dataset`` / ``run_construct`` / ``run_train``
  submit a job to the scheduler. All are gated (they run arbitrary config code,
  download data, and — for training — consume GPUs for a long time). They share
  the same validated enqueue path (``queue_ops.validate_and_enqueue``); the
  per-type schemas differ only in their command-specific flags.
- Dataset metadata (READ): ``list_dataset_servers`` / ``dataset_info`` answer
  "what are this dataset's splits, example counts, and features?" by querying a
  running dataset server (local or cluster).

Job state and TTY are read in-process (``job_records`` + the shared
``routes.jobs.read_tty_tail`` tail reader); enqueue goes through the same
validated path the HTTP route uses (``queue_ops.validate_and_enqueue``).
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from typing import Any, Dict, List, Optional

from .. import (
    construct_ops,
    dataset_ops,
    eval_ops,
    gpu_monitor,
    job_records,
    queue_ops,
    queue_store,
)
from ..routes.jobs import read_tty_tail
from . import _dataset_servers
from .registry import CONFIRM, EXTENDED, READ, Proposal, ToolRegistry, ToolSpec

log = logging.getLogger("forgather_server.agent.tools_jobs")

# Keep the TTY tail tiny: this goes straight into the model's context, unlike
# the 32 MiB the HTTP /tty route may return to a browser.
_OUTPUT_MAX_BYTES = 16 * 1024
_DEFAULT_TAIL_LINES = 200
_DEFAULT_JOBS_LIMIT = 50

# wait_for_job: poll server-side so the model doesn't burn tokens polling.
_WAIT_POLL_SECONDS = 3.0
_WAIT_DEFAULT_TIMEOUT = 120.0
_WAIT_MAX_TIMEOUT = 600.0
_WAIT_TAIL_LINES = 80

_ALL_STATUSES = job_records.RUNNING_STATUSES | job_records.TERMINAL_STATUSES


# ---- job visibility --------------------------------------------------------


def _record_summary(r: job_records.JobRecord) -> Dict[str, Any]:
    # A small, safe projection: omit dynamic_args / job_params / tty_log_path /
    # auth_token (context size + secret hygiene).
    return {
        "queue_id": r.queue_id,
        "job_type": r.job_type,
        "project_dir": r.project_dir,
        "config": r.config,
        "status": r.status,
        "requested_gpus": r.requested_gpus,
        "gpu_indices": list(r.gpu_indices),
        "submitted_at": r.submitted_at,
        "started_at": r.started_at,
        "finished_at": r.finished_at,
        "exit_code": r.exit_code,
        "error": r.error,
    }


def _list_jobs(args: Dict[str, Any]) -> Any:
    status = args.get("status") or None
    job_type = args.get("job_type") or None
    limit = args.get("limit")
    limit = _DEFAULT_JOBS_LIMIT if limit in (None, "") else int(limit)
    if status is not None and status not in (_ALL_STATUSES | {"queued"}):
        raise ValueError(
            f"unknown status {status!r}; expected one of "
            f"{sorted(_ALL_STATUSES | {'queued'})}"
        )

    rows: List[Dict[str, Any]] = [_record_summary(r) for r in job_records.list_records()]
    # Not-yet-dispatched queue items show as status "queued".
    for it in queue_store.list_items():
        rows.append(
            {
                "queue_id": it.queue_id,
                "job_type": it.job_type,
                "project_dir": it.project_dir,
                "config": it.config,
                "status": "queued",
                "requested_gpus": it.requested_gpus,
                "gpu_indices": [],
                "submitted_at": it.submitted_at,
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "error": None,
            }
        )

    if status is not None:
        rows = [r for r in rows if r["status"] == status]
    if job_type is not None:
        rows = [r for r in rows if r["job_type"] == job_type]
    # Newest first by the most recent timestamp we have for the row.
    rows.sort(
        key=lambda r: (r["finished_at"] or r["started_at"] or r["submitted_at"] or 0),
        reverse=True,
    )
    return {"jobs": rows[:limit], "total": len(rows)}


def _read_job_output(args: Dict[str, Any]) -> Any:
    queue_id = args["queue_id"]
    tail_lines = args.get("tail_lines")
    tail_lines = _DEFAULT_TAIL_LINES if tail_lines in (None, "") else int(tail_lines)

    rec = job_records.get_record(queue_id)
    if rec is None:
        raise ValueError(
            f"no job with queue_id {queue_id!r} (use list_jobs to find one)"
        )
    if not rec.tty_log_path:
        raise ValueError(f"job {queue_id} has no console output recorded (yet)")
    tail = read_tty_tail(
        rec.tty_log_path, max_bytes=_OUTPUT_MAX_BYTES, tail_lines=tail_lines
    )
    return {
        "queue_id": queue_id,
        "status": rec.status,
        "exit_code": rec.exit_code,
        "tail": tail,
    }


async def _wait_for_job(args: Dict[str, Any]) -> Any:
    # Block server-side (not on the model) until the job reaches the target
    # state or the timeout elapses. asyncio.sleep yields the event loop, so the
    # scheduler keeps advancing the job while we wait.
    #
    # until="terminal" (default): wait for done/failed/aborted — for jobs that
    #   complete (run_dataset / run_construct / run_train / run_eval).
    # until="running": wait for the job to come UP — for long-running services
    #   (a dataset / inference / diloco server) that never go terminal while
    #   healthy. A job that fails before coming up still returns (terminal),
    #   so the agent learns it failed instead of waiting out the timeout.
    queue_id = args["queue_id"]
    until = (args.get("until") or "terminal").strip().lower()
    if until not in ("terminal", "running"):
        raise ValueError(f"until must be 'terminal' or 'running', got {until!r}")
    raw = args.get("timeout_seconds")
    timeout = _WAIT_DEFAULT_TIMEOUT if raw in (None, "") else float(raw)
    timeout = max(1.0, min(timeout, _WAIT_MAX_TIMEOUT))

    if job_records.get_record(queue_id) is None:
        raise ValueError(
            f"no job with queue_id {queue_id!r} (use list_jobs to find one)"
        )

    def _reached(rec) -> bool:
        if rec.status in job_records.TERMINAL_STATUSES:
            return True
        return until == "running" and rec.status == "running"

    waited = 0.0
    while True:
        rec = job_records.get_record(queue_id)
        if rec is None:
            raise ValueError(f"job {queue_id} disappeared while waiting")
        reached = _reached(rec)
        if reached or waited >= timeout:
            return {
                "queue_id": queue_id,
                "status": rec.status,
                "exit_code": rec.exit_code,
                "timed_out": not reached,
                "waited_seconds": round(waited, 1),
                "tail": read_tty_tail(
                    rec.tty_log_path,
                    max_bytes=_OUTPUT_MAX_BYTES,
                    tail_lines=_WAIT_TAIL_LINES,
                ),
            }
        step = min(_WAIT_POLL_SECONDS, timeout - waited)
        await asyncio.sleep(step)
        waited += step


# ---- run a config as a scheduler job (CONFIRM) -----------------------------


def _enqueue_note(item: queue_store.QueueItem, what: str) -> str:
    return (
        f"enqueued {item.job_type} job {item.queue_id} ({what}). "
        f"Poll list_jobs / read_job_output('{item.queue_id}') for progress, or "
        f"wait_for_job('{item.queue_id}'). Do not report it finished until its "
        f"status is terminal (done / failed / aborted)."
    )


def _run_dataset(args: Dict[str, Any]) -> Proposal:
    project_dir = args["project_dir"]
    config_name = args["config_name"]
    target = args.get("target") or "train_dataset_split"
    examples = args.get("examples")
    truncate = args.get("truncate")
    tokenizer_path = args.get("tokenizer_path") or None
    dynamic_args = args.get("dynamic_args") or {}

    # Preview the exact command the job will run (no side effect).
    cmd = dataset_ops.build_dataset_command(
        project_dir=project_dir,
        config_name=config_name,
        dynamic_args=dynamic_args,
        target=target,
        examples=int(examples) if examples not in (None, "") else None,
        truncate=int(truncate) if truncate not in (None, "") else None,
        tokenizer_path=tokenizer_path,
    )
    cmd_str = shlex.join(cmd)

    job_params: Dict[str, Any] = {"target": target}
    if examples not in (None, ""):
        job_params["examples"] = int(examples)
    if truncate not in (None, ""):
        job_params["truncate"] = int(truncate)
    if tokenizer_path:
        job_params["tokenizer_path"] = tokenizer_path

    def commit() -> str:
        item = queue_ops.validate_and_enqueue(
            project_dir=project_dir,
            config=config_name,
            dynamic_args=dynamic_args,
            requested_gpus=0,  # dataset jobs are CPU-only
            job_type="dataset",
            job_params=job_params,
            enforce_fs_root=True,
        )
        return _enqueue_note(item, f"target={target}") + (
            " The first build downloads + builds the dataset and can be slow."
        )

    return Proposal(
        title=f"Build dataset: {config_name} ({target})",
        summary="Run a `dataset` job: materializes the target, which downloads/"
        "builds the data and prints sample examples.",
        extra={
            "command": cmd_str,
            "target": target,
            "warning": "The FIRST build downloads and builds the dataset and "
            "can take a long time (large datasets especially). It runs as a "
            "background scheduler job; watch it with list_jobs / read_job_output.",
        },
        commit=commit,
    )


def _run_construct(args: Dict[str, Any]) -> Proposal:
    project_dir = args["project_dir"]
    config_name = args["config_name"]
    target = args.get("target") or "main"
    call = bool(args.get("call", False))
    dynamic_args = args.get("dynamic_args") or {}
    gpus = args.get("gpus")
    requested_gpus = 0 if gpus in (None, "") else int(gpus)

    # Preview the exact command (cheap: only loads the dynamic-args schema).
    cmd = construct_ops.build_construct_command(
        project_dir=project_dir,
        config_name=config_name,
        target=target,
        dynamic_args=dynamic_args,
        call=call,
    )
    cmd_str = shlex.join(cmd)

    job_params: Dict[str, Any] = {"target": target, "call": call}

    def commit() -> str:
        item = queue_ops.validate_and_enqueue(
            project_dir=project_dir,
            config=config_name,
            dynamic_args=dynamic_args,
            requested_gpus=requested_gpus,
            job_type="construct",
            job_params=job_params,
            enforce_fs_root=True,
        )
        return _enqueue_note(item, f"target={target}")

    return Proposal(
        title=f"Construct: {config_name} ({target})",
        summary="Run a `construct` job: materializes the target node and prints "
        "its repr (with --call, invokes it). A debug/inspection run — it executes "
        "the config's constructors.",
        extra={
            "command": cmd_str,
            "target": target,
            "call": call,
            "requested_gpus": requested_gpus,
            "warning": "Runs the config's constructor code as a background "
            "scheduler job; watch it with list_jobs / read_job_output.",
        },
        commit=commit,
    )


def _infer_requested_gpus(project_dir: str, config_name: str) -> int:
    """Default GPU reservation from the config's nproc_per_node, like the
    Submit modal / `forgather submit`. Falls back to 1 when it's a keyword
    ("gpu"/"cpu"/"auto") or can't be read."""
    try:
        from .. import config_ops

        n = config_ops.load_output_dir_info(project_dir, config_name).nproc_per_node
        if isinstance(n, int) and n > 0:
            return n
    except Exception:
        log.debug("could not infer nproc_per_node for %s", config_name, exc_info=True)
    return 1


def _run_train(args: Dict[str, Any]) -> Proposal:
    project_dir = args["project_dir"]
    config_name = args["config_name"]
    dynamic_args = args.get("dynamic_args") or {}
    nproc = args.get("nproc")
    priority = args.get("priority")
    priority = 0 if priority in (None, "") else int(priority)
    dataset_server_id = args.get("dataset_server_id") or None

    # requested_gpus: explicit wins; otherwise infer from the config's
    # nproc_per_node (matches `forgather submit` and the Submit modal).
    gpus = args.get("gpus")
    requested_gpus = (
        int(gpus) if gpus not in (None, "")
        else _infer_requested_gpus(project_dir, config_name)
    )

    job_params: Dict[str, Any] = {}
    if nproc not in (None, ""):
        job_params["nproc"] = str(nproc)
    # Dataset source: default (None) = the in-process loader. A server id from
    # list_dataset_servers routes training data through that dataset server.
    dataset_source = (
        {"kind": "server", "server_id": dataset_server_id} if dataset_server_id else None
    )

    # Accurate preview: this SCHEDULES a background job (the equivalent of
    # `forgather submit`, i.e. `train --schedule`) — it does NOT run training
    # in the foreground. (Plain `forgather train` would be foreground.)
    cmd = ["forgather", "-p", project_dir, "-t", config_name, "submit",
           "--requested-gpus", str(requested_gpus)]
    if priority:
        cmd += ["--priority", str(priority)]
    if nproc not in (None, ""):
        cmd += ["--nproc", str(nproc)]
    if dataset_server_id:
        cmd += ["--dataset-source", dataset_server_id]
    cmd_str = shlex.join(cmd)

    def commit() -> str:
        item = queue_ops.validate_and_enqueue(
            project_dir=project_dir,
            config=config_name,
            dynamic_args=dynamic_args,
            requested_gpus=requested_gpus,
            priority=priority,
            job_type="training",
            job_params=job_params,
            dataset_source=dataset_source,
            enforce_fs_root=True,
        )
        return _enqueue_note(item, f"{requested_gpus} GPU(s)") + (
            " Training can run for a long time; check on it periodically rather "
            "than blocking."
        )

    return Proposal(
        title=f"Train: {config_name}",
        summary="SCHEDULE a `training` job (a background scheduler job, like "
        "`forgather submit`/`train --schedule` — not a foreground run). This "
        "trains the model: long-running and reserves GPUs.",
        extra={
            "command": cmd_str,
            "requested_gpus": requested_gpus,
            "priority": priority,
            "dataset_source": dataset_server_id or "local",
            "dynamic_args": dynamic_args or None,
            "warning": "Training is long-running and reserves "
            f"{requested_gpus} GPU(s). It is QUEUED to the scheduler (background), "
            "not run in the foreground; watch it with list_jobs / "
            "read_job_output / job_status and the training dashboard.",
        },
        commit=commit,
    )


# ---- dataset metadata (via dataset server) ---------------------------------


def _list_dataset_servers(_args: Dict[str, Any]) -> Any:
    return {"servers": _dataset_servers.list_servers()}


def _dataset_info(args: Dict[str, Any]) -> Any:
    dataset = (args.get("dataset") or "").strip()
    if not dataset:
        raise ValueError("dataset is required (the HF name/path from the config)")
    return _dataset_servers.info(
        dataset, server_id=args.get("server_id") or None, split=args.get("split") or None
    )


# ---- GPU status (READ) -----------------------------------------------------


def _gpu_status(_args: Dict[str, Any]) -> Any:
    # Compact per-GPU projection — enough to advise requested_gpus / which card
    # is free, without dumping every process into model context.
    gpus = []
    for g in gpu_monitor.snapshot():
        total = g.total_mem_bytes or 0
        used = g.used_mem_bytes or 0
        gpus.append(
            {
                "index": g.index,
                "name": g.name,
                "total_mem_bytes": total,
                "used_mem_bytes": used,
                "free_mem_bytes": max(0, total - used),
                "util_pct": g.util_pct,
                "mem_util_pct": g.mem_util_pct,
                "temp_c": g.temp_c,
                "power_w": g.power_w,
                "disabled": g.disabled,
                "excluded": g.excluded,
                "min_priority": g.min_priority,
                "unified_memory": g.unified_memory,
                "process_count": len(g.processes or []),
            }
        )
    return {"gpus": gpus}


# ---- eval configs + run an eval job (CONFIRM) ------------------------------


def _list_eval_configs(_args: Dict[str, Any]) -> Any:
    from dataclasses import asdict

    return {"eval_configs": [asdict(e) for e in eval_ops.list_eval_configs()]}


_EVAL_PASSTHROUGH = ("batch_size", "max_length", "stride", "checkpoint_path", "trainer")


def _run_eval(args: Dict[str, Any]) -> Proposal:
    eval_name = (args.get("eval_name") or "").strip()
    model_path = args["model_path"]
    if not eval_name:
        raise ValueError("eval_name is required (a name from list_eval_configs)")
    entries = {e.name: e for e in eval_ops.list_eval_configs()}
    entry = entries.get(eval_name)
    if entry is None:
        raise ValueError(
            f"unknown eval config {eval_name!r}; available: {sorted(entries)}"
        )
    gpus = args.get("gpus")
    requested_gpus = 1 if gpus in (None, "") else int(gpus)
    passthrough = {k: args[k] for k in _EVAL_PASSTHROUGH if args.get(k) not in (None, "")}

    job_params: Dict[str, Any] = {
        "eval_project": entry.project_dir,
        "eval_template": entry.template,
        "model_path": model_path,
        **passthrough,
    }
    # Preview the exact subprocess argv (no side effect).
    cmd = eval_ops.build_eval_command(
        eval_project=entry.project_dir,
        eval_template=entry.template,
        model_path=model_path,
        **passthrough,
    )
    cmd_str = shlex.join(cmd)

    def commit() -> str:
        item = queue_ops.validate_and_enqueue(
            project_dir=entry.project_dir,  # display hint + fs-root anchor
            config=entry.name,
            dynamic_args={},
            requested_gpus=requested_gpus,
            job_type="eval",
            job_params=job_params,
            enforce_fs_root=True,
        )
        return _enqueue_note(item, f"eval {eval_name} on {model_path}")

    return Proposal(
        title=f"Evaluate: {eval_name}",
        summary="Run an `eval` job: scores the model against the eval config's "
        "dataset (perplexity / loss / bpb). Runs as a background scheduler job.",
        extra={
            "command": cmd_str,
            "eval_name": eval_name,
            "model_path": model_path,
            "requested_gpus": requested_gpus,
        },
        commit=commit,
    )


# ---- trainer control of a running job (CONFIRM) ----------------------------

# action -> (trainer_control function name, human description)
_CONTROL_ACTIONS = {
    "save": ("save_checkpoint", "request a checkpoint save"),
    "stop": ("graceful_stop", "gracefully stop (saves a final checkpoint)"),
    "save-stop": ("save_and_stop", "save a checkpoint, then stop"),
    "abort": ("abort", "abort immediately (NO final checkpoint)"),
}


def _control_job(args: Dict[str, Any]) -> Proposal:
    queue_id = args["queue_id"]
    action = (args.get("action") or "").strip()
    if action not in _CONTROL_ACTIONS:
        raise ValueError(
            f"unknown action {action!r}; expected one of {sorted(_CONTROL_ACTIONS)}"
        )
    rec = job_records.get_record(queue_id)
    if rec is None:
        raise ValueError(f"no job with queue_id {queue_id!r} (use list_jobs)")
    if rec.job_type != "training":
        raise ValueError(
            f"control_job only applies to training jobs (job {queue_id} is "
            f"{rec.job_type!r})"
        )
    if rec.job_id is None:
        raise ValueError(
            f"job {queue_id} is not yet correlated to a trainer endpoint; wait "
            "until it is running, then retry"
        )
    fn_name, desc = _CONTROL_ACTIONS[action]
    job_id = rec.job_id

    def commit() -> str:
        from forgather import trainer_control

        resp = getattr(trainer_control, fn_name)(job_id)
        ok = getattr(resp, "success", None)
        msg = getattr(resp, "message", str(resp))
        return f"{action} job {queue_id}: success={ok} — {msg}"

    return Proposal(
        title=f"{action} training job {queue_id}",
        summary=f"Trainer control: {desc}.",
        extra={"queue_id": queue_id, "action": action, "config": rec.config},
        commit=commit,
    )


# ---- clean up finished job records (CONFIRM) -------------------------------


def _cleanup_jobs(args: Dict[str, Any]) -> Proposal:
    queue_ids = list(args.get("queue_ids") or [])
    all_terminal = bool(args.get("all_terminal", False))
    if not queue_ids and not all_terminal:
        raise ValueError(
            "pass queue_ids (the finished jobs you spawned) or all_terminal=true"
        )
    if queue_ids and all_terminal:
        raise ValueError("pass either queue_ids or all_terminal, not both")

    if all_terminal:
        targets = [
            r.queue_id for r in job_records.list_records()
            if r.status in job_records.TERMINAL_STATUSES
        ]
        scope = f"all {len(targets)} completed job record(s)"
    else:
        targets, problems = [], []
        for qid in queue_ids:
            rec = job_records.get_record(qid)
            if rec is None:
                problems.append(f"{qid} (no such record)")
            elif rec.status not in job_records.TERMINAL_STATUSES:
                problems.append(f"{qid} (still {rec.status})")
            else:
                targets.append(qid)
        if not targets:
            raise ValueError(
                "no removable (terminal) job among the given queue_ids: "
                + "; ".join(problems)
            )
        scope = f"{len(targets)} job record(s)"

    def commit() -> str:
        from fastapi import HTTPException

        from ..routes import jobs as jobs_routes

        if all_terminal:
            res = jobs_routes.cleanup_jobs()
            return f"removed {res['count']} completed job record(s)."
        removed, errors = [], []
        for qid in targets:
            try:
                jobs_routes.remove_job(qid)
                removed.append(qid)
            except HTTPException as e:
                errors.append(f"{qid}: {e.detail}")
        msg = f"removed {len(removed)} job record(s)"
        if errors:
            msg += f"; {len(errors)} could not be removed: {errors}"
        return msg

    return Proposal(
        title="Clean up jobs",
        summary=(
            f"Remove {scope} from the Jobs history — terminal records and their "
            "captured TTY logs. Running jobs are never touched."
        ),
        extra={"queue_ids": targets, "all_terminal": all_terminal},
        commit=commit,
    )


# ---- registration ----------------------------------------------------------


def register_all(reg: ToolRegistry) -> None:
    reg.register(
        ToolSpec(
            name="list_jobs",
            description=(
                "List scheduler jobs (queued + running + finished) with their "
                "status, type, project/config, and timing. Use to watch a job "
                "you submitted (e.g. with run_dataset) until it reaches a "
                "terminal status (done / failed / aborted). Optional filters: "
                "status, job_type; limit defaults to 50, newest first."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Filter by status: queued|starting|running|done|failed|aborted."},
                    "job_type": {"type": "string", "description": "Filter by job type, e.g. \"dataset\"."},
                    "limit": {"type": "integer", "description": "Max rows (default 50)."},
                },
            },
            handler=_list_jobs,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="read_job_output",
            description=(
                "Read the tail of a job's console/TTY output by queue_id (from "
                "list_jobs). Use to see what a dataset run printed (examples, "
                "download progress) or why it failed. Returns the last "
                "tail_lines lines (default 200, capped to a small size to "
                "protect context) plus the job's status and exit_code."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "queue_id": {"type": "string"},
                    "tail_lines": {"type": "integer", "description": "Number of trailing lines (default 200)."},
                },
                "required": ["queue_id"],
            },
            handler=_read_job_output,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="wait_for_job",
            description=(
                "Wait for a job to reach a target state, blocking on the server "
                "(NOT by repeatedly calling list_jobs — that wastes tokens). "
                "until='terminal' (default) waits for done/failed/aborted — use "
                "for jobs that COMPLETE (run_dataset / run_construct / run_train "
                "/ run_eval). until='running' waits for the job to come UP — use "
                "for long-running SERVICES started with start_service (a dataset "
                "/ inference / diloco server never goes terminal while healthy, "
                "so waiting for 'terminal' would just time out). Either way a job "
                "that fails returns immediately. Polls until the target state or "
                "timeout_seconds (default 120, max 600); returns status, "
                "exit_code, timed_out, and an output tail. If it times out, call "
                "it again to keep waiting."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "queue_id": {"type": "string"},
                    "until": {
                        "type": "string",
                        "enum": ["terminal", "running"],
                        "description": "terminal = wait for completion (default); running = wait for a service to come up.",
                    },
                    "timeout_seconds": {"type": "number", "description": "Max seconds to wait this call (default 120, capped at 600)."},
                },
                "required": ["queue_id"],
            },
            handler=_wait_for_job,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="run_dataset",
            description=(
                "Run a dataset config as a scheduler job (the equivalent of "
                "`forgather ... dataset --target ...`). Use to materialize a "
                "split — which downloads/builds the data — and to smoke-test it "
                "by printing examples. target defaults to train_dataset_split "
                "(a raw split needing no tokenizer); tokenized splits "
                "(train_dataset/eval_dataset/test_dataset) need tokenizer_path. "
                "examples = how many to print; truncate = max chars per example. "
                "Approval required (it runs code and downloads data). It returns "
                "immediately with a queue_id; the job runs in the background, so "
                "tell the user the first build can be slow, then poll list_jobs "
                "/ read_job_output and only report success once status is "
                "terminal."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string"},
                    "config_name": {"type": "string"},
                    "target": {"type": "string", "description": "Dataset target (default \"train_dataset_split\"). Raw: train/validation/test_dataset_split; tokenized: train_dataset/eval_dataset/test_dataset (need tokenizer_path)."},
                    "examples": {"type": "integer", "description": "Number of examples to print (-n)."},
                    "truncate": {"type": "integer", "description": "Max characters per example (--truncate)."},
                    "tokenizer_path": {"type": "string", "description": "Tokenizer path (-T), required only for tokenized targets."},
                    "dynamic_args": {"type": "object", "description": "Config-specific args (from inspect_config's dynamic_args), keyed by dest."},
                },
                "required": ["project_dir", "config_name"],
            },
            handler=_run_dataset,
            risk=CONFIRM,
        )
    )
    reg.register(
        ToolSpec(
            name="run_construct",
            description=(
                "Run a config's `construct` target as a scheduler job (the "
                "equivalent of `forgather ... construct --target ...`). Use to "
                "materialize and inspect a node from the config — e.g. build the "
                "model, a tokenizer, or any named target — without launching a "
                "full training run. target defaults to \"main\"; set call=true to "
                "invoke the materialized object. gpus defaults to 0 (most targets "
                "materialize on meta/CPU); request a GPU only if the target "
                "allocates real tensors. Approval required (it executes the "
                "config's constructor code). Returns a queue_id; watch it with "
                "list_jobs / read_job_output / wait_for_job and only report "
                "success once status is terminal."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string"},
                    "config_name": {"type": "string"},
                    "target": {"type": "string", "description": "Target node to construct (default \"main\")."},
                    "call": {"type": "boolean", "description": "Invoke the materialized object after constructing it (default false)."},
                    "gpus": {"type": "integer", "description": "GPUs to reserve (default 0)."},
                    "dynamic_args": {"type": "object", "description": "Config-specific args (from inspect_config's dynamic_args), keyed by dest."},
                },
                "required": ["project_dir", "config_name"],
            },
            handler=_run_construct,
            risk=CONFIRM,
        )
    )
    reg.register(
        ToolSpec(
            name="run_train",
            description=(
                "SCHEDULE a training job — the equivalent of `forgather ... "
                "submit` (i.e. `forgather train --schedule`), NOT a foreground "
                "`forgather train`. It QUEUES a background scheduler job that "
                "trains the model (long-running, reserves GPUs, may take hours); "
                "it does not run training in the foreground. gpus = GPUs to "
                "reserve; if omitted it defaults to the config's nproc_per_node "
                "(like the Submit modal), falling back to 1 — set 0 for a CPU "
                "smoke-test. priority (default 0) orders the scheduler queue. "
                "nproc optionally overrides processes-per-node. dataset_server_id "
                "(from list_dataset_servers) routes training data through a "
                "dataset server; omit to use the in-process loader. Approval "
                "required. Returns immediately with a queue_id; tell the user it "
                "is long-running, then check on it with list_jobs / job_status / "
                "read_job_output periodically (do NOT block on wait_for_job for a "
                "full training run) and only report success once status is "
                "terminal."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string"},
                    "config_name": {"type": "string"},
                    "gpus": {"type": "integer", "description": "GPUs to reserve (default: config's nproc_per_node, else 1; 0 = CPU smoke-test)."},
                    "priority": {"type": "integer", "description": "Scheduler priority (default 0; higher dispatches first)."},
                    "nproc": {"type": "string", "description": "Override processes-per-node (int, or \"gpu\"/\"cpu\"/\"auto\")."},
                    "dataset_server_id": {"type": "string", "description": "Optional dataset-server id (from list_dataset_servers) to source training data; omit for the in-process loader."},
                    "dynamic_args": {"type": "object", "description": "Config-specific args (from inspect_config's dynamic_args), keyed by dest."},
                },
                "required": ["project_dir", "config_name"],
            },
            handler=_run_train,
            risk=CONFIRM,
        )
    )
    reg.register(
        ToolSpec(
            name="list_dataset_servers",
            description=(
                "List dataset servers available to query (local, user-"
                "registered, and cluster), with whether each is currently "
                "reachable. Use before dataset_info to pick a server, or to "
                "tell the user none is running."
            ),
            json_schema={"type": "object", "properties": {}},
            handler=_list_dataset_servers,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="dataset_info",
            description=(
                "Get a dataset's splits, number of examples per split, and "
                "feature/column names by querying a dataset server. Use this "
                "when defining or smoke-testing a dataset project (these are "
                "not obvious from the config). Pass the dataset's HF name/path "
                "(read it from the config's load_dataset args via "
                "inspect_config / render_config_pp). The data must already be "
                "built/cached and a dataset server reachable (see "
                "list_dataset_servers) — run_dataset on a raw split first to "
                "build it. If none is reachable, you'll get an error to relay "
                "to the user."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "dataset": {"type": "string", "description": "HF dataset name or path (from the config's load_dataset args)."},
                    "server_id": {"type": "string", "description": "Optional dataset-server id from list_dataset_servers (default: first reachable)."},
                    "split": {"type": "string", "description": "Optional split to use for the features fallback load."},
                },
                "required": ["dataset"],
            },
            handler=_dataset_info,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="gpu_status",
            description=(
                "Snapshot the GPUs: per-card name, total/used/free memory, "
                "utilization, temperature, and whether the card is disabled or "
                "excluded from scheduling. Use to advise requested_gpus before "
                "run_train / run_eval, or to see what is busy."
            ),
            json_schema={"type": "object", "properties": {}},
            handler=_gpu_status,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="control_job",
            description=(
                "Control a RUNNING training job by queue_id: save a checkpoint, "
                "gracefully stop (saves a final checkpoint), save-and-stop, or "
                "abort (immediate, no final checkpoint). Approval required. Only "
                "valid for training jobs that have started (are correlated to a "
                "trainer endpoint)."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "queue_id": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["save", "stop", "save-stop", "abort"],
                        "description": "save | stop | save-stop | abort.",
                    },
                },
                "required": ["queue_id", "action"],
            },
            handler=_control_job,
            risk=CONFIRM,
        )
    )
    reg.register(
        ToolSpec(
            name="cleanup_jobs",
            description=(
                "Remove FINISHED job records from the Jobs list (terminal: "
                "done/failed/aborted) and their captured TTY logs. Approval "
                "required. Prefer passing queue_ids — the specific jobs you "
                "spawned and are done reporting on — so you don't clear jobs the "
                "user started. Set all_terminal=true only to clear every "
                "completed job (the 'Clean completed' button). Running jobs are "
                "never affected. Use this to tidy up after short-lived jobs "
                "(dataset builds, construct/eval runs); don't remove a job whose "
                "output the user may still want."
            ),
            summary="Remove finished job records (yours by queue_id, or all). CONFIRM.",
            json_schema={
                "type": "object",
                "properties": {
                    "queue_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Finished jobs to remove (the ones you spawned).",
                    },
                    "all_terminal": {
                        "type": "boolean",
                        "description": "Remove every completed job instead of specific ids (default false).",
                    },
                },
            },
            handler=_cleanup_jobs,
            risk=CONFIRM,
        )
    )
    reg.register(
        ToolSpec(
            name="list_eval_configs",
            description=(
                "List the available evaluation configs (name, project_dir, "
                "template, description, default batch_size/max_length/stride). "
                "Use to pick an eval_name for run_eval."
            ),
            summary="List available evaluation configs (for run_eval).",
            json_schema={"type": "object", "properties": {}},
            handler=_list_eval_configs,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="run_eval",
            description=(
                "Run an evaluation job on the scheduler (the equivalent of "
                "`forgather eval test ... -M <model>`). Scores a model against an "
                "eval config's dataset (perplexity / loss / bpb). eval_name is a "
                "name from list_eval_configs; model_path is the model output dir "
                "or checkpoint to evaluate. gpus defaults to 1 (0 = CPU). "
                "Optional: batch_size, max_length, stride, checkpoint_path, "
                "trainer. Approval required. Returns a queue_id; watch it with "
                "list_jobs / read_job_output and read results with "
                "list_evaluations once it is done."
            ),
            summary="Run an evaluation job for a model (CONFIRM-gated).",
            json_schema={
                "type": "object",
                "properties": {
                    "eval_name": {"type": "string", "description": "Eval config name (from list_eval_configs)."},
                    "model_path": {"type": "string", "description": "Model output dir or checkpoint to evaluate."},
                    "gpus": {"type": "integer", "description": "GPUs to reserve (default 1; 0 = CPU)."},
                    "batch_size": {"type": "integer"},
                    "max_length": {"type": "integer"},
                    "stride": {"type": "integer"},
                    "checkpoint_path": {"type": "string"},
                    "trainer": {"type": "string", "description": "simple | ddp | pipeline (default ddp)."},
                },
                "required": ["eval_name", "model_path"],
            },
            handler=_run_eval,
            risk=CONFIRM,
        )
    )
