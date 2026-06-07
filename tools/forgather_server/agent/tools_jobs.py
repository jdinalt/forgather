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

from .. import construct_ops, dataset_ops, job_records, queue_ops, queue_store
from ..routes.jobs import read_tty_tail
from . import _dataset_servers
from .registry import CONFIRM, READ, Proposal, ToolRegistry, ToolSpec

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
    # Block server-side (not on the model) until the job is terminal or the
    # timeout elapses. asyncio.sleep yields the event loop, so the scheduler
    # keeps advancing the job while we wait.
    queue_id = args["queue_id"]
    raw = args.get("timeout_seconds")
    timeout = _WAIT_DEFAULT_TIMEOUT if raw in (None, "") else float(raw)
    timeout = max(1.0, min(timeout, _WAIT_MAX_TIMEOUT))

    if job_records.get_record(queue_id) is None:
        raise ValueError(
            f"no job with queue_id {queue_id!r} (use list_jobs to find one)"
        )
    waited = 0.0
    while True:
        rec = job_records.get_record(queue_id)
        if rec is None:
            raise ValueError(f"job {queue_id} disappeared while waiting")
        terminal = rec.status in job_records.TERMINAL_STATUSES
        if terminal or waited >= timeout:
            return {
                "queue_id": queue_id,
                "status": rec.status,
                "exit_code": rec.exit_code,
                "timed_out": not terminal,
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


def _run_train(args: Dict[str, Any]) -> Proposal:
    project_dir = args["project_dir"]
    config_name = args["config_name"]
    dynamic_args = args.get("dynamic_args") or {}
    gpus = args.get("gpus")
    requested_gpus = 1 if gpus in (None, "") else int(gpus)
    nproc = args.get("nproc")

    # No cheap preview command for training (build_command materializes the
    # config). Show the equivalent CLI invocation instead.
    cmd_str = shlex.join(
        ["forgather", "-p", project_dir, "-t", config_name, "train"]
    )

    job_params: Dict[str, Any] = {}
    if nproc not in (None, ""):
        job_params["nproc"] = nproc

    def commit() -> str:
        item = queue_ops.validate_and_enqueue(
            project_dir=project_dir,
            config=config_name,
            dynamic_args=dynamic_args,
            requested_gpus=requested_gpus,
            job_type="training",
            job_params=job_params,
            enforce_fs_root=True,
        )
        return _enqueue_note(item, f"{requested_gpus} GPU(s)") + (
            " Training can run for a long time; check on it periodically rather "
            "than blocking."
        )

    return Proposal(
        title=f"Train: {config_name}",
        summary="Run a `training` job on the scheduler. This trains the model — "
        "a long, resource-intensive run that reserves GPUs.",
        extra={
            "command": cmd_str,
            "requested_gpus": requested_gpus,
            "dynamic_args": dynamic_args or None,
            "warning": "Training is long-running and reserves "
            f"{requested_gpus} GPU(s). It runs as a background scheduler job; "
            "watch it with list_jobs / read_job_output and the training "
            "dashboard.",
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
                "Wait for a job to finish, blocking on the server (NOT by "
                "repeatedly calling list_jobs — that wastes tokens). Polls "
                "internally until the job reaches a terminal status "
                "(done/failed/aborted) or timeout_seconds elapses (default "
                "120, max 600). Returns the final status, exit_code, "
                "timed_out, and a tail of the output. If it times out (e.g. a "
                "long first build), call it again to keep waiting. Use this "
                "after run_dataset instead of looping on list_jobs."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "queue_id": {"type": "string"},
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
                "Run a config as a training job on the scheduler (the equivalent "
                "of `forgather ... train`). This TRAINS the model: a long, "
                "resource-intensive run that reserves GPUs and may take hours. "
                "gpus defaults to 1; set 0 for a CPU smoke-test, or more for "
                "multi-GPU. nproc optionally overrides processes-per-node. "
                "Approval required. Returns immediately with a queue_id; the run "
                "continues in the background, so tell the user it is long-running, "
                "then check on it with list_jobs / read_job_output periodically "
                "(do NOT block on wait_for_job for a full training run) and only "
                "report success once status is terminal."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string"},
                    "config_name": {"type": "string"},
                    "gpus": {"type": "integer", "description": "GPUs to reserve (default 1; 0 = CPU smoke-test)."},
                    "nproc": {"type": "string", "description": "Override processes-per-node (int, or \"gpu\"/\"cpu\"/\"auto\")."},
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
