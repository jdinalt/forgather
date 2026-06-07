"""Agent tools for the job scheduler and dataset workflow.

Three groups, all wrapping existing server machinery:

- Job visibility (READ): ``list_jobs`` and ``read_job_output`` let the agent
  watch a job it submitted — its status and a tail of its console/TTY output.
- ``run_dataset`` (CONFIRM): submit a ``dataset`` build/inspect job to the
  scheduler (gated, since it downloads/builds data and runs code).
- Dataset metadata (READ): ``list_dataset_servers`` / ``dataset_info`` answer
  "what are this dataset's splits, example counts, and features?" by querying a
  running dataset server (local or cluster).

Job state and TTY are read in-process (``job_records`` + the shared
``routes.jobs.read_tty_tail`` tail reader); enqueue goes through the same
validated path the HTTP route uses (``queue_ops.validate_and_enqueue``).
"""

from __future__ import annotations

import logging
import shlex
from typing import Any, Dict, List, Optional

from .. import dataset_ops, job_records, queue_ops, queue_store
from ..routes.jobs import read_tty_tail
from . import _dataset_servers
from .registry import CONFIRM, READ, Proposal, ToolRegistry, ToolSpec

log = logging.getLogger("forgather_server.agent.tools_jobs")

# Keep the TTY tail tiny: this goes straight into the model's context, unlike
# the 32 MiB the HTTP /tty route may return to a browser.
_OUTPUT_MAX_BYTES = 16 * 1024
_DEFAULT_TAIL_LINES = 200
_DEFAULT_JOBS_LIMIT = 50

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


# ---- run a dataset job (CONFIRM) -------------------------------------------


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
        return (
            f"enqueued dataset job {item.queue_id} (target={target}). "
            f"Poll list_jobs / read_job_output('{item.queue_id}') for progress; "
            f"the first build downloads + builds the dataset and can be slow. "
            f"Do not report it finished until its status is terminal "
            f"(done / failed / aborted)."
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
