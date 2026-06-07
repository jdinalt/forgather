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
from typing import Any, Dict, List, Optional

from .. import job_records, queue_store
from ..routes.jobs import read_tty_tail
from .registry import READ, ToolRegistry, ToolSpec

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
