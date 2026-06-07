"""Agent tools for training results: models, runs, checkpoints, evals.

All READ. They wrap the in-process ``models_catalog`` catalog functions (the
same ones the Models/Runs/Checkpoints panels use) plus the trainer-control
``get_job_status`` proxy, so the agent can answer "how did training go?",
"which checkpoint is best/latest?", and "what is this run doing right now?".

``models_catalog`` returns dataclasses; we ``asdict`` them for JSON. The
catalog functions already fail soft (return ``[]`` on a missing/parse-error
dir), so these tools surface an empty list rather than raising for an
unbuilt output dir.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List

from .. import job_records, models_catalog
from .registry import EXTENDED, READ, ToolRegistry, ToolSpec

log = logging.getLogger("forgather_server.agent.tools_models")

# Keep a run's TTY tail tiny — it goes straight into model context.
_RUN_TTY_MAX_BYTES = 16 * 1024
_DEFAULT_RUN_TAIL_LINES = 200


def _rows(entries: List[Any]) -> List[Dict[str, Any]]:
    return [asdict(e) for e in entries]


def _list_models(args: Dict[str, Any]) -> Any:
    return {"models": _rows(models_catalog.get_project_models(args["project_dir"]))}


def _list_runs(args: Dict[str, Any]) -> Any:
    return {"runs": _rows(models_catalog.get_model_runs(args["output_dir"]))}


def _list_checkpoints(args: Dict[str, Any]) -> Any:
    return {"checkpoints": _rows(models_catalog.list_run_checkpoints(args["output_dir"]))}


def _list_evaluations(args: Dict[str, Any]) -> Any:
    return {"evaluations": _rows(models_catalog.list_model_evaluations(args["output_dir"]))}


def _run_summary(args: Dict[str, Any]) -> Any:
    # Already a dict ({summary, log_path, config_path, pp_path}).
    return models_catalog.get_run_summary(args["run_dir"])


def _read_run_tty(args: Dict[str, Any]) -> Any:
    run_dir = args["run_dir"]
    tail_lines = args.get("tail_lines")
    tail_lines = _DEFAULT_RUN_TAIL_LINES if tail_lines in (None, "") else int(tail_lines)
    try:
        text = models_catalog.read_run_tty(run_dir, max_bytes=_RUN_TTY_MAX_BYTES)
    except FileNotFoundError:
        raise ValueError(f"no console log (tty.log) found under {run_dir!r}")
    lines = text.splitlines()
    if len(lines) > tail_lines:
        lines = lines[-tail_lines:]
    return {"run_dir": run_dir, "tail": "\n".join(lines)}


def _job_status(args: Dict[str, Any]) -> Any:
    # Live trainer status (step/loss/...) for a running job. Resolve our
    # queue_id to the correlated trainer job_id first (mirrors the HTTP route).
    queue_id = args["queue_id"]
    rec = job_records.get_record(queue_id)
    if rec is None:
        raise ValueError(f"no job with queue_id {queue_id!r} (use list_jobs)")
    if rec.job_id is None:
        return {
            "queue_id": queue_id,
            "status": rec.status,
            "trainer": None,
            "note": "not yet correlated to a trainer endpoint (still starting)",
        }
    # Import lazily: trainer_control pulls in heavier deps and isn't needed
    # for the catalog-only tools.
    from forgather import trainer_control

    try:
        status = trainer_control.get_job_status(rec.job_id)
    except Exception as e:  # trainer unreachable / not serving /status yet
        return {"queue_id": queue_id, "status": rec.status, "error": f"{type(e).__name__}: {e}"}
    return {"queue_id": queue_id, "status": rec.status, "trainer": status}


def register_all(reg: ToolRegistry) -> None:
    reg.register(
        ToolSpec(
            name="list_models",
            description=(
                "List the trained models (output dirs) of a project, each with "
                "its configs and run / checkpoint / evaluation counts. Use to "
                "find a model's output_dir before list_runs / list_checkpoints / "
                "list_evaluations. project_dir is the project directory."
            ),
            json_schema={
                "type": "object",
                "properties": {"project_dir": {"type": "string"}},
                "required": ["project_dir"],
            },
            handler=_list_models,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="run_summary",
            description=(
                "Summarize a training run from its log: best/last loss, steps, "
                "perplexity and related stats. The primary answer to \"how did "
                "this run go?\". run_dir comes from list_runs (each run entry's "
                "run_dir). Returns {summary, log_path, config_path, pp_path}."
            ),
            json_schema={
                "type": "object",
                "properties": {"run_dir": {"type": "string"}},
                "required": ["run_dir"],
            },
            handler=_run_summary,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="list_checkpoints",
            description=(
                "List the checkpoints saved under a model output_dir: step, "
                "size, world_size, timestamp, and whether a manifest is present. "
                "Use to pick the best/latest checkpoint (e.g. to resume, eval, or "
                "serve). output_dir comes from list_models."
            ),
            json_schema={
                "type": "object",
                "properties": {"output_dir": {"type": "string"}},
                "required": ["output_dir"],
            },
            handler=_list_checkpoints,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="job_status",
            description=(
                "Get the LIVE trainer status (step, loss, etc.) of a running "
                "training job by queue_id. Unlike read_job_output (raw TTY tail) "
                "this is the structured trainer /status. Returns trainer:null "
                "with a note while the job is still starting, or an error field "
                "if the trainer isn't reachable yet."
            ),
            json_schema={
                "type": "object",
                "properties": {"queue_id": {"type": "string"}},
                "required": ["queue_id"],
            },
            handler=_job_status,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="list_runs",
            description=(
                "List the training runs recorded under a model output_dir, with "
                "timestamps, hostname, and log paths. Feed a run's run_dir to "
                "run_summary or read_run_tty."
            ),
            summary="List training runs under an output_dir (for run_summary).",
            json_schema={
                "type": "object",
                "properties": {"output_dir": {"type": "string"}},
                "required": ["output_dir"],
            },
            handler=_list_runs,
            risk=READ,
            tier=EXTENDED,
        )
    )
    reg.register(
        ToolSpec(
            name="list_evaluations",
            description=(
                "List the evaluations recorded under a model output_dir, with "
                "their results (eval_loss, perplexity, bpb, dataset, checkpoint, "
                "etc.). Use to report or compare a model's eval results."
            ),
            summary="List a model's evaluation results under an output_dir.",
            json_schema={
                "type": "object",
                "properties": {"output_dir": {"type": "string"}},
                "required": ["output_dir"],
            },
            handler=_list_evaluations,
            risk=READ,
            tier=EXTENDED,
        )
    )
    reg.register(
        ToolSpec(
            name="read_run_tty",
            description=(
                "Read the tail of a finished/older run's console log (tty.log) "
                "by run_dir — for runs not (or no longer) tracked as scheduler "
                "jobs (use read_job_output for an active queue_id instead). "
                "Returns the last tail_lines lines (default 200), capped to a "
                "small size to protect context."
            ),
            summary="Tail a run's tty.log by run_dir (older runs not in the job list).",
            json_schema={
                "type": "object",
                "properties": {
                    "run_dir": {"type": "string"},
                    "tail_lines": {"type": "integer", "description": "Trailing lines (default 200)."},
                },
                "required": ["run_dir"],
            },
            handler=_read_run_tty,
            risk=READ,
            tier=EXTENDED,
        )
    )
