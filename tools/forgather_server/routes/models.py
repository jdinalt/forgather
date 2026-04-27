"""Per-project model output endpoints.

Read-only views into what a project has produced on disk: models grouped
by ``output_dir``, runs under each model, run summaries, and checkpoints.
All paths are derived from each config's materialized meta, so external
``output_dir`` values (common in finetune setups) show up just like the
default ``<project>/output_models/`` case.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .. import eval_ops, models_catalog

router = APIRouter(tags=["models"])


class ModelEntryModel(BaseModel):
    output_dir: str
    model_name: str
    configs: List[str]
    exists: bool
    run_count: int = 0
    checkpoint_count: int = 0
    eval_count: int = 0
    total_size_bytes: int = 0
    parse_errors: Dict[str, str] = {}


class EvalResultModel(BaseModel):
    """Mirrors :class:`forgather.eval_config.EvalResult`.

    Kept in lock-step with the library dataclass — adding a field there
    means adding it here (the route layer does ``asdict(result)`` and
    feeds the payload straight in).
    """

    eval_name: str
    config_name: str
    description: str
    dataset_proj: str
    dataset_config: str
    dataset_target: str
    model_path: str
    checkpoint_path: Optional[str] = None
    batch_size: int
    max_length: int
    stride: int
    dtype: str
    attn_implementation: str
    trainer: str
    world_size: int
    eval_loss: Optional[float] = None
    perplexity: Optional[float] = None
    wall_time_s: Optional[float] = None
    timestamp: Optional[str] = None


class EvalEntryModel(BaseModel):
    eval_dir: str
    eval_id: str
    result: Optional[EvalResultModel] = None
    parse_error: Optional[str] = None


class EvalConfigEntryModel(BaseModel):
    """One available eval config for the evaluate-a-model picker."""

    name: str
    project_dir: str
    template: str
    description: str
    default_batch_size: int
    default_max_length: int
    default_stride: int


class RunEntryModel(BaseModel):
    run_dir: str
    run_id: str
    started_at: float
    has_logs: bool
    hostname: Optional[str] = None
    tty_log_path: Optional[str] = None


class CheckpointEntryModel(BaseModel):
    checkpoint_dir: str
    step: int
    size_bytes: int = 0
    world_size: Optional[int] = None
    timestamp: Optional[str] = None
    manifest_present: bool = False


class RunSummaryModel(BaseModel):
    summary: Dict[str, Any]
    log_path: Optional[str] = None
    config_path: Optional[str] = None
    pp_path: Optional[str] = None


@router.get("/project/models", response_model=List[ModelEntryModel])
def list_project_models(project_dir: str):
    """Every distinct ``output_dir`` produced by configs in this project."""
    try:
        entries = models_catalog.get_project_models(project_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return [
        ModelEntryModel(
            output_dir=m.output_dir,
            model_name=m.model_name,
            configs=m.configs,
            exists=m.exists,
            run_count=m.run_count,
            checkpoint_count=m.checkpoint_count,
            eval_count=m.eval_count,
            total_size_bytes=m.total_size_bytes,
            parse_errors=m.parse_errors,
        )
        for m in entries
    ]


@router.get("/model/evaluations", response_model=List[EvalEntryModel])
def list_model_evaluations(output_dir: str):
    """List parsed ``<output_dir>/evals/*/results.json`` entries.

    ``result`` is a full :class:`forgather.eval_config.EvalResult`
    payload; ``parse_error`` is populated when the directory had no /
    invalid results.json (partial runs stay visible rather than vanish).
    """
    try:
        evals = models_catalog.list_model_evaluations(output_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    out: List[EvalEntryModel] = []
    for ev in evals:
        result_model: Optional[EvalResultModel] = None
        if ev.result is not None:
            result_model = EvalResultModel(**asdict(ev.result))
        out.append(
            EvalEntryModel(
                eval_dir=ev.eval_dir,
                eval_id=ev.eval_id,
                result=result_model,
                parse_error=ev.parse_error,
            )
        )
    return out


@router.get("/model/runs", response_model=List[RunEntryModel])
def list_model_runs(output_dir: str):
    """List every run under ``<output_dir>/runs/``, newest first."""
    try:
        runs = models_catalog.get_model_runs(output_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return [
        RunEntryModel(
            run_dir=r.run_dir,
            run_id=r.run_id,
            started_at=r.started_at,
            has_logs=r.has_logs,
            hostname=r.hostname,
            tty_log_path=r.tty_log_path,
        )
        for r in runs
    ]


@router.get("/run/tty", response_class=PlainTextResponse)
def get_run_tty(run_dir: str):
    """Return the TTY log captured for a historical run.

    Tail-only: capped at 8 MiB from the end of the file. 404 when the run
    has no ``tty.log``.
    """
    try:
        return models_catalog.read_run_tty(run_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/run/summary", response_model=RunSummaryModel)
def get_run_summary(run_dir: str):
    """Summary stats + paths to the run's config / pp / log artifacts.

    ``summary`` is ``{}`` when ``trainer_logs.json`` is missing (e.g.
    still-setting-up run), and ``{"error": ...}`` when parsing fails.
    """
    try:
        data = models_catalog.get_run_summary(run_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RunSummaryModel(**data)


@router.get("/eval-configs", response_model=List[EvalConfigEntryModel])
def list_eval_configs():
    """Every discoverable eval config (same set as ``forgather eval list``)."""
    try:
        entries = eval_ops.list_eval_configs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return [
        EvalConfigEntryModel(
            name=e.name,
            project_dir=e.project_dir,
            template=e.template,
            description=e.description,
            default_batch_size=e.default_batch_size,
            default_max_length=e.default_max_length,
            default_stride=e.default_stride,
        )
        for e in entries
    ]


@router.get("/model/checkpoints", response_model=List[CheckpointEntryModel])
def list_model_checkpoints(output_dir: str):
    """List ``checkpoint-N`` directories, newest-step first."""
    try:
        checkpoints = models_catalog.list_run_checkpoints(output_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return [
        CheckpointEntryModel(
            checkpoint_dir=c.checkpoint_dir,
            step=c.step,
            size_bytes=c.size_bytes,
            world_size=c.world_size,
            timestamp=c.timestamp,
            manifest_present=c.manifest_present,
        )
        for c in checkpoints
    ]
