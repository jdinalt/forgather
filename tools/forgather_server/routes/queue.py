"""Queue management endpoints (waiting list only).

POST /api/queue              enqueue a training job
GET  /api/queue              list queued items
DELETE /api/queue/{id}       cancel a queued item (or abort if running)
GET  /api/queue/scheduler    dispatcher on/off + counters
POST /api/queue/scheduler    enable / disable the dispatcher
GET  /api/config/dynamic-args  form schema for the submit UI

Once the scheduler dispatches a queue item it disappears from this
endpoint and shows up under :mod:`routes.jobs` as a JobRecord. TTY
streaming + control therefore live with the Jobs API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import config_ops, queue_store, scheduler

router = APIRouter(tags=["queue"])


class DynamicArgModel(BaseModel):
    dest: str
    cli_name: str
    type: str
    help: Optional[str] = None
    default: Any = None
    choices: Optional[List[Any]] = None


class QueueItemModel(BaseModel):
    queue_id: str
    project_dir: str
    config: str
    dynamic_args: Dict[str, Any]
    requested_gpus: int
    priority: int
    submitted_at: float
    job_type: str = "training"
    job_params: Dict[str, Any] = Field(default_factory=dict)


class EnqueueRequest(BaseModel):
    project_dir: str
    config: str
    dynamic_args: Dict[str, Any] = Field(default_factory=dict)
    requested_gpus: int = 1
    priority: int = 0
    # Omit for legacy-style training enqueues (defaults to "training").
    # See QueueItem for the per-type shape of job_params.
    job_type: str = "training"
    job_params: Dict[str, Any] = Field(default_factory=dict)


class SchedulerStatusModel(BaseModel):
    enabled: bool
    tick_count: int
    last_tick_at: Optional[float]
    running_count: int


class SchedulerRequest(BaseModel):
    enabled: bool


def _to_model(item: queue_store.QueueItem) -> QueueItemModel:
    return QueueItemModel(
        queue_id=item.queue_id,
        project_dir=item.project_dir,
        config=item.config,
        dynamic_args=item.dynamic_args,
        requested_gpus=item.requested_gpus,
        priority=item.priority,
        submitted_at=item.submitted_at,
        job_type=item.job_type,
        job_params=item.job_params,
    )


_SUPPORTED_JOB_TYPES = {
    "training",
    "eval",
    "inference",
    "tensorboard",
    "mkdocs",
    "convert",
    "finalize",
}
# Required keys per job type. ``job_params`` for training is always empty
# (the real parameters live in ``project_dir``/``config``/``dynamic_args``).
_REQUIRED_EVAL_PARAMS = {"eval_project", "eval_template", "model_path"}
_REQUIRED_INFERENCE_PARAMS = {"model_path", "port"}
_REQUIRED_TENSORBOARD_PARAMS = {"logdir", "port"}
_REQUIRED_MKDOCS_PARAMS = {"config_file", "port"}
_REQUIRED_CONVERT_PARAMS = {"src_model_path", "dst_model_path"}
_REQUIRED_FINALIZE_PARAMS = {"source", "dest"}
# Types that accept ``requested_gpus == 0``. Everything else still needs
# at least one GPU (training / eval / inference all spawn CUDA workloads).
# Convert / finalize default to CPU (they're pure I/O + tensor reshape
# work) but the user can opt into a GPU via the modal's device field.
_ZERO_GPU_JOB_TYPES = {"tensorboard", "mkdocs", "convert", "finalize"}


@router.get("/queue", response_model=List[QueueItemModel])
def list_queue():
    items = queue_store.list_items()
    items.sort(key=lambda it: (-it.priority, it.submitted_at))
    return [_to_model(it) for it in items]


@router.post("/queue", response_model=QueueItemModel)
def enqueue(req: EnqueueRequest):
    if req.job_type not in _SUPPORTED_JOB_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported job_type: {req.job_type!r}",
        )
    # GPU requirement is type-dependent. CPU-only viewers (tensorboard)
    # run with zero reserved GPUs; everything else still needs >= 1.
    min_gpus = 0 if req.job_type in _ZERO_GPU_JOB_TYPES else 1
    if req.requested_gpus < min_gpus:
        raise HTTPException(
            status_code=400,
            detail=(
                f"requested_gpus must be >= {min_gpus} for " f"{req.job_type} jobs"
            ),
        )
    if req.job_type == "eval":
        missing = _REQUIRED_EVAL_PARAMS - set(req.job_params.keys())
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"eval job_params missing: {sorted(missing)}",
            )
    elif req.job_type == "inference":
        missing = _REQUIRED_INFERENCE_PARAMS - set(req.job_params.keys())
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"inference job_params missing: {sorted(missing)}",
            )
    elif req.job_type == "tensorboard":
        missing = _REQUIRED_TENSORBOARD_PARAMS - set(req.job_params.keys())
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"tensorboard job_params missing: {sorted(missing)}",
            )
    elif req.job_type == "mkdocs":
        missing = _REQUIRED_MKDOCS_PARAMS - set(req.job_params.keys())
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"mkdocs job_params missing: {sorted(missing)}",
            )
    elif req.job_type == "convert":
        missing = _REQUIRED_CONVERT_PARAMS - set(req.job_params.keys())
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"convert job_params missing: {sorted(missing)}",
            )
    elif req.job_type == "finalize":
        missing = _REQUIRED_FINALIZE_PARAMS - set(req.job_params.keys())
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"finalize job_params missing: {sorted(missing)}",
            )
    item = queue_store.QueueItem.new(
        project_dir=req.project_dir,
        config=req.config,
        dynamic_args=req.dynamic_args,
        requested_gpus=req.requested_gpus,
        priority=req.priority,
        job_type=req.job_type,
        job_params=req.job_params,
    )
    queue_store.add_item(item)
    return _to_model(item)


@router.delete("/queue/{queue_id}")
def abort(queue_id: str):
    """Cancel a queued item, or abort it if it has already dispatched.

    Either path returns ``{aborted: queue_id}``. 404 if unknown.
    """
    if not scheduler.abort_or_cancel(queue_id):
        raise HTTPException(
            status_code=404,
            detail=f"queue item {queue_id} not found or already terminal",
        )
    return {"aborted": queue_id}


@router.get("/queue/scheduler", response_model=SchedulerStatusModel)
def scheduler_status():
    s = scheduler.get_state()
    return SchedulerStatusModel(
        enabled=s.enabled,
        tick_count=s.tick_count,
        last_tick_at=s.last_tick_at,
        running_count=len(s.running),
    )


@router.post("/queue/scheduler", response_model=SchedulerStatusModel)
def scheduler_toggle(req: SchedulerRequest):
    scheduler.set_enabled(req.enabled)
    return scheduler_status()


@router.get("/config/dynamic-args", response_model=List[DynamicArgModel])
def dynamic_args(project_dir: str, config: str):
    args = config_ops.load_dynamic_args(project_dir, config)
    return [
        DynamicArgModel(
            dest=a.dest,
            cli_name=a.cli_name,
            type=a.type,
            help=a.help,
            default=a.default,
            choices=a.choices,
        )
        for a in args
    ]
