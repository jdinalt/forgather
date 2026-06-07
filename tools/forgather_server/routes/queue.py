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

from .. import auth as auth_mod
from .. import config_ops
from .. import paths as fs_paths
from .. import queue_ops, queue_store, scheduler


def _enforce_fs_root(path) -> None:
    """403 if the path isn't under the configured fs-root allowlist."""
    if fs_paths.is_path_in_fs_root(path):
        return
    raise HTTPException(
        status_code=403,
        detail=f"path is outside the configured filesystem roots: {path}",
    )


router = APIRouter(tags=["queue"])


class DynamicArgModel(BaseModel):
    dest: str
    cli_name: str
    type: str
    help: Optional[str] = None
    default: Any = None
    choices: Optional[List[Any]] = None
    # Colon-separated organizational path; webui groups + collapses by it.
    group: Optional[str] = None
    # Enforced server-side for training enqueues; the CLI also blocks the
    # train action when missing. Skipped for ``pp`` so placeholder defaults
    # still materialize.
    required: bool = False
    # Inclusive numeric bounds for int / float args. Webui flags violations
    # in red and blocks Submit; server enqueue rejects with HTTP 400.
    min: Optional[float] = None
    max: Optional[float] = None


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
    # Submit-modal dataset-source choice. ``None`` (or {"kind":"local"})
    # leaves the training process to load datasets in-process; a
    # ``{"kind":"server","server_id":"..."}`` value is resolved server-
    # side to ``FORGATHER_DATASET_SERVER[_TOKEN]`` env vars that are
    # merged into ``job_params.extra_env`` here. Only meaningful for
    # ``job_type == "training"`` — other job types ignore it.
    dataset_source: Optional[Dict[str, Any]] = None


class SchedulerStatusModel(BaseModel):
    enabled: bool
    tick_count: int
    last_tick_at: Optional[float]
    running_count: int


class SchedulerRequest(BaseModel):
    enabled: bool


def _to_model(item: queue_store.QueueItem) -> QueueItemModel:
    # In demo mode strip bearer tokens from the dicts we ship to the
    # webui. Two leak vectors here: inference job_params can carry
    # ``auth_token`` directly (the inference modal builds this dict);
    # job_params["extra_env"] can carry e.g.
    # ``FORGATHER_DATASET_SERVER_TOKEN`` after the dataset-source
    # resolver merges it in at submit time. Both get caught by the
    # substring scrubber in ``redact_sensitive_in_demo``.
    return QueueItemModel(
        queue_id=item.queue_id,
        project_dir=item.project_dir,
        config=item.config,
        dynamic_args=auth_mod.redact_sensitive_in_demo(item.dynamic_args),
        requested_gpus=item.requested_gpus,
        priority=item.priority,
        submitted_at=item.submitted_at,
        job_type=item.job_type,
        job_params=auth_mod.redact_sensitive_in_demo(item.job_params),
    )


@router.get("/queue", response_model=List[QueueItemModel])
def list_queue():
    items = queue_store.list_items()
    items.sort(key=lambda it: (-it.priority, it.submitted_at))
    return [_to_model(it) for it in items]


@router.post("/queue", response_model=QueueItemModel)
def enqueue(req: EnqueueRequest):
    # Validation + enqueue lives in queue_ops so the agent's run_dataset tool
    # applies the same checks. Map its exceptions back to HTTP status codes:
    # fs-root failures stay 403; everything else is 400.
    try:
        item = queue_ops.validate_and_enqueue(
            project_dir=req.project_dir,
            config=req.config,
            dynamic_args=req.dynamic_args,
            requested_gpus=req.requested_gpus,
            priority=req.priority,
            job_type=req.job_type,
            job_params=req.job_params,
            dataset_source=req.dataset_source,
        )
    except queue_ops.EnqueueForbidden as e:
        raise HTTPException(status_code=403, detail=str(e))
    except queue_ops.EnqueueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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
    _enforce_fs_root(project_dir)
    args = config_ops.load_dynamic_args(project_dir, config)
    return [
        DynamicArgModel(
            dest=a.dest,
            cli_name=a.cli_name,
            type=a.type,
            help=a.help,
            default=a.default,
            choices=a.choices,
            group=a.group,
            required=a.required,
            min=a.min,
            max=a.max,
        )
        for a in args
    ]
