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

import math
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import auth as auth_mod
from .. import config_ops, dataset_source
from .. import paths as fs_paths
from .. import queue_store, scheduler
from ..dataset_source import DatasetSourceError


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


_SUPPORTED_JOB_TYPES = {
    "training",
    "eval",
    "inference",
    "dataset_server",
    "diloco_server",
    "tensorboard",
    "mkdocs",
    "convert",
    "finalize",
    "update",
    "model",
    "dataset",
    "construct",
}
# Required job_params keys per job type. Training jobs are absent because
# their real parameters live in project_dir/config/dynamic_args; model and
# dataset jobs are absent because every flag is optional (defaults match
# the CLI parsers).
_REQUIRED_PARAMS_BY_TYPE = {
    "eval": {"eval_project", "eval_template", "model_path"},
    "inference": {"model_path", "port"},
    "dataset_server": {"port"},
    "diloco_server": {"output_dir", "port", "num_workers"},
    "tensorboard": {"logdir", "port"},
    "mkdocs": {"config_file", "port"},
    "convert": {"src_model_path", "dst_model_path"},
    "finalize": {"source", "dest"},
    "update": {"src_model_path", "dst_model_path"},
}
_VALID_MODEL_SUBCOMMANDS = {"construct", "test"}
# Types that accept ``requested_gpus == 0``. Inference is included now
# that the spawn path passes ``-d cpu`` when no GPU is reserved (the
# inference server's _default_device() detects CPU correctly, and CPU
# inference is a legitimate path on hosts without a CUDA driver — e.g.
# laptops / Chromebooks running ``--gpus none`` under docker). Training
# and eval still require >= 1 GPU here even though the underlying
# CLIs now support CPU; gating those at the queue layer is a separate
# concern (they need user-facing knobs we haven't added yet).
# Convert / finalize default to CPU (they're pure I/O + tensor reshape
# work) but the user can opt into a GPU via the modal's device field.
# ``model`` defaults to CPU/meta too — the user opts into a GPU via the
# device field; ``dataset`` is always CPU-only.
_ZERO_GPU_JOB_TYPES = {
    "tensorboard",
    "mkdocs",
    "convert",
    "finalize",
    "update",
    "model",
    "dataset",
    "dataset_server",
    "diloco_server",
    "construct",
    "inference",
    # Training and eval support CPU dispatch now that:
    #   * the train CLI / launcher.build_command falls back from
    #     nproc_per_node="gpu" to 1 when no GPU is reserved
    #     (see distributed.py CPU/gloo backend fix);
    #   * the eval CLI / eval_ops.build_eval_command does the same
    #     for --trainer ddp / pipeline.
    # Useful for CPU debugging of training/eval pipelines on hosts
    # without a CUDA driver.
    "training",
    "eval",
}


@router.get("/queue", response_model=List[QueueItemModel])
def list_queue():
    items = queue_store.list_items()
    items.sort(key=lambda it: (-it.priority, it.submitted_at))
    return [_to_model(it) for it in items]


@router.post("/queue", response_model=QueueItemModel)
def enqueue(req: EnqueueRequest):
    # Same fs-root gate the GET /api/config/dynamic-args endpoint gets:
    # the enqueue path below runs template-preprocess via
    # load_dynamic_args(req.project_dir, ...), which follows
    # ``-- include`` chains anywhere project_dir points. Demo mode
    # already 403s the POST via the mutation gate, but the fs-root
    # allowlist is meant to apply outside demo too (operator runs the
    # server with --fs-root for a tighter local sandbox); without this
    # check that intent is bypassed.
    _enforce_fs_root(req.project_dir)
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
    # Required dynamic-args are enforced here rather than at form-render
    # time so any client (CLI, scripted enqueues) gets the same guarantee.
    # Training, model, and dataset jobs all materialize the same dynamic
    # args; other types don't consume them.
    if req.job_type in ("training", "model", "dataset", "construct"):
        # If the schema can't load (template parse error, missing config,
        # etc.) we surface the failure as 400 rather than silently treating
        # it as an empty schema — otherwise required-field enforcement is
        # bypassed exactly when the config is broken.
        try:
            schema = config_ops.load_dynamic_args(req.project_dir, req.config)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"could not load dynamic-args schema for " f"{req.config!r}: {e}"
                ),
            )
        missing = [
            a.cli_name
            for a in schema
            if a.required and req.dynamic_args.get(a.dest) in (None, "")
        ]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"required dynamic arg(s) missing: {missing}",
            )
        # Numeric bounds: only checked when the user has actually supplied a
        # value (template defaults may legitimately sit outside any
        # newly-tightened bound). Same closed-interval semantics as the
        # webui — both endpoints inclusive.
        bound_violations: list[str] = []
        for a in schema:
            if a.type not in ("int", "float"):
                continue
            if a.min is None and a.max is None:
                continue
            v = req.dynamic_args.get(a.dest)
            if v is None or isinstance(v, bool):
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            # Reject NaN / Inf — neither passes ordinary `<` / `>` checks
            # against a finite bound, so they would silently sneak past
            # without this guard.
            if math.isnan(fv) or math.isinf(fv):
                bound_violations.append(f"{a.cli_name}: not a finite number")
                continue
            if a.min is not None and fv < a.min:
                bound_violations.append(f"{a.cli_name} >= {a.min}")
            if a.max is not None and fv > a.max:
                bound_violations.append(f"{a.cli_name} <= {a.max}")
        if bound_violations:
            raise HTTPException(
                status_code=400,
                detail=f"dynamic arg constraint violated: {bound_violations}",
            )
    required = _REQUIRED_PARAMS_BY_TYPE.get(req.job_type)
    if required is not None:
        missing = required - set(req.job_params.keys())
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"{req.job_type} job_params missing: {sorted(missing)}",
            )
    if req.job_type == "model":
        sub = req.job_params.get("subcommand", "construct")
        if sub not in _VALID_MODEL_SUBCOMMANDS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"model subcommand must be one of "
                    f"{sorted(_VALID_MODEL_SUBCOMMANDS)}; got {sub!r}"
                ),
            )

    # Merge dataset-source env vars into job_params.extra_env. Applies
    # to every job type whose subprocess goes through
    # ``fast_load_iterable_dataset`` — training, eval, and the
    # ``forgather model`` / ``forgather dataset`` CLI paths. Inference,
    # mkdocs, tensorboard etc. ignore the field entirely. Resolution
    # failures (stale ids) become 400s so the operator sees a clear
    # error rather than a job that silently fell back to in-process
    # loading.
    job_params = dict(req.job_params)
    if (
        req.job_type in ("training", "eval", "model", "dataset", "construct")
        and req.dataset_source
    ):
        try:
            ds_env = dataset_source.resolve_to_env(req.dataset_source)
        except DatasetSourceError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if ds_env:
            merged_env = dict(job_params.get("extra_env") or {})
            # Caller-supplied extra_env wins on conflict — same precedence
            # we use elsewhere (CLI-style explicit-over-default).
            for k, v in ds_env.items():
                merged_env.setdefault(k, v)
            job_params["extra_env"] = merged_env

    item = queue_store.QueueItem.new(
        project_dir=req.project_dir,
        config=req.config,
        dynamic_args=req.dynamic_args,
        requested_gpus=req.requested_gpus,
        priority=req.priority,
        job_type=req.job_type,
        job_params=job_params,
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
