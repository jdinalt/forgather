"""Validate-and-enqueue core, shared by the HTTP route and the agent.

``routes/queue.py`` and the agent's ``run_dataset`` tool must enqueue jobs the
same way — same job-type/required-arg/numeric-bound/required-param checks, same
dataset-source env merge, same fs-root gate. This module holds that logic in a
framework-agnostic form: it raises plain exceptions (``EnqueueError`` /
``EnqueueForbidden``) instead of ``HTTPException``; the route maps them to
400 / 403, the agent surfaces them as a tool error.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from . import config_ops
from . import paths as fs_paths
from . import queue_store
from .dataset_source import DatasetSourceError, resolve_to_env


class EnqueueError(ValueError):
    """An enqueue request failed validation (maps to HTTP 400)."""


class EnqueueForbidden(EnqueueError):
    """An enqueue request targets a path outside the fs-root allowlist
    (maps to HTTP 403)."""


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
# Required job_params keys per job type. Training jobs are absent because their
# real parameters live in project_dir/config/dynamic_args; model and dataset
# jobs are absent because every flag is optional (defaults match the CLI).
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
# Types that accept ``requested_gpus == 0`` (see the original route comment for
# the per-type rationale; training/eval support CPU dispatch now too).
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
    "training",
    "eval",
}
# Job types that materialize the config's dynamic args (and so get required /
# numeric-bound enforcement) and that honor a dataset-source.
_DYNAMIC_ARG_JOB_TYPES = ("training", "model", "dataset", "construct")
_DATASET_SOURCE_JOB_TYPES = ("training", "eval", "model", "dataset", "construct")


def _enforce_fs_root(path: str) -> None:
    if not fs_paths.is_path_in_fs_root(path):
        raise EnqueueForbidden(
            f"path is outside the configured filesystem roots: {path}"
        )


def _validate_dynamic_args(
    project_dir: str, config: str, dynamic_args: Dict[str, Any]
) -> None:
    # Surface a schema load failure as a validation error rather than silently
    # treating it as an empty schema (which would bypass required-field checks
    # exactly when the config is broken).
    try:
        schema = config_ops.load_dynamic_args(project_dir, config)
    except Exception as e:
        raise EnqueueError(
            f"could not load dynamic-args schema for {config!r}: {e}"
        )
    missing = [
        a.cli_name
        for a in schema
        if a.required and dynamic_args.get(a.dest) in (None, "")
    ]
    if missing:
        raise EnqueueError(f"required dynamic arg(s) missing: {missing}")
    # Numeric bounds: only when the user supplied a value (template defaults may
    # legitimately sit outside a newly-tightened bound). Inclusive on both ends.
    violations: List[str] = []
    for a in schema:
        if a.type not in ("int", "float"):
            continue
        if a.min is None and a.max is None:
            continue
        v = dynamic_args.get(a.dest)
        if v is None or isinstance(v, bool):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv) or math.isinf(fv):
            violations.append(f"{a.cli_name}: not a finite number")
            continue
        if a.min is not None and fv < a.min:
            violations.append(f"{a.cli_name} >= {a.min}")
        if a.max is not None and fv > a.max:
            violations.append(f"{a.cli_name} <= {a.max}")
    if violations:
        raise EnqueueError(f"dynamic arg constraint violated: {violations}")


def validate_and_enqueue(
    *,
    project_dir: str,
    config: str,
    dynamic_args: Dict[str, Any],
    requested_gpus: int,
    priority: int = 0,
    job_type: str = "training",
    job_params: Optional[Dict[str, Any]] = None,
    dataset_source: Optional[Dict[str, Any]] = None,
    enforce_fs_root: bool = True,
) -> queue_store.QueueItem:
    """Validate an enqueue request and add it to the queue.

    Raises :class:`EnqueueForbidden` (fs-root) or :class:`EnqueueError`
    (any other validation failure). Returns the created ``QueueItem`` on
    success.
    """
    dynamic_args = dynamic_args or {}
    job_params = dict(job_params or {})

    if enforce_fs_root:
        _enforce_fs_root(project_dir)
    if job_type not in _SUPPORTED_JOB_TYPES:
        raise EnqueueError(f"unsupported job_type: {job_type!r}")
    min_gpus = 0 if job_type in _ZERO_GPU_JOB_TYPES else 1
    if requested_gpus < min_gpus:
        raise EnqueueError(
            f"requested_gpus must be >= {min_gpus} for {job_type} jobs"
        )
    if job_type in _DYNAMIC_ARG_JOB_TYPES:
        _validate_dynamic_args(project_dir, config, dynamic_args)
    required = _REQUIRED_PARAMS_BY_TYPE.get(job_type)
    if required is not None:
        missing = required - set(job_params.keys())
        if missing:
            raise EnqueueError(f"{job_type} job_params missing: {sorted(missing)}")
    if job_type == "model":
        sub = job_params.get("subcommand", "construct")
        if sub not in _VALID_MODEL_SUBCOMMANDS:
            raise EnqueueError(
                f"model subcommand must be one of "
                f"{sorted(_VALID_MODEL_SUBCOMMANDS)}; got {sub!r}"
            )

    # Merge dataset-source env vars into job_params.extra_env for the job types
    # whose subprocess loads datasets. Resolution failures (stale ids) are
    # validation errors so the caller sees a clear message rather than a job
    # that silently fell back to in-process loading.
    if job_type in _DATASET_SOURCE_JOB_TYPES and dataset_source:
        try:
            ds_env = resolve_to_env(dataset_source)
        except DatasetSourceError as e:
            raise EnqueueError(str(e))
        if ds_env:
            merged_env = dict(job_params.get("extra_env") or {})
            for k, v in ds_env.items():
                merged_env.setdefault(k, v)  # caller-supplied wins
            job_params["extra_env"] = merged_env

    # DiLoCo single-worker submits get a memorable worker_id default (the
    # pool-style modal / CLI already mint names client-side).
    if job_type == "training":
        diloco = job_params.get("diloco")
        if isinstance(diloco, dict) and diloco.get("server_addr"):
            wid = (diloco.get("worker_id") or "").strip()
            if not wid:
                from forgather.utils import generate_name

                diloco = dict(diloco)
                diloco["worker_id"] = generate_name()
                job_params["diloco"] = diloco

    item = queue_store.QueueItem.new(
        project_dir=project_dir,
        config=config,
        dynamic_args=dynamic_args,
        requested_gpus=requested_gpus,
        priority=priority,
        job_type=job_type,
        job_params=job_params,
    )
    queue_store.add_item(item)
    return item
