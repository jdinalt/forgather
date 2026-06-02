"""Persistent records of jobs the server has launched.

Distinct from :mod:`queue_store`, which only holds items still waiting for
GPUs. When the scheduler dispatches a queue item, its metadata moves here
as a :class:`JobRecord` and the queue entry is removed.

Also distinct from :class:`forgather.trainer_control.JobInfo`, which
represents the trainer's own ``endpoint.json`` file under
``~/.config/forgather/jobs/``. The unified ``/api/jobs`` view merges JobRecords
with TrainerControlClient discoveries by PID lineage.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional

from ._atomic import atomic_write_text
from .paths import server_state_dir

LOCAL_NODE = platform.node()

RUNNING_STATUSES = {"starting", "running"}
TERMINAL_STATUSES = {"done", "failed", "aborted"}


def _records_file():
    return server_state_dir() / "job_records.json"


_lock = Lock()


@dataclass
class JobRecord:
    # Carried from the queue submission.
    queue_id: str
    project_dir: str = ""
    config: str = ""
    dynamic_args: Dict[str, Any] = field(default_factory=dict)
    requested_gpus: int = 1
    priority: int = 0
    submitted_at: float = 0.0
    # See ``QueueItem`` for the shape of these two fields — they are
    # mirrored through the dispatch path and persisted on the record so
    # the Jobs view can render eval jobs differently from training jobs.
    job_type: str = "training"
    job_params: Dict[str, Any] = field(default_factory=dict)

    # Where it ran.
    node: Optional[str] = None
    gpu_indices: List[int] = field(default_factory=list)
    pid: Optional[int] = None  # torchrun's PID

    # Lifecycle.
    status: str = "starting"
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None

    # True for records the scheduler did NOT spawn: a trainer launched outside
    # the server (e.g. a foreground ``forgather train`` with
    # TrainerControlCallback) that the scheduler promoted from its discovered
    # control endpoint. Such a record has no Popen handle and no
    # scheduler-reserved GPUs; it is reaped by PID liveness like a re-attached
    # job.
    externally_launched: bool = False

    # IO + correlation with TrainerControlClient endpoints.
    tty_log_path: Optional[str] = None
    job_id: Optional[str] = None  # set after PID-lineage match with an endpoint
    logs_dir: Optional[str] = None
    output_dir: Optional[str] = None

    # URL path prefix the server's reverse proxy uses to reach this job's
    # spawned HTTP service (currently only TensorBoard). Set at dispatch
    # time so the proxy can recover ``host``/``port`` from ``job_params``
    # and TB knows what prefix its own internal links should carry.
    path_prefix: Optional[str] = None

    # Inference-job bearer token. Stored on the record so the same-origin
    # proxy can re-add ``Authorization: Bearer <token>`` when forwarding
    # browser requests. None for non-inference jobs and for inference jobs
    # spawned with --no-auth.
    auth_token: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobRecord":
        allowed = {k: data.get(k) for k in cls.__dataclass_fields__.keys() if k in data}
        return cls(**allowed)


def _read_raw() -> List[JobRecord]:
    path = _records_file()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    out: List[JobRecord] = []
    for raw in data:
        if isinstance(raw, dict):
            try:
                out.append(JobRecord.from_dict(raw))
            except Exception:
                continue
    return out


def _write_raw(records: List[JobRecord]) -> None:
    atomic_write_text(
        _records_file(),
        json.dumps([r.to_dict() for r in records], indent=2),
        mode=0o600,
    )


def list_records() -> List[JobRecord]:
    with _lock:
        return _read_raw()


def add_record(record: JobRecord) -> JobRecord:
    with _lock:
        records = _read_raw()
        # If a record with the same queue_id already exists, replace it
        # (defensive: dispatch should never double-create).
        records = [r for r in records if r.queue_id != record.queue_id]
        records.append(record)
        _write_raw(records)
    return record


def get_record(queue_id: str) -> Optional[JobRecord]:
    with _lock:
        for r in _read_raw():
            if r.queue_id == queue_id:
                return r
    return None


def update_record(queue_id: str, **changes: Any) -> Optional[JobRecord]:
    with _lock:
        records = _read_raw()
        for i, r in enumerate(records):
            if r.queue_id == queue_id:
                for k, v in changes.items():
                    setattr(r, k, v)
                records[i] = r
                _write_raw(records)
                return r
    return None


def update_if_not_terminal(queue_id: str, **changes: Any) -> Optional[JobRecord]:
    """Atomic compare-and-swap: apply *changes* only if the record's
    on-disk status is not already terminal.

    Used by the scheduler reap path to avoid clobbering a concurrent
    abort. Returns the updated record, or ``None`` if the record was
    not found or was already in a terminal state (in which case the
    caller's update was a no-op).
    """
    with _lock:
        records = _read_raw()
        for i, r in enumerate(records):
            if r.queue_id == queue_id:
                if r.status in TERMINAL_STATUSES:
                    return None
                for k, v in changes.items():
                    setattr(r, k, v)
                records[i] = r
                _write_raw(records)
                return r
    return None


def remove_record(queue_id: str) -> bool:
    with _lock:
        records = _read_raw()
        kept = [r for r in records if r.queue_id != queue_id]
        if len(kept) == len(records):
            return False
        _write_raw(kept)
    return True
