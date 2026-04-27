"""Persistent dispatch queue.

State lives at ``~/.forgather/server/queue.json`` as a list of QueueItem
dicts. Items here are *waiting* to dispatch — once the scheduler picks one
up it's moved into :mod:`job_records` (and removed from this file). That
keeps "queue" honest as the waiting list.
"""

from __future__ import annotations

import json
import platform
import time
import uuid
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional

from ._atomic import atomic_write_text
from .paths import queue_file

LOCAL_NODE = platform.node()

_lock = Lock()


@dataclass
class QueueItem:
    queue_id: str
    project_dir: str
    config: str
    dynamic_args: Dict[str, Any] = field(default_factory=dict)
    requested_gpus: int = 1
    priority: int = 0
    submitted_at: float = 0.0
    # Kind of subprocess-style Forgather job this entry runs. ``"training"``
    # (the historical default) spawns ``scripts/train_script.py`` and
    # correlates with TrainerControlClient. ``"eval"`` spawns
    # ``scripts/eval_script.py`` and is fire-and-forget (no endpoint.json,
    # no save / stop control). New fields are appended to stay
    # backwards-compatible with queue.json files written before this field
    # existed — ``from_dict`` already filters unknown keys.
    job_type: str = "training"
    # Type-specific payload the scheduler reads at dispatch. For
    # ``job_type == "eval"`` this holds eval_project / eval_template /
    # model_path / checkpoint_path / trainer / batch_size / max_length /
    # max_steps / dtype / attn_implementation / compile / output_dir.
    # Empty for training jobs (they use ``project_dir`` + ``config`` +
    # ``dynamic_args`` directly).
    job_params: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new(
        project_dir: str,
        config: str,
        dynamic_args: Dict[str, Any],
        requested_gpus: int,
        priority: int,
        job_type: str = "training",
        job_params: Optional[Dict[str, Any]] = None,
    ) -> "QueueItem":
        return QueueItem(
            queue_id=f"q_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}",
            project_dir=project_dir,
            config=config,
            dynamic_args=dict(dynamic_args),
            requested_gpus=max(0, int(requested_gpus)),
            priority=int(priority),
            submitted_at=time.time(),
            job_type=job_type,
            job_params=dict(job_params or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueueItem":
        allowed = {k: data.get(k) for k in cls.__dataclass_fields__.keys() if k in data}
        return cls(**allowed)


def _read_raw() -> List[QueueItem]:
    path = queue_file()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    items: List[QueueItem] = []
    for raw in data:
        if isinstance(raw, dict):
            try:
                items.append(QueueItem.from_dict(raw))
            except Exception:
                continue
    return items


def _write_raw(items: List[QueueItem]) -> None:
    atomic_write_text(
        queue_file(), json.dumps([it.to_dict() for it in items], indent=2)
    )


def list_items() -> List[QueueItem]:
    with _lock:
        return _read_raw()


def add_item(item: QueueItem) -> QueueItem:
    with _lock:
        items = _read_raw()
        items.append(item)
        _write_raw(items)
    return item


def get_item(queue_id: str) -> Optional[QueueItem]:
    with _lock:
        for it in _read_raw():
            if it.queue_id == queue_id:
                return it
    return None


def remove_item(queue_id: str) -> bool:
    with _lock:
        items = _read_raw()
        kept = [it for it in items if it.queue_id != queue_id]
        if len(kept) == len(items):
            return False
        _write_raw(kept)
    return True
