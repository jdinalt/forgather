"""Persistent per-config overrides cache.

Each (project_dir, config_name) pair gets its own JSON file under
``~/.forgather/server/overrides/``. Files are self-describing (they include
the project_dir and config they belong to) so they can be inspected with
ordinary tools.

File layout::

    {
        "project_dir": "/abs/path/to/project",
        "config": "train_tiny_llama.yaml",
        "values": {"max_steps": 42, "lr": 1e-4},
        "updated_at": 1713500000.123
    }

Key derivation: SHA-256 of ``"{abspath(project_dir)}\\0{config_name}"``
truncated to 16 hex chars. Stable across restarts; does not depend on
template resolution.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from ._atomic import atomic_write_text
from .paths import overrides_dir

_lock = Lock()


def _key(project_dir: str, config: str) -> str:
    raw = f"{os.path.abspath(project_dir)}\0{config}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _path(project_dir: str, config: str) -> Path:
    return overrides_dir() / f"{_key(project_dir, config)}.json"


def _read(path: Path) -> Optional[Dict[str, Any]]:
    """Return parsed payload, or None on any error (missing / corrupt)."""
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, json.JSONDecodeError):
        return None


def get_overrides(project_dir: str, config: str) -> Dict[str, Any]:
    """Return the stored values dict, or ``{}`` if no cache file exists."""
    with _lock:
        data = _read(_path(project_dir, config))
    if data is None:
        return {}
    v = data.get("values")
    return dict(v) if isinstance(v, dict) else {}


def get_overrides_payload(project_dir: str, config: str) -> Dict[str, Any]:
    """Return the full stored payload (values + updated_at), or a null stub."""
    with _lock:
        data = _read(_path(project_dir, config))
    if data is None:
        return {"values": {}, "updated_at": None}
    return {
        "values": data.get("values") if isinstance(data.get("values"), dict) else {},
        "updated_at": data.get("updated_at"),
    }


def set_overrides(
    project_dir: str, config: str, values: Dict[str, Any]
) -> Dict[str, Any]:
    """Persist *values* and return the stored payload."""
    abs_dir = os.path.abspath(project_dir)
    now = time.time()
    payload: Dict[str, Any] = {
        "project_dir": abs_dir,
        "config": config,
        "values": dict(values),
        "updated_at": now,
    }
    p = _path(project_dir, config)
    with _lock:
        atomic_write_text(p, json.dumps(payload, indent=2))
    return {"values": dict(values), "updated_at": now}


def clear_overrides(project_dir: str, config: str) -> bool:
    """Remove the cache file. Returns True if a file was actually removed."""
    p = _path(project_dir, config)
    with _lock:
        try:
            p.unlink()
            return True
        except FileNotFoundError:
            return False
