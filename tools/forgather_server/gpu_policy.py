"""Persistent runtime scheduling policy for individual GPUs.

Stored at ``~/.config/forgather/server/gpu_policy.json``.  The file is a JSON
object with a single ``"gpus"`` key whose value is a dict keyed by GPU
index (as a string) containing ``disabled`` and ``min_priority`` fields.

Example on-disk format::

    {
      "gpus": {
        "0": { "disabled": false, "min_priority": 0 },
        "2": { "disabled": true,  "min_priority": 0 },
        "5": { "disabled": false, "min_priority": 10 }
      }
    }

All writes are atomic (write temp + os.replace).  A module-level Lock
serialises concurrent mutation.  A corrupt or missing file silently
yields default policies so the server never hard-fails on startup.

Public API
----------
get_policy(index)               -> GpuPolicy
set_policy(index, *, ...)       -> GpuPolicy   (partial update)
clear_policy(index)             -> bool
all_policies()                  -> dict[int, GpuPolicy]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from threading import Lock
from typing import Optional

from ._atomic import atomic_write_text
from .paths import gpu_policy_file

log = logging.getLogger("forgather_server.gpu_policy")

_lock = Lock()


@dataclass
class GpuPolicy:
    disabled: bool = False
    min_priority: int = 0  # inclusive lower bound; 0 means no restriction


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_raw() -> dict:
    """Return the parsed ``gpus`` sub-dict from disk, or {} on any failure."""
    path = gpu_policy_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    gpus = data.get("gpus", {})
    if not isinstance(gpus, dict):
        return {}
    return gpus


def _write_raw(gpus: dict) -> None:
    atomic_write_text(
        gpu_policy_file(), json.dumps({"gpus": gpus}, indent=2), mode=0o600
    )


def _entry_to_policy(entry: object) -> GpuPolicy:
    """Convert a raw dict entry to a GpuPolicy, filling in defaults."""
    if not isinstance(entry, dict):
        return GpuPolicy()
    return GpuPolicy(
        disabled=bool(entry.get("disabled", False)),
        min_priority=int(entry.get("min_priority", 0)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_policy(index: int) -> GpuPolicy:
    """Return the policy for *index*, defaulting to no restrictions."""
    with _lock:
        gpus = _read_raw()
    entry = gpus.get(str(index))
    if entry is None:
        return GpuPolicy()
    return _entry_to_policy(entry)


def set_policy(
    index: int,
    *,
    disabled: Optional[bool] = None,
    min_priority: Optional[int] = None,
) -> GpuPolicy:
    """Update the policy for *index*.

    Only the fields explicitly passed (not ``None``) are modified; the rest
    are left at their current values (defaulting to ``GpuPolicy`` defaults
    if this GPU has no entry yet).  Returns the full resulting policy.
    """
    with _lock:
        gpus = _read_raw()
        key = str(index)
        current = _entry_to_policy(gpus.get(key))
        new_disabled = disabled if disabled is not None else current.disabled
        new_min_priority = (
            min_priority if min_priority is not None else current.min_priority
        )
        gpus[key] = {"disabled": new_disabled, "min_priority": new_min_priority}
        _write_raw(gpus)
    return GpuPolicy(disabled=new_disabled, min_priority=new_min_priority)


def clear_policy(index: int) -> bool:
    """Remove the policy entry for *index*.

    Returns ``True`` if an entry was present and removed, ``False`` if there
    was nothing to remove.
    """
    with _lock:
        gpus = _read_raw()
        key = str(index)
        if key not in gpus:
            return False
        del gpus[key]
        _write_raw(gpus)
    return True


def all_policies() -> dict[int, GpuPolicy]:
    """Return a dict mapping GPU index (int) -> GpuPolicy for all stored entries."""
    with _lock:
        gpus = _read_raw()
    return {int(k): _entry_to_policy(v) for k, v in gpus.items()}
