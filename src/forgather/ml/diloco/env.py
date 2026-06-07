"""Environment-driven DiLoCo enablement check.

Single source of truth for "is DiLoCo active in this process?". The
worker process learns DiLoCo state via env vars set by the scheduler
(or the CLI's ``forgather diloco`` subcommand) — ``DILOCO_SERVER``
is the canonical signal.

The template-side equivalent is ``ns.diloco_is_enabled``, set once in
``lm_training_project.yaml``'s ``[globals]`` block. Both paths must
agree because they're driven by the same env var; if the canonical
signal ever changes (new env var, multi-server config, scheduler-set
flag, etc.) update both places — Python here and the
``ns.diloco_is_enabled`` set in lm_training_project.yaml.
"""

from __future__ import annotations

import os


def diloco_is_enabled() -> bool:
    """Return True if DiLoCo is active in the current process.

    Driven by ``DILOCO_SERVER`` (set by the scheduler / CLI). Empty /
    whitespace-only values count as not-set, matching the template-side
    truthiness check.
    """
    return bool(os.environ.get("DILOCO_SERVER", "").strip())


def diloco_server_addr() -> str:
    """Return the DiLoCo server address from the env, or ``""`` if unset.

    Stripped — matches ``diloco_is_enabled``'s notion of "set". Callers
    that need the address typically also need the bool check (e.g.
    ``if not addr: return``); this returns the empty string so
    ``not addr`` works as a single check rather than needing both
    ``diloco_is_enabled()`` and ``os.environ[...]``.
    """
    return os.environ.get("DILOCO_SERVER", "").strip()


def diloco_backend() -> str:
    """Return the sync-backend selector: ``"http"`` (default) or
    ``"shared_memory"``. Set via ``DILOCO_BACKEND``."""
    return os.environ.get("DILOCO_BACKEND", "http").strip().lower() or "http"


def diloco_shm_group_dir() -> str:
    """Shared-memory group directory (the per-host rendezvous), or ``""``."""
    return os.environ.get("DILOCO_SHM_GROUP_DIR", "").strip()


def diloco_shm_group_size() -> int:
    """Number of co-located workers in the shared-memory group, or ``0`` if
    unset. Raises ``ValueError`` if set to a non-integer."""
    raw = os.environ.get("DILOCO_SHM_GROUP_SIZE", "").strip()
    return int(raw) if raw else 0


def diloco_shm_init_checkpoint() -> str:
    """Optional override for the shared-memory init checkpoint dir, or ``""``.

    When unset, the aggregator seeds the region from the checkpoint the
    coordinator advertises in ``/info`` (``model_checkpoint_dir``)."""
    return os.environ.get("DILOCO_SHM_INIT_CHECKPOINT", "").strip()
