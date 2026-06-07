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
    """Return the sync-backend selector: ``"http"`` (default),
    ``"shared_memory"``, or ``"collective"``. Set via ``DILOCO_BACKEND``."""
    return os.environ.get("DILOCO_BACKEND", "http").strip().lower() or "http"


def diloco_replicate() -> int:
    """The collective backend's replicate degree (number of replicas in the
    ``diloco`` mesh axis), or 1 if unset. Set via ``DILOCO_REPLICATE``; the
    launch sizes one torchrun world as ``replicate × inner``. Raises
    ``ValueError`` if set to a non-integer."""
    raw = os.environ.get("DILOCO_REPLICATE", "").strip()
    return int(raw) if raw else 1


def diloco_inner_axis() -> str:
    """The inner parallelism axis composed with the ``diloco`` replicate axis:
    ``"data_parallel"`` (default) or ``"pipeline_parallel"``. Set via
    ``DILOCO_INNER_AXIS`` for a pipeline run (mesh =
    ``(diloco, pipeline_parallel)``); the trainer parallelizes over the inner
    sub-mesh while the collective runs over the diloco axis."""
    raw = os.environ.get("DILOCO_INNER_AXIS", "").strip().lower()
    return raw or "data_parallel"


def diloco_apply_collective_worker_id() -> None:
    """In the collective regime (``DILOCO_REPLICATE`` > 1), rewrite
    ``DILOCO_WORKER_ID`` to a per-replica-distinct ``{base}_r{diloco_rank}``.

    The N collective replicas run as one torchrun world and share the env, so a
    single ``DILOCO_WORKER_ID`` would put every replica on the same output dir /
    run logs / checkpoints and the same dataset shard. Making it distinct here —
    once, at the torchrun entrypoint, BEFORE config preprocessing reads it for
    the output-dir derivation — is the single source of per-replica identity:
    the config, the ``DiLoCoCallback``, and the work-unit dispatch all then read
    the same distinct id. Idempotent; a no-op when the degree is 1 or the base
    is unset (the downstream 'unset' guidance still fires)."""
    replicate = diloco_replicate()
    if replicate <= 1:
        return
    base = os.environ.get("DILOCO_WORKER_ID", "").strip()
    if not base:
        return
    world = int(os.environ.get("WORLD_SIZE", "1") or "1")
    rank = int(os.environ.get("RANK", "0") or "0")
    inner = max(1, world // replicate)
    suffix = f"_r{rank // inner}"
    if not base.endswith(suffix):
        os.environ["DILOCO_WORKER_ID"] = f"{base}{suffix}"


def diloco_init_checkpoint() -> str:
    """Optional override for a non-HTTP backend's init checkpoint dir, or ``""``.

    Used by the collective backend (the worker-process replicas seed from this
    dir on rank 0). When unset, the backend seeds from the checkpoint the
    coordinator advertises in ``/info`` (``model_checkpoint_dir``). The
    shared-memory backend has its own ``DILOCO_SHM_INIT_CHECKPOINT`` for the same
    role."""
    return os.environ.get("DILOCO_INIT_CHECKPOINT", "").strip()


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


def diloco_report_sync_state() -> bool:
    """Whether to report per-worker sync-state on the heartbeat (default True).

    Set ``DILOCO_REPORT_SYNC_STATE`` to a falsy value (0/false/no/off) to omit it
    — a small payload trim if the coordinator's diagnostics aren't needed."""
    raw = os.environ.get("DILOCO_REPORT_SYNC_STATE", "").strip().lower()
    return raw not in ("0", "false", "no", "off")
