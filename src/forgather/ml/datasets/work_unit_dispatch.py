"""DiLoCo work-unit dispatch — composable-level enablement.

Under the DiLoCo work-unit dispatch design
(``docs/design/diloco-work-unit-dispatch.md``), workers don't
hand-partition the train dataset. Each worker asks the DiLoCo server
for the next available work unit, drains that unit's row range from
the underlying storage backend, and asks for another.

This module hosts the entry point that wires a
``ComposableIterableDataset`` up to a DiLoCo server. The dispatch
itself lives **inside** the composable (``_wud_iter_window``); this
module is the seam between operator config (env vars set by the
scheduler) and the composable's ``enable_work_dispatch`` method.

The wrap is applied from ``preprocess_dataset`` after slice/shard are
settled, **not** at backend-construction time. That ordering matters:
the slice bounds the composable carries become part of the
``dataset_id`` hash, so two workers using different slices of the same
source dataset key separate queues.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

from .composable_iterable_dataset import ComposableIterableDataset

logger = logging.getLogger(__name__)


class DiLoCoWorkDispatchUnavailable(RuntimeError):
    """Raised when ``DILOCO_SERVER`` is set but work-dispatch can't be
    wired up.

    The wrap has several preconditions: ``DILOCO_WORKER_ID`` set, a
    composable carrying ``_load_args`` (i.e. built by
    ``fast_load_iterable_dataset``), and a reachable
    ``/datasets/register`` endpoint. When any of them fails on a
    DiLoCo-enabled run, the worker can't share the dataset with its
    peers and silently falling back to the bare composable would mean
    every worker iterates the full row stream on identical rows —
    broken-data-parallelism dressed up as a feature.

    Surface as a fatal startup error so the operator sees the
    misconfiguration in the TTY pane instead of training islands that
    look healthy from outside.
    """

    pass


def unit_range(unit_id: int, total_units: int, length: int) -> Tuple[int, int]:
    """Deterministic ``[start, end)`` row range for ``unit_id``.

    Every worker that knows ``(total_units, length)`` computes the same
    range for the same ``unit_id``, so the DiLoCo server only has to
    hand out integers — no per-unit row-range bookkeeping or shape
    negotiation beyond the bitmap.

    The composable applies the returned range relative to its current
    view bounds: ``backend.seek(view_start + unit_start)`` and yield
    ``unit_end - unit_start`` rows. So ``length`` here is the
    post-slice view length, not the underlying backend's full row
    count.
    """
    if total_units < 1:
        raise ValueError(f"total_units must be >= 1, got {total_units}")
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}")
    if not (0 <= unit_id < total_units):
        raise ValueError(f"unit_id {unit_id} out of range [0, {total_units})")
    start = (unit_id * length) // total_units
    end = ((unit_id + 1) * length) // total_units
    return start, end


def maybe_enable_work_dispatch(
    ds: ComposableIterableDataset,
) -> ComposableIterableDataset:
    """Enable DiLoCo work-unit dispatch on ``ds`` from env vars.

    Reads ``DILOCO_SERVER`` and ``DILOCO_WORKER_ID``; when both are set
    and ``ds`` carries the load identity needed to compute a
    ``dataset_id``, calls ``ds.enable_work_dispatch(client, worker_id)``
    and returns ``ds``. When ``DILOCO_SERVER`` is unset (the common
    case for non-DiLoCo runs), this is a no-op — ``ds`` is returned
    unchanged.

    Every failure path raises ``DiLoCoWorkDispatchUnavailable``:
    silently returning the bare composable on a DiLoCo-enabled run
    would put every worker on identical rows. See the
    ``feedback_no_silent_fallback`` memory.

    The wrap uses its own ``DiLoCoClient`` — it does NOT share state
    with the ``DiLoCoCallback`` that manages the parameter-sync
    worker. The two are decoupled; they share ``worker_id`` and the
    server address as common identity but never call into each other.
    """
    from forgather.ml.diloco import diloco_server_addr

    server_addr = diloco_server_addr()
    if not server_addr:
        # No DiLoCo server: vanilla single-node run, or eval/test load.
        # Nothing to dispatch.
        return ds

    worker_id = os.environ.get("DILOCO_WORKER_ID", "").strip()
    if not worker_id:
        raise DiLoCoWorkDispatchUnavailable(
            "DILOCO_SERVER is set but DILOCO_WORKER_ID is unset. The "
            "scheduler emits both whenever DiLoCo is enabled; this "
            "almost always means a CLI-spawned worker forgot "
            "--diloco-worker-id, or DILOCO_WORKER_ID was clobbered. "
            "Set it explicitly (any string unique per worker)."
        )

    if ds._load_args is None:
        raise DiLoCoWorkDispatchUnavailable(
            "DiLoCo work-dispatch needs a stable dataset identity to "
            "key the server-side queue, but this dataset wasn't built "
            "through fast_load_iterable_dataset and carries no "
            "load_args. Construct the dataset via the standard loader "
            "or disable DiLoCo for this run."
        )

    # Deferred import keeps the loader path light when DiLoCo isn't in
    # use.
    from forgather.ml.diloco.client import DiLoCoClient

    client = DiLoCoClient(server_addr)
    logger.info(
        "Enabling DiLoCo work-unit dispatch on %s (server=%s, worker_id=%s)",
        type(ds).__name__,
        server_addr,
        worker_id,
    )
    ds.enable_work_dispatch(client, worker_id)
    return ds
