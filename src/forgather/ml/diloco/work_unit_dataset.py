"""``WorkUnitDataset`` — DiLoCo-coordinated streaming dataset wrapper.

Under the work-unit dispatch design
(``docs/design/diloco-work-unit-dispatch.md``), DiLoCo workers no
longer hand-partition the train dataset with ``--num-shards N
--shard-index I``. Instead each worker asks the DiLoCo server for the
next available work unit, streams the row range for that unit from
the dataset server, and asks for another. Issuance is one-way: once
the server hands out a unit it's consumed from the queue regardless
of worker fate, so within an epoch no row is ever trained on twice.

This wrapper is the worker-side glue. It composes with
``ComposableIterableDataset`` from
``forgather.ml.datasets.composable_iterable_dataset``: under the
hood we call ``base.shuffle(seed)`` once at the start of ``__iter__``
(same seed every worker uses → same shuffle order across the fleet)
and then ``shuffled.slice(start, end)`` for each unit.

Per-unit transient errors are deliberately swallowed: the unit is
already consumed from the server's bitmap, so propagating the
exception would just crash the training loop with no chance of
recovering the rows. We log and advance to the next unit instead. The
design doc's "consequences of no reset within a queue" section
covers the trade-off explicitly.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


def unit_range(unit_id: int, total_units: int, length: int) -> tuple[int, int]:
    """Deterministic row range for a unit.

    ``start = (unit_id * length) // total_units``,
    ``end = ((unit_id + 1) * length) // total_units``.

    Every worker that knows ``(total_units, length)`` computes the
    same range for the same ``unit_id``, so the DiLoCo server only
    has to hand out integers — no shape negotiation, no per-unit
    bookkeeping beyond the bitmap.
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


class WorkUnitDataset(IterableDataset):
    """Iterable wrapper that yields rows for server-dispatched work units.

    Parameters
    ----------
    base : ComposableIterableDataset (or duck-typed equivalent)
        Underlying dataset. Must support ``.shuffle(seed)`` returning
        a new wrapper and ``.slice(start, end)`` on that wrapper.
        ``__len__()`` is consulted once at construction; the value
        must match what was reported as ``hint.length`` to the DiLoCo
        server, otherwise the per-unit ranges will be misaligned.
    client : DiLoCoClient
        Connected DiLoCo client. ``request_work`` /
        ``complete_work`` are called per unit; nothing else.
    worker_id : str
        This worker's identity (same value used on ``/register``).
    dataset_id : str
        Stable hash from
        :func:`forgather.ml.datasets.dataset_id.compute_dataset_id`.
    shuffle_seed : int
        The seed this worker passed to ``/datasets/register``. Same
        seed across the fleet → same shuffle order → unit ranges
        line up.
    total_units : int
        K — number of work units in the queue. Comes from the
        ``/datasets/register`` response. Must match the server's
        value for this ``(dataset_id, shuffle_seed)`` queue.
    length : int
        Row count of the underlying dataset. Must equal what was
        reported in ``hint.length``.
    """

    def __init__(
        self,
        *,
        base,
        client,
        worker_id: str,
        dataset_id: str,
        shuffle_seed: int,
        total_units: int,
        length: int,
    ):
        if total_units < 1:
            raise ValueError(f"total_units must be >= 1, got {total_units}")
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        self._base = base
        self._client = client
        self._worker_id = worker_id
        self._dataset_id = dataset_id
        self._shuffle_seed = int(shuffle_seed)
        self._total_units = int(total_units)
        self._length = int(length)
        # Shuffled view, cached so we don't re-build the wrapper on
        # every unit. ``.slice(start, end)`` is cheap; ``.shuffle()``
        # returns a new wrapper and re-snapshots backend state.
        self._shuffled = base.shuffle(self._shuffle_seed)

    def __len__(self) -> int:
        # Full-dataset length — the trainer's step-count math is built
        # against this. An ``__iter__`` pass may yield fewer rows
        # (queue can exhaust mid-epoch), but advertising a smaller
        # length would underestimate steps-per-epoch.
        return self._length

    def __iter__(self) -> Iterator[dict]:
        while True:
            try:
                resp = self._client.request_work(
                    self._worker_id, self._dataset_id, self._shuffle_seed
                )
            except Exception as exc:
                # Connection blip mid-iteration → no unit was issued.
                # Surface the error so the training loop can decide
                # (the higher-level retry logic in DiLoCoCallback /
                # DiLoCoWorker handles transient outages).
                logger.error(
                    "DiLoCo work request failed: %s — re-raising to training loop",
                    exc,
                )
                raise

            if resp.get("exhausted"):
                logger.info(
                    "Work queue exhausted for (%s, seed=%d) — ending iteration",
                    self._dataset_id,
                    self._shuffle_seed,
                )
                return

            unit_id = int(resp["unit_id"])
            start, end = unit_range(unit_id, self._total_units, self._length)
            view = self._shuffled.slice(start, end)

            try:
                count = 0
                for row in view:
                    yield row
                    count += 1
                logger.debug(
                    "Drained unit %d: yielded %d rows [%d:%d) for (%s, seed=%d)",
                    unit_id,
                    count,
                    start,
                    end,
                    self._dataset_id,
                    self._shuffle_seed,
                )
            except Exception as exc:
                # Per-unit drain error — the unit is already consumed
                # from the queue's bitmap. Swallow + log + move on to
                # the next unit; propagating would crash the training
                # loop and the rows are unrecoverable for this epoch
                # anyway. See "consequences of no reset within a
                # queue" in the design doc.
                logger.warning(
                    "Unit %d drain failed (partial loss for this epoch): %s",
                    unit_id,
                    exc,
                )

            # Diagnostic-only completion ack. Failure is non-fatal —
            # the server's completed bitmap just stays out of sync
            # for this unit, which only affects the diagnostic UI.
            try:
                self._client.complete_work(
                    self._worker_id,
                    self._dataset_id,
                    self._shuffle_seed,
                    unit_id,
                )
            except Exception as exc:
                logger.debug("complete_work for unit %d failed: %s", unit_id, exc)

    # Trainer checkpoint-resume integration: WorkUnitDataset has no
    # per-iteration position to checkpoint. The queue lives on the
    # DiLoCo server (and is itself checkpointed there). Resuming
    # training just asks for the next available unit; any rows the
    # worker was draining mid-checkpoint are silently lost for this
    # epoch — see design doc, "Worker checkpoint resume doesn't
    # replay rows" (actually a feature under DiLoCo: replaying old
    # rows after the global params have advanced would only add
    # gradient noise).

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        # Stub; nothing to restore. Don't error on a non-empty state
        # — older checkpoints written by ``ComposableIterableDataset``
        # may carry position / count data we don't need now.
        return None
