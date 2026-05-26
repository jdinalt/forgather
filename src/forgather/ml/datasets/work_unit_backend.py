"""``WorkUnitBackend`` — DiLoCo-coordinated iterable-dataset backend.

Under the DiLoCo work-unit dispatch design
(``docs/design/diloco-work-unit-dispatch.md``), workers no longer
hand-partition the train dataset with ``--num-shards / --shard-index``.
Each worker asks the DiLoCo server for the next available work unit,
streams that unit's row range from the underlying storage backend
(typically ``ResilientRemoteBackend`` against a dataset server), and
asks for another. Issuance is one-way: once the server hands out a
unit it's consumed from the queue regardless of worker fate, so within
an epoch no row is trained on twice.

This is a **storage-layer** wrapper, not a wrapper-layer one. It
implements :class:`IterableDatasetBackend` so the higher-level
``ComposableIterableDataset`` (map / filter / shard / shuffle buffer /
state-dict) sees a normal backend and operates on the rows that come
out of it. The wrap point is ``fast_load_iterable_dataset``: when
``DILOCO_SERVER`` is set in the env and the split is the training
split, the loader's backend construction is decorated with
:func:`maybe_wrap_for_work_dispatch` before the composable wrapper is
built. The dataset_id, length, and shuffle_seed are negotiated with
the DiLoCo server at that point.

Per-unit transient errors during iteration are swallowed and logged:
the unit is already consumed from the server's bitmap, so propagating
would crash the training loop with no chance of recovering the rows.
See the design doc's "consequences of no reset within a queue" section
for the trade-off rationale.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, Optional, Tuple

from .iterable_backend import IterableDatasetBackend

logger = logging.getLogger(__name__)


def unit_range(unit_id: int, total_units: int, length: int) -> Tuple[int, int]:
    """Deterministic ``[start, end)`` row range for ``unit_id``.

    Every worker that knows ``(total_units, length)`` computes the same
    range for the same ``unit_id``, so the DiLoCo server only has to
    hand out integers — no per-unit row-range bookkeeping or shape
    negotiation beyond the bitmap.
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


class WorkUnitBackend(IterableDatasetBackend):
    """Wraps another backend to inject DiLoCo work-unit dispatch.

    Implements ``IterableDatasetBackend`` so the higher-level
    ``ComposableIterableDataset`` (map / filter / shard / state_dict)
    sees a normal backend. Only ``__iter__`` differs in behavior — it
    drives row emission from the DiLoCo server's work-queue rather
    than sequential reads from the underlying backend.

    Parameters
    ----------
    wrapped : IterableDatasetBackend
        The underlying backend (typically a ``ResilientRemoteBackend``
        — phase 1 of the dispatch design requires a dataset_server-
        backed wrapped backend so the worker can seek to arbitrary
        positions cheaply).
    client : DiLoCoClient
        Connected DiLoCo client. ``request_work`` and ``complete_work``
        are called per unit; nothing else.
    worker_id : str
        This worker's identity (same value passed to ``/register``).
    dataset_id : str
        Stable hash from
        :func:`forgather.ml.datasets.dataset_id.compute_dataset_id`.
    shuffle_seed : int
        The seed this worker registered the queue with. The wrapped
        backend should have already been shuffled with this seed —
        WorkUnitBackend does not re-shuffle on its own.
    total_units : int
        K — number of units in the queue. Comes from the
        ``/datasets/register`` response.
    length : int
        Total row count of the wrapped backend; used to compute
        per-unit ``(start, end)`` ranges.
    """

    def __init__(
        self,
        wrapped: IterableDatasetBackend,
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
        self._wrapped = wrapped
        self._client = client
        self._worker_id = worker_id
        self._dataset_id = dataset_id
        self._shuffle_seed = int(shuffle_seed)
        self._total_units = int(total_units)
        self._length = int(length)
        # Running count of rows yielded across all dispatched units in
        # the current iteration. ``position()`` returns this so
        # ``ComposableIterableDataset._iter_window`` (which does
        # ``yielded_idx = backend.position() - 1; if yielded_idx < start: continue``)
        # advances through any view-slice filter correctly. Semantics
        # mismatch the dataset's actual row positions — the count is
        # "rows yielded by this iter", not "row index in the source
        # dataset" — but is consistent with how the composable uses
        # ``position()``.
        self._yielded = 0

    # ----- Backend interface --------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        """Drive iteration from the DiLoCo server's work queue.

        Loops: request_work → wrapped.seek(start) → iterate the wrapped
        backend yielding ``limit`` rows → complete_work → next request.
        Exits when the server returns ``{exhausted: true}``.

        Per-unit drain errors are swallowed (the unit is already
        consumed; propagating would crash training). request_work
        failures propagate (no unit was issued — the higher-level
        retry logic handles it).
        """
        # Reset the per-iter yielded counter. Each ``__iter__`` pass
        # advances ``position()`` from 0 monotonically — that's the
        # contract ``ComposableIterableDataset._iter_window`` relies
        # on for its view-slice filter.
        self._yielded = 0
        while True:
            try:
                resp = self._client.request_work(
                    self._worker_id, self._dataset_id, self._shuffle_seed
                )
            except Exception as exc:
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
            limit = end - start

            try:
                view = self._wrapped.seek(start)
                yielded = 0
                for row in view:
                    if yielded >= limit:
                        break
                    self._yielded += 1
                    yield row
                    yielded += 1
                logger.debug(
                    "Drained unit %d: yielded %d rows [%d:%d) for (%s, seed=%d) — total yielded so far: %d",
                    unit_id,
                    yielded,
                    start,
                    end,
                    self._dataset_id,
                    self._shuffle_seed,
                    self._yielded,
                )
            except Exception as exc:
                # Per-unit drain error — the unit is already consumed
                # from the queue's bitmap. Swallow + log + move on.
                logger.warning(
                    "Unit %d drain failed (partial loss for this epoch): %s",
                    unit_id,
                    exc,
                )

            # Diagnostic-only completion ack. Failure is non-fatal.
            try:
                self._client.complete_work(
                    self._worker_id,
                    self._dataset_id,
                    self._shuffle_seed,
                    unit_id,
                )
            except Exception as exc:
                logger.debug("complete_work for unit %d failed: %s", unit_id, exc)

    def __len__(self) -> int:
        # Full-dataset length — the trainer's step-count math is built
        # against this. An ``__iter__`` pass may yield fewer rows
        # (queue can exhaust mid-epoch), but advertising a smaller
        # length would underestimate steps-per-epoch.
        return self._length

    def shuffle(self, seed: Optional[int] = None) -> "WorkUnitBackend":
        """Return a new WorkUnitBackend wrapping a re-shuffled wrapped backend.

        The shuffle_seed for the queue is fixed at construction time —
        that's what's registered with the DiLoCo server. The composable
        wrapper might also call shuffle to apply its own seed; we
        propagate to the wrapped backend so any row-level shuffle
        happens, but the queue keying doesn't change (workers must
        agree on the seed they used at register time).

        Practically: under DiLoCo work-unit dispatch the inner shuffle
        is already done at backend-construction time, before this
        wrap is applied. The composable rarely calls shuffle again at
        runtime.
        """
        return WorkUnitBackend(
            wrapped=self._wrapped.shuffle(seed),
            client=self._client,
            worker_id=self._worker_id,
            dataset_id=self._dataset_id,
            shuffle_seed=self._shuffle_seed,
            total_units=self._total_units,
            length=self._length,
        )

    def seek(self, position: int) -> "WorkUnitBackend":
        """``seek`` is a no-op for DiLoCo work-unit dispatch.

        Iteration positions are server-determined (the dispatch loop
        seeks the wrapped backend to per-unit start offsets). A
        composable-level seek (e.g. for checkpoint resume) doesn't
        apply: resume just asks the server for the next available
        unit, see the design doc's "Worker checkpoint resume doesn't
        replay rows" note.

        Returns self so the contract that ``seek`` returns a backend
        instance is satisfied. Any value the caller passes is ignored.
        """
        if position != 0:
            logger.debug(
                "WorkUnitBackend.seek(%d) ignored — positions are server-driven",
                position,
            )
        return self

    def position(self) -> int:
        """Running count of rows yielded by the current ``__iter__`` pass.

        The flat dataset-position notion doesn't apply under work-unit
        dispatch (rows come from server-dispatched slices, not sequential
        positions). But ``ComposableIterableDataset._iter_window`` uses
        ``backend.position()`` to compute ``yielded_idx = position - 1``
        for its view-slice filter: returning a row counter that
        monotonically increases with each yield keeps that loop
        consistent. Returning 0 (the previous stub) made every yielded
        row look pre-slice and the composable silently discarded them
        all — the dataset_server got hit but no rows reached the
        trainer, GPU stayed cold.

        Semantic note: the counter is "rows yielded by this iter", not
        "row index in the source dataset". Under work-dispatch a
        view-slice's ``start`` becomes "discard the first N dispatched
        rows" rather than "skip dataset rows [0, N)". For the manual
        ``train[10000:]``-style slicing used by the standard project
        templates this is a minor semantic difference (the first
        10000 dispatched rows are dropped during warmup), and the
        operator's intent of "leave the first N for eval" still holds
        because the eval load uses a different ``dataset_id`` /
        queue. Full slice-aware dispatch is a follow-up; see the
        design doc.
        """
        return self._yielded

    # ----- Optional metadata --------------------------------------------

    @property
    def column_names(self):
        # Pass through to the wrapped backend so the composable can
        # answer column-aware APIs.
        return getattr(self._wrapped, "column_names", None)

    @property
    def n_shards(self) -> int:
        return getattr(self._wrapped, "n_shards", 1)


# ---------------------------------------------------------------------------
# Self-configuring wrap helper
# ---------------------------------------------------------------------------


def maybe_wrap_for_work_dispatch(
    backend: IterableDatasetBackend,
    *,
    path: str,
    name: Optional[str] = None,
    split: Optional[str] = None,
    data_files=None,
    revision: Optional[str] = None,
    shuffle_seed: int = 0,
) -> IterableDatasetBackend:
    """Optionally wrap ``backend`` with ``WorkUnitBackend`` based on env vars.

    Called from inside ``fast_load_iterable_dataset`` after the inner
    backend (``ResilientRemoteBackend`` typically) has been built and
    before the ``ComposableIterableDataset`` is constructed around it.
    Reads inputs from env vars to avoid polluting the loader's
    signature with DiLoCo-specific knobs:

    - ``DILOCO_SERVER`` (host:port required) — DiLoCo server addr.
      Presence of this var is the gate; when unset the wrap is a no-op.
    - ``DILOCO_WORKER_ID`` (required) — worker identity. Set by the
      scheduler when DiLoCo is enabled, with a queue_id fallback.

    Returns ``backend`` unchanged when no DiLoCo server is configured —
    non-DiLoCo runs see no behavior change. Errors during
    ``/datasets/register`` are logged at ERROR and the backend is
    returned unchanged, so a server hiccup doesn't crash training (just
    disables work-dispatch for the run).

    The wrap uses its **own** ``DiLoCoClient`` — it does NOT share
    state with the ``DiLoCoCallback`` that manages the parameter-sync
    worker. The two are decoupled; they share ``worker_id`` and the
    server address as common identity but never call into each other.

    Parameters
    ----------
    backend
        The freshly-constructed backend to potentially wrap.
    path, name, split, data_files, revision
        The same dataset-identity tuple passed to
        ``fast_load_iterable_dataset``. Used to compute the canonical
        ``dataset_id``.
    shuffle_seed
        The seed the queue is keyed by. Phase 1 default is 0;
        multi-epoch rotation is a follow-up.
    """
    server_addr = os.environ.get("DILOCO_SERVER", "").strip()
    if not server_addr:
        # No DiLoCo server in this process: vanilla single-node run, or
        # eval/test load. Nothing to dispatch.
        return backend
    worker_id = os.environ.get("DILOCO_WORKER_ID", "").strip()
    if not worker_id:
        logger.error(
            "DILOCO_SERVER is set but DILOCO_WORKER_ID is unset — "
            "skipping work-unit dispatch wrap. The scheduler should "
            "have emitted both; check the DiLoCo callback's startup "
            "diagnostics."
        )
        return backend

    try:
        length = len(backend)
    except TypeError:
        logger.error(
            "DiLoCo work-dispatch enabled but backend has no __len__; "
            "work-unit dispatch needs a fixed dataset length. "
            "Skipping wrap."
        )
        return backend

    # Deferred imports keep the loader path light when DiLoCo isn't
    # in use.
    from forgather.ml.datasets.dataset_id import compute_dataset_id
    from forgather.ml.diloco.client import DiLoCoClient

    try:
        dataset_id = compute_dataset_id(
            path=path,
            name=name,
            split=split,
            data_files=data_files,
            revision=revision,
        )
    except ValueError as exc:
        logger.error(
            "Could not compute dataset_id from load args: %s. "
            "Skipping work-unit dispatch wrap.",
            exc,
        )
        return backend

    # Transient client — not shared with the DiLoCoCallback's worker.
    client = DiLoCoClient(server_addr)
    try:
        reply = client.register_dataset(
            worker_id=worker_id,
            dataset_id=dataset_id,
            shuffle_seed=int(shuffle_seed),
            hint={"length": length},
        )
    except Exception as exc:
        logger.error(
            "/datasets/register failed for dataset_id=%s: %s. "
            "Skipping work-unit dispatch wrap.",
            dataset_id,
            exc,
        )
        return backend

    total_units = int(reply["total_units"])
    logger.info(
        "Wrapping backend with WorkUnitBackend "
        "(dataset_id=%s, shuffle_seed=%d, K=%d, length=%d, worker_id=%s)",
        dataset_id,
        shuffle_seed,
        total_units,
        length,
        worker_id,
    )
    return WorkUnitBackend(
        wrapped=backend,
        client=client,
        worker_id=worker_id,
        dataset_id=dataset_id,
        shuffle_seed=shuffle_seed,
        total_units=total_units,
        length=length,
    )
