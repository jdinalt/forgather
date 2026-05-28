"""Tests for composable-level DiLoCo work-unit dispatch.

Covers ``ComposableIterableDataset.enable_work_dispatch`` and the
``maybe_enable_work_dispatch`` env-driven helper in
``forgather.ml.datasets.work_unit_dispatch``.

The dispatch lives inside the composable's ``_iter_window`` (replaces
the sequential walk with a server-driven work-queue loop) so these
tests target the composable directly, not a backend wrapper.
"""

from __future__ import annotations

import logging
import os
from typing import List
from unittest.mock import patch

import pytest

from forgather.ml.datasets.composable_iterable_dataset import (
    ComposableIterableDataset,
)
from forgather.ml.datasets.iterable_backend import IterableDatasetBackend
from forgather.ml.datasets.work_unit_dispatch import (
    DiLoCoWorkDispatchUnavailable,
    maybe_enable_work_dispatch,
    unit_range,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeBackend(IterableDatasetBackend):
    """Minimal in-memory backend; ``seek`` returns a fresh instance."""

    def __init__(self, n: int = 100, _start: int = 0):
        self._n = n
        self._pos = _start
        # Track every seek() target the dispatch loop asked for, so
        # tests can assert "yes, we actually seeked to that unit".
        self.seek_log: List[int] = []

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        while self._pos < self._n:
            i = self._pos
            self._pos += 1
            yield {"i": i}

    def shuffle(self, seed=None):
        return self

    def seek(self, position: int):
        new = FakeBackend(self._n, _start=position)
        new.seek_log = self.seek_log
        new.seek_log.append(position)
        return new

    def position(self) -> int:
        return self._pos


class FakeClient:
    def __init__(self, K: int = 4):
        self.K = K
        self.next_unit = 0
        self.calls: List[tuple] = []
        self.complete_log: List[int] = []
        # Optional injections for failure tests.
        self.register_exc: Exception | None = None
        self.request_exc: Exception | None = None

    def register_dataset(self, *, worker_id, dataset_id, shuffle_seed, hint):
        if self.register_exc is not None:
            raise self.register_exc
        self.calls.append(("register", worker_id, dataset_id, shuffle_seed, hint))
        return {"total_units": self.K}

    def request_work(self, worker_id, dataset_id, shuffle_seed):
        if self.request_exc is not None:
            raise self.request_exc
        self.calls.append(("request", worker_id, dataset_id, shuffle_seed))
        if self.next_unit >= self.K:
            return {"exhausted": True}
        u = self.next_unit
        self.next_unit += 1
        return {"unit_id": u}

    def complete_work(self, worker_id, dataset_id, shuffle_seed, unit_id):
        self.calls.append(("complete", worker_id, dataset_id, shuffle_seed, unit_id))
        self.complete_log.append(unit_id)
        return {"ack": True}


def _load_args(path="dataset/test", split="train"):
    return {
        "path": path,
        "name": None,
        "split": split,
        "data_files": None,
        "revision": None,
    }


# ---------------------------------------------------------------------------
# unit_range — deterministic geometry helper
# ---------------------------------------------------------------------------


class TestUnitRange:
    def test_evenly_divides(self):
        assert unit_range(0, 4, 100) == (0, 25)
        assert unit_range(3, 4, 100) == (75, 100)

    def test_remainder_distributed_via_floor_division(self):
        # length=10, K=4 → 2,2,2,4 in absolute terms via the (i*L)//K
        # / ((i+1)*L)//K formula.
        ranges = [unit_range(i, 4, 10) for i in range(4)]
        # No gaps, no overlaps, covers [0, 10).
        for (a, b), (c, d) in zip(ranges, ranges[1:]):
            assert b == c
        assert ranges[0][0] == 0 and ranges[-1][1] == 10

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            unit_range(-1, 4, 100)
        with pytest.raises(ValueError):
            unit_range(4, 4, 100)

    def test_total_units_must_be_positive(self):
        with pytest.raises(ValueError):
            unit_range(0, 0, 100)


# ---------------------------------------------------------------------------
# enable_work_dispatch preconditions
# ---------------------------------------------------------------------------


class TestEnablePreconditions:
    def test_requires_load_args(self):
        ds = ComposableIterableDataset(FakeBackend())
        with pytest.raises(RuntimeError, match="load_args"):
            ds.enable_work_dispatch(client=FakeClient(), worker_id="w0")

    def test_refuses_after_shard(self):
        ds = ComposableIterableDataset(FakeBackend(), load_args=_load_args())
        ds = ds.shard(num_shards=2, index=0)
        with pytest.raises(RuntimeError, match="sharded dataset"):
            ds.enable_work_dispatch(client=FakeClient(), worker_id="w0")

    def test_shard_refuses_after_enable(self):
        ds = ComposableIterableDataset(FakeBackend(), load_args=_load_args())
        ds.enable_work_dispatch(client=FakeClient(), worker_id="w0")
        with pytest.raises(RuntimeError, match="DiLoCo work-unit dispatch enabled"):
            ds.shard(num_shards=2, index=0)


# ---------------------------------------------------------------------------
# Dispatch loop behavior
# ---------------------------------------------------------------------------


class TestDispatchIteration:
    def test_yields_all_rows_via_K_units(self):
        client = FakeClient(K=4)
        ds = ComposableIterableDataset(FakeBackend(100), load_args=_load_args())
        ds.enable_work_dispatch(client, "w0")
        rows = [r["i"] for r in ds]
        assert rows == list(range(100))
        # K register calls (1) + K+1 request calls (one extra for exhausted)
        # + K complete calls.
        register_calls = [c for c in client.calls if c[0] == "register"]
        request_calls = [c for c in client.calls if c[0] == "request"]
        complete_calls = [c for c in client.calls if c[0] == "complete"]
        assert len(register_calls) == 1
        assert len(request_calls) == 5  # 4 units + 1 exhausted signal
        assert len(complete_calls) == 4

    def test_exhaustion_ends_iteration(self):
        # K=2, but the backend has 4 units' worth of data — we only
        # see the rows for units 0 and 1, then the loop exits cleanly.
        client = FakeClient(K=2)
        ds = ComposableIterableDataset(FakeBackend(100), load_args=_load_args())
        ds.enable_work_dispatch(client, "w0")
        rows = [r["i"] for r in ds]
        # K=2 means u0=[0,50), u1=[50,100). All 100 rows yielded.
        assert rows == list(range(100))

    def test_slice_composition_train_10000(self):
        """Regression test for the slice-math bug the refactor fixes.

        Old backend-layer wrap dispatched over the full backend (rows
        0..N-1) and the composable's slice filter then *discarded* the
        first 10000 yielded rows via a yield-counter trick. New
        composable-level dispatch operates on the post-slice view
        ``[10000, N)`` directly — workers never seek to or yield rows
        in ``[0, 10000)``.
        """
        client = FakeClient(K=4)
        ds = ComposableIterableDataset(FakeBackend(100), load_args=_load_args())
        ds = ds.slice(40, None)  # window [40, 100), L=60
        ds.enable_work_dispatch(client, "w0")
        rows = [r["i"] for r in ds]
        assert len(rows) == 60
        assert min(rows) == 40 and max(rows) == 99
        # The backend should never have been seeked to a position in
        # [0, 40) — that's the kludge the old design relied on.
        assert all(s >= 40 for s in ds._backend.seek_log)

    def test_percentage_slice(self):
        client = FakeClient(K=4)
        ds = ComposableIterableDataset(FakeBackend(100), load_args=_load_args())
        ds = ds.slice("25%", "75%")  # bounds resolve to [25, 75)
        ds.enable_work_dispatch(client, "w0")
        rows = [r["i"] for r in ds]
        assert len(rows) == 50
        assert min(rows) == 25 and max(rows) == 74

    def test_per_unit_drain_error_swallowed(self, caplog):
        """A drain failure mid-unit must not crash training — the unit
        is already consumed from the server's bitmap, so propagating
        would crash the run with no chance to recover.

        Instrument a backend that raises on seek(50) — the third unit
        for K=4 over 100 rows. The dispatch loop should swallow + log
        and move on to unit 3.
        """

        class FlakyBackend(FakeBackend):
            def seek(self, position: int):
                if position == 50:
                    raise RuntimeError("simulated drain failure")
                # Return a FlakyBackend (not FakeBackend) so subsequent
                # seeks still hit our override.
                new = FlakyBackend(self._n, _start=position)
                new.seek_log = self.seek_log
                new.seek_log.append(position)
                return new

        client = FakeClient(K=4)
        ds = ComposableIterableDataset(FlakyBackend(100), load_args=_load_args())
        ds.enable_work_dispatch(client, "w0")
        with caplog.at_level(
            logging.WARNING,
            logger="forgather.ml.datasets.composable_iterable_dataset",
        ):
            rows = [r["i"] for r in ds]
        # Other 3 units yielded their 25 rows each = 75 total.
        assert len(rows) == 75
        assert any("Unit 2 drain failed" in r.message for r in caplog.records)

    def test_register_failure_propagates(self):
        client = FakeClient(K=4)
        client.register_exc = RuntimeError("server unreachable")
        ds = ComposableIterableDataset(FakeBackend(100), load_args=_load_args())
        ds.enable_work_dispatch(client, "w0")
        with pytest.raises(RuntimeError, match="server unreachable"):
            list(ds)

    def test_request_failure_propagates(self):
        client = FakeClient(K=4)
        client.request_exc = ConnectionError("network blip")
        ds = ComposableIterableDataset(FakeBackend(100), load_args=_load_args())
        ds.enable_work_dispatch(client, "w0")
        with pytest.raises(ConnectionError, match="network blip"):
            list(ds)


# ---------------------------------------------------------------------------
# dataset_id keying and set_epoch lazy re-register
# ---------------------------------------------------------------------------


class TestDatasetIdKeying:
    def test_distinct_slices_get_distinct_dataset_ids(self):
        """Two slices of the same source dataset must produce distinct
        dataset_ids so the server gives them separate queues."""
        client = FakeClient(K=4)
        ds_a = ComposableIterableDataset(FakeBackend(100), load_args=_load_args())
        ds_a = ds_a.slice(0, 50)
        ds_a.enable_work_dispatch(client, "w0")
        list(ds_a)

        client_b = FakeClient(K=4)
        ds_b = ComposableIterableDataset(FakeBackend(100), load_args=_load_args())
        ds_b = ds_b.slice(50, 100)
        ds_b.enable_work_dispatch(client_b, "w0")
        list(ds_b)

        id_a = client.calls[0][2]  # register: ('register', worker_id, dataset_id, ...)
        id_b = client_b.calls[0][2]
        assert id_a != id_b, "Different slices must produce different dataset_ids"

    def test_set_epoch_registers_new_queue(self):
        """``set_epoch(N)`` changes the effective shuffle seed, which
        should trigger a fresh ``/datasets/register`` for the new
        ``(dataset_id, seed)`` pair without any explicit handling."""
        client = FakeClient(K=4)
        ds = ComposableIterableDataset(FakeBackend(100), load_args=_load_args())
        ds = ds.shuffle(seed=7)
        ds.enable_work_dispatch(client, "w0")
        list(ds)
        seed_epoch0 = client.calls[0][3]

        # Reset the unit counter so the new epoch can drain too.
        client.next_unit = 0
        ds.set_epoch(1)
        list(ds)
        # First register call's seed (epoch 0), and there must be a
        # second register call with a different seed.
        register_calls = [c for c in client.calls if c[0] == "register"]
        assert len(register_calls) == 2
        assert register_calls[0][3] == seed_epoch0
        assert register_calls[1][3] == seed_epoch0 + 1


# ---------------------------------------------------------------------------
# maybe_enable_work_dispatch — env-driven entry point
# ---------------------------------------------------------------------------


class TestDataLoaderWorkers:
    """Regression for the multi-DataLoader-worker case.

    Forked DataLoader workers each fork the composable and call
    ``__iter__``. Under conventional sharding the composable narrows
    the per-worker view via ``_worker_view_bounds()``. Under DiLoCo
    dispatch each fork should see the FULL view (post-slice, no
    DataLoader subdivision) — they all register the same
    ``(dataset_id, seed)`` with identical ``hint["length"]`` (server
    would otherwise return 409 on the second register) and compete
    for units in one queue. The unit math is over the full L, so K
    issued by the server matches what every fork uses.

    These tests fake ``torch.utils.data.get_worker_info`` to simulate
    a DataLoader fork with ``num_workers > 1`` without actually
    spinning up subprocess workers.
    """

    def test_dispatch_ignores_worker_subdivision(self, monkeypatch):
        from forgather.ml.datasets import composable_iterable_dataset as cid

        # Simulate worker 0 of a 2-worker DataLoader. Without the fix,
        # _worker_view_bounds() would narrow L to 50 (half of 100),
        # the register call would send hint.length=50, and the unit
        # math would use the wrong L.
        class FakeWorkerInfo:
            id = 0
            num_workers = 2

        monkeypatch.setattr(
            cid.torch.utils.data,
            "get_worker_info",
            lambda: FakeWorkerInfo(),
        )

        client = FakeClient(K=4)
        ds = ComposableIterableDataset(FakeBackend(100), load_args=_load_args())
        ds.enable_work_dispatch(client, "w0")
        # Iterate — should drive the dispatch loop over the FULL
        # length, not the per-DataLoader-worker subwindow.
        rows = [r["i"] for r in ds]
        # Verify the register hint reflects the full view, not the
        # per-worker slice.
        register = next(c for c in client.calls if c[0] == "register")
        assert register[4]["length"] == 100, (
            f"Expected register hint.length=100 (full view), got "
            f"{register[4]['length']} (worker-subdivided)"
        )
        # And every row from 0..99 should be yielded (single worker
        # in this test; in a real two-worker run each fork would get
        # roughly half the units atomically from the server).
        assert rows == list(range(100))


class TestMaybeEnableWorkDispatch:
    def test_no_server_returns_unchanged(self, monkeypatch):
        monkeypatch.delenv("DILOCO_SERVER", raising=False)
        monkeypatch.delenv("DILOCO_WORKER_ID", raising=False)
        ds = ComposableIterableDataset(FakeBackend(), load_args=_load_args())
        out = maybe_enable_work_dispatch(ds)
        assert out is ds
        assert ds._wud_client is None  # no dispatch wired up

    def test_missing_worker_id_raises(self, monkeypatch):
        monkeypatch.setenv("DILOCO_SERVER", "localhost:8000")
        monkeypatch.delenv("DILOCO_WORKER_ID", raising=False)
        ds = ComposableIterableDataset(FakeBackend(), load_args=_load_args())
        with pytest.raises(DiLoCoWorkDispatchUnavailable, match="DILOCO_WORKER_ID"):
            maybe_enable_work_dispatch(ds)

    def test_missing_load_args_raises(self, monkeypatch):
        monkeypatch.setenv("DILOCO_SERVER", "localhost:8000")
        monkeypatch.setenv("DILOCO_WORKER_ID", "w0")
        ds = ComposableIterableDataset(FakeBackend())  # no load_args
        with pytest.raises(DiLoCoWorkDispatchUnavailable, match="load_args"):
            maybe_enable_work_dispatch(ds)

    def test_happy_path_enables_dispatch(self, monkeypatch):
        monkeypatch.setenv("DILOCO_SERVER", "localhost:8000")
        monkeypatch.setenv("DILOCO_WORKER_ID", "w0")
        ds = ComposableIterableDataset(FakeBackend(), load_args=_load_args())
        # Patch DiLoCoClient so we don't make a real network call.
        with patch("forgather.ml.diloco.client.DiLoCoClient") as MockClient:
            instance = MockClient.return_value
            out = maybe_enable_work_dispatch(ds)
        assert out is ds
        assert ds._wud_client is instance
        assert ds._wud_worker_id == "w0"
