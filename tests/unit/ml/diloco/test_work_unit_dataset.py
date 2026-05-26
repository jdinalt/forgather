"""Tests for ``WorkUnitDataset`` — the worker-side wrapper that pulls
per-unit row ranges from the DiLoCo server."""

from __future__ import annotations

import logging
from typing import Optional

import pytest

from forgather.ml.diloco.work_unit_dataset import WorkUnitDataset, unit_range

# ---------------------------------------------------------------------------
# unit_range
# ---------------------------------------------------------------------------


class TestUnitRange:
    def test_clean_split(self):
        # K=4, length=16 → 4 rows per unit, no remainder.
        assert unit_range(0, 4, 16) == (0, 4)
        assert unit_range(1, 4, 16) == (4, 8)
        assert unit_range(2, 4, 16) == (8, 12)
        assert unit_range(3, 4, 16) == (12, 16)

    def test_uneven_split_handles_remainder(self):
        # K=4, length=17 → last unit gets the extra row.
        assert unit_range(0, 4, 17) == (0, 4)
        assert unit_range(3, 4, 17) == (12, 17)

    def test_length_equals_k(self):
        # K=4, length=4 → one row per unit.
        for u in range(4):
            assert unit_range(u, 4, 4) == (u, u + 1)

    def test_length_smaller_than_k_yields_empty_units(self):
        # K=4, length=2 → some units are empty ranges. That's fine;
        # WorkUnitDataset yields nothing for those units and moves on.
        assert unit_range(0, 4, 2) == (0, 0)
        assert unit_range(1, 4, 2) == (0, 1)
        assert unit_range(2, 4, 2) == (1, 1)
        assert unit_range(3, 4, 2) == (1, 2)

    def test_out_of_range_unit_id_raises(self):
        with pytest.raises(ValueError, match="unit_id"):
            unit_range(4, 4, 16)  # K=4 means valid ids are 0..3
        with pytest.raises(ValueError, match="unit_id"):
            unit_range(-1, 4, 16)


# ---------------------------------------------------------------------------
# WorkUnitDataset
# ---------------------------------------------------------------------------


class FakeBase:
    """Minimal duck-type for ComposableIterableDataset.

    Carries a row list. ``shuffle(seed)`` returns self (deterministic
    in tests); ``slice(start, end)`` returns a new FakeBase with the
    sliced row range. ``__iter__`` yields dicts.
    """

    def __init__(self, rows, seed: Optional[int] = None):
        self._rows = list(rows)
        self.seed = seed

    def __len__(self):
        return len(self._rows)

    def shuffle(self, seed):
        # Real ComposableIterableDataset returns a new wrapper. For
        # tests we just stamp the seed and return self — that's enough
        # to assert the wrapper passes the right seed through.
        return FakeBase(self._rows, seed=seed)

    def slice(self, start, end):
        return FakeBase(self._rows[start:end], seed=self.seed)

    def __iter__(self):
        for row in self._rows:
            yield {"row": row}


class FakeClient:
    """In-memory DiLoCo client stub.

    Hands out unit_ids in ascending order until ``total_units`` is
    reached, then returns ``{exhausted: true}``. Records every
    request_work / complete_work call for assertions.
    """

    def __init__(self, total_units: int):
        self.total_units = total_units
        self._next = 0
        self.requests: list[tuple] = []
        self.completes: list[tuple] = []

    def request_work(self, worker_id, dataset_id, shuffle_seed):
        self.requests.append((worker_id, dataset_id, shuffle_seed))
        if self._next >= self.total_units:
            return {"exhausted": True}
        unit_id = self._next
        self._next += 1
        return {"unit_id": unit_id}

    def complete_work(self, worker_id, dataset_id, shuffle_seed, unit_id):
        self.completes.append((worker_id, dataset_id, shuffle_seed, unit_id))
        return {"ack": True}


def _make_wud(rows, K=4, worker_id="w0"):
    base = FakeBase(rows)
    client = FakeClient(total_units=K)
    wud = WorkUnitDataset(
        base=base,
        client=client,
        worker_id=worker_id,
        dataset_id="ds-test",
        shuffle_seed=42,
        total_units=K,
        length=len(rows),
    )
    return wud, client


class TestIteration:
    def test_yields_all_rows_in_order(self):
        # 16 rows, K=4 → 4 units × 4 rows. Drained sequentially the
        # wrapper yields rows 0..15 exactly once each.
        wud, client = _make_wud(list(range(16)), K=4)
        rows = [r["row"] for r in wud]
        assert rows == list(range(16))
        # 4 unit requests + 1 final "exhausted" check = 5 requests.
        assert len(client.requests) == 5
        # All units completed (diagnostic ack).
        assert len(client.completes) == 4
        assert [c[3] for c in client.completes] == [0, 1, 2, 3]

    def test_exhaustion_ends_iteration(self):
        wud, client = _make_wud(list(range(8)), K=2)
        rows = list(wud)
        assert len(rows) == 8
        # Final request returned {exhausted: True}.
        assert client.requests[-1] == ("w0", "ds-test", 42)
        assert client._next == 2  # all 2 units issued

    def test_uneven_split_yields_all_rows(self):
        # K=4, length=17 → first 3 units have 4 rows, last has 5.
        wud, _ = _make_wud(list(range(17)), K=4)
        rows = [r["row"] for r in wud]
        assert rows == list(range(17))

    def test_empty_dataset_raises(self):
        # WorkUnitDataset rejects length=0 at construction — a queue
        # over zero rows is degenerate.
        with pytest.raises(ValueError, match="length"):
            WorkUnitDataset(
                base=FakeBase([]),
                client=FakeClient(total_units=4),
                worker_id="w",
                dataset_id="d",
                shuffle_seed=0,
                total_units=4,
                length=0,
            )

    def test_shuffle_seed_threaded_through(self):
        # The wrapper's __iter__ calls base.shuffle(seed) once at
        # construction. The fake base stores the seed; the slices
        # inherit it.
        wud, _ = _make_wud(list(range(4)), K=4)
        assert wud._shuffled.seed == 42

    def test_request_work_failure_propagates(self, caplog):
        """A failed request_work (server unreachable) must NOT be
        silently swallowed — the unit was never issued, so no row
        loss. Let the training loop see it."""

        class Boom(FakeClient):
            def request_work(self, *a, **k):
                raise ConnectionError("network down")

        wud = WorkUnitDataset(
            base=FakeBase([1, 2, 3]),
            client=Boom(total_units=4),
            worker_id="w0",
            dataset_id="d",
            shuffle_seed=0,
            total_units=4,
            length=3,
        )
        with pytest.raises(ConnectionError, match="network down"):
            list(wud)

    def test_per_unit_drain_error_swallowed(self, caplog):
        """A mid-unit drain exception is logged and skipped — the unit
        is already consumed from the queue, so propagating would
        crash the training loop with no chance of recovery."""

        class FlakyBase:
            def __init__(self):
                self._calls = 0

            def __len__(self):
                return 8

            def shuffle(self, seed):
                return self

            def slice(self, start, end):
                return self

            def __iter__(self):
                # Raise on the 2nd unit's drain attempt, succeed otherwise.
                self._calls += 1
                if self._calls == 2:
                    raise RuntimeError("simulated dataloader blip")
                yield {"row": self._calls}

        client = FakeClient(total_units=3)
        wud = WorkUnitDataset(
            base=FlakyBase(),
            client=client,
            worker_id="w0",
            dataset_id="d",
            shuffle_seed=0,
            total_units=3,
            length=8,
        )
        with caplog.at_level(logging.WARNING):
            rows = list(wud)
        # First and third units yielded rows; second errored mid-drain
        # and was skipped.
        assert len(rows) == 2
        # The error was logged as a warning.
        assert any("drain failed" in rec.message for rec in caplog.records)
        # All three units still get completion acks (diagnostic-only,
        # idempotent; ack on the failed unit is best-effort).
        assert len(client.completes) == 3

    def test_complete_work_failure_does_not_propagate(self, caplog):
        """A failed complete_work (server hiccup, etc.) is
        diagnostic-only — it should NOT abort iteration."""

        class BadCompleteClient(FakeClient):
            def complete_work(self, *a, **k):
                raise ConnectionError("complete failed")

        wud = WorkUnitDataset(
            base=FakeBase(list(range(8))),
            client=BadCompleteClient(total_units=2),
            worker_id="w0",
            dataset_id="d",
            shuffle_seed=0,
            total_units=2,
            length=8,
        )
        with caplog.at_level(logging.DEBUG):
            rows = list(wud)
        assert len(rows) == 8


# ---------------------------------------------------------------------------
# Trainer integration surface
# ---------------------------------------------------------------------------


class TestTrainerIntegration:
    def test_len_returns_full_dataset_length(self):
        wud, _ = _make_wud(list(range(100)), K=4)
        assert len(wud) == 100

    def test_state_dict_is_stub(self):
        # state_dict / load_state_dict are no-ops; queue position lives
        # server-side. Resume just asks for the next available unit.
        wud, _ = _make_wud(list(range(8)), K=2)
        assert wud.state_dict() == {}
        # load_state_dict is permissive — old wrapper state dicts
        # shouldn't crash on restore.
        wud.load_state_dict({"position": 42, "count": 7})  # no error
