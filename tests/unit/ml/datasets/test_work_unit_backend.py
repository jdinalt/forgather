"""Tests for ``WorkUnitBackend`` — the backend-layer DiLoCo dispatch wrap.

Backend-level so it composes under ``ComposableIterableDataset`` like
any other ``IterableDatasetBackend``. Iteration is driven by work units
issued from a DiLoCo server; the wrapped backend supplies the actual
rows via its ``seek`` + iter primitives.
"""

from __future__ import annotations

import logging
import os
from unittest.mock import patch

import pytest

from forgather.ml.datasets.iterable_backend import IterableDatasetBackend
from forgather.ml.datasets.work_unit_backend import (
    WorkUnitBackend,
    maybe_wrap_for_work_dispatch,
    unit_range,
)

# ---------------------------------------------------------------------------
# unit_range
# ---------------------------------------------------------------------------


class TestUnitRange:
    def test_clean_split(self):
        assert unit_range(0, 4, 16) == (0, 4)
        assert unit_range(3, 4, 16) == (12, 16)

    def test_uneven_split_handles_remainder(self):
        # K=4, length=17 → last unit gets the extra row.
        assert unit_range(0, 4, 17) == (0, 4)
        assert unit_range(3, 4, 17) == (12, 17)

    def test_length_smaller_than_k(self):
        # K=4, length=2 — some units empty, all rows covered exactly once.
        assert unit_range(0, 4, 2) == (0, 0)
        assert unit_range(3, 4, 2) == (1, 2)

    def test_out_of_range_unit_id_raises(self):
        with pytest.raises(ValueError, match="unit_id"):
            unit_range(4, 4, 16)
        with pytest.raises(ValueError, match="unit_id"):
            unit_range(-1, 4, 16)


# ---------------------------------------------------------------------------
# WorkUnitBackend
# ---------------------------------------------------------------------------


class FakeBackend(IterableDatasetBackend):
    """Minimal IterableDatasetBackend backed by an in-memory row list.

    ``seek`` returns a new FakeBackend pointing at the new position;
    ``__iter__`` yields from ``_position`` forward.
    """

    def __init__(self, rows, position=0, seed=None):
        self._rows = list(rows)
        self._position = position
        self._seed = seed

    def __iter__(self):
        for row in self._rows[self._position :]:
            self._position += 1
            yield {"row": row}

    def __len__(self):
        return len(self._rows)

    def shuffle(self, seed=None):
        # Trivial shuffle: just stamp the seed (real backends would
        # permute). Returns a new instance positioned at 0.
        return FakeBackend(self._rows, position=0, seed=seed)

    def seek(self, position):
        return FakeBackend(self._rows, position=position, seed=self._seed)

    def position(self):
        return self._position


class FakeClient:
    """In-memory DiLoCo client stub."""

    def __init__(self, total_units: int):
        self.total_units = total_units
        self._next = 0
        self.requests = []
        self.completes = []

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


def _make_backend(rows, K=4):
    inner = FakeBackend(rows)
    client = FakeClient(total_units=K)
    backend = WorkUnitBackend(
        wrapped=inner,
        client=client,
        worker_id="w0",
        dataset_id="ds-test",
        shuffle_seed=42,
        total_units=K,
        length=len(rows),
    )
    return backend, inner, client


class TestIteration:
    def test_yields_all_rows_in_order(self):
        # 16 rows, K=4 → 4 units of 4 rows each.
        backend, inner, client = _make_backend(list(range(16)), K=4)
        rows = [r["row"] for r in backend]
        assert rows == list(range(16))
        # 4 requests + 1 exhaustion check = 5 calls.
        assert len(client.requests) == 5
        assert len(client.completes) == 4
        assert [c[3] for c in client.completes] == [0, 1, 2, 3]

    def test_uneven_split(self):
        # 17 rows, K=4 → first 3 units have 4 rows, last has 5.
        backend, _, _ = _make_backend(list(range(17)), K=4)
        rows = [r["row"] for r in backend]
        assert rows == list(range(17))

    def test_exhaustion_ends_iteration(self):
        backend, _, client = _make_backend(list(range(8)), K=2)
        rows = list(backend)
        assert len(rows) == 8
        # All units issued; final request returned exhausted.
        assert client._next == 2

    def test_request_failure_propagates(self):
        """A failed request_work means no unit was issued — let the
        training loop see it (no row loss)."""

        class Boom(FakeClient):
            def request_work(self, *a, **k):
                raise ConnectionError("network down")

        backend = WorkUnitBackend(
            wrapped=FakeBackend([1, 2, 3]),
            client=Boom(total_units=4),
            worker_id="w0",
            dataset_id="d",
            shuffle_seed=0,
            total_units=4,
            length=3,
        )
        with pytest.raises(ConnectionError, match="network down"):
            list(backend)

    def test_per_unit_drain_error_swallowed(self, caplog):
        """A mid-unit drain exception is logged and skipped — the unit
        is already consumed from the queue."""

        class FlakyBackend(IterableDatasetBackend):
            def __init__(self):
                self._calls = 0

            def __len__(self):
                return 8

            def shuffle(self, seed=None):
                return self

            def seek(self, position):
                self._calls += 1
                return self

            def position(self):
                return 0

            def __iter__(self):
                # Raise on the 2nd unit's drain, succeed on 1 and 3.
                if self._calls == 2:
                    raise RuntimeError("simulated backend blip")
                yield {"row": self._calls}

        client = FakeClient(total_units=3)
        backend = WorkUnitBackend(
            wrapped=FlakyBackend(),
            client=client,
            worker_id="w0",
            dataset_id="d",
            shuffle_seed=0,
            total_units=3,
            length=8,
        )
        with caplog.at_level(
            logging.WARNING, logger="forgather.ml.datasets.work_unit_backend"
        ):
            rows = list(backend)
        # Units 0 and 2 succeeded; unit 1 errored mid-drain.
        assert len(rows) == 2
        assert any("drain failed" in rec.message for rec in caplog.records)
        # All units got the completion ack regardless (best-effort).
        assert len(client.completes) == 3


class TestBackendInterface:
    def test_len_returns_wrapped_length(self):
        backend, _, _ = _make_backend(list(range(100)), K=4)
        assert len(backend) == 100

    def test_seek_is_noop(self):
        backend, _, _ = _make_backend(list(range(16)), K=4)
        result = backend.seek(5)
        assert result is backend  # contract returns a backend; we return self

    def test_position_is_zero(self):
        # The composable's state_dict reads position(); under work-
        # dispatch there's no meaningful flat position, so we return 0.
        backend, _, _ = _make_backend(list(range(16)), K=4)
        assert backend.position() == 0

    def test_shuffle_wraps_new_wrapped_backend(self):
        backend, inner, _ = _make_backend(list(range(16)), K=4)
        shuffled = backend.shuffle(123)
        # Returns a new WorkUnitBackend whose wrapped is shuffled.
        assert isinstance(shuffled, WorkUnitBackend)
        assert shuffled._wrapped is not inner
        # Same queue-keying state preserved.
        assert shuffled._dataset_id == backend._dataset_id
        assert shuffled._shuffle_seed == backend._shuffle_seed
        assert shuffled._total_units == backend._total_units
        assert shuffled._length == backend._length

    def test_column_names_passthrough(self):
        class WithCols(FakeBackend):
            @property
            def column_names(self):
                return ["a", "b", "c"]

        backend = WorkUnitBackend(
            wrapped=WithCols([1, 2]),
            client=FakeClient(2),
            worker_id="w",
            dataset_id="d",
            shuffle_seed=0,
            total_units=2,
            length=2,
        )
        assert backend.column_names == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# maybe_wrap_for_work_dispatch — the loader hook
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_diloco_env(monkeypatch):
    for k in ("DILOCO_WORK_DISPATCH", "DILOCO_SERVER", "DILOCO_WORKER_ID"):
        monkeypatch.delenv(k, raising=False)


def _fake_backend(length=1000):
    return FakeBackend(list(range(length)))


class TestMaybeWrap:
    def test_opt_out_returns_unchanged(self):
        b = _fake_backend()
        out = maybe_wrap_for_work_dispatch(b, path="x")
        assert out is b

    def test_missing_server_addr_returns_unchanged(self, monkeypatch, caplog):
        monkeypatch.setenv("DILOCO_WORK_DISPATCH", "1")
        monkeypatch.setenv("DILOCO_WORKER_ID", "w0")
        b = _fake_backend()
        with caplog.at_level(
            logging.ERROR, logger="forgather.ml.datasets.work_unit_backend"
        ):
            out = maybe_wrap_for_work_dispatch(b, path="x")
        assert out is b
        assert any("DILOCO_SERVER" in rec.message for rec in caplog.records)

    def test_missing_worker_id_returns_unchanged(self, monkeypatch, caplog):
        monkeypatch.setenv("DILOCO_WORK_DISPATCH", "1")
        monkeypatch.setenv("DILOCO_SERVER", "h:1")
        b = _fake_backend()
        with caplog.at_level(
            logging.ERROR, logger="forgather.ml.datasets.work_unit_backend"
        ):
            out = maybe_wrap_for_work_dispatch(b, path="x")
        assert out is b
        assert any("DILOCO_WORKER_ID" in rec.message for rec in caplog.records)

    def test_no_len_returns_unchanged(self, monkeypatch, caplog):
        monkeypatch.setenv("DILOCO_WORK_DISPATCH", "1")
        monkeypatch.setenv("DILOCO_SERVER", "h:1")
        monkeypatch.setenv("DILOCO_WORKER_ID", "w0")

        class NoLen(IterableDatasetBackend):
            def __iter__(self):
                yield {}

            def __len__(self):
                raise TypeError("no len")

            def shuffle(self, seed=None):
                return self

            def seek(self, p):
                return self

            def position(self):
                return 0

        b = NoLen()
        with caplog.at_level(
            logging.ERROR, logger="forgather.ml.datasets.work_unit_backend"
        ):
            out = maybe_wrap_for_work_dispatch(b, path="x")
        assert out is b
        assert any("__len__" in rec.message for rec in caplog.records)

    def test_wraps_on_happy_path(self, monkeypatch):
        monkeypatch.setenv("DILOCO_WORK_DISPATCH", "1")
        monkeypatch.setenv("DILOCO_SERVER", "diloco-host:8512")
        monkeypatch.setenv("DILOCO_WORKER_ID", "alpha")
        b = _fake_backend(length=2000)
        with patch("forgather.ml.diloco.client.DiLoCoClient") as MockClient:
            MockClient.return_value.register_dataset.return_value = {"total_units": 64}
            out = maybe_wrap_for_work_dispatch(b, path="test/dataset", split="train")
        assert isinstance(out, WorkUnitBackend)
        assert out._total_units == 64
        assert out._length == 2000
        assert out._worker_id == "alpha"
        # Underlying backend preserved.
        assert out._wrapped is b
        # register_dataset got the right shape.
        call = MockClient.return_value.register_dataset.call_args
        assert call.kwargs["worker_id"] == "alpha"
        assert call.kwargs["shuffle_seed"] == 0
        assert call.kwargs["hint"] == {"length": 2000}
        assert len(call.kwargs["dataset_id"]) == 16

    def test_register_failure_returns_unchanged(self, monkeypatch, caplog):
        monkeypatch.setenv("DILOCO_WORK_DISPATCH", "1")
        monkeypatch.setenv("DILOCO_SERVER", "h:1")
        monkeypatch.setenv("DILOCO_WORKER_ID", "w0")
        b = _fake_backend()
        with patch("forgather.ml.diloco.client.DiLoCoClient") as MockClient:
            MockClient.return_value.register_dataset.side_effect = ConnectionError(
                "server unreachable"
            )
            with caplog.at_level(
                logging.ERROR, logger="forgather.ml.datasets.work_unit_backend"
            ):
                out = maybe_wrap_for_work_dispatch(b, path="x")
        assert out is b
        assert any("/datasets/register failed" in rec.message for rec in caplog.records)

    def test_invalid_path_returns_unchanged(self, monkeypatch, caplog):
        """compute_dataset_id rejects an empty path; the wrap falls
        back gracefully rather than crashing the loader."""
        monkeypatch.setenv("DILOCO_WORK_DISPATCH", "1")
        monkeypatch.setenv("DILOCO_SERVER", "h:1")
        monkeypatch.setenv("DILOCO_WORKER_ID", "w0")
        b = _fake_backend()
        with caplog.at_level(
            logging.ERROR, logger="forgather.ml.datasets.work_unit_backend"
        ):
            out = maybe_wrap_for_work_dispatch(b, path="")
        assert out is b
        assert any("dataset_id" in rec.message for rec in caplog.records)
