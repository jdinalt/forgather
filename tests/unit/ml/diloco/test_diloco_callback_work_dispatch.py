"""Tests for the work-unit dispatch opt-in on ``DiLoCoCallback``.

Covers:
- Constructor / env-var opt-in.
- 409 from ``DiLoCoWorker.start`` (worker_id collision) propagates as
  a clean fatal that the trainer loop turns into an exit.
- When ``work_dispatch=True``, the callback wraps
  ``trainer.train_dataloader.dataset`` with a ``WorkUnitDataset``
  after the worker has started.
- Edge cases: no train_dataloader, missing _load_args, missing
  __len__ — all log + skip the wrap rather than crash.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from forgather.ml.diloco.client import DiLoCoRegisterCollisionError
from forgather.ml.diloco.work_unit_dataset import WorkUnitDataset
from forgather.ml.trainer.callbacks.diloco_callback import DiLoCoCallback


@pytest.fixture
def diloco_logs_propagate(monkeypatch):
    """Let pytest's caplog see records from the DiLoCoCallback logger.

    The callback module calls ``prefix_logger_rank`` at import time
    which sets ``logger.propagate = False`` — necessary for the
    rank-prefixed production handler, but it also blocks pytest's
    caplog (which hooks the root logger). Temporarily flip
    propagation for the test.
    """
    import logging

    cb_logger = logging.getLogger("forgather.ml.trainer.callbacks.diloco_callback")
    monkeypatch.setattr(cb_logger, "propagate", True)
    yield


class TinyModel(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x)


def _kwargs(model, optimizer, train_dataloader=None):
    return {
        "model": model,
        "optimizer": optimizer,
        "train_dataloader": train_dataloader,
    }


_WORKER_PATCH = "forgather.ml.diloco.worker.DiLoCoWorker"


# ---------------------------------------------------------------------------
# Opt-in plumbing
# ---------------------------------------------------------------------------


class TestOptIn:
    def test_default_off(self):
        cb = DiLoCoCallback()
        assert cb.work_dispatch is False

    def test_explicit_true(self):
        cb = DiLoCoCallback(work_dispatch=True)
        assert cb.work_dispatch is True

    def test_explicit_false_overrides_env(self):
        with patch.dict(os.environ, {"DILOCO_WORK_DISPATCH": "1"}):
            cb = DiLoCoCallback(work_dispatch=False)
        assert cb.work_dispatch is False

    def test_env_var_on(self):
        with patch.dict(os.environ, {"DILOCO_WORK_DISPATCH": "1"}):
            cb = DiLoCoCallback()
        assert cb.work_dispatch is True

    def test_env_var_truthy_values(self):
        for val in ("1", "true", "yes"):
            with patch.dict(os.environ, {"DILOCO_WORK_DISPATCH": val}):
                cb = DiLoCoCallback()
            assert cb.work_dispatch is True, f"DILOCO_WORK_DISPATCH={val!r}"

    def test_env_var_falsy(self):
        with patch.dict(os.environ, {"DILOCO_WORK_DISPATCH": "0"}):
            cb = DiLoCoCallback()
        assert cb.work_dispatch is False


# ---------------------------------------------------------------------------
# 409 propagation (worker_id collision)
# ---------------------------------------------------------------------------


class TestCollisionPropagation:
    def test_collision_re_raised_from_on_train_begin(self):
        """A 409 from DiLoCoWorker.start (worker_id collision) is
        re-raised so the trainer's run loop aborts cleanly. The
        callback also clears ``_worker`` so subsequent on_save / etc.
        don't operate on a half-initialized instance."""
        cb = DiLoCoCallback(server_addr="host:8512")
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        with patch(_WORKER_PATCH) as MockWorker:
            inst = MockWorker.return_value
            inst.start.side_effect = DiLoCoRegisterCollisionError(
                "HTTP 409: worker_id 'alpha' is already registered; …",
                diagnostic="worker_id 'alpha' is already registered; …",
            )

            with pytest.raises(DiLoCoRegisterCollisionError) as exc_info:
                cb.on_train_begin(
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    **_kwargs(model, optimizer),
                )
            assert "alpha" in str(exc_info.value)
            assert cb._worker is None  # cleaned up

    def test_non_collision_errors_still_propagate(self):
        """Any other ConnectionError-shaped failure from start() also
        propagates — not callback-swallowed."""
        cb = DiLoCoCallback(server_addr="host:8512")
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        with patch(_WORKER_PATCH) as MockWorker:
            inst = MockWorker.return_value
            inst.start.side_effect = ConnectionError("dns failure")

            with pytest.raises(ConnectionError, match="dns failure"):
                cb.on_train_begin(
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    **_kwargs(model, optimizer),
                )


# ---------------------------------------------------------------------------
# Dataset wrap behavior
# ---------------------------------------------------------------------------


class _FakeBackend:
    """Minimal duck-type for the ResilientRemoteBackend that
    ComposableIterableDataset holds. Only ``_load_args`` matters
    here."""

    def __init__(self, load_args):
        self._load_args = load_args


class _FakeDataset:
    """Stand-in for a ComposableIterableDataset wrapping a
    ResilientRemoteBackend. Carries _backend (with _load_args), a
    length, and shuffle / slice methods so WorkUnitDataset can
    operate on it."""

    def __init__(self, load_args, length=1000):
        self._backend = _FakeBackend(load_args)
        self._length = length

    def __len__(self):
        return self._length

    def shuffle(self, seed):
        # WorkUnitDataset.__init__ calls base.shuffle(seed) once. Return
        # self to keep the test fixture simple — the wrap test doesn't
        # need to exercise the inner iter.
        return self


def _fake_train_dataloader(dataset):
    """Stand-in for a torch DataLoader. Only ``.dataset`` is touched
    by the callback's wrap path."""
    dl = MagicMock()
    dl.dataset = dataset
    return dl


class TestDatasetWrap:
    def test_wrap_installed_when_work_dispatch_true(self):
        cb = DiLoCoCallback(server_addr="host:8512", work_dispatch=True)
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        dataset = _FakeDataset(
            load_args={
                "path": "test/dataset",
                "name": None,
                "split": "train",
                "data_files": None,
                "revision": None,
            },
            length=2000,
        )
        train_dataloader = _fake_train_dataloader(dataset)

        with patch(_WORKER_PATCH) as MockWorker:
            inst = MockWorker.return_value
            inst.worker_id = "w0"
            inst.client = MagicMock()
            inst.client.register_dataset.return_value = {"total_units": 16}

            cb.on_train_begin(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                **_kwargs(model, optimizer, train_dataloader=train_dataloader),
            )

        # After on_train_begin, the dataloader's dataset is now the
        # WorkUnitDataset wrapper.
        assert isinstance(train_dataloader.dataset, WorkUnitDataset)
        assert train_dataloader.dataset._total_units == 16
        assert train_dataloader.dataset._length == 2000
        # register_dataset was called with the canonical dataset_id
        # (16 hex chars) and the right hint length.
        call = inst.client.register_dataset.call_args
        assert call.kwargs["worker_id"] == "w0"
        assert len(call.kwargs["dataset_id"]) == 16
        assert call.kwargs["shuffle_seed"] == 0
        assert call.kwargs["hint"] == {"length": 2000}

    def test_no_wrap_when_work_dispatch_false(self):
        cb = DiLoCoCallback(server_addr="host:8512", work_dispatch=False)
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        dataset = _FakeDataset(load_args={"path": "x"}, length=100)
        train_dataloader = _fake_train_dataloader(dataset)

        with patch(_WORKER_PATCH) as MockWorker:
            MockWorker.return_value.worker_id = "w0"
            cb.on_train_begin(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                **_kwargs(model, optimizer, train_dataloader=train_dataloader),
            )

        # Dataset untouched.
        assert train_dataloader.dataset is dataset

    def test_no_wrap_when_dataset_has_no_load_args(self, caplog, diloco_logs_propagate):
        """A non-dataset_server dataset (local path, etc.) has no
        _backend._load_args — phase 1 of work-dispatch requires
        dataset_server. The callback should log + skip the wrap, not
        crash. Worker continues running normally; the operator just
        gets the manual-shard path."""
        import logging

        cb = DiLoCoCallback(server_addr="host:8512", work_dispatch=True)
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        # Plain dataset without _backend or with empty _load_args.
        dataset = MagicMock(spec=[])  # no _backend attribute
        train_dataloader = _fake_train_dataloader(dataset)

        with patch(_WORKER_PATCH) as MockWorker:
            MockWorker.return_value.worker_id = "w0"
            MockWorker.return_value.client = MagicMock()
            with caplog.at_level(
                logging.ERROR,
                logger="forgather.ml.trainer.callbacks.diloco_callback",
            ):
                cb.on_train_begin(
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    **_kwargs(model, optimizer, train_dataloader=train_dataloader),
                )

        # Dataset untouched.
        assert train_dataloader.dataset is dataset
        # register_dataset never called.
        MockWorker.return_value.client.register_dataset.assert_not_called()
        # Error logged.
        assert any("_load_args" in rec.message for rec in caplog.records)

    def test_no_wrap_when_no_train_dataloader(self, caplog, diloco_logs_propagate):
        import logging

        cb = DiLoCoCallback(server_addr="host:8512", work_dispatch=True)
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        with patch(_WORKER_PATCH) as MockWorker:
            MockWorker.return_value.worker_id = "w0"
            with caplog.at_level(
                logging.ERROR,
                logger="forgather.ml.trainer.callbacks.diloco_callback",
            ):
                cb.on_train_begin(
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    **_kwargs(model, optimizer, train_dataloader=None),
                )
        # Error logged but training continues (worker is up).
        assert any("no train_dataloader" in rec.message for rec in caplog.records)

    def test_register_dataset_failure_falls_back_cleanly(
        self, caplog, diloco_logs_propagate
    ):
        """If /datasets/register fails (e.g. server hiccup), the
        callback logs and skips the wrap — training continues with
        the unwrapped dataset."""
        import logging

        cb = DiLoCoCallback(server_addr="host:8512", work_dispatch=True)
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        dataset = _FakeDataset(load_args={"path": "x"}, length=100)
        train_dataloader = _fake_train_dataloader(dataset)

        with patch(_WORKER_PATCH) as MockWorker:
            inst = MockWorker.return_value
            inst.worker_id = "w0"
            inst.client = MagicMock()
            inst.client.register_dataset.side_effect = ConnectionError("server hiccup")

            with caplog.at_level(
                logging.ERROR,
                logger="forgather.ml.trainer.callbacks.diloco_callback",
            ):
                cb.on_train_begin(
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    **_kwargs(model, optimizer, train_dataloader=train_dataloader),
                )

        assert train_dataloader.dataset is dataset  # unwrapped
        assert any("/datasets/register failed" in rec.message for rec in caplog.records)
