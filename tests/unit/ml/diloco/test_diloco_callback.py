"""Tests for DiLoCoCallback - trainer integration for DiLoCo distributed training."""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from forgather.ml.trainer.callbacks.diloco_callback import DiLoCoCallback
from forgather.ml.trainer.trainer_types import TrainerControl, TrainerState


class TinyModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self, dim=8):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear2(self.linear1(x))


def _make_args():
    """Create a minimal mock training arguments object."""
    args = MagicMock()
    args.output_dir = "/tmp/test_diloco"
    return args


def _make_state():
    """Create a minimal TrainerState."""
    return TrainerState(
        logging_steps=100,
        eval_steps=500,
        train_batch_size=32,
        max_steps=1000,
        num_train_epochs=1,
        max_eval_steps=-1,
    )


def _make_control():
    """Create a minimal TrainerControl."""
    return TrainerControl()


# Patch target: the import inside on_train_begin resolves to this module
_WORKER_PATCH = "forgather.ml.diloco.worker.DiLoCoWorker"


class TestNoOpBehavior:
    """Callback should be inactive when no server_addr is configured."""

    def test_inactive_when_no_server(self):
        """Callback with no server_addr has active=False."""
        cb = DiLoCoCallback()
        assert not cb.active

    def test_inactive_with_empty_string(self):
        """Callback with empty server_addr has active=False."""
        cb = DiLoCoCallback(server_addr="")
        assert not cb.active

    def test_on_train_begin_noop(self):
        """on_train_begin does nothing when inactive."""
        cb = DiLoCoCallback()
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_train_begin(args, state, control, model=model, optimizer=optimizer)
        assert cb._worker is None

    def test_on_log_noop(self):
        """on_log does nothing when inactive."""
        cb = DiLoCoCallback()
        args, state, control = _make_args(), _make_state(), _make_control()
        logs = {"loss": 1.0}

        cb.on_log(args, state, control, logs=logs)
        assert "diloco/sync_count" not in logs

    def test_on_train_end_noop(self):
        """on_train_end does nothing when inactive."""
        cb = DiLoCoCallback()
        args, state, control = _make_args(), _make_state(), _make_control()
        # Should not raise
        cb.on_train_end(args, state, control)

    def test_state_dict_empty_when_inactive(self):
        """state_dict returns empty dict when no worker is active."""
        cb = DiLoCoCallback()
        assert cb.state_dict() == {}

    def test_load_state_dict_empty_noop(self):
        """load_state_dict with empty dict is a no-op."""
        cb = DiLoCoCallback()
        cb.load_state_dict({})
        assert cb._pending_state is None


class TestEnvVarConfiguration:
    """Environment variable reading and constructor override."""

    def test_server_addr_from_env(self):
        """DILOCO_SERVER env var provides server_addr."""
        with patch.dict(os.environ, {"DILOCO_SERVER": "myhost:9000"}):
            cb = DiLoCoCallback()
            assert cb.server_addr == "myhost:9000"
            assert cb.active

    def test_explicit_overrides_env(self):
        """Explicit server_addr overrides DILOCO_SERVER env var."""
        with patch.dict(os.environ, {"DILOCO_SERVER": "envhost:9000"}):
            cb = DiLoCoCallback(server_addr="explicit:8512")
            assert cb.server_addr == "explicit:8512"

    def test_sync_every_from_env(self):
        """DILOCO_SYNC_EVERY env var provides sync_every."""
        with patch.dict(os.environ, {"DILOCO_SYNC_EVERY": "200"}):
            cb = DiLoCoCallback()
            assert cb.sync_every == 200

    def test_sync_every_explicit_overrides_env(self):
        """Explicit sync_every overrides env var."""
        with patch.dict(os.environ, {"DILOCO_SYNC_EVERY": "200"}):
            cb = DiLoCoCallback(sync_every=300)
            assert cb.sync_every == 300

    def test_worker_id_from_env(self):
        """DILOCO_WORKER_ID env var provides worker_id."""
        with patch.dict(os.environ, {"DILOCO_WORKER_ID": "w42"}):
            cb = DiLoCoCallback()
            assert cb.worker_id == "w42"

    def test_bf16_comm_from_env(self):
        """DILOCO_BF16_COMM env var provides bf16_comm."""
        with patch.dict(os.environ, {"DILOCO_BF16_COMM": "false"}):
            cb = DiLoCoCallback()
            assert cb.bf16_comm is False

    def test_bf16_comm_default_true(self):
        """bf16_comm defaults to True when env var is unset."""
        cb = DiLoCoCallback()
        assert cb.bf16_comm is True

    def test_dylu_from_env(self):
        """DILOCO_DYLU env var provides dylu."""
        with patch.dict(os.environ, {"DILOCO_DYLU": "1"}):
            cb = DiLoCoCallback()
            assert cb.dylu is True

    def test_heartbeat_interval_from_env(self):
        """DILOCO_HEARTBEAT_INTERVAL env var provides heartbeat_interval."""
        with patch.dict(os.environ, {"DILOCO_HEARTBEAT_INTERVAL": "15.0"}):
            cb = DiLoCoCallback()
            assert cb.heartbeat_interval == 15.0

    def test_num_fragments_from_env(self):
        """DILOCO_NUM_FRAGMENTS env var provides num_fragments."""
        with patch.dict(os.environ, {"DILOCO_NUM_FRAGMENTS": "4"}):
            cb = DiLoCoCallback()
            assert cb.num_fragments == 4

    def test_defaults_without_env(self):
        """Default values when no env vars are set."""
        # Clear any DILOCO_* env vars
        env = {k: v for k, v in os.environ.items() if not k.startswith("DILOCO_")}
        with patch.dict(os.environ, env, clear=True):
            cb = DiLoCoCallback()
            assert cb.server_addr == ""
            assert cb.sync_every == 500
            assert cb.worker_id is None
            assert cb.bf16_comm is True
            assert cb.dylu is False
            assert cb.heartbeat_interval == 30.0
            assert cb.num_fragments == 1
            assert cb.timeout == 600
            assert cb.max_sync_retries == 3


class TestWorkerLifecycle:
    """Worker created/started in on_train_begin, stopped in on_train_end."""

    @patch(_WORKER_PATCH)
    def test_worker_created_on_train_begin(self, MockWorker):
        """on_train_begin creates and starts a DiLoCoWorker."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {}

        cb = DiLoCoCallback(server_addr="host:8512", sync_every=100)
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_train_begin(args, state, control, model=model, optimizer=optimizer)

        MockWorker.assert_called_once_with(
            model=model,
            optimizer=optimizer,
            server_addr="host:8512",
            sync_every=100,
            worker_id=None,
            bf16_comm=True,
            timeout=600,
            dylu=False,
            heartbeat_interval=30.0,
            num_fragments=1,
            max_sync_retries=3,
        )
        mock_instance.start.assert_called_once()

    @patch(_WORKER_PATCH)
    def test_worker_stopped_on_train_end(self, MockWorker):
        """on_train_end stops the worker."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {}

        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_train_begin(args, state, control, model=model, optimizer=optimizer)
        cb.on_train_end(args, state, control)

        mock_instance.stop.assert_called_once()
        assert cb._worker is None

    @patch(_WORKER_PATCH)
    def test_worker_not_created_without_model(self, MockWorker):
        """on_train_begin without model in kwargs logs error, no worker created."""
        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()

        cb.on_train_begin(args, state, control)
        MockWorker.assert_not_called()
        assert cb._worker is None

    @patch(_WORKER_PATCH)
    def test_custom_parameters_passed_to_worker(self, MockWorker):
        """All callback parameters are forwarded to DiLoCoWorker."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {}

        cb = DiLoCoCallback(
            server_addr="remote:9999",
            sync_every=200,
            worker_id="test_worker",
            bf16_comm=False,
            dylu=True,
            heartbeat_interval=10.0,
            num_fragments=4,
            timeout=300,
            max_sync_retries=5,
        )
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_train_begin(args, state, control, model=model, optimizer=optimizer)

        MockWorker.assert_called_once_with(
            model=model,
            optimizer=optimizer,
            server_addr="remote:9999",
            sync_every=200,
            worker_id="test_worker",
            bf16_comm=False,
            timeout=300,
            dylu=True,
            heartbeat_interval=10.0,
            num_fragments=4,
            max_sync_retries=5,
        )


class TestMetricsInjection:
    """Sync metrics injected into logs dict."""

    @patch(_WORKER_PATCH)
    def test_metrics_injected_on_log(self, MockWorker):
        """on_log adds sync_metrics to the logs dict."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {
            "diloco/sync_count": 5,
            "diloco/local_step": 42,
            "diloco/total_sync_time": 10.5,
        }

        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_train_begin(args, state, control, model=model, optimizer=optimizer)

        logs = {"loss": 1.5, "lr": 1e-4}
        cb.on_log(args, state, control, logs=logs)

        assert logs["diloco/sync_count"] == 5
        assert logs["diloco/local_step"] == 42
        assert logs["diloco/total_sync_time"] == 10.5
        # Original logs preserved
        assert logs["loss"] == 1.5
        assert logs["lr"] == 1e-4

    def test_no_metrics_when_inactive(self):
        """on_log does not modify logs when no worker is active."""
        cb = DiLoCoCallback()
        args, state, control = _make_args(), _make_state(), _make_control()
        logs = {"loss": 1.5}

        cb.on_log(args, state, control, logs=logs)
        assert logs == {"loss": 1.5}

    @patch(_WORKER_PATCH)
    def test_no_crash_when_logs_is_none(self, MockWorker):
        """on_log handles None logs gracefully."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {"diloco/sync_count": 1}

        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_train_begin(args, state, control, model=model, optimizer=optimizer)
        # Should not raise
        cb.on_log(args, state, control, logs=None)


class TestStatefulProtocol:
    """state_dict/load_state_dict, deferred restore, empty state when inactive."""

    @patch(_WORKER_PATCH)
    def test_state_dict_captures_worker_state(self, MockWorker):
        """state_dict returns worker metrics and config."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {}
        mock_instance._sync_count = 10
        mock_instance._local_step = 42
        mock_instance.sync_every = 500
        mock_instance.worker_id = "w1"
        mock_instance._total_sync_time = 30.5
        mock_instance._sync_retries = 2
        mock_instance._reconnections = 1
        mock_instance._dylu_adjustments = 3
        mock_instance._fragment_syncs = 8

        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_train_begin(args, state, control, model=model, optimizer=optimizer)

        sd = cb.state_dict()
        assert sd["sync_count"] == 10
        assert sd["local_step"] == 42
        assert sd["sync_every"] == 500
        assert sd["worker_id"] == "w1"
        assert sd["total_sync_time"] == 30.5
        assert sd["sync_retries"] == 2
        assert sd["reconnections"] == 1
        assert sd["dylu_adjustments"] == 3
        assert sd["fragment_syncs"] == 8

    def test_state_dict_empty_when_no_worker(self):
        """state_dict returns {} when worker not active."""
        cb = DiLoCoCallback()
        assert cb.state_dict() == {}

    def test_load_state_dict_defers_state(self):
        """load_state_dict stores state in _pending_state."""
        cb = DiLoCoCallback(server_addr="host:8512")
        saved = {
            "sync_count": 5,
            "local_step": 100,
            "sync_every": 250,
            "total_sync_time": 20.0,
        }

        cb.load_state_dict(saved)
        assert cb._pending_state == saved

    @patch(_WORKER_PATCH)
    def test_deferred_state_applied_on_train_begin(self, MockWorker):
        """Pending state from load_state_dict is applied when worker starts."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {}

        cb = DiLoCoCallback(server_addr="host:8512")

        # Simulate checkpoint load (before on_train_begin)
        saved = {
            "sync_count": 7,
            "local_step": 50,
            "sync_every": 300,
            "total_sync_time": 15.0,
            "sync_retries": 1,
            "reconnections": 0,
            "dylu_adjustments": 2,
            "fragment_syncs": 4,
        }
        cb.load_state_dict(saved)

        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_train_begin(args, state, control, model=model, optimizer=optimizer)

        # Verify state was applied to mock worker
        assert mock_instance._sync_count == 7
        assert mock_instance._local_step == 50
        assert mock_instance.sync_every == 300
        assert mock_instance._total_sync_time == 15.0
        assert mock_instance._sync_retries == 1
        assert mock_instance._reconnections == 0
        assert mock_instance._dylu_adjustments == 2
        assert mock_instance._fragment_syncs == 4

        # Pending state should be cleared
        assert cb._pending_state is None

    @patch(_WORKER_PATCH)
    def test_no_pending_state_when_not_loaded(self, MockWorker):
        """on_train_begin works fine without any pending state."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {}

        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_train_begin(args, state, control, model=model, optimizer=optimizer)

        # Worker should not have had state set on it
        mock_instance.start.assert_called_once()
        assert cb._pending_state is None

    def test_load_empty_state_dict_no_pending(self):
        """Loading an empty state_dict does not set _pending_state."""
        cb = DiLoCoCallback(server_addr="host:8512")
        cb.load_state_dict({})
        assert cb._pending_state is None

    @patch(_WORKER_PATCH)
    def test_roundtrip_state_dict(self, MockWorker):
        """state_dict output can be loaded back via load_state_dict."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {}
        mock_instance._sync_count = 3
        mock_instance._local_step = 20
        mock_instance.sync_every = 500
        mock_instance.worker_id = "w_test"
        mock_instance._total_sync_time = 5.0
        mock_instance._sync_retries = 0
        mock_instance._reconnections = 0
        mock_instance._dylu_adjustments = 0
        mock_instance._fragment_syncs = 0

        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_train_begin(args, state, control, model=model, optimizer=optimizer)

        sd = cb.state_dict()
        assert sd  # not empty

        # Create a new callback and load state
        cb2 = DiLoCoCallback(server_addr="host:8512")
        cb2.load_state_dict(sd)
        assert cb2._pending_state == sd
