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


# Patch targets: the imports inside on_load_model_weights resolve to these modules
_WORKER_PATCH = "forgather.ml.diloco.worker.DiLoCoWorker"
_CLIENT_PATCH = "forgather.ml.diloco.client.DiLoCoClient"


def _info(
    sync_every=500,
    dylu=False,
    bf16_comm=True,
    num_fragments_default=1,
    heartbeat_timeout=0,
):
    """A minimal /info payload as the server would return it. The four
    server-authoritative settings live under expected_client_settings."""
    return {
        "mode": "sync",
        "num_parameters": 64,
        "model_hash": "deadbeef",
        "settings_authority": "server",
        "expected_client_settings": {
            "sync_every": sync_every,
            "dylu": dylu,
            "bf16_comm": bf16_comm,
            "num_fragments_min": 1,
            "num_fragments_default": num_fragments_default,
            "heartbeat_timeout": heartbeat_timeout,
        },
    }


def _stub_info(MockClient, **kwargs):
    """Point the mocked DiLoCoClient's get_info at an _info() payload."""
    MockClient.return_value.get_info.return_value = _info(**kwargs)


class TestFailFastWhenUnconfigured:
    """The callback was reworked from "silent no-op when DILOCO_SERVER
    is unset" to "fail fast on misconfiguration." The previous silent
    no-op was masking distributed-training islands (two workers,
    callback present, no server, no sync, no warning). The template
    is now responsible for gating the include on DILOCO_SERVER; if
    the gate is missing and we reach here without a server_addr,
    we raise."""

    def test_inactive_property_still_works(self):
        """``active`` is still False without a server_addr — used by
        the template to decide whether to include the callback at all."""
        cb = DiLoCoCallback()
        assert not cb.active
        cb2 = DiLoCoCallback(server_addr="")
        assert not cb2.active

    def test_on_train_begin_raises_when_no_server(self):
        """on_train_begin raises DiLoCoServerUnreachable when the
        callback was constructed without a server_addr. Previously this
        was a silent no-op."""
        from forgather.ml.diloco.client import DiLoCoServerUnreachable

        cb = DiLoCoCallback()
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with pytest.raises(DiLoCoServerUnreachable, match="DILOCO_SERVER"):
            cb.on_load_model_weights(
                args, state, control, model=model, optimizer=optimizer
            )

    def test_on_log_noop_when_inactive(self):
        """on_log is still a no-op when the worker was never started
        (it's called on every step, so raising would be catastrophic)."""
        cb = DiLoCoCallback()
        args, state, control = _make_args(), _make_state(), _make_control()
        logs = {"loss": 1.0}
        cb.on_log(args, state, control, logs=logs)
        assert "diloco/sync_count" not in logs

    def test_on_train_end_noop_when_inactive(self):
        """on_train_end is still a no-op when the worker was never
        started (covers the on_train_begin-raised path so cleanup
        doesn't double-fault)."""
        cb = DiLoCoCallback()
        args, state, control = _make_args(), _make_state(), _make_control()
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

    @patch(_WORKER_PATCH)
    @patch(_CLIENT_PATCH)
    def test_on_train_begin_raises_when_server_unreachable(
        self, MockClient, MockWorker
    ):
        """on_train_begin raises DiLoCoServerUnreachable when the
        /info probe at startup fails. Surfaces the failure while
        the operator's still watching the TTY, not 500 steps in."""
        from forgather.ml.diloco.client import DiLoCoServerUnreachable

        MockClient.return_value.get_info.side_effect = ConnectionError("refused")

        cb = DiLoCoCallback(server_addr="unreachable:9999")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with pytest.raises(DiLoCoServerUnreachable, match="/info round-trip"):
            cb.on_load_model_weights(
                args, state, control, model=model, optimizer=optimizer
            )
        # Worker was never started since the probe came first.
        MockWorker.return_value.start.assert_not_called()


class TestEnvVarConfiguration:
    """Environment variable reading for the client-local knobs. The
    server-authoritative settings (sync_every / bf16_comm / dylu /
    num_fragments) are no longer constructor args or env vars — the
    worker reads them from /info (see TestServerAuthoritativeSettings)."""

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

    def test_worker_id_from_env(self):
        """DILOCO_WORKER_ID env var provides worker_id."""
        with patch.dict(os.environ, {"DILOCO_WORKER_ID": "w42"}):
            cb = DiLoCoCallback()
            assert cb.worker_id == "w42"

    def test_heartbeat_interval_from_env(self):
        """DILOCO_HEARTBEAT_INTERVAL env var provides heartbeat_interval."""
        with patch.dict(os.environ, {"DILOCO_HEARTBEAT_INTERVAL": "15.0"}):
            cb = DiLoCoCallback()
            assert cb.heartbeat_interval == 15.0

    def test_server_authoritative_settings_not_constructor_args(self):
        """The removed must-match settings are not accepted as kwargs."""
        for kw in ("sync_every", "bf16_comm", "dylu", "num_fragments"):
            with pytest.raises(TypeError):
                DiLoCoCallback(server_addr="host:8512", **{kw: 1})

    def test_defaults_without_env(self):
        """Default values when no env vars are set. The server-authoritative
        settings stay None until /info is read in on_train_begin."""
        # Clear any DILOCO_* env vars
        env = {k: v for k, v in os.environ.items() if not k.startswith("DILOCO_")}
        with patch.dict(os.environ, env, clear=True):
            cb = DiLoCoCallback()
            assert cb.server_addr == ""
            assert cb.worker_id is None
            assert cb.heartbeat_interval == 30.0
            assert cb.timeout == 600
            assert cb.max_sync_retries == 3
            # Not resolved until on_train_begin negotiates with the server.
            assert cb.sync_every is None
            assert cb.bf16_comm is None
            assert cb.dylu is None
            assert cb.num_fragments is None


class TestServerAuthoritativeSettings:
    """sync_every / bf16_comm / dylu / num_fragments are taken verbatim
    from the server's /info, with no client override."""

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_worker_built_with_server_settings(self, MockWorker, MockClient):
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {}
        _stub_info(
            MockClient,
            sync_every=250,
            dylu=True,
            bf16_comm=False,
            num_fragments_default=3,
        )

        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        cb.on_load_model_weights(args, state, control, model=model, optimizer=optimizer)

        _, kwargs = MockWorker.call_args
        assert kwargs["sync_every"] == 250
        assert kwargs["dylu"] is True
        # The legacy ``bf16_comm=False`` from /info maps to the
        # post-#130 ``upload_dtype="fp32"`` via the callback's
        # back-compat shim in ``_resolve_server_settings``.
        assert kwargs["upload_dtype"] == "fp32"
        assert kwargs["num_fragments"] == 3
        # And the callback's own copies were updated.
        assert cb.sync_every == 250
        assert cb.dylu is True
        assert cb.upload_dtype == "fp32"
        assert cb.bf16_comm is False  # legacy mirror, still reported

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_missing_sync_every_is_fatal(self, MockWorker, MockClient):
        """A server that doesn't advertise sync_every (too old) is fatal,
        not silently defaulted."""
        from forgather.ml.diloco.client import DiLoCoServerUnreachable

        MockClient.return_value.get_info.return_value = _info(sync_every=None)
        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with pytest.raises(DiLoCoServerUnreachable, match="sync_every"):
            cb.on_load_model_weights(
                args, state, control, model=model, optimizer=optimizer
            )
        MockWorker.return_value.start.assert_not_called()

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_heartbeat_interval_at_or_above_timeout_raises(
        self, MockWorker, MockClient
    ):
        """A heartbeat cadence >= the server's death timeout is rejected
        up front (it guarantees spurious eviction)."""
        _stub_info(MockClient, heartbeat_timeout=20)
        cb = DiLoCoCallback(server_addr="host:8512", heartbeat_interval=30.0)
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with pytest.raises(ValueError, match="heartbeat_interval"):
            cb.on_load_model_weights(
                args, state, control, model=model, optimizer=optimizer
            )
        MockWorker.return_value.start.assert_not_called()

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_heartbeat_timeout_zero_disables_validation(self, MockWorker, MockClient):
        """heartbeat_timeout=0 means death detection is off — any cadence
        is allowed."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {}
        _stub_info(MockClient, heartbeat_timeout=0)
        cb = DiLoCoCallback(server_addr="host:8512", heartbeat_interval=999.0)
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        cb.on_load_model_weights(args, state, control, model=model, optimizer=optimizer)
        mock_instance.start.assert_called_once()


class TestWorkerLifecycle:
    """Worker created/started in on_train_begin, stopped in on_train_end.

    The /status pre-probe must be stubbed in every test here — the
    callback's fail-fast path otherwise tries to hit the network and
    blows up before reaching the worker construction we care about."""

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_worker_created_on_train_begin(self, MockWorker, MockClient):
        """on_train_begin creates and starts a DiLoCoWorker."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {}
        _stub_info(MockClient)  # server advertises sync_every=500, defaults

        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_load_model_weights(args, state, control, model=model, optimizer=optimizer)

        MockWorker.assert_called_once_with(
            model=model,
            optimizer=optimizer,
            server_addr="host:8512",
            sync_every=500,  # from /info
            worker_id=None,
            upload_dtype="bf16",
            upload_sr=False,
            download_dtype="fp32",
            download_sr=False,
            timeout=600,
            dylu=False,
            heartbeat_interval=30.0,
            num_fragments=1,
            max_sync_retries=3,
            backend=None,
            report_sync_state=True,
            param_view=None,
            auth_token=None,
            verify_tls=True,
            output_dir="/tmp/test_diloco",
        )
        mock_instance.start.assert_called_once()
        # Pre-probe (now /info) should have happened first.
        MockClient.return_value.get_info.assert_called_once()

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_worker_output_dir_reported_absolute(self, MockWorker, MockClient):
        """The worker's reported output_dir must be os.path.abspath(...) of
        args.output_dir — byte-identical to what the control callback writes
        to its endpoint (and thus the job's output_dir), so the webui's
        output_dir-based job correlation string-matches (issue #103). A raw
        relative output_dir would never match the abspath'd job value."""
        import os

        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {}
        _stub_info(MockClient)

        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        args.output_dir = "output_models/tinyv2_w0"  # relative
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_load_model_weights(args, state, control, model=model, optimizer=optimizer)

        _, kwargs = MockWorker.call_args
        assert kwargs["output_dir"] == os.path.abspath("output_models/tinyv2_w0")
        assert os.path.isabs(kwargs["output_dir"])

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_worker_stopped_on_train_end(self, MockWorker, MockClient):
        """on_train_end stops the worker."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {}
        _stub_info(MockClient)

        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_load_model_weights(args, state, control, model=model, optimizer=optimizer)
        cb.on_train_end(args, state, control)

        mock_instance.stop.assert_called_once()
        assert cb._worker is None

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_missing_model_raises(self, MockWorker, MockClient):
        """on_train_begin without model in kwargs raises (used to be
        a silent log+return — now a fatal RuntimeError)."""
        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()

        with pytest.raises(RuntimeError, match="model or optimizer"):
            cb.on_load_model_weights(args, state, control)
        MockWorker.assert_not_called()
        assert cb._worker is None

    def test_on_train_begin_requires_worker_started(self):
        """on_train_begin is a defensive assert: if on_load_model_weights
        never ran (worker still None — e.g. checkpoint_components didn't
        exclude 'model'), it fails loud rather than training a model the
        server never filled."""
        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        with pytest.raises(RuntimeError, match="not started by on_load_model_weights"):
            cb.on_train_begin(args, state, control)

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_on_train_begin_noop_after_load(self, MockWorker, MockClient):
        """Once on_load_model_weights has started the worker, on_train_begin
        passes (no exception, no second worker build)."""
        MockWorker.return_value.sync_metrics = {}
        _stub_info(MockClient)
        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        cb.on_load_model_weights(args, state, control, model=model, optimizer=optimizer)
        MockWorker.assert_called_once()
        cb.on_train_begin(args, state, control, model=model, optimizer=optimizer)
        MockWorker.assert_called_once()  # not rebuilt

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_custom_parameters_passed_to_worker(self, MockWorker, MockClient):
        """Client-local params come from the constructor; the four
        server-authoritative ones come from /info."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {}
        _stub_info(
            MockClient,
            sync_every=200,
            dylu=True,
            bf16_comm=False,
            num_fragments_default=4,
        )

        cb = DiLoCoCallback(
            server_addr="remote:9999",
            worker_id="test_worker",
            heartbeat_interval=10.0,
            timeout=300,
            max_sync_retries=5,
        )
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_load_model_weights(args, state, control, model=model, optimizer=optimizer)

        MockWorker.assert_called_once_with(
            model=model,
            optimizer=optimizer,
            server_addr="remote:9999",
            sync_every=200,  # from /info
            worker_id="test_worker",
            # bf16_comm=False on the legacy /info path maps to
            # upload_dtype="fp32" via the callback's back-compat shim.
            upload_dtype="fp32",
            upload_sr=False,
            download_dtype="fp32",
            download_sr=False,
            timeout=300,
            dylu=True,  # from /info
            heartbeat_interval=10.0,
            num_fragments=4,  # from /info
            max_sync_retries=5,
            backend=None,
            report_sync_state=True,
            param_view=None,
            auth_token=None,
            verify_tls=True,
            output_dir="/tmp/test_diloco",
        )


class TestPipelineDetection:
    """Issue #84: when the trainer kwarg has ``pipeline_modules``, the
    callback constructs a ``PipelineParamView`` and registers as one
    rank of a ``pp_world_size``-sized group."""

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_pipeline_trainer_builds_param_view_and_group_kwargs(
        self, MockWorker, MockClient
    ):
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {}
        _stub_info(MockClient)

        # Fake pipeline trainer: only the attributes the callback reads.
        fake_trainer = MagicMock()
        fake_trainer.pipeline_modules = [TinyModel(), TinyModel()]
        fake_trainer.sharing_metadata = []
        fake_trainer.dist.rank = 1
        fake_trainer.dist.world_size = 3

        cb = DiLoCoCallback(server_addr="host:8512", worker_id="alpha")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_load_model_weights(
            args,
            state,
            control,
            model=model,
            optimizer=optimizer,
            trainer=fake_trainer,
        )

        # Worker was constructed with a ParamView, group args, and a
        # rank-suffixed worker_id derived from the operator's "alpha".
        call_kwargs = MockWorker.call_args.kwargs
        assert call_kwargs["worker_id"] == "alpha_pp1"
        assert call_kwargs["group_id"] == "alpha"
        assert call_kwargs["pp_rank"] == 1
        assert call_kwargs["pp_world_size"] == 3
        # PipelineParamView instance
        from forgather.ml.diloco.param_view import PipelineParamView

        assert isinstance(call_kwargs["param_view"], PipelineParamView)

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_non_pipeline_trainer_keeps_solo_path(self, MockWorker, MockClient):
        """A trainer kwarg without ``pipeline_modules`` (or with it set
        to None / empty) takes the solo-worker path: no ParamView, no
        group kwargs."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {}
        _stub_info(MockClient)

        fake_trainer = MagicMock()
        fake_trainer.pipeline_modules = None  # not a pipeline trainer

        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_load_model_weights(
            args,
            state,
            control,
            model=model,
            optimizer=optimizer,
            trainer=fake_trainer,
        )

        call_kwargs = MockWorker.call_args.kwargs
        # param_view stays None (worker will build a SimpleModelParamView)
        assert call_kwargs["param_view"] is None
        # No group kwargs (worker defaults to pp_world_size=1)
        assert "group_id" not in call_kwargs
        assert "pp_rank" not in call_kwargs
        assert "pp_world_size" not in call_kwargs


class TestMetricsInjection:
    """Sync metrics injected into logs dict."""

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_metrics_injected_on_log(self, MockWorker, MockClient):
        """on_log adds sync_metrics to the logs dict."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {
            "diloco/sync_count": 5,
            "diloco/local_step": 42,
            "diloco/total_sync_time": 10.5,
        }
        _stub_info(MockClient)

        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_load_model_weights(args, state, control, model=model, optimizer=optimizer)

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

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_no_crash_when_logs_is_none(self, MockWorker, MockClient):
        """on_log handles None logs gracefully."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {"diloco/sync_count": 1}
        _stub_info(MockClient)

        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_load_model_weights(args, state, control, model=model, optimizer=optimizer)
        # Should not raise
        cb.on_log(args, state, control, logs=None)


class TestStatefulProtocol:
    """state_dict/load_state_dict, deferred restore, empty state when inactive."""

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_state_dict_captures_worker_state(self, MockWorker, MockClient):
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
        _stub_info(MockClient)

        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_load_model_weights(args, state, control, model=model, optimizer=optimizer)

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

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_deferred_state_applied_on_train_begin(self, MockWorker, MockClient):
        """Pending state from load_state_dict is applied when worker starts."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {}
        _stub_info(MockClient)

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

        cb.on_load_model_weights(args, state, control, model=model, optimizer=optimizer)

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

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_no_pending_state_when_not_loaded(self, MockWorker, MockClient):
        """on_train_begin works fine without any pending state."""
        mock_instance = MockWorker.return_value
        mock_instance.sync_metrics = {}
        _stub_info(MockClient)

        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_load_model_weights(args, state, control, model=model, optimizer=optimizer)

        # Worker should not have had state set on it
        mock_instance.start.assert_called_once()
        assert cb._pending_state is None

    def test_load_empty_state_dict_no_pending(self):
        """Loading an empty state_dict does not set _pending_state."""
        cb = DiLoCoCallback(server_addr="host:8512")
        cb.load_state_dict({})
        assert cb._pending_state is None

    @patch(_CLIENT_PATCH)
    @patch(_WORKER_PATCH)
    def test_roundtrip_state_dict(self, MockWorker, MockClient):
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
        _stub_info(MockClient)

        cb = DiLoCoCallback(server_addr="host:8512")
        args, state, control = _make_args(), _make_state(), _make_control()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        cb.on_load_model_weights(args, state, control, model=model, optimizer=optimizer)

        sd = cb.state_dict()
        assert sd  # not empty

        # Create a new callback and load state
        cb2 = DiLoCoCallback(server_addr="host:8512")
        cb2.load_state_dict(sd)
        assert cb2._pending_state == sd


class TestBuildStatsSnapshot:
    """``DiLoCoCallback._build_stats_snapshot`` maps trainer state + log dict
    onto the normalized stats schema the server aggregator consumes
    (``diloco/stats.py``). The fields that mirror the trainer-control
    endpoint (``global_step`` / ``epoch`` / ``learning_rate``) are required
    so the webui's per-worker stats row can render from the DiLoCo server
    alone — without needing a JobRecord on the local node."""

    def _state(self, **over):
        # TrainerState carries global_step / epoch / max_steps;
        # num_input_tokens_seen and total_flos may or may not be set.
        s = _make_state()
        for k, v in over.items():
            setattr(s, k, v)
        return s

    def test_trainer_state_fields_mapped(self):
        state = self._state(
            global_step=1234,
            epoch=1.75,
            num_input_tokens_seen=99_999,
            total_flos=1e15,
            max_steps=8030,
        )
        snap = DiLoCoCallback._build_stats_snapshot(state, logs=None)
        assert snap["step_total"] == 1234
        # ``global_step`` is the same numeric value under the conventional
        # trainer-side name — the webui's per-worker stats row reads this
        # key directly without translating ``step_total``.
        assert snap["global_step"] == 1234
        assert snap["epoch"] == 1.75
        assert snap["tokens_total"] == 99_999
        assert snap["flos_total"] == 1e15
        assert snap["max_steps"] == 8030

    def test_log_fields_passed_through(self):
        state = self._state(global_step=10, epoch=0.5)
        logs = {
            "loss": 2.5,
            "grad_norm": 0.8,
            "tok_per_sec": 1234.5,
            "mfu": 0.42,
            "learning_rate": 1.5e-4,
            "tokens": 500,
            "peak_mem": 1_000_000_000,
        }
        snap = DiLoCoCallback._build_stats_snapshot(state, logs=logs)
        assert snap["loss"] == 2.5
        assert snap["grad_norm"] == 0.8
        assert snap["tok_per_sec"] == 1234.5
        assert snap["mfu"] == 0.42
        # ``learning_rate`` is the trainer-control parity field — webui's
        # JobStatsRow reads it as the ``lr`` pill.
        assert snap["learning_rate"] == 1.5e-4
        assert snap["tokens_window"] == 500
        assert snap["peak_mem"] == 1_000_000_000.0

    def test_missing_logs_dict_still_yields_state_fields(self):
        """``on_log`` may pass ``logs=None`` on the first call; the snapshot
        still needs to carry the trainer-state fields (otherwise progress
        and step never reach the server until the first log dict arrives)."""
        state = self._state(global_step=42, epoch=0.1)
        snap = DiLoCoCallback._build_stats_snapshot(state, logs=None)
        assert snap["global_step"] == 42
        assert snap["epoch"] == 0.1
        assert "loss" not in snap

    def test_eval_path_preserves_learning_rate_absent(self):
        """The eval path doesn't carry a learning_rate (no log dict). The
        snapshot must not synthesize one — missing ``learning_rate`` is the
        signal that the previous gauge value should remain on the server."""
        state = self._state(global_step=500, epoch=0.5)
        snap = DiLoCoCallback._build_stats_snapshot(state, logs=None, eval_loss=1.42)
        assert snap["eval_loss"] == 1.42
        assert snap["eval_step"] == 500
        assert "learning_rate" not in snap
