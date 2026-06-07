"""Tests for DiLoCoWorker - pseudo-gradient computation and sync logic."""

import threading
import time

import pytest
import torch
import torch.nn as nn

from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.server import DiLoCoServer
from forgather.ml.diloco.worker import DiLoCoWorker

from .conftest import make_initial_checkpoint


class TinyModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self, dim=8):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear2(self.linear1(x))


@pytest.fixture
def server_with_model(tmp_path):
    """Create a server initialized with a TinyModel's state dict."""
    torch.manual_seed(42)
    model = TinyModel(dim=8)
    sd = model.state_dict()

    def simple_sgd(params):
        return torch.optim.SGD(params, lr=1.0, momentum=0.0)

    ckpt = make_initial_checkpoint(sd, tmp_path)
    server = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=1,
        port=0,
        outer_optimizer_factory=simple_sgd,
    )
    server.start()
    time.sleep(0.2)

    yield server, model

    server.stop()


@pytest.fixture
def two_worker_server_with_model(tmp_path):
    """Create a server expecting 2 workers."""
    torch.manual_seed(42)
    model = TinyModel(dim=8)
    sd = model.state_dict()

    def simple_sgd(params):
        return torch.optim.SGD(params, lr=1.0, momentum=0.0)

    ckpt = make_initial_checkpoint(sd, tmp_path)
    server = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=2,
        port=0,
        outer_optimizer_factory=simple_sgd,
    )
    server.start()
    time.sleep(0.2)

    yield server, model

    server.stop()


class TestWorkerInfoOutputDir:
    """The worker reports its local output_dir at registration purely so
    the webui can correlate it to its forgather job by output_dir when the
    worker-id was renamed away from the job's queue_id (issue #103)."""

    def test_output_dir_included_when_set(self):
        model = TinyModel(dim=4)
        worker = DiLoCoWorker(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            server_addr="dummy:8512",
            output_dir="/runs/tinyv2_w0",
        )
        assert worker._get_worker_info()["output_dir"] == "/runs/tinyv2_w0"

    def test_output_dir_omitted_when_unset(self):
        model = TinyModel(dim=4)
        worker = DiLoCoWorker(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            server_addr="dummy:8512",
        )
        # Omitted entirely (not a null) so the server stores None and old
        # workers that never report it behave identically.
        assert "output_dir" not in worker._get_worker_info()


class TestPseudoGradientComputation:
    def test_pseudograd_is_global_minus_local(self):
        """Verify pseudo-gradient = global_params - local_params."""
        model = TinyModel(dim=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        worker = DiLoCoWorker(
            model,
            optimizer,
            server_addr="dummy:8512",
            sync_every=100,
            bf16_comm=False,
        )

        # Manually set global params snapshot
        worker._global_params = {
            name: torch.ones_like(p.data) for name, p in model.named_parameters()
        }

        # Model params are random (different from ones)
        pseudograds = worker._compute_pseudogradients()

        for name, p in model.named_parameters():
            expected = torch.ones_like(p.data) - p.data.cpu()
            assert torch.allclose(
                pseudograds[name], expected
            ), f"Pseudo-gradient mismatch for {name}"

    def test_pseudograd_bf16_casting(self):
        """The wire cast lives in the backend now: _compute_pseudogradients
        returns the raw fp32 difference, and the backend casts to bf16 when
        upload_dtype=bf16 (bf16_comm=True)."""
        model = TinyModel(dim=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        worker = DiLoCoWorker(
            model,
            optimizer,
            server_addr="dummy:8512",
            sync_every=100,
            bf16_comm=True,
        )

        worker._global_params = {
            name: p.data.clone().cpu() for name, p in model.named_parameters()
        }

        raw = worker._compute_pseudogradients()
        for name, pg in raw.items():
            assert pg.dtype == torch.float32, f"{name} should be raw fp32"

        # bf16_comm=True -> the default backend casts the wire payload to bf16.
        assert worker.backend.upload_dtype == "bf16"
        wire = worker.backend._cast_upload(raw)
        for name, pg in wire.items():
            assert pg.dtype == torch.bfloat16, f"{name} should be bfloat16 on wire"

    def test_pseudograd_full_precision(self):
        """Verify pseudo-gradients stay float32 when bf16_comm=False."""
        model = TinyModel(dim=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        worker = DiLoCoWorker(
            model,
            optimizer,
            server_addr="dummy:8512",
            sync_every=100,
            bf16_comm=False,
        )

        worker._global_params = {
            name: p.data.clone().cpu() for name, p in model.named_parameters()
        }

        pseudograds = worker._compute_pseudogradients()

        for name, pg in pseudograds.items():
            assert pg.dtype == torch.float32, f"{name} should be float32"

    def test_zero_pseudograd_when_no_update(self):
        """If model hasn't changed, pseudo-gradients should be zero."""
        model = TinyModel(dim=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        worker = DiLoCoWorker(
            model,
            optimizer,
            server_addr="dummy:8512",
            sync_every=100,
            bf16_comm=False,
        )

        # Global params = current model params
        worker._global_params = {
            name: p.data.clone().cpu() for name, p in model.named_parameters()
        }

        pseudograds = worker._compute_pseudogradients()

        for name, pg in pseudograds.items():
            assert torch.allclose(
                pg, torch.zeros_like(pg)
            ), f"Pseudo-gradient for {name} should be zero"


class TestApplyGlobalParams:
    def test_apply_updates_model(self):
        """Verify _apply_global_params loads new values into model."""
        model = TinyModel(dim=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        worker = DiLoCoWorker(
            model,
            optimizer,
            server_addr="dummy:8512",
            sync_every=100,
        )

        # Create new params (all ones)
        new_params = {
            name: torch.ones_like(p.data) for name, p in model.named_parameters()
        }

        worker._apply_global_params(new_params)

        for name, p in model.named_parameters():
            assert torch.allclose(
                p.data, torch.ones_like(p.data)
            ), f"Parameter {name} should be all ones after apply"


class TestOptimizerHook:
    def test_hook_increments_step(self, server_with_model):
        """Verify the optimizer hook counts steps correctly."""
        server, model = server_with_model

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Use a large sync_every so we don't trigger sync
        with DiLoCoWorker(
            model,
            optimizer,
            server_addr=f"localhost:{server.port}",
            sync_every=1000,
            bf16_comm=False,
        ) as worker:
            # Do a few optimizer steps
            for i in range(5):
                x = torch.randn(2, 8)
                loss = model(x).sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            assert worker._local_step == 5

    def test_hook_triggers_sync(self, server_with_model):
        """Verify sync is triggered after sync_every steps."""
        server, model = server_with_model

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with DiLoCoWorker(
            model,
            optimizer,
            server_addr=f"localhost:{server.port}",
            sync_every=3,
            bf16_comm=False,
        ) as worker:
            # Do 6 steps -> should trigger 2 syncs
            for i in range(6):
                x = torch.randn(2, 8)
                loss = model(x).sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            assert worker._sync_count == 2
            assert worker._local_step == 0  # Reset after last sync


class TestEndToEndSync:
    def test_single_worker_training_loop(self, server_with_model):
        """Full training loop with single worker - verify loss decreases and sync works."""
        server, model = server_with_model

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Fixed target for reproducible test
        torch.manual_seed(99)
        target = torch.randn(4, 8)
        x = torch.randn(4, 8)

        with DiLoCoWorker(
            model,
            optimizer,
            server_addr=f"localhost:{server.port}",
            sync_every=5,
            bf16_comm=False,
        ) as worker:
            losses = []
            for step in range(15):
                output = model(x)
                loss = nn.functional.mse_loss(output, target)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Should have synced 3 times
            assert worker._sync_count == 3

        # Loss should generally decrease (may not be monotonic due to syncs)
        assert (
            losses[-1] < losses[0]
        ), f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_two_worker_sync(self, two_worker_server_with_model):
        """Two workers training and synchronizing."""
        server, ref_model = two_worker_server_with_model

        # Create two independent models with same initial params
        torch.manual_seed(42)
        model0 = TinyModel(dim=8)
        model1 = TinyModel(dim=8)
        # Load same initial state
        initial_sd = ref_model.state_dict()
        model0.load_state_dict({k: v.clone() for k, v in initial_sd.items()})
        model1.load_state_dict({k: v.clone() for k, v in initial_sd.items()})

        opt0 = torch.optim.SGD(model0.parameters(), lr=0.01)
        opt1 = torch.optim.SGD(model1.parameters(), lr=0.01)

        # Fixed data
        torch.manual_seed(99)
        x = torch.randn(4, 8)
        target = torch.randn(4, 8)

        sync_every = 3
        errors = [None, None]

        def train_worker(idx, model, optimizer, sync_count_expected):
            try:
                with DiLoCoWorker(
                    model,
                    optimizer,
                    server_addr=f"localhost:{server.port}",
                    sync_every=sync_every,
                    worker_id=f"worker_{idx}",
                    bf16_comm=False,
                ) as worker:
                    for step in range(sync_every):
                        output = model(x)
                        loss = nn.functional.mse_loss(output, target)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    assert worker._sync_count == sync_count_expected
            except Exception as e:
                errors[idx] = e

        t0 = threading.Thread(target=train_worker, args=(0, model0, opt0, 1))
        t1 = threading.Thread(target=train_worker, args=(1, model1, opt1, 1))

        t0.start()
        t1.start()
        t0.join(timeout=15)
        t1.join(timeout=15)

        assert errors[0] is None, f"Worker 0 error: {errors[0]}"
        assert errors[1] is None, f"Worker 1 error: {errors[1]}"

        # After sync, both models should have the same params (the updated global params)
        for (name0, p0), (name1, p1) in zip(
            model0.named_parameters(), model1.named_parameters()
        ):
            assert torch.allclose(
                p0.data, p1.data, atol=1e-5
            ), f"After sync, {name0} should match between workers"

        # And the server should be at sync_round 1
        assert server._sync_round == 1


class TestWorkerMetrics:
    def test_sync_metrics(self, server_with_model):
        server, model = server_with_model

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with DiLoCoWorker(
            model,
            optimizer,
            server_addr=f"localhost:{server.port}",
            sync_every=3,
            bf16_comm=False,
        ) as worker:
            for _ in range(3):
                x = torch.randn(2, 8)
                loss = model(x).sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            metrics = worker.sync_metrics
            assert metrics["diloco/sync_count"] == 1
            assert metrics["diloco/local_step"] == 0
            assert metrics["diloco/last_sync_time"] > 0
            # The backend-reported sent_bytes flows into last_send_mb. fp32
            # upload -> wire size == raw size (4 bytes/param).
            n_params = sum(p.numel() for p in model.parameters())
            assert metrics["diloco/last_send_mb"] == pytest.approx(n_params * 4 / 1e6)

    def test_last_send_mb_reflects_bf16_wire_size(self, server_with_model):
        """End-to-end: with bf16 upload, the cast happens in the backend and the
        metric reports the bf16 wire size (half the fp32 raw size) — proving
        result.sent_bytes flows into diloco/last_send_mb."""
        server, model = server_with_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        n_params = sum(p.numel() for p in model.parameters())

        with DiLoCoWorker(
            model,
            optimizer,
            server_addr=f"localhost:{server.port}",
            sync_every=3,
            bf16_comm=True,
        ) as worker:
            for _ in range(3):
                x = torch.randn(2, 8)
                model(x).sum().backward()
                optimizer.step()
                optimizer.zero_grad()

            assert worker.sync_metrics["diloco/last_send_mb"] == pytest.approx(
                n_params * 2 / 1e6
            )

    def test_force_sync(self, server_with_model):
        server, model = server_with_model

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with DiLoCoWorker(
            model,
            optimizer,
            server_addr=f"localhost:{server.port}",
            sync_every=1000,
            bf16_comm=False,
        ) as worker:
            # Do a single step
            x = torch.randn(2, 8)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            assert worker._sync_count == 0

            # Force sync
            worker.force_sync()
            assert worker._sync_count == 1


class TestDDPRankAwareness:
    """DDP rank-awareness: only the leader (rank 0) registers /
    syncs / heartbeats; followers participate only in the broadcast.

    Real torch.distributed initialization is heavyweight, so these
    tests construct the worker and then patch ``_is_leader`` /
    ``_is_dist`` to simulate a follower. The contract is small enough
    to verify this way: ``start()`` on a follower must NOT call
    ``client.register``, ``stop()`` must NOT call
    ``client.deregister``, and the optimizer hook must still be
    installed so the follower participates in the broadcast at sync
    time.
    """

    def test_follower_does_not_register(self, server_with_model):
        server, model = server_with_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        from unittest.mock import patch

        worker = DiLoCoWorker(
            model,
            optimizer,
            server_addr=f"localhost:{server.port}",
            sync_every=1000,
            bf16_comm=False,
            heartbeat_interval=0,
        )
        # Patch rank to simulate follower.
        worker._is_dist = False  # skip broadcast collective in unit
        worker._is_leader = False

        with (
            patch.object(worker.client, "register") as m_register,
            patch.object(worker.client, "deregister") as m_deregister,
        ):
            worker.start()
            try:
                # Follower does NOT call register.
                m_register.assert_not_called()
                # But it DOES install an optimizer hook (so it
                # participates in the broadcast at sync time).
                assert len(worker._hooks) == 1
                assert worker._active is True
                # And heartbeat thread is not started.
                assert worker._heartbeat_thread is None
            finally:
                worker.stop()
            # Follower does NOT call deregister.
            m_deregister.assert_not_called()

    def test_refuses_streaming_fragments_under_ddp(self, server_with_model):
        """Streaming-fragment sync (num_fragments > 1) isn't yet
        DDP-rank-aware — followers would race the leader's HTTP
        submissions. Construction-time refusal is the safer behavior
        until that path gets a leader/follower split of its own.

        Simulates a DDP context by passing a stub that satisfies the
        ``dist.is_available() and dist.is_initialized()`` check so the
        worker sees world_size > 1.
        """
        server, model = server_with_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        from unittest.mock import patch

        with (
            patch("forgather.ml.diloco.worker.dist.is_available", return_value=True),
            patch("forgather.ml.diloco.worker.dist.is_initialized", return_value=True),
            patch("forgather.ml.diloco.worker.dist.get_rank", return_value=0),
            patch("forgather.ml.diloco.worker.dist.get_world_size", return_value=2),
        ):
            with pytest.raises(ValueError, match="streaming-fragment sync"):
                DiLoCoWorker(
                    model,
                    optimizer,
                    server_addr=f"localhost:{server.port}",
                    sync_every=1000,
                    num_fragments=4,
                    bf16_comm=False,
                    heartbeat_interval=0,
                )

    def test_leader_registers_normally(self, server_with_model):
        """Default behavior (no distributed group): worker IS the
        leader, registers as before."""
        server, model = server_with_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        worker = DiLoCoWorker(
            model,
            optimizer,
            server_addr=f"localhost:{server.port}",
            sync_every=1000,
            bf16_comm=False,
            heartbeat_interval=0,
        )
        # Confirm default rank state.
        assert worker._is_leader is True
        assert worker._ddp_rank == 0
        worker.start()
        try:
            assert worker._active is True
            assert len(worker._hooks) == 1
        finally:
            worker.stop()


class TestPipelineGroupGuards:
    """Issue #84: the worker's construction-time refusals for unsafe
    combinations under pipeline groups."""

    def test_pure_pipeline_world_size_equals_pp_world_size_is_allowed(
        self, server_with_model
    ):
        """Pure pipeline parallel: ``torch.distributed`` is initialized
        with world_size == pp_world_size, each process IS one pipeline
        rank. The worker MUST construct without complaint — the
        within-stage DDP guard only fires when world_size exceeds
        pp_world_size (genuine stage replication). Regression test
        for the pp=2 + torchrun --nproc_per_node=2 startup failure."""
        from unittest.mock import patch

        from forgather.ml.diloco.param_view import PipelineParamView

        server, model = server_with_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        view = PipelineParamView([model])

        with (
            patch("forgather.ml.diloco.worker.dist.is_available", return_value=True),
            patch("forgather.ml.diloco.worker.dist.is_initialized", return_value=True),
            patch("forgather.ml.diloco.worker.dist.get_rank", return_value=1),
            patch("forgather.ml.diloco.worker.dist.get_world_size", return_value=2),
        ):
            # Must not raise. (Network registration is irrelevant — we
            # construct without calling start().)
            DiLoCoWorker(
                model,
                optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=1000,
                bf16_comm=False,
                heartbeat_interval=0,
                param_view=view,
                group_id="alpha",
                pp_rank=1,
                pp_world_size=2,
            )

    def test_pipeline_plus_within_stage_ddp_is_rejected(self, server_with_model):
        """When world_size > pp_world_size the extra processes are
        within-stage DDP replicas — refused at construction time
        until the composition is implemented."""
        from unittest.mock import patch

        from forgather.ml.diloco.param_view import PipelineParamView

        server, model = server_with_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        view = PipelineParamView([model])

        with (
            patch("forgather.ml.diloco.worker.dist.is_available", return_value=True),
            patch("forgather.ml.diloco.worker.dist.is_initialized", return_value=True),
            patch("forgather.ml.diloco.worker.dist.get_rank", return_value=0),
            # world_size=4, pp_world_size=2 → 2 DDP replicas per pp rank
            patch("forgather.ml.diloco.worker.dist.get_world_size", return_value=4),
        ):
            with pytest.raises(ValueError, match="within-stage DDP"):
                DiLoCoWorker(
                    model,
                    optimizer,
                    server_addr=f"localhost:{server.port}",
                    sync_every=1000,
                    bf16_comm=False,
                    heartbeat_interval=0,
                    param_view=view,
                    group_id="alpha",
                    pp_rank=0,
                    pp_world_size=2,
                )

    def test_pipeline_plus_dylu_is_rejected(self, server_with_model):
        """DyLU + pipeline groups would desync the group barrier."""
        from forgather.ml.diloco.param_view import PipelineParamView

        server, model = server_with_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        view = PipelineParamView([model])

        with pytest.raises(ValueError, match="DyLU"):
            DiLoCoWorker(
                model,
                optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=1000,
                bf16_comm=False,
                heartbeat_interval=0,
                param_view=view,
                group_id="alpha",
                pp_rank=0,
                pp_world_size=2,
                dylu=True,
            )


class TestSetStats:
    """The pending-stats slot merges on_log + on_evaluate snapshots so an
    eval reported between heartbeats isn't clobbered before it ships."""

    def _bare_worker(self):
        # Exercise the stats slot in isolation, without standing up a server.
        w = DiLoCoWorker.__new__(DiLoCoWorker)
        w._pending_stats = None
        w._pending_stats_lock = threading.Lock()
        return w

    def test_merge_not_clobber(self):
        w = self._bare_worker()
        w.set_stats({"loss": 1.0, "tokens_total": 10, "step_total": 3})
        w.set_stats({"eval_loss": 2.0, "eval_step": 3})  # on_evaluate
        w.set_stats({"loss": 1.1, "tokens_total": 20, "step_total": 4})  # on_log
        merged = w._consume_stats()
        # eval survived the later on_log; train fields are the latest values.
        assert merged == {
            "loss": 1.1,
            "tokens_total": 20,
            "step_total": 4,
            "eval_loss": 2.0,
            "eval_step": 3,
        }

    def test_consume_clears(self):
        w = self._bare_worker()
        w.set_stats({"loss": 1.0})
        assert w._consume_stats() == {"loss": 1.0}
        assert w._consume_stats() is None

    def test_empty_snapshot_ignored(self):
        w = self._bare_worker()
        w.set_stats(None)
        w.set_stats({})
        assert w._consume_stats() is None
