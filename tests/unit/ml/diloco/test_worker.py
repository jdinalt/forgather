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
        """Verify pseudo-gradients are cast to bf16 when bf16_comm=True."""
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

        pseudograds = worker._compute_pseudogradients()

        for name, pg in pseudograds.items():
            assert pg.dtype == torch.bfloat16, f"{name} should be bfloat16"

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
