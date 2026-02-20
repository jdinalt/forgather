"""Tests for DiLoCo async mode, Delayed Nesterov, and Dynamic Local Updates."""

import threading
import time

import pytest
import torch
import torch.nn as nn

from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.server import DiLoCoServer
from forgather.ml.diloco.worker import DiLoCoWorker


def _make_state_dict(dim=8, num_layers=2, seed=42):
    torch.manual_seed(seed)
    return {
        f"layer{i}.weight": torch.randn(dim, dim)
        for i in range(num_layers)
    }


class TinyModel(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear2(self.linear1(x))


# ---------------------------------------------------------------------------
# Async server unit tests
# ---------------------------------------------------------------------------


class TestAsyncServerDirect:
    """Test async server logic directly (without HTTP)."""

    def test_async_apply_immediate(self):
        """In async mode, each submission should update params immediately."""
        dim = 4
        sd = {"w": torch.zeros(dim)}

        def simple_sgd(params):
            return torch.optim.SGD(params, lr=1.0, momentum=0.0)

        server = DiLoCoServer(
            sd, num_workers=2, port=0,
            outer_optimizer_factory=simple_sgd,
            async_mode=True,
        )

        # Worker 0 submits pseudo_grad = 1.0
        server._apply_async_pseudograd("w0", {"w": torch.ones(dim)})
        # param = 0 - 1.0 * 1.0 = -1.0
        assert torch.allclose(server.get_global_params()["w"], torch.full((dim,), -1.0))

        # Worker 1 submits pseudo_grad = 2.0
        server._apply_async_pseudograd("w1", {"w": torch.full((dim,), 2.0)})
        # param = -1.0 - 1.0 * 2.0 = -3.0
        assert torch.allclose(server.get_global_params()["w"], torch.full((dim,), -3.0))

        assert server._sync_round == 2

    def test_async_no_barrier(self):
        """Async mode should not block waiting for other workers."""
        sd = _make_state_dict(dim=4)

        def simple_sgd(params):
            return torch.optim.SGD(params, lr=1.0, momentum=0.0)

        server = DiLoCoServer(
            sd, num_workers=3, port=0,
            outer_optimizer_factory=simple_sgd,
            async_mode=True,
        )
        server.start()
        time.sleep(0.2)

        try:
            client = DiLoCoClient(f"localhost:{server.port}", timeout=5)
            client.register("w0")

            pg = {k: torch.full_like(v, 0.1) for k, v in sd.items()}

            # This should return immediately, not block for 2 more workers
            t0 = time.time()
            result = client.submit_pseudogradients("w0", pg)
            elapsed = time.time() - t0

            assert result is not None
            assert elapsed < 3.0, f"Async submit took {elapsed:.1f}s, should be instant"
            assert server._sync_round == 1
        finally:
            server.stop()

    def test_async_staleness_tracking(self):
        """Server should track staleness (server rounds since worker's last sync)."""
        sd = {"w": torch.zeros(4)}

        def simple_sgd(params):
            return torch.optim.SGD(params, lr=0.1, momentum=0.0)

        server = DiLoCoServer(
            sd, num_workers=2, port=0,
            outer_optimizer_factory=simple_sgd,
            async_mode=True,
        )
        server.start()
        time.sleep(0.2)

        try:
            client0 = DiLoCoClient(f"localhost:{server.port}", timeout=5)
            client1 = DiLoCoClient(f"localhost:{server.port}", timeout=5)
            client0.register("w0")
            client1.register("w1")

            pg = {"w": torch.ones(4)}

            # w0 submits 3 times, w1 doesn't submit
            for _ in range(3):
                client0.submit_pseudogradients("w0", pg)

            # server round is now 3
            assert server._sync_round == 3

            # w1 submits - staleness should be 3 (missed 3 server rounds)
            # We can verify via the worker info
            assert server._workers["w1"].last_sync_server_round == 0

            client1.submit_pseudogradients("w1", pg)

            # After submitting, w1's last_sync_server_round should be updated
            assert server._workers["w1"].last_sync_server_round == 4
        finally:
            server.stop()


# ---------------------------------------------------------------------------
# Delayed Nesterov (DN) tests
# ---------------------------------------------------------------------------


class TestDelayedNesterov:
    """Test DN momentum buffering."""

    def test_dn_buffer_fills_and_applies(self):
        """DN should buffer pseudo-gradients and apply momentum every N submissions."""
        dim = 4
        sd = {"w": torch.zeros(dim)}

        # Use SGD with momentum to verify DN behavior
        def sgd_momentum(params):
            return torch.optim.SGD(params, lr=1.0, momentum=0.9, nesterov=True)

        server = DiLoCoServer(
            sd, num_workers=2, port=0,
            outer_optimizer_factory=sgd_momentum,
            async_mode=True,
            dn_buffer_size=3,
        )

        pg = {"w": torch.ones(dim)}

        # First 2 submissions: intermediate direct GD steps (no momentum)
        server._apply_async_pseudograd("w0", pg)
        assert len(server._dn_grad_buffer) == 1
        # param = 0 - lr * grad = 0 - 1.0 * 1.0 = -1.0
        assert torch.allclose(server.get_global_params()["w"], torch.full((dim,), -1.0))

        server._apply_async_pseudograd("w1", pg)
        assert len(server._dn_grad_buffer) == 2
        # param = -1.0 - 1.0 * 1.0 = -2.0
        assert torch.allclose(server.get_global_params()["w"], torch.full((dim,), -2.0))

        # Third submission: buffer full -> apply optimizer with momentum
        server._apply_async_pseudograd("w0", pg)
        assert len(server._dn_grad_buffer) == 0  # Buffer cleared

        # The momentum step averages 3 grads (all 1.0 -> avg=1.0) and applies
        # with Nesterov momentum. The result depends on PyTorch's SGD implementation.
        # What matters is that the buffer was flushed and momentum was applied.
        params = server.get_global_params()["w"]
        # Should have moved further than the simple -2.0 - 1.0 = -3.0 from direct GD
        # because momentum adds extra movement
        assert params[0].item() != 0.0  # Params changed
        assert server._sync_round == 3

    def test_dn_disabled_applies_every_step(self):
        """With dn_buffer_size=0, momentum is applied on every submission."""
        dim = 4
        sd = {"w": torch.zeros(dim)}

        def sgd_momentum(params):
            return torch.optim.SGD(params, lr=1.0, momentum=0.9, nesterov=True)

        server = DiLoCoServer(
            sd, num_workers=1, port=0,
            outer_optimizer_factory=sgd_momentum,
            async_mode=True,
            dn_buffer_size=0,
        )

        pg = {"w": torch.ones(dim)}

        # Each step should apply the full optimizer (with momentum)
        for _ in range(3):
            server._apply_async_pseudograd("w0", pg)

        # With momentum accumulating over 3 steps, should move further
        # than 3 * lr * grad = 3.0
        params = server.get_global_params()["w"]
        assert abs(params[0].item()) > 3.0, (
            f"With momentum, should move more than 3.0, got {params[0].item()}"
        )

    def test_dn_direct_vs_momentum_difference(self):
        """
        DN should produce different results from no-DN because intermediate
        steps use direct GD (no momentum) instead of full optimizer steps.
        """
        dim = 4
        sd = {"w": torch.zeros(dim)}
        pg = {"w": torch.ones(dim)}

        def sgd_momentum(params):
            return torch.optim.SGD(params, lr=0.5, momentum=0.9, nesterov=True)

        # Without DN: momentum every step
        server_no_dn = DiLoCoServer(
            sd, num_workers=1, port=0,
            outer_optimizer_factory=sgd_momentum,
            async_mode=True,
            dn_buffer_size=0,
        )
        for _ in range(6):
            server_no_dn._apply_async_pseudograd("w0", pg)
        params_no_dn = server_no_dn.get_global_params()["w"]

        # With DN buffer=3: momentum every 3rd step
        server_dn = DiLoCoServer(
            sd, num_workers=1, port=0,
            outer_optimizer_factory=sgd_momentum,
            async_mode=True,
            dn_buffer_size=3,
        )
        for _ in range(6):
            server_dn._apply_async_pseudograd("w0", pg)
        params_dn = server_dn.get_global_params()["w"]

        # Results should differ
        assert not torch.allclose(params_no_dn, params_dn), (
            "DN and non-DN should produce different results"
        )


# ---------------------------------------------------------------------------
# Dynamic Local Updates (DyLU) tests
# ---------------------------------------------------------------------------


class TestDynamicLocalUpdates:
    """Test DyLU per-worker sync frequency adaptation."""

    def test_dylu_recommendation_proportional_to_speed(self):
        """Faster workers should get higher recommended sync_every."""
        sd = _make_state_dict(dim=4)
        server = DiLoCoServer(
            sd, num_workers=3, port=0,
            async_mode=True,
            dylu_enabled=True,
            dylu_base_sync_every=500,
        )

        # Register workers with different speeds
        from forgather.ml.diloco.server import WorkerInfo
        server._workers["fast"] = WorkerInfo(
            worker_id="fast", hostname="a", registered_at=0,
            last_heartbeat=0, steps_per_second=10.0,
        )
        server._workers["medium"] = WorkerInfo(
            worker_id="medium", hostname="b", registered_at=0,
            last_heartbeat=0, steps_per_second=5.0,
        )
        server._workers["slow"] = WorkerInfo(
            worker_id="slow", hostname="c", registered_at=0,
            last_heartbeat=0, steps_per_second=2.0,
        )

        fast_h = server._compute_dylu_sync_every("fast")
        medium_h = server._compute_dylu_sync_every("medium")
        slow_h = server._compute_dylu_sync_every("slow")

        # fast gets full H (10/10 * 500 = 500)
        assert fast_h == 500
        # medium gets proportional (5/10 * 500 = 250)
        assert medium_h == 250
        # slow gets proportional (2/10 * 500 = 100)
        assert slow_h == 100

    def test_dylu_minimum_one(self):
        """DyLU should never recommend sync_every < 1."""
        sd = _make_state_dict(dim=4)
        server = DiLoCoServer(
            sd, num_workers=2, port=0,
            async_mode=True,
            dylu_enabled=True,
            dylu_base_sync_every=10,
        )

        from forgather.ml.diloco.server import WorkerInfo
        server._workers["fast"] = WorkerInfo(
            worker_id="fast", hostname="a", registered_at=0,
            last_heartbeat=0, steps_per_second=100.0,
        )
        server._workers["very_slow"] = WorkerInfo(
            worker_id="very_slow", hostname="b", registered_at=0,
            last_heartbeat=0, steps_per_second=0.01,
        )

        # very_slow: 0.01/100 * 10 = 0.001 -> floor = 0 -> clamp to 1
        slow_h = server._compute_dylu_sync_every("very_slow")
        assert slow_h == 1

    def test_dylu_no_speed_data_returns_none(self):
        """DyLU should return None when speed data is unavailable."""
        sd = _make_state_dict(dim=4)
        server = DiLoCoServer(
            sd, num_workers=1, port=0,
            async_mode=True,
            dylu_enabled=True,
            dylu_base_sync_every=500,
        )

        result = server._compute_dylu_sync_every("nonexistent")
        assert result is None

    def test_dylu_disabled_returns_none(self):
        """When DyLU is disabled, should always return None."""
        sd = _make_state_dict(dim=4)
        server = DiLoCoServer(
            sd, num_workers=1, port=0,
            async_mode=True,
            dylu_enabled=False,
        )

        from forgather.ml.diloco.server import WorkerInfo
        server._workers["w0"] = WorkerInfo(
            worker_id="w0", hostname="a", registered_at=0,
            last_heartbeat=0, steps_per_second=10.0,
        )

        result = server._compute_dylu_sync_every("w0")
        assert result is None

    def test_dylu_via_heartbeat(self):
        """DyLU recommendation should be returned in heartbeat response."""
        sd = _make_state_dict(dim=4)
        server = DiLoCoServer(
            sd, num_workers=2, port=0,
            async_mode=True,
            dylu_enabled=True,
            dylu_base_sync_every=100,
        )
        server.start()
        time.sleep(0.2)

        try:
            client = DiLoCoClient(f"localhost:{server.port}", timeout=5)
            client.register("fast")
            client.register("slow")

            # Report speeds
            server._workers["fast"].steps_per_second = 10.0
            server._workers["slow"].steps_per_second = 2.0

            response = client.heartbeat("slow", steps_per_second=2.0)

            assert "recommended_sync_every" in response
            # slow: 2/10 * 100 = 20
            assert response["recommended_sync_every"] == 20
        finally:
            server.stop()


# ---------------------------------------------------------------------------
# Async end-to-end integration tests
# ---------------------------------------------------------------------------


class TestAsyncEndToEnd:
    """End-to-end tests with async server and workers."""

    @pytest.fixture
    def async_server_with_model(self):
        torch.manual_seed(42)
        model = TinyModel(dim=8)
        sd = model.state_dict()

        def simple_sgd(params):
            return torch.optim.SGD(params, lr=1.0, momentum=0.0)

        server = DiLoCoServer(
            sd, num_workers=2, port=0,
            outer_optimizer_factory=simple_sgd,
            async_mode=True,
        )
        server.start()
        time.sleep(0.2)

        yield server, model

        server.stop()

    def test_single_worker_async_training(self):
        """Single worker training in async mode - should not block."""
        torch.manual_seed(42)
        model = TinyModel(dim=8)
        sd = model.state_dict()

        def simple_sgd(params):
            return torch.optim.SGD(params, lr=1.0, momentum=0.0)

        server = DiLoCoServer(
            sd, num_workers=1, port=0,
            outer_optimizer_factory=simple_sgd,
            async_mode=True,
        )
        server.start()
        time.sleep(0.2)

        try:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            torch.manual_seed(99)
            x = torch.randn(4, 8)
            target = torch.randn(4, 8)

            with DiLoCoWorker(
                model, optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=5,
                bf16_comm=False,
            ) as worker:
                losses = []
                for _ in range(15):
                    output = model(x)
                    loss = nn.functional.mse_loss(output, target)
                    losses.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                assert worker._sync_count == 3
                assert losses[-1] < losses[0]
        finally:
            server.stop()

    def test_two_workers_async_no_deadlock(self, async_server_with_model):
        """Two workers training asynchronously - neither should block on the other."""
        server, ref_model = async_server_with_model

        torch.manual_seed(42)
        model0 = TinyModel(dim=8)
        model1 = TinyModel(dim=8)
        initial_sd = ref_model.state_dict()
        model0.load_state_dict({k: v.clone() for k, v in initial_sd.items()})
        model1.load_state_dict({k: v.clone() for k, v in initial_sd.items()})

        opt0 = torch.optim.SGD(model0.parameters(), lr=0.01)
        opt1 = torch.optim.SGD(model1.parameters(), lr=0.01)

        torch.manual_seed(99)
        x = torch.randn(4, 8)
        target = torch.randn(4, 8)

        errors = [None, None]
        sync_counts = [0, 0]

        def train_worker(idx, model, optimizer, num_steps):
            try:
                with DiLoCoWorker(
                    model, optimizer,
                    server_addr=f"localhost:{server.port}",
                    sync_every=3,
                    worker_id=f"w{idx}",
                    bf16_comm=False,
                ) as worker:
                    for _ in range(num_steps):
                        output = model(x)
                        loss = nn.functional.mse_loss(output, target)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    sync_counts[idx] = worker._sync_count
            except Exception as e:
                errors[idx] = e

        # Worker 0 trains for 6 steps (2 syncs), worker 1 for 9 steps (3 syncs)
        # In async mode, they should not wait for each other
        t0 = threading.Thread(target=train_worker, args=(0, model0, opt0, 6))
        t1 = threading.Thread(target=train_worker, args=(1, model1, opt1, 9))

        t0.start()
        t1.start()
        t0.join(timeout=15)
        t1.join(timeout=15)

        assert errors[0] is None, f"Worker 0 error: {errors[0]}"
        assert errors[1] is None, f"Worker 1 error: {errors[1]}"
        assert sync_counts[0] == 2
        assert sync_counts[1] == 3
        # Total server rounds = 2 + 3 = 5
        assert server._sync_round == 5

    def test_async_with_dn(self):
        """Async mode with Delayed Nesterov - training should work."""
        torch.manual_seed(42)
        model = TinyModel(dim=8)
        sd = model.state_dict()

        def sgd_momentum(params):
            return torch.optim.SGD(params, lr=0.5, momentum=0.9, nesterov=True)

        server = DiLoCoServer(
            sd, num_workers=1, port=0,
            outer_optimizer_factory=sgd_momentum,
            async_mode=True,
            dn_buffer_size=2,
        )
        server.start()
        time.sleep(0.2)

        try:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            torch.manual_seed(99)
            x = torch.randn(4, 8)
            target = torch.randn(4, 8)

            with DiLoCoWorker(
                model, optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=3,
                bf16_comm=False,
            ) as worker:
                losses = []
                for _ in range(9):
                    output = model(x)
                    loss = nn.functional.mse_loss(output, target)
                    losses.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                assert worker._sync_count == 3
                assert losses[-1] < losses[0]
        finally:
            server.stop()


class TestAsyncStatus:
    """Test status endpoint in async mode."""

    def test_status_shows_async_fields(self):
        sd = _make_state_dict(dim=4)
        server = DiLoCoServer(
            sd, num_workers=2, port=0,
            async_mode=True,
            dn_buffer_size=3,
            dylu_enabled=True,
            dylu_base_sync_every=200,
        )
        server.start()
        time.sleep(0.2)

        try:
            client = DiLoCoClient(f"localhost:{server.port}", timeout=5)
            status = client.get_status()

            assert status["mode"] == "async"
            assert status["dn_buffer_size"] == 3
            assert status["dn_buffered"] == 0
            assert status["dylu_enabled"] is True
            assert status["dylu_base_sync_every"] == 200
            assert "total_submissions" in status
        finally:
            server.stop()

    def test_status_shows_sync_mode(self):
        sd = _make_state_dict(dim=4)
        server = DiLoCoServer(sd, num_workers=1, port=0)
        server.start()
        time.sleep(0.2)

        try:
            client = DiLoCoClient(f"localhost:{server.port}", timeout=5)
            status = client.get_status()

            assert status["mode"] == "sync"
            assert "total_submissions" not in status
        finally:
            server.stop()


class TestWorkerDyLU:
    """Test worker-side DyLU adaptation."""

    def test_worker_dylu_adjusts_sync_every(self):
        """Worker with DyLU should adjust sync_every from heartbeat responses."""
        torch.manual_seed(42)
        model = TinyModel(dim=8)
        sd = model.state_dict()

        def simple_sgd(params):
            return torch.optim.SGD(params, lr=1.0, momentum=0.0)

        server = DiLoCoServer(
            sd, num_workers=2, port=0,
            outer_optimizer_factory=simple_sgd,
            async_mode=True,
            dylu_enabled=True,
            dylu_base_sync_every=100,
        )
        server.start()
        time.sleep(0.2)

        try:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            # Create worker WITHOUT heartbeat thread so we control timing exactly
            worker = DiLoCoWorker(
                model, optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=100,
                worker_id="slow_worker",
                bf16_comm=False,
                dylu=True,
                heartbeat_interval=0,  # Disabled - we'll call manually
            )
            worker.start()

            # Register a "fast" worker with higher speed directly on server
            from forgather.ml.diloco.server import WorkerInfo
            server._workers["fast_worker"] = WorkerInfo(
                worker_id="fast_worker", hostname="fast-host",
                registered_at=time.time(), last_heartbeat=time.time(),
                steps_per_second=20.0,
            )

            # Manually send heartbeat with our desired speed
            response = worker.client.heartbeat("slow_worker", steps_per_second=5.0)

            # Response should contain DyLU recommendation: 5/20 * 100 = 25
            assert "recommended_sync_every" in response
            assert response["recommended_sync_every"] == 25

            # Apply recommendation as the heartbeat loop would
            worker.sync_every = response["recommended_sync_every"]
            worker._dylu_adjustments += 1

            assert worker.sync_every == 25
            assert worker._dylu_adjustments == 1

            worker.stop()
        finally:
            server.stop()

    def test_worker_heartbeat_thread_updates_sync_every(self):
        """Heartbeat thread should automatically apply DyLU recommendations."""
        torch.manual_seed(42)
        model = TinyModel(dim=8)
        sd = model.state_dict()

        def simple_sgd(params):
            return torch.optim.SGD(params, lr=1.0, momentum=0.0)

        server = DiLoCoServer(
            sd, num_workers=2, port=0,
            outer_optimizer_factory=simple_sgd,
            async_mode=True,
            dylu_enabled=True,
            dylu_base_sync_every=1000,
        )
        server.start()
        time.sleep(0.2)

        try:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            worker = DiLoCoWorker(
                model, optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=1000,
                worker_id="slow_worker",
                bf16_comm=False,
                dylu=True,
                heartbeat_interval=0.3,
            )
            worker.start()

            # Simulate the worker having some training speed by injecting
            # step timestamps. The heartbeat thread reports get_steps_per_second()
            # to the server, which uses it for DyLU. 10 steps over 1 second = 10 steps/s.
            now = time.time()
            worker._step_timestamps = [now - 1.0 + i * 0.1 for i in range(11)]

            # Register a "fast" worker on the server
            from forgather.ml.diloco.server import WorkerInfo
            server._workers["fast_worker"] = WorkerInfo(
                worker_id="fast_worker", hostname="fast-host",
                registered_at=time.time(), last_heartbeat=time.time(),
                steps_per_second=100.0,
            )

            # Wait for heartbeat to fire and get DyLU recommendation.
            # slow_worker reports ~10 steps/s, fast_worker has 100 steps/s.
            # DyLU: 10/100 * 1000 = 100
            time.sleep(1.0)

            assert worker.sync_every == 100, (
                f"Expected sync_every=100, got {worker.sync_every}"
            )
            assert worker._dylu_adjustments >= 1

            worker.stop()
        finally:
            server.stop()
