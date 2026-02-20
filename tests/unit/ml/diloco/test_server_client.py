"""Tests for DiLoCo server-client communication."""

import threading
import time

import pytest
import torch

from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.server import DiLoCoServer


def _make_state_dict(dim=8, num_layers=2, seed=42):
    torch.manual_seed(seed)
    return {
        f"layer{i}.weight": torch.randn(dim, dim)
        for i in range(num_layers)
    }


@pytest.fixture
def server_and_client():
    """Create a server with 1 worker and a connected client."""
    sd = _make_state_dict()

    def simple_sgd(params):
        return torch.optim.SGD(params, lr=1.0, momentum=0.0)

    server = DiLoCoServer(sd, num_workers=1, port=0, outer_optimizer_factory=simple_sgd)
    server.start()

    time.sleep(0.2)  # Let server start

    client = DiLoCoClient(f"localhost:{server.port}", timeout=10)

    yield server, client, sd

    server.stop()


@pytest.fixture
def two_worker_server():
    """Create a server expecting 2 workers."""
    sd = _make_state_dict()

    def simple_sgd(params):
        return torch.optim.SGD(params, lr=1.0, momentum=0.0)

    server = DiLoCoServer(sd, num_workers=2, port=0, outer_optimizer_factory=simple_sgd)
    server.start()

    time.sleep(0.2)

    client0 = DiLoCoClient(f"localhost:{server.port}", timeout=10)
    client1 = DiLoCoClient(f"localhost:{server.port}", timeout=10)

    yield server, client0, client1, sd

    server.stop()


class TestRegistration:
    def test_register_and_get_params(self, server_and_client):
        server, client, sd = server_and_client

        params = client.register("worker_0")
        assert set(params.keys()) == set(sd.keys())
        for key in sd:
            assert torch.allclose(params[key], sd[key].float())

    def test_register_with_info(self, server_and_client):
        server, client, sd = server_and_client

        params = client.register("worker_0", {"hostname": "test-host", "num_gpus": 2})
        assert len(params) == len(sd)

        # Verify worker is registered on server
        assert "worker_0" in server._workers
        assert server._workers["worker_0"].hostname == "test-host"


class TestPseudogradientSubmission:
    def test_single_worker_sync(self, server_and_client):
        """Test single worker pseudo-gradient submission and param update."""
        server, client, sd = server_and_client

        client.register("worker_0")

        # Pseudo-grad: pretend worker moved params by -0.1
        pseudograds = {k: torch.full_like(v, 0.1) for k, v in sd.items()}

        new_params = client.submit_pseudogradients("worker_0", pseudograds)

        # With SGD(lr=1.0): new_param = old_param - 1.0 * pseudo_grad
        for key in sd:
            expected = sd[key].float() - 0.1
            assert torch.allclose(new_params[key], expected, atol=1e-5), (
                f"Key {key}: expected {expected.flatten()[:4]}, got {new_params[key].flatten()[:4]}"
            )

    def test_two_worker_sync(self, two_worker_server):
        """Test two workers synchronizing pseudo-gradients."""
        server, client0, client1, sd = two_worker_server

        client0.register("worker_0")
        client1.register("worker_1")

        # Worker 0: pseudo_grad = 0.2
        pg0 = {k: torch.full_like(v, 0.2) for k, v in sd.items()}
        # Worker 1: pseudo_grad = 0.4
        pg1 = {k: torch.full_like(v, 0.4) for k, v in sd.items()}

        # Submit from both workers in parallel (they block until both submit)
        results = [None, None]
        errors = [None, None]

        def submit(idx, client, worker_id, pg):
            try:
                results[idx] = client.submit_pseudogradients(worker_id, pg)
            except Exception as e:
                errors[idx] = e

        t0 = threading.Thread(target=submit, args=(0, client0, "worker_0", pg0))
        t1 = threading.Thread(target=submit, args=(1, client1, "worker_1", pg1))

        t0.start()
        t1.start()
        t0.join(timeout=10)
        t1.join(timeout=10)

        assert errors[0] is None, f"Worker 0 error: {errors[0]}"
        assert errors[1] is None, f"Worker 1 error: {errors[1]}"
        assert results[0] is not None
        assert results[1] is not None

        # Average pseudo_grad = (0.2 + 0.4) / 2 = 0.3
        # new_param = old_param - 1.0 * 0.3 = old_param - 0.3
        for key in sd:
            expected = sd[key].float() - 0.3
            assert torch.allclose(results[0][key], expected, atol=1e-5)
            # Both workers should get same params
            assert torch.allclose(results[0][key], results[1][key])

    def test_bf16_pseudogradients(self, server_and_client):
        """Test that bf16 pseudo-gradients work correctly."""
        server, client, sd = server_and_client

        client.register("worker_0")

        # Send bf16 pseudo-gradients
        pseudograds = {k: torch.full_like(v, 0.5).to(torch.bfloat16) for k, v in sd.items()}

        new_params = client.submit_pseudogradients("worker_0", pseudograds)

        # Server should still produce float32 results
        for key in new_params:
            assert new_params[key].dtype == torch.float32


class TestStatusAndHeartbeat:
    def test_status(self, server_and_client):
        server, client, sd = server_and_client

        status = client.get_status()
        assert status["status"] == "running"
        assert status["num_workers"] == 1
        assert status["sync_round"] == 0

    def test_heartbeat(self, server_and_client):
        server, client, sd = server_and_client

        client.register("worker_0")
        result = client.heartbeat("worker_0", steps_per_second=3.5)

        assert result["status"] == "ok"
        assert result["sync_round"] == 0

        # Check server recorded the speed
        assert server._workers["worker_0"].steps_per_second == 3.5


class TestGetGlobalParams:
    def test_get_global_params(self, server_and_client):
        server, client, sd = server_and_client

        params = client.get_global_params()
        for key in sd:
            assert torch.allclose(params[key], sd[key].float())


class TestDeregistration:
    def test_deregister(self, server_and_client):
        server, client, sd = server_and_client

        client.register("worker_0")
        assert "worker_0" in server._workers

        client.deregister("worker_0")
        assert "worker_0" not in server._workers


class TestMultipleRounds:
    def test_multiple_sync_rounds(self, server_and_client):
        """Test multiple sequential sync rounds with a single worker."""
        server, client, sd = server_and_client

        client.register("worker_0")

        current_params = {k: v.float().clone() for k, v in sd.items()}

        for round_num in range(3):
            # Pseudo-gradient of 0.1 each round
            pg = {k: torch.full_like(v, 0.1) for k, v in current_params.items()}
            new_params = client.submit_pseudogradients("worker_0", pg)

            # Verify: param = param - lr * grad = param - 1.0 * 0.1 = param - 0.1
            for key in current_params:
                expected = current_params[key] - 0.1
                assert torch.allclose(new_params[key], expected, atol=1e-5)

            current_params = {k: v.clone() for k, v in new_params.items()}

        assert server._sync_round == 3
