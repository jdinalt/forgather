"""Tests for DiLoCo server-client communication."""

import os
import threading
import time

import pytest
import torch

from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.server import DiLoCoServer

from .conftest import make_initial_checkpoint


def _make_state_dict(dim=8, num_layers=2, seed=42):
    torch.manual_seed(seed)
    return {f"layer{i}.weight": torch.randn(dim, dim) for i in range(num_layers)}


@pytest.fixture
def server_and_client(tmp_path):
    """Create a server with 1 worker and a connected client."""
    sd = _make_state_dict()
    ckpt = make_initial_checkpoint(sd, tmp_path)

    def simple_sgd(params):
        return torch.optim.SGD(params, lr=1.0, momentum=0.0)

    server = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=1,
        port=0,
        outer_optimizer_factory=simple_sgd,
    )
    server.start()

    time.sleep(0.2)  # Let server start

    client = DiLoCoClient(f"localhost:{server.port}", timeout=10)

    yield server, client, sd

    server.stop()


@pytest.fixture
def two_worker_server(tmp_path):
    """Create a server expecting 2 workers."""
    sd = _make_state_dict()
    ckpt = make_initial_checkpoint(sd, tmp_path)

    def simple_sgd(params):
        return torch.optim.SGD(params, lr=1.0, momentum=0.0)

    server = DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        num_workers=2,
        port=0,
        outer_optimizer_factory=simple_sgd,
    )
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

    def test_output_dir_stored_and_exposed_in_status(self, server_and_client):
        """A reported output_dir is stored on the WorkerInfo and surfaced
        per-worker in /status, so the webui can correlate a renamed worker
        to its job by output_dir (issue #103)."""
        server, client, sd = server_and_client

        client.register("worker_0", {"output_dir": "/runs/tinyv2_w0"})
        assert server._workers["worker_0"].output_dir == "/runs/tinyv2_w0"

        status = client.get_status()
        assert status["workers"]["worker_0"]["output_dir"] == "/runs/tinyv2_w0"

    def test_output_dir_none_when_not_reported(self, server_and_client):
        """Workers that don't report output_dir (older clients) store None
        and /status carries null — no correlation, but no error."""
        server, client, sd = server_and_client

        client.register("worker_0", {"hostname": "test-host"})
        assert server._workers["worker_0"].output_dir is None
        status = client.get_status()
        assert status["workers"]["worker_0"]["output_dir"] is None

    def test_known_workers_endpoint_tracks_running(self, server_and_client):
        """A registered worker appears in /known_workers as running; after
        it deregisters it stays in the roster but flips to not-running, so
        the webui can offer it for checkpoint-resuming relaunch (#103)."""
        server, client, sd = server_and_client

        client.register("worker_0", {"output_dir": "/runs/model_worker_0"})
        by_id = {w["worker_id"]: w for w in client.get_known_workers()["workers"]}
        assert by_id["worker_0"]["running"] is True
        assert by_id["worker_0"]["output_dir"] == "/runs/model_worker_0"

        client.deregister("worker_0")
        by_id = {w["worker_id"]: w for w in client.get_known_workers()["workers"]}
        assert "worker_0" in by_id, "deregistered worker dropped from roster"
        assert by_id["worker_0"]["running"] is False
        # output_dir is retained so a relaunch knows which checkpoint dir.
        assert by_id["worker_0"]["output_dir"] == "/runs/model_worker_0"


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
            assert torch.allclose(
                new_params[key], expected, atol=1e-5
            ), f"Key {key}: expected {expected.flatten()[:4]}, got {new_params[key].flatten()[:4]}"

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
        pseudograds = {
            k: torch.full_like(v, 0.5).to(torch.bfloat16) for k, v in sd.items()
        }

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


class TestUnifiedStats:
    """End-to-end: worker reports stats on the heartbeat, server aggregates
    them and exposes them via /status (and logs the stream)."""

    def test_heartbeat_stats_folded_into_status(self, two_worker_server):
        server, c0, c1, sd = two_worker_server
        c0.register("worker_0")
        c1.register("worker_1")

        c0.heartbeat(
            "worker_0",
            steps_per_second=2.0,
            stats={
                "tokens_total": 1000,
                "step_total": 10,
                "tok_per_sec": 100.0,
                "mfu": 0.2,
                "loss": 3.0,
                "tokens_window": 100,
            },
        )
        c1.heartbeat(
            "worker_1",
            steps_per_second=2.0,
            stats={
                "tokens_total": 500,
                "step_total": 5,
                "tok_per_sec": 150.0,
                "mfu": 0.3,
                "loss": 3.0,
                "tokens_window": 100,
            },
        )

        status = c0.get_status()
        agg = status["aggregate_stats"]
        assert agg["total_tokens"] == 1500
        assert agg["total_steps"] == 15
        assert agg["tok_per_sec"] == pytest.approx(250.0)
        # MFU is a token-weighted mean (equal tokens_window here): (0.2+0.3)/2.
        assert agg["mfu"] == pytest.approx(0.25)
        assert agg["train_loss"] == pytest.approx(3.0)
        assert agg["num_reporting"] == 2

        # Per-worker snapshot is exposed too.
        assert status["workers"]["worker_0"]["stats"]["tokens_total"] == 1000

    def test_heartbeat_without_stats_is_fine(self, server_and_client):
        server, client, sd = server_and_client
        client.register("worker_0")
        client.heartbeat("worker_0", steps_per_second=1.0)
        agg = client.get_status()["aggregate_stats"]
        assert agg["total_tokens"] == 0
        assert agg["train_loss"] is None

    def test_nonfinite_stats_do_not_break_status(self, server_and_client):
        # A buggy/hostile worker reporting NaN/Inf must not poison the
        # aggregate or produce a /status body the JSON client can't parse.
        server, client, sd = server_and_client
        client.register("worker_0")
        client.heartbeat(
            "worker_0",
            stats={
                "loss": float("nan"),
                "tok_per_sec": float("inf"),
                "tokens_total": 1000,
                "tokens_window": 100,
            },
        )
        agg = client.get_status()["aggregate_stats"]
        assert agg["train_loss"] is None  # NaN dropped, EMA stays clean
        assert agg["tok_per_sec"] == 0.0  # Inf dropped
        assert agg["total_tokens"] == 1000  # the valid field still counted

    def test_eval_loss_reported(self, server_and_client):
        server, client, sd = server_and_client
        client.register("worker_0")
        client.heartbeat(
            "worker_0",
            stats={"eval_loss": 2.5, "eval_step": 42, "tokens_window": 100},
        )
        agg = client.get_status()["aggregate_stats"]
        assert agg["eval_loss"] == pytest.approx(2.5)
        assert agg["eval_step"] == 42

    def test_stats_log_written(self, server_and_client):
        server, client, sd = server_and_client
        client.register("worker_0")
        client.heartbeat(
            "worker_0",
            stats={"tokens_total": 1000, "step_total": 10, "loss": 3.0},
        )
        log_path = os.path.join(server.output_dir, "logs", "diloco_server_stats.json")
        # The log record is written after the heartbeat response is sent (file
        # IO is kept off the heartbeat latency path), so poll briefly for it.
        deadline = time.time() + 5.0
        while not os.path.isfile(log_path) and time.time() < deadline:
            time.sleep(0.05)
        assert os.path.isfile(log_path)
        deadline = time.time() + 5.0
        content = ""
        while '"global_step": 10' not in content and time.time() < deadline:
            with open(log_path) as f:
                content = f.read()
            if '"global_step": 10' not in content:
                time.sleep(0.05)
        # At least one record carrying the aggregate total step.
        assert '"global_step": 10' in content

    def test_stats_history_endpoint(self, server_and_client):
        server, client, sd = server_and_client
        client.register("worker_0")
        # Empty before anything is logged.
        assert client.get_stats_history()["records"] == []

        client.heartbeat(
            "worker_0",
            stats={"tokens_total": 1000, "step_total": 10, "loss": 3.0},
        )
        # The record is written after the heartbeat response — poll for it.
        deadline = time.time() + 5.0
        hist = client.get_stats_history()
        while not hist["records"] and time.time() < deadline:
            time.sleep(0.05)
            hist = client.get_stats_history()
        assert hist["count"] >= 1
        rec = hist["records"][-1]
        assert rec["global_step"] == 10
        assert rec["total_tokens"] == 1000

    def test_stale_log_file_does_not_break_logging(self, tmp_path):
        # Regression: a stats log left by a prior run in the same output_dir
        # must not make the fresh "x"-mode open raise FileExistsError and
        # silently disable logging (empty history → no webui plot).
        # Pre-create a stale (closed, empty) stats log.
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        (logs_dir / "diloco_server_stats.json").write_text("[\n\n]")

        sd = _make_state_dict()
        ckpt = make_initial_checkpoint(sd, tmp_path)
        server = DiLoCoServer(
            output_dir=str(tmp_path),
            from_checkpoint=ckpt,
            num_workers=1,
            port=0,
            outer_optimizer_factory=lambda p: torch.optim.SGD(p, lr=1.0),
        )
        server.start()
        time.sleep(0.2)
        try:
            client = DiLoCoClient(f"localhost:{server.port}", timeout=10)
            client.register("worker_0")
            client.heartbeat(
                "worker_0",
                stats={"tokens_total": 1000, "step_total": 10, "loss": 3.0},
            )
            deadline = time.time() + 5.0
            hist = client.get_stats_history()
            while not hist["records"] and time.time() < deadline:
                time.sleep(0.05)
                hist = client.get_stats_history()
            assert hist["records"], "logging silently disabled by stale log file"
            assert hist["records"][-1]["global_step"] == 10
        finally:
            server.stop()

    def test_evicted_worker_drops_from_throughput(self, two_worker_server):
        server, c0, c1, sd = two_worker_server
        c0.register("worker_0")
        c1.register("worker_1")
        c0.heartbeat("worker_0", stats={"tok_per_sec": 100.0})
        c1.heartbeat("worker_1", stats={"tok_per_sec": 150.0})
        assert c0.get_status()["aggregate_stats"]["tok_per_sec"] == pytest.approx(250.0)

        # Evict worker_1; its live-gauge contribution drops out.
        server._handle_worker_death("worker_1")
        assert c0.get_status()["aggregate_stats"]["tok_per_sec"] == pytest.approx(100.0)


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
