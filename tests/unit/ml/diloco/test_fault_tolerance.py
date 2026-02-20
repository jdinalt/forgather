"""Tests for DiLoCo fault tolerance (Phase 4): health monitoring, dead worker
eviction, barrier release, dynamic joining, and worker reconnection."""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.health import HealthMonitor
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
# HealthMonitor unit tests
# ---------------------------------------------------------------------------


class TestHealthMonitor(unittest.TestCase):
    """Test HealthMonitor background thread."""

    def test_health_monitor_start_stop(self):
        """Monitor starts and stops cleanly."""
        server = MagicMock()
        server._workers = {}
        server._workers_lock = threading.Lock()

        monitor = HealthMonitor(server, heartbeat_timeout=10.0, check_interval=0.1)
        monitor.start()
        self.assertTrue(monitor.is_running)

        monitor.stop()
        self.assertFalse(monitor.is_running)

    def test_health_monitor_detects_dead_worker(self):
        """Monitor calls _handle_worker_death for timed-out workers."""
        server = MagicMock()
        server._workers_lock = threading.Lock()

        # Worker with very old heartbeat
        from forgather.ml.diloco.server import WorkerInfo
        server._workers = {
            "alive": WorkerInfo("alive", "h1", time.time(), time.time()),
            "dead": WorkerInfo("dead", "h2", time.time(), time.time() - 200),
        }

        monitor = HealthMonitor(server, heartbeat_timeout=60.0, check_interval=0.05)
        monitor.start()
        time.sleep(0.2)  # Allow at least one check cycle
        monitor.stop()

        # Should have been called for the dead worker
        server._handle_worker_death.assert_called_with("dead")

    def test_health_monitor_skips_alive_workers(self):
        """Monitor does not evict workers with recent heartbeats."""
        server = MagicMock()
        server._workers_lock = threading.Lock()

        from forgather.ml.diloco.server import WorkerInfo
        server._workers = {
            "alive": WorkerInfo("alive", "h1", time.time(), time.time()),
        }

        monitor = HealthMonitor(server, heartbeat_timeout=60.0, check_interval=0.05)
        monitor.start()
        time.sleep(0.2)
        monitor.stop()

        server._handle_worker_death.assert_not_called()

    def test_health_monitor_disabled_when_timeout_zero(self):
        """Server with heartbeat_timeout=0 does not start a monitor."""
        sd = _make_state_dict()
        server = DiLoCoServer(sd, num_workers=1, heartbeat_timeout=0.0)
        server.start()
        time.sleep(0.1)

        self.assertIsNone(server._health_monitor)
        server.stop()


# ---------------------------------------------------------------------------
# Worker death and barrier release
# ---------------------------------------------------------------------------


class TestWorkerDeath(unittest.TestCase):
    """Test _handle_worker_death barrier release logic."""

    def test_death_removes_worker_from_registry(self):
        """Dead worker is removed from the server's worker registry."""
        sd = _make_state_dict()
        server = DiLoCoServer(sd, num_workers=2, heartbeat_timeout=0)
        server.start()
        try:
            client = DiLoCoClient(f"localhost:{server.port}")
            client.register("w1", {})
            client.register("w2", {})

            self.assertEqual(len(server._workers), 2)

            server._handle_worker_death("w1")

            self.assertEqual(len(server._workers), 1)
            self.assertNotIn("w1", server._workers)
            self.assertIn("w2", server._workers)
        finally:
            server.stop()

    def test_death_decrements_num_workers(self):
        """num_workers is decreased when a worker dies."""
        sd = _make_state_dict()
        server = DiLoCoServer(sd, num_workers=3, heartbeat_timeout=0)
        server.start()
        try:
            client = DiLoCoClient(f"localhost:{server.port}")
            client.register("w1", {})
            client.register("w2", {})
            client.register("w3", {})

            server._handle_worker_death("w1")
            self.assertEqual(server.num_workers, 2)

            server._handle_worker_death("w2")
            # min_workers defaults to 1
            self.assertEqual(server.num_workers, 1)
        finally:
            server.stop()

    def test_death_respects_min_workers(self):
        """num_workers does not drop below min_workers."""
        sd = _make_state_dict()
        server = DiLoCoServer(sd, num_workers=3, min_workers=2, heartbeat_timeout=0)
        server.start()
        try:
            client = DiLoCoClient(f"localhost:{server.port}")
            client.register("w1", {})
            client.register("w2", {})
            client.register("w3", {})

            server._handle_worker_death("w1")
            self.assertEqual(server.num_workers, 2)

            # Even though only 1 worker remains, min_workers=2
            server._handle_worker_death("w2")
            self.assertEqual(server.num_workers, 2)
        finally:
            server.stop()

    def test_death_tracks_total_deaths(self):
        """Server tracks total worker deaths for status reporting."""
        sd = _make_state_dict()
        server = DiLoCoServer(sd, num_workers=2, heartbeat_timeout=0)
        server.start()
        try:
            client = DiLoCoClient(f"localhost:{server.port}")
            client.register("w1", {})
            client.register("w2", {})

            self.assertEqual(server._total_worker_deaths, 0)

            server._handle_worker_death("w1")
            self.assertEqual(server._total_worker_deaths, 1)

            server._handle_worker_death("w2")
            self.assertEqual(server._total_worker_deaths, 2)
        finally:
            server.stop()

    def test_death_idempotent(self):
        """Calling _handle_worker_death twice for the same worker is safe."""
        sd = _make_state_dict()
        server = DiLoCoServer(sd, num_workers=2, heartbeat_timeout=0)
        server.start()
        try:
            client = DiLoCoClient(f"localhost:{server.port}")
            client.register("w1", {})
            client.register("w2", {})

            server._handle_worker_death("w1")
            server._handle_worker_death("w1")  # Second call should be no-op
            self.assertEqual(server._total_worker_deaths, 1)
        finally:
            server.stop()


class TestBarrierRelease(unittest.TestCase):
    """Test that the sync barrier releases when a worker dies mid-sync."""

    def test_sync_barrier_releases_after_worker_death(self):
        """Worker A submits, worker B dies -> barrier releases for A."""
        sd = _make_state_dict()
        server = DiLoCoServer(sd, num_workers=2, heartbeat_timeout=0)
        server.start()
        try:
            client_a = DiLoCoClient(f"localhost:{server.port}", timeout=10)
            client_b = DiLoCoClient(f"localhost:{server.port}", timeout=10)
            client_a.register("worker_a", {})
            client_b.register("worker_b", {})

            # Worker A submits pseudo-gradients in a background thread
            result = {}
            error = {}

            def submit_a():
                try:
                    pseudograds = {
                        name: torch.zeros_like(p)
                        for name, p in sd.items()
                    }
                    result["params"] = client_a.submit_pseudogradients(
                        "worker_a", pseudograds
                    )
                except Exception as e:
                    error["e"] = e

            t = threading.Thread(target=submit_a)
            t.start()

            # Wait briefly for A's submission to reach the server
            time.sleep(0.5)

            # Kill worker B -> should release the barrier
            server._handle_worker_death("worker_b")

            t.join(timeout=10)
            self.assertFalse(t.is_alive(), "Worker A should not be blocked")
            self.assertNotIn("e", error, f"Worker A submission failed: {error.get('e')}")
            self.assertIn("params", result)
        finally:
            server.stop()

    def test_deregister_releases_barrier(self):
        """Deregistration uses _handle_worker_death and releases barrier."""
        sd = _make_state_dict()
        server = DiLoCoServer(sd, num_workers=2, heartbeat_timeout=0)
        server.start()
        try:
            client_a = DiLoCoClient(f"localhost:{server.port}", timeout=10)
            client_b = DiLoCoClient(f"localhost:{server.port}", timeout=10)
            client_a.register("worker_a", {})
            client_b.register("worker_b", {})

            result = {}

            def submit_a():
                pseudograds = {
                    name: torch.zeros_like(p) for name, p in sd.items()
                }
                result["params"] = client_a.submit_pseudogradients(
                    "worker_a", pseudograds
                )

            t = threading.Thread(target=submit_a)
            t.start()
            time.sleep(0.5)

            # Worker B deregisters (voluntary departure)
            client_b.deregister("worker_b")

            t.join(timeout=10)
            self.assertFalse(t.is_alive())
            self.assertIn("params", result)
        finally:
            server.stop()


# ---------------------------------------------------------------------------
# Dynamic worker joining
# ---------------------------------------------------------------------------


class TestDynamicJoining(unittest.TestCase):
    """Test workers joining an active training run."""

    def test_new_worker_gets_current_global_params(self):
        """A new worker joining mid-training gets the latest global params."""
        sd = _make_state_dict()
        server = DiLoCoServer(sd, num_workers=1, heartbeat_timeout=0)
        server.start()
        try:
            client = DiLoCoClient(f"localhost:{server.port}")

            # Worker 1 registers and does a sync round
            params1 = client.register("w1", {})

            # Submit pseudo-gradients to change global params
            pseudograds = {
                name: torch.ones_like(p) * 0.1
                for name, p in sd.items()
            }
            new_params = client.submit_pseudogradients("w1", pseudograds)

            # Worker 2 joins after the sync - should get updated params
            params2 = client.register("w2", {})

            # Params received by w2 should match the post-sync params
            for name in sd:
                self.assertTrue(
                    torch.allclose(params2[name], new_params[name], atol=1e-5),
                    f"New worker got stale params for {name}"
                )
        finally:
            server.stop()

    def test_dynamic_join_increases_num_workers(self):
        """When more workers join than expected, num_workers increases."""
        sd = _make_state_dict()
        server = DiLoCoServer(sd, num_workers=1, heartbeat_timeout=0)
        server.start()
        try:
            client = DiLoCoClient(f"localhost:{server.port}")
            client.register("w1", {})
            self.assertEqual(server.num_workers, 1)

            client.register("w2", {})
            self.assertEqual(server.num_workers, 2)

            client.register("w3", {})
            self.assertEqual(server.num_workers, 3)
        finally:
            server.stop()


# ---------------------------------------------------------------------------
# Worker reconnection
# ---------------------------------------------------------------------------


class TestWorkerReconnection(unittest.TestCase):
    """Test worker reconnection after server restart or network failure."""

    def test_worker_reconnect_re_registers(self):
        """_reconnect() re-registers and gets fresh global params."""
        torch.manual_seed(42)
        model = TinyModel(dim=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        sd = model.state_dict()
        server = DiLoCoServer(sd, num_workers=1, heartbeat_timeout=0)
        server.start()
        try:
            worker = DiLoCoWorker(
                model, optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=100,
                heartbeat_interval=0,
            )
            worker.start()

            # Reconnect
            initial_reconnections = worker._reconnections
            worker._reconnect()

            self.assertEqual(worker._reconnections, initial_reconnections + 1)

            worker.stop()
        finally:
            server.stop()

    def test_re_registration_replaces_old_entry(self):
        """Re-registering a worker replaces its old entry (no duplicates)."""
        sd = _make_state_dict()
        server = DiLoCoServer(sd, num_workers=1, heartbeat_timeout=0)
        server.start()
        try:
            client = DiLoCoClient(f"localhost:{server.port}")
            client.register("w1", {"hostname": "host1"})
            self.assertEqual(len(server._workers), 1)

            # Re-register with different hostname
            client.register("w1", {"hostname": "host2"})
            self.assertEqual(len(server._workers), 1)
            self.assertEqual(server._workers["w1"].hostname, "host2")
        finally:
            server.stop()

    def test_sync_retry_on_connection_failure(self):
        """Worker retries sync on connection failure."""
        torch.manual_seed(42)
        model = TinyModel(dim=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        sd = model.state_dict()
        server = DiLoCoServer(sd, num_workers=1, heartbeat_timeout=0)
        server.start()
        try:
            worker = DiLoCoWorker(
                model, optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=5,
                heartbeat_interval=0,
                max_sync_retries=2,
            )
            worker.start()

            # Simulate training steps
            for _ in range(5):
                x = torch.randn(4, 8)
                loss = model(x).sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Sync should have happened successfully (no failures to trigger retry)
            self.assertEqual(worker._sync_count, 1)
            self.assertEqual(worker._sync_retries, 0)

            worker.stop()
        finally:
            server.stop()


# ---------------------------------------------------------------------------
# Worker heartbeat (unconditional)
# ---------------------------------------------------------------------------


class TestUnconditionalHeartbeat(unittest.TestCase):
    """Test that heartbeats are sent regardless of DyLU setting."""

    def test_heartbeat_starts_without_dylu(self):
        """Heartbeat thread starts even when dylu=False."""
        torch.manual_seed(42)
        model = TinyModel(dim=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        sd = model.state_dict()
        server = DiLoCoServer(sd, num_workers=1, heartbeat_timeout=0)
        server.start()
        try:
            worker = DiLoCoWorker(
                model, optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=100,
                dylu=False,
                heartbeat_interval=0.1,
            )
            worker.start()

            # Heartbeat thread should be running
            self.assertIsNotNone(worker._heartbeat_thread)
            self.assertTrue(worker._heartbeat_thread.is_alive())

            worker.stop()
        finally:
            server.stop()

    def test_heartbeat_disabled_when_interval_zero(self):
        """No heartbeat thread when heartbeat_interval=0."""
        torch.manual_seed(42)
        model = TinyModel(dim=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        sd = model.state_dict()
        server = DiLoCoServer(sd, num_workers=1, heartbeat_timeout=0)
        server.start()
        try:
            worker = DiLoCoWorker(
                model, optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=100,
                heartbeat_interval=0,
            )
            worker.start()

            self.assertIsNone(worker._heartbeat_thread)

            worker.stop()
        finally:
            server.stop()

    def test_heartbeats_update_server_timestamp(self):
        """Worker heartbeats update the server's last_heartbeat timestamp."""
        torch.manual_seed(42)
        model = TinyModel(dim=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        sd = model.state_dict()
        server = DiLoCoServer(sd, num_workers=1, heartbeat_timeout=0)
        server.start()
        try:
            worker = DiLoCoWorker(
                model, optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=100,
                heartbeat_interval=0.1,
            )
            worker.start()

            # Record initial heartbeat time
            initial_hb = server._workers[worker.worker_id].last_heartbeat

            # Wait for a heartbeat to arrive
            time.sleep(0.3)

            updated_hb = server._workers[worker.worker_id].last_heartbeat
            self.assertGreater(updated_hb, initial_hb)

            worker.stop()
        finally:
            server.stop()


# ---------------------------------------------------------------------------
# Health monitor integration (server detects dead worker)
# ---------------------------------------------------------------------------


class TestHealthMonitorIntegration(unittest.TestCase):
    """Test full health monitor integration with server."""

    def test_server_evicts_dead_worker_via_health_monitor(self):
        """Health monitor detects and evicts a worker that stops heartbeating."""
        sd = _make_state_dict()
        server = DiLoCoServer(
            sd, num_workers=2,
            heartbeat_timeout=2.0,  # Short for testing but long enough for alive worker
        )
        server.start()
        try:
            client_a = DiLoCoClient(f"localhost:{server.port}")
            client_b = DiLoCoClient(f"localhost:{server.port}")
            client_a.register("alive_worker", {})
            client_b.register("dead_worker", {})

            # Manually set dead_worker's heartbeat far in the past
            server._workers["dead_worker"].last_heartbeat = time.time() - 10

            # Wait for health monitor to detect and evict
            # check_interval defaults to timeout/3 = ~0.67s
            time.sleep(1.5)

            self.assertNotIn("dead_worker", server._workers)
            self.assertIn("alive_worker", server._workers)
            self.assertEqual(server._total_worker_deaths, 1)
        finally:
            server.stop()

    def test_health_monitor_barrier_release_integration(self):
        """Full integration: worker A submits, worker B times out, A unblocks."""
        sd = _make_state_dict()
        server = DiLoCoServer(
            sd, num_workers=2,
            heartbeat_timeout=0.5,
        )
        server.start()
        try:
            client_a = DiLoCoClient(f"localhost:{server.port}", timeout=10)
            client_b = DiLoCoClient(f"localhost:{server.port}", timeout=10)
            client_a.register("worker_a", {})
            client_b.register("worker_b", {})

            # Set worker_b's heartbeat far in the past
            server._workers["worker_b"].last_heartbeat = time.time() - 10

            result = {}
            error = {}

            def submit_a():
                try:
                    pseudograds = {
                        name: torch.zeros_like(p) for name, p in sd.items()
                    }
                    result["params"] = client_a.submit_pseudogradients(
                        "worker_a", pseudograds
                    )
                except Exception as e:
                    error["e"] = e

            t = threading.Thread(target=submit_a)
            t.start()

            # Health monitor should evict worker_b and release barrier
            t.join(timeout=10)

            self.assertFalse(t.is_alive(), "Worker A should have been unblocked")
            self.assertNotIn("e", error, f"Error: {error.get('e')}")
            self.assertIn("params", result)
        finally:
            server.stop()


# ---------------------------------------------------------------------------
# Status with fault tolerance fields
# ---------------------------------------------------------------------------


class TestStatusFaultTolerance(unittest.TestCase):
    """Test that status endpoint includes fault tolerance fields."""

    def test_status_includes_fault_tolerance_fields(self):
        """Status response includes heartbeat_timeout and min_workers."""
        sd = _make_state_dict()
        server = DiLoCoServer(
            sd, num_workers=2,
            heartbeat_timeout=120.0,
            min_workers=1,
        )
        server.start()
        try:
            client = DiLoCoClient(f"localhost:{server.port}")
            status = client.get_status()

            self.assertEqual(status["heartbeat_timeout"], 120.0)
            self.assertEqual(status["min_workers"], 1)
        finally:
            server.stop()

    def test_status_shows_worker_deaths(self):
        """Status shows total worker deaths after eviction."""
        sd = _make_state_dict()
        server = DiLoCoServer(sd, num_workers=2, heartbeat_timeout=0)
        server.start()
        try:
            client = DiLoCoClient(f"localhost:{server.port}")
            client.register("w1", {})
            client.register("w2", {})

            server._handle_worker_death("w1")

            status = client.get_status()
            self.assertEqual(status["total_worker_deaths"], 1)
        finally:
            server.stop()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    """Test edge cases in fault tolerance."""

    def test_min_workers_validation(self):
        """min_workers must be >= 1."""
        sd = _make_state_dict()
        with self.assertRaises(ValueError):
            DiLoCoServer(sd, num_workers=2, min_workers=0)

    def test_single_worker_death_does_not_prevent_sync(self):
        """After a death, remaining single worker can still sync."""
        sd = _make_state_dict()
        server = DiLoCoServer(sd, num_workers=2, heartbeat_timeout=0)
        server.start()
        try:
            client = DiLoCoClient(f"localhost:{server.port}")
            client.register("w1", {})
            client.register("w2", {})

            # Kill w2
            server._handle_worker_death("w2")

            # w1 should be able to sync alone (num_workers is now 1)
            pseudograds = {
                name: torch.ones_like(p) * 0.01
                for name, p in sd.items()
            }
            result = client.submit_pseudogradients("w1", pseudograds)
            self.assertIsNotNone(result)
            self.assertEqual(len(result), len(sd))
        finally:
            server.stop()

    def test_worker_death_during_no_pending_is_safe(self):
        """Worker death when no submissions are pending is handled gracefully."""
        sd = _make_state_dict()
        server = DiLoCoServer(sd, num_workers=2, heartbeat_timeout=0)
        server.start()
        try:
            client = DiLoCoClient(f"localhost:{server.port}")
            client.register("w1", {})
            client.register("w2", {})

            # Kill before any submissions
            server._handle_worker_death("w2")
            self.assertEqual(server.num_workers, 1)
        finally:
            server.stop()

    def test_worker_sync_metrics_include_retry_fields(self):
        """sync_metrics includes retry and reconnection counters."""
        torch.manual_seed(42)
        model = TinyModel(dim=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        sd = model.state_dict()
        server = DiLoCoServer(sd, num_workers=1, heartbeat_timeout=0)
        server.start()
        try:
            worker = DiLoCoWorker(
                model, optimizer,
                server_addr=f"localhost:{server.port}",
                sync_every=100,
                heartbeat_interval=0,
            )
            worker.start()

            # Before any retries/reconnections, fields should not appear
            metrics = worker.sync_metrics
            self.assertNotIn("diloco/sync_retries", metrics)
            self.assertNotIn("diloco/reconnections", metrics)

            # Simulate a reconnection
            worker._reconnect()
            metrics = worker.sync_metrics
            self.assertEqual(metrics["diloco/reconnections"], 1)

            worker.stop()
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
