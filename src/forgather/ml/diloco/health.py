"""
Health monitoring for DiLoCo workers.

Runs a background thread on the server that periodically checks worker
heartbeat timestamps and evicts workers that have gone silent. When a dead
worker is evicted in sync mode, the synchronization barrier is re-evaluated
so remaining workers can proceed.

Usage:
    monitor = HealthMonitor(server, heartbeat_timeout=120.0, check_interval=30.0)
    monitor.start()
    # ... server runs ...
    monitor.stop()
"""

import logging
import threading
import time
from typing import List

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Background health checker for DiLoCo worker liveness.

    Periodically scans registered workers' last heartbeat timestamps. Workers
    whose heartbeats are older than ``heartbeat_timeout`` are considered dead
    and evicted from the server via ``server._handle_worker_death()``.

    The check runs every ``check_interval`` seconds. A reasonable default is
    ``check_interval = heartbeat_timeout / 3`` so that a dead worker is
    detected within one timeout period.

    Args:
        server: The DiLoCoServer instance to monitor.
        heartbeat_timeout: Seconds since last heartbeat before a worker is
            considered dead. Should be at least 2-3x the worker's heartbeat
            interval to allow for network delays and large model transfers.
        check_interval: Seconds between health checks. Defaults to
            ``heartbeat_timeout / 3``.
    """

    def __init__(
        self,
        server,
        heartbeat_timeout: float = 120.0,
        check_interval: float = 0.0,
    ):
        self._server = server
        self.heartbeat_timeout = heartbeat_timeout
        self.check_interval = check_interval if check_interval > 0 else heartbeat_timeout / 3
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        """Start the health monitoring thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="diloco-health-monitor"
        )
        self._thread.start()
        logger.info(
            f"Health monitor started: timeout={self.heartbeat_timeout}s, "
            f"check_interval={self.check_interval:.1f}s"
        )

    def stop(self):
        """Stop the health monitoring thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Health monitor stopped")

    def _monitor_loop(self):
        """Background loop that periodically checks worker health."""
        while not self._stop_event.wait(timeout=self.check_interval):
            try:
                self._check_workers()
            except Exception as e:
                logger.error(f"Health monitor error: {e}", exc_info=True)

    def _check_workers(self):
        """Check all registered workers and evict dead ones."""
        now = time.time()
        dead_workers: List[str] = []

        with self._server._workers_lock:
            for wid, info in self._server._workers.items():
                elapsed = now - info.last_heartbeat
                if elapsed > self.heartbeat_timeout:
                    dead_workers.append(wid)
                    logger.warning(
                        f"Worker {wid} heartbeat timeout: "
                        f"{elapsed:.0f}s since last heartbeat "
                        f"(timeout={self.heartbeat_timeout}s)"
                    )

        for wid in dead_workers:
            self._server._handle_worker_death(wid)

    @property
    def is_running(self) -> bool:
        """Whether the health monitor thread is alive."""
        return self._thread is not None and self._thread.is_alive()
