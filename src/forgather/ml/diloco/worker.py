"""
DiLoCo Worker - Composable wrapper for any trainer/optimizer.

Wraps around a model and optimizer to periodically synchronize with a DiLoCo
parameter server. Uses optimizer post-step hooks so it works transparently
with any existing Forgather trainer (single GPU, DDP, pipeline).

Supports both synchronous and asynchronous DiLoCo modes (determined by the
server). In async mode, the worker can optionally adapt its sync frequency
dynamically via DyLU (Dynamic Local Updates) based on server recommendations.

Usage:
    model = MyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    with DiLoCoWorker(model, optimizer, server_addr="host:8512", sync_every=500) as diloco:
        trainer.train()  # Normal training - DiLoCo syncs happen automatically

For pipeline-parallel workers, only rank 0 communicates with the server.
Rank 0 gathers/scatters parameters from/to other pipeline ranks.
"""

import logging
import platform
import threading
import time
import uuid
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .client import DiLoCoClient

logger = logging.getLogger(__name__)


class DiLoCoWorker:
    """
    Composable DiLoCo wrapper that hooks into any optimizer.

    On every optimizer.step(), a post-step hook increments a local step counter.
    When sync_every steps have been taken, the worker:
    1. Computes pseudo-gradients: global_params - local_params
    2. Optionally casts to bfloat16 for bandwidth reduction
    3. Submits to the server (blocks in sync mode, returns immediately in async)
    4. Receives updated global params and loads them into the model

    Args:
        model: The model being trained.
        optimizer: The (inner) optimizer being used for training.
        server_addr: DiLoCo server address as "host:port".
        sync_every: Number of optimizer steps between syncs (H in DiLoCo paper).
        worker_id: Unique worker ID. Auto-generated if None.
        bf16_comm: If True, cast pseudo-gradients to bfloat16 before sending.
            Halves bandwidth with minimal quality loss.
        timeout: Client timeout in seconds for server communication.
        dylu: If True, dynamically adjust sync_every based on server
            recommendations from DyLU (Dynamic Local Updates). The server
            computes a recommended sync interval proportional to this worker's
            speed relative to the fastest worker. Requires periodic heartbeats.
        heartbeat_interval: Seconds between heartbeat messages to the server.
            Heartbeats report training speed and receive DyLU recommendations.
            Only active when dylu=True. Set to 0 to disable.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        server_addr: str,
        sync_every: int = 500,
        worker_id: Optional[str] = None,
        bf16_comm: bool = True,
        timeout: float = 600,
        dylu: bool = False,
        heartbeat_interval: float = 30.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.sync_every = sync_every
        self._initial_sync_every = sync_every
        self.bf16_comm = bf16_comm
        self.worker_id = worker_id or self._generate_worker_id()
        self.dylu = dylu
        self.heartbeat_interval = heartbeat_interval

        self.client = DiLoCoClient(server_addr, timeout=timeout)

        # State
        self._global_params: Dict[str, torch.Tensor] = {}
        self._local_step: int = 0
        self._sync_count: int = 0
        self._hooks: List = []
        self._active = False

        # Metrics
        self._last_sync_time: float = 0
        self._total_sync_time: float = 0
        self._last_sync_send_bytes: int = 0
        self._last_sync_recv_bytes: int = 0
        self._step_timestamps: List[float] = []
        self._last_staleness: int = 0
        self._dylu_adjustments: int = 0

        # Heartbeat thread (for DyLU speed reporting)
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()

    @staticmethod
    def _generate_worker_id() -> str:
        hostname = platform.node()
        short_uuid = uuid.uuid4().hex[:8]
        return f"worker_{hostname}_{short_uuid}"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        """Register with server, load global params, install optimizer hooks."""
        if self._active:
            logger.warning("DiLoCoWorker already active")
            return

        logger.info(f"DiLoCoWorker {self.worker_id}: registering with server...")

        # Register and get global params
        worker_info = self._get_worker_info()
        global_params = self.client.register(self.worker_id, worker_info)

        # Load global params into model
        self._apply_global_params(global_params)
        self._save_global_params_snapshot()

        # Install optimizer hook
        hook = self.optimizer.register_step_post_hook(self._post_step_hook)
        self._hooks.append(hook)

        self._active = True
        self._local_step = 0

        # Start heartbeat thread if DyLU is enabled
        if self.dylu and self.heartbeat_interval > 0:
            self._heartbeat_stop.clear()
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True
            )
            self._heartbeat_thread.start()

        logger.info(
            f"DiLoCoWorker {self.worker_id}: active. "
            f"Syncing every {self.sync_every} steps, "
            f"bf16_comm={self.bf16_comm}, dylu={self.dylu}"
        )

    def stop(self):
        """Remove hooks and deregister from server."""
        if not self._active:
            return

        # Stop heartbeat thread
        if self._heartbeat_thread is not None:
            self._heartbeat_stop.set()
            self._heartbeat_thread.join(timeout=5)
            self._heartbeat_thread = None

        # Remove optimizer hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        # Deregister
        self.client.deregister(self.worker_id)

        self._active = False

        logger.info(
            f"DiLoCoWorker {self.worker_id}: stopped after "
            f"{self._sync_count} sync rounds, "
            f"total sync time: {self._total_sync_time:.1f}s"
        )
        if self._dylu_adjustments > 0:
            logger.info(
                f"  DyLU adjustments: {self._dylu_adjustments}, "
                f"final sync_every: {self.sync_every} "
                f"(initial: {self._initial_sync_every})"
            )

    def _get_worker_info(self) -> dict:
        """Gather worker metadata for registration."""
        info = {
            "hostname": platform.node(),
            "sync_every": self.sync_every,
            "bf16_comm": self.bf16_comm,
            "dylu": self.dylu,
        }

        # Add GPU info if available
        if torch.cuda.is_available():
            info["num_gpus"] = torch.cuda.device_count()
            info["gpu_names"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]

        return info

    def _save_global_params_snapshot(self):
        """Save a CPU copy of current model params as the global reference point."""
        self._global_params = {
            name: p.data.detach().clone().cpu()
            for name, p in self.model.named_parameters()
        }

    def _compute_pseudogradients(self) -> Dict[str, torch.Tensor]:
        """
        Compute pseudo-gradients: global_params - local_params.

        This represents the negative of the accumulated local update direction.
        The server's outer optimizer uses these as gradients to update the
        global parameters toward where workers have moved.
        """
        pseudograds = {}
        for name, p in self.model.named_parameters():
            pg = self._global_params[name] - p.data.cpu()
            if self.bf16_comm:
                pg = pg.to(torch.bfloat16)
            pseudograds[name] = pg
        return pseudograds

    def _apply_global_params(self, global_params: Dict[str, torch.Tensor]):
        """Load updated global params into the model."""
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in global_params:
                    p.data.copy_(global_params[name].to(dtype=p.dtype, device=p.device))
                else:
                    logger.warning(f"Parameter {name} not found in global params")

    def _post_step_hook(self, optimizer, args, kwargs):
        """Optimizer post-step hook. Triggers sync when sync_every steps reached."""
        self._local_step += 1
        self._step_timestamps.append(time.time())

        if self._local_step >= self.sync_every:
            self._sync()

    def _sync(self):
        """Perform a sync round with the server."""
        t0 = time.time()

        logger.info(
            f"DiLoCoWorker {self.worker_id}: starting sync "
            f"(round {self._sync_count + 1}, after {self._local_step} local steps)"
        )

        # Compute pseudo-gradients
        pseudograds = self._compute_pseudogradients()

        # Track send size
        send_bytes = sum(
            p.numel() * p.element_size() for p in pseudograds.values()
        )
        self._last_sync_send_bytes = send_bytes

        # Submit and receive updated global params
        new_global_params = self.client.submit_pseudogradients(
            self.worker_id, pseudograds
        )

        # Track receive size
        recv_bytes = sum(
            p.numel() * p.element_size() for p in new_global_params.values()
        )
        self._last_sync_recv_bytes = recv_bytes

        # Apply new global params to model
        self._apply_global_params(new_global_params)
        self._save_global_params_snapshot()

        # Reset local step counter
        self._local_step = 0
        self._sync_count += 1
        self._step_timestamps.clear()

        elapsed = time.time() - t0
        self._last_sync_time = elapsed
        self._total_sync_time += elapsed

        logger.info(
            f"DiLoCoWorker {self.worker_id}: sync round {self._sync_count} complete. "
            f"Sent {send_bytes / 1e6:.1f} MB, received {recv_bytes / 1e6:.1f} MB, "
            f"took {elapsed:.1f}s"
        )

    def _heartbeat_loop(self):
        """Background thread that sends periodic heartbeats to the server."""
        while not self._heartbeat_stop.wait(timeout=self.heartbeat_interval):
            if not self._active:
                break
            try:
                speed = self.get_steps_per_second()
                response = self.client.heartbeat(self.worker_id, steps_per_second=speed)

                # Apply DyLU recommendation if present
                if self.dylu and "recommended_sync_every" in response:
                    new_sync_every = response["recommended_sync_every"]
                    if new_sync_every != self.sync_every:
                        old = self.sync_every
                        self.sync_every = new_sync_every
                        self._dylu_adjustments += 1
                        logger.info(
                            f"DiLoCoWorker {self.worker_id}: DyLU adjusted "
                            f"sync_every {old} -> {new_sync_every}"
                        )
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

    def get_steps_per_second(self) -> float:
        """Compute current training speed from step timestamps."""
        if len(self._step_timestamps) < 2:
            return 0.0
        duration = self._step_timestamps[-1] - self._step_timestamps[0]
        if duration <= 0:
            return 0.0
        return (len(self._step_timestamps) - 1) / duration

    @property
    def sync_metrics(self) -> dict:
        """Return current sync metrics for logging."""
        metrics = {
            "diloco/sync_count": self._sync_count,
            "diloco/local_step": self._local_step,
            "diloco/last_sync_time": self._last_sync_time,
            "diloco/total_sync_time": self._total_sync_time,
            "diloco/last_send_mb": self._last_sync_send_bytes / 1e6,
            "diloco/last_recv_mb": self._last_sync_recv_bytes / 1e6,
            "diloco/steps_per_second": self.get_steps_per_second(),
            "diloco/sync_every": self.sync_every,
        }
        if self.dylu:
            metrics["diloco/dylu_adjustments"] = self._dylu_adjustments
        return metrics

    def force_sync(self):
        """Force an immediate sync regardless of step count."""
        if not self._active:
            raise RuntimeError("DiLoCoWorker is not active")
        self._sync()
