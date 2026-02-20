"""
DiLoCo Worker - Composable wrapper for any trainer/optimizer.

Wraps around a model and optimizer to periodically synchronize with a DiLoCo
parameter server. Uses optimizer post-step hooks so it works transparently
with any existing Forgather trainer (single GPU, DDP, pipeline).

Supports both synchronous and asynchronous DiLoCo modes (determined by the
server). In async mode, the worker can optionally adapt its sync frequency
dynamically via DyLU (Dynamic Local Updates) based on server recommendations.

**Streaming mode** (num_fragments > 1): Splits the model into N fragments and
syncs them at staggered intervals, enabling communication-computation overlap.
Fragment submissions happen in background threads while training continues.

Usage:
    model = MyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Standard mode
    with DiLoCoWorker(model, optimizer, server_addr="host:8512", sync_every=500) as diloco:
        trainer.train()  # Normal training - DiLoCo syncs happen automatically

    # Streaming mode (4 fragments)
    with DiLoCoWorker(model, optimizer, server_addr="host:8512",
                      sync_every=500, num_fragments=4) as diloco:
        trainer.train()  # Fragments sync in background

For pipeline-parallel workers, only rank 0 communicates with the server.
Rank 0 gathers/scatters parameters from/to other pipeline ranks.
"""

import logging
import platform
import queue
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .client import DiLoCoClient
from .fragments import FragmentManager

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

    **Streaming mode** (num_fragments > 1): The model is split into N fragments.
    Every sync_every/N steps, one fragment is synced with the server in a
    background thread while training continues. This overlaps communication
    with computation, hiding transfer latency.

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
            Heartbeats report training speed, enable server-side health
            monitoring, and receive DyLU recommendations. Set to 0 to disable.
        num_fragments: Number of fragments for streaming sync. When > 1,
            enables streaming mode where fragments sync at staggered intervals
            in background threads. Default 1 (standard non-streaming mode).
        max_sync_retries: Maximum retry attempts for sync failures. On
            connection error, the worker will re-register with the server
            and retry the sync. 0 means no retries (fail immediately).
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
        num_fragments: int = 1,
        max_sync_retries: int = 3,
    ):
        self.model = model
        self.optimizer = optimizer
        self.sync_every = sync_every
        self._initial_sync_every = sync_every
        self.bf16_comm = bf16_comm
        self.worker_id = worker_id or self._generate_worker_id()
        self.dylu = dylu
        self.heartbeat_interval = heartbeat_interval
        self.num_fragments = num_fragments
        self.max_sync_retries = max_sync_retries

        self.client = DiLoCoClient(server_addr, timeout=timeout)

        # Fragment manager (None if num_fragments <= 1)
        self._fragment_manager: Optional[FragmentManager] = None
        if num_fragments > 1:
            self._fragment_manager = FragmentManager(model, num_fragments)

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
        self._fragment_syncs: int = 0
        self._sync_retries: int = 0
        self._reconnections: int = 0

        # Heartbeat thread (for health monitoring and DyLU)
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()

        # Streaming state: at most one fragment in-flight at a time.
        # The background thread submits pseudo-gradients and stores the
        # result. The main thread applies the result before starting the
        # next fragment submission.
        self._inflight_thread: Optional[threading.Thread] = None
        self._inflight_result: Optional[Tuple[int, Optional[Dict[str, torch.Tensor]]]] = None

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

        # Start heartbeat thread (enables server-side health monitoring
        # and DyLU speed reporting)
        if self.heartbeat_interval > 0:
            self._heartbeat_stop.clear()
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True
            )
            self._heartbeat_thread.start()

        streaming_info = ""
        if self._fragment_manager is not None:
            frag_interval = self.sync_every // self.num_fragments
            streaming_info = (
                f", streaming={self.num_fragments} fragments "
                f"(interval={frag_interval} steps)"
            )

        logger.info(
            f"DiLoCoWorker {self.worker_id}: active. "
            f"Syncing every {self.sync_every} steps, "
            f"bf16_comm={self.bf16_comm}, dylu={self.dylu}{streaming_info}"
        )

    def stop(self):
        """Remove hooks and deregister from server."""
        if not self._active:
            return

        # Wait for any in-flight fragment to complete
        self._wait_and_apply_inflight_fragment()

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
        if self._fragment_syncs > 0:
            logger.info(
                f"  Fragment syncs: {self._fragment_syncs} "
                f"({self.num_fragments} fragments)"
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

        if self._fragment_manager is None:
            # Standard path: full model sync
            if self._local_step >= self.sync_every:
                self._sync()
        else:
            # Streaming path: check for fragment sync schedule
            frag_id = self._fragment_manager.get_fragment_schedule(
                self._local_step, self.sync_every
            )
            if frag_id is not None:
                self._sync_fragment(frag_id)

            # Reset step counter at sync_every boundary (all fragments submitted)
            if self._local_step >= self.sync_every:
                self._local_step = 0
                self._sync_count += 1
                self._step_timestamps.clear()

    def _sync(self):
        """Perform a sync round with the server, with retry on failure.

        On connection error, attempts to re-register with the server and
        retry the sync up to max_sync_retries times. If all retries fail,
        logs the error and continues training (the sync is skipped).
        """
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

        # Submit with retry on connection failure
        new_global_params = None
        retry_delay = 2.0
        for attempt in range(self.max_sync_retries + 1):
            try:
                new_global_params = self.client.submit_pseudogradients(
                    self.worker_id, pseudograds
                )
                break
            except ConnectionError as e:
                if attempt < self.max_sync_retries:
                    self._sync_retries += 1
                    logger.warning(
                        f"DiLoCoWorker {self.worker_id}: sync failed "
                        f"(attempt {attempt + 1}/{self.max_sync_retries + 1}): {e}. "
                        f"Reconnecting in {retry_delay:.0f}s..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    self._reconnect()
                    # Recompute pseudo-gradients (params may have changed
                    # after reconnect)
                    pseudograds = self._compute_pseudogradients()
                else:
                    logger.error(
                        f"DiLoCoWorker {self.worker_id}: sync failed after "
                        f"{self.max_sync_retries + 1} attempts: {e}. "
                        f"Skipping this sync round."
                    )

        if new_global_params is not None:
            # Track receive size
            recv_bytes = sum(
                p.numel() * p.element_size() for p in new_global_params.values()
            )
            self._last_sync_recv_bytes = recv_bytes

            # Apply new global params to model
            self._apply_global_params(new_global_params)
            self._save_global_params_snapshot()

            elapsed = time.time() - t0
            self._last_sync_time = elapsed
            self._total_sync_time += elapsed

            logger.info(
                f"DiLoCoWorker {self.worker_id}: sync round {self._sync_count + 1} complete. "
                f"Sent {send_bytes / 1e6:.1f} MB, received {recv_bytes / 1e6:.1f} MB, "
                f"took {elapsed:.1f}s"
            )

        # Reset local step counter (even on failure, to avoid repeated sync attempts)
        self._local_step = 0
        self._sync_count += 1
        self._step_timestamps.clear()

    def _reconnect(self):
        """Re-register with the server after a connection failure.

        Fetches the current global parameters and updates the local
        snapshot. This handles server restarts where the server may have
        a newer state than this worker's snapshot.
        """
        logger.info(f"DiLoCoWorker {self.worker_id}: attempting reconnection...")
        try:
            worker_info = self._get_worker_info()
            global_params = self.client.register(self.worker_id, worker_info)
            self._apply_global_params(global_params)
            self._save_global_params_snapshot()
            self._reconnections += 1
            logger.info(f"DiLoCoWorker {self.worker_id}: reconnected successfully")
        except Exception as e:
            logger.warning(
                f"DiLoCoWorker {self.worker_id}: reconnection failed: {e}"
            )

    # --- Streaming fragment methods ---

    def _sync_fragment(self, fragment_id: int):
        """
        Sync a single fragment with the server.

        1. Wait for any in-flight fragment to complete and apply its result
        2. Compute pseudo-gradients for this fragment
        3. Submit in a background thread (overlap with next training steps)
        """
        t0 = time.time()

        # Apply any pending result from the previous fragment
        self._wait_and_apply_inflight_fragment()

        # Compute pseudo-gradients for this fragment
        pseudograds = self._fragment_manager.compute_fragment_pseudogradients(
            fragment_id, self._global_params, self.model, self.bf16_comm
        )

        send_bytes = sum(
            p.numel() * p.element_size() for p in pseudograds.values()
        )

        logger.info(
            f"DiLoCoWorker {self.worker_id}: submitting fragment {fragment_id} "
            f"({send_bytes / 1e6:.1f} MB, step {self._local_step})"
        )

        # Submit in background thread
        self._inflight_thread = threading.Thread(
            target=self._submit_fragment_background,
            args=(fragment_id, pseudograds, t0),
            daemon=True,
        )
        self._inflight_thread.start()

    def _submit_fragment_background(
        self,
        fragment_id: int,
        pseudograds: Dict[str, torch.Tensor],
        start_time: float,
    ):
        """Background thread: submit fragment pseudo-gradients to server."""
        try:
            result = self.client.submit_fragment_pseudogradients(
                self.worker_id, fragment_id, pseudograds
            )
            self._inflight_result = (fragment_id, result)

            elapsed = time.time() - start_time
            self._total_sync_time += elapsed

            logger.debug(
                f"DiLoCoWorker {self.worker_id}: fragment {fragment_id} "
                f"sync complete ({elapsed:.1f}s)"
            )
        except Exception as e:
            logger.error(
                f"DiLoCoWorker {self.worker_id}: fragment {fragment_id} "
                f"submission failed: {e}"
            )
            self._inflight_result = (fragment_id, None)

    def _wait_and_apply_inflight_fragment(self):
        """Wait for the in-flight fragment thread and apply its result."""
        if self._inflight_thread is None:
            return

        self._inflight_thread.join()
        self._inflight_thread = None

        if self._inflight_result is not None:
            frag_id, new_params = self._inflight_result
            self._inflight_result = None

            if new_params is not None:
                self._fragment_manager.apply_fragment_global_params(
                    frag_id, new_params, self.model, self._global_params
                )
                self._fragment_syncs += 1

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
        if self._fragment_manager is not None:
            metrics["diloco/num_fragments"] = self.num_fragments
            metrics["diloco/fragment_syncs"] = self._fragment_syncs
        if self._sync_retries > 0:
            metrics["diloco/sync_retries"] = self._sync_retries
        if self._reconnections > 0:
            metrics["diloco/reconnections"] = self._reconnections
        return metrics

    def force_sync(self):
        """Force an immediate full-model sync regardless of step count."""
        if not self._active:
            raise RuntimeError("DiLoCoWorker is not active")
        # Wait for any pending fragment first
        self._wait_and_apply_inflight_fragment()
        self._sync()
