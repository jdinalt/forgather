"""
DiLoCoCallback - Trainer callback for DiLoCo distributed training integration.

Manages the DiLoCoWorker lifecycle within the Forgather trainer ecosystem.
Implements both TrainerCallback (for lifecycle events) and Stateful (for
checkpoint persistence). When no server_addr is configured, the callback
is a no-op, allowing a single configuration to work for both DiLoCo and
standalone training.

Usage:
    from forgather.ml.trainer.callbacks import DiLoCoCallback

    # Explicit configuration
    callback = DiLoCoCallback(server_addr="host:8512", sync_every=500)

    # Or configure via environment variables (set by `forgather diloco worker`)
    callback = DiLoCoCallback()

    trainer = Trainer(model=model, args=args, callbacks=[callback])
    trainer.train()
"""

import logging
import os
from typing import Any, Dict, Optional

from ..trainer_types import (
    MinimalTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean from an environment variable."""
    val = os.environ.get(name, "")
    if not val:
        return default
    return val.lower() in ("1", "true", "yes")


def _env_float(name: str, default: float) -> float:
    """Read a float from an environment variable."""
    val = os.environ.get(name, "")
    if not val:
        return default
    return float(val)


def _env_int(name: str, default: int) -> int:
    """Read an int from an environment variable."""
    val = os.environ.get(name, "")
    if not val:
        return default
    return int(val)


class DiLoCoCallback(TrainerCallback):
    """
    Trainer callback that manages a DiLoCoWorker for distributed local-SGD training.

    Implements the Stateful protocol for checkpoint persistence. The checkpoint
    manager auto-discovers Stateful callbacks and saves/restores their state.

    When ``server_addr`` is empty (and DILOCO_SERVER is unset), all methods are
    no-ops. This allows a single training configuration to work both with and
    without a DiLoCo server.

    Args:
        server_addr: DiLoCo server address ("host:port"). Falls back to
            DILOCO_SERVER env var.
        sync_every: Local optimizer steps between syncs. Falls back to
            DILOCO_SYNC_EVERY env var. Default 500.
        worker_id: Unique worker ID. Falls back to DILOCO_WORKER_ID env var.
            Auto-generated if unset.
        bf16_comm: Cast pseudo-gradients to bfloat16. Falls back to
            DILOCO_BF16_COMM env var. Default True.
        dylu: Enable Dynamic Local Updates. Falls back to DILOCO_DYLU env var.
            Default False.
        heartbeat_interval: Seconds between heartbeats. Falls back to
            DILOCO_HEARTBEAT_INTERVAL env var. Default 30.0.
        num_fragments: Number of streaming fragments. Falls back to
            DILOCO_NUM_FRAGMENTS env var. Default 1 (no streaming).
        timeout: Client timeout in seconds. Default 600.
        max_sync_retries: Max retries for sync failures. Default 3.
    """

    def __init__(
        self,
        server_addr: Optional[str] = None,
        sync_every: Optional[int] = None,
        worker_id: Optional[str] = None,
        bf16_comm: Optional[bool] = None,
        dylu: Optional[bool] = None,
        heartbeat_interval: Optional[float] = None,
        num_fragments: Optional[int] = None,
        timeout: float = 600,
        max_sync_retries: int = 3,
    ):
        # Resolve with env var fallbacks
        self.server_addr = server_addr or os.environ.get("DILOCO_SERVER", "")
        self.sync_every = (
            sync_every if sync_every is not None
            else _env_int("DILOCO_SYNC_EVERY", 500)
        )
        self.worker_id = worker_id or os.environ.get("DILOCO_WORKER_ID", "") or None
        self.bf16_comm = (
            bf16_comm if bf16_comm is not None
            else _env_bool("DILOCO_BF16_COMM", True)
        )
        self.dylu = (
            dylu if dylu is not None
            else _env_bool("DILOCO_DYLU", False)
        )
        self.heartbeat_interval = (
            heartbeat_interval if heartbeat_interval is not None
            else _env_float("DILOCO_HEARTBEAT_INTERVAL", 30.0)
        )
        self.num_fragments = (
            num_fragments if num_fragments is not None
            else _env_int("DILOCO_NUM_FRAGMENTS", 1)
        )
        self.timeout = timeout
        self.max_sync_retries = max_sync_retries

        # Worker instance (created in on_train_begin)
        self._worker = None

        # Deferred checkpoint state (loaded before on_train_begin)
        self._pending_state: Optional[Dict[str, Any]] = None

    @property
    def active(self) -> bool:
        """Whether DiLoCo integration is configured (server_addr is set)."""
        return bool(self.server_addr)

    def on_train_begin(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Create and start the DiLoCoWorker."""
        if not self.active:
            logger.info("DiLoCoCallback: no server_addr configured, running as no-op")
            return

        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        if model is None or optimizer is None:
            logger.error(
                "DiLoCoCallback: model or optimizer not provided in kwargs. "
                "Cannot initialize DiLoCoWorker."
            )
            return

        from forgather.ml.diloco.worker import DiLoCoWorker

        self._worker = DiLoCoWorker(
            model=model,
            optimizer=optimizer,
            server_addr=self.server_addr,
            sync_every=self.sync_every,
            worker_id=self.worker_id,
            bf16_comm=self.bf16_comm,
            timeout=self.timeout,
            dylu=self.dylu,
            heartbeat_interval=self.heartbeat_interval,
            num_fragments=self.num_fragments,
            max_sync_retries=self.max_sync_retries,
        )
        self._worker.start()

        # Apply deferred checkpoint state
        if self._pending_state is not None:
            self._apply_pending_state()
            self._pending_state = None

        logger.info(
            f"DiLoCoCallback: worker started "
            f"(server={self.server_addr}, sync_every={self.sync_every})"
        )

    def _apply_pending_state(self):
        """Apply deferred state from load_state_dict to the active worker."""
        if self._worker is None or self._pending_state is None:
            return

        st = self._pending_state
        self._worker._sync_count = st.get("sync_count", 0)
        self._worker._local_step = st.get("local_step", 0)
        self._worker._total_sync_time = st.get("total_sync_time", 0.0)
        self._worker._sync_retries = st.get("sync_retries", 0)
        self._worker._reconnections = st.get("reconnections", 0)
        self._worker._dylu_adjustments = st.get("dylu_adjustments", 0)
        self._worker._fragment_syncs = st.get("fragment_syncs", 0)

        # Restore sync_every (may have been adjusted by DyLU)
        if "sync_every" in st:
            self._worker.sync_every = st["sync_every"]

        logger.info(
            f"DiLoCoCallback: restored state from checkpoint "
            f"(sync_count={st.get('sync_count', 0)}, "
            f"local_step={st.get('local_step', 0)})"
        )

    def on_log(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ):
        """Inject DiLoCo sync metrics into the logs dict."""
        if self._worker is not None and logs is not None:
            logs.update(self._worker.sync_metrics)

    def on_train_end(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Stop the DiLoCoWorker."""
        if self._worker is not None:
            self._worker.stop()
            logger.info("DiLoCoCallback: worker stopped")
            self._worker = None

    # -- Stateful protocol --

    def state_dict(self) -> Dict[str, Any]:
        """Save DiLoCo state for checkpointing.

        Does NOT save global_params snapshot -- the server provides fresh
        params when the worker re-registers on resume.
        """
        if self._worker is None:
            return {}

        return {
            "sync_count": self._worker._sync_count,
            "local_step": self._worker._local_step,
            "sync_every": self._worker.sync_every,
            "worker_id": self._worker.worker_id,
            "total_sync_time": self._worker._total_sync_time,
            "sync_retries": self._worker._sync_retries,
            "reconnections": self._worker._reconnections,
            "dylu_adjustments": self._worker._dylu_adjustments,
            "fragment_syncs": self._worker._fragment_syncs,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Defer state restoration until on_train_begin.

        Checkpoint loading happens during _prepare() before on_train_begin,
        so the worker doesn't exist yet. We store the state and apply it
        once the worker is created.
        """
        if not state_dict:
            return
        self._pending_state = state_dict
        logger.debug(
            "DiLoCoCallback: checkpoint state deferred until on_train_begin"
        )
