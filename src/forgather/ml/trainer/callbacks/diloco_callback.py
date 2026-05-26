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

from forgather.ml.distributed import prefix_logger_rank

from ..trainer_types import (
    MinimalTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

logger = logging.getLogger(__name__)
prefix_logger_rank(logger, show_all_ranks=True)


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

    Parameters
    ----------
    server_addr : str, optional
        DiLoCo server address (``"host:port"``). Falls back to
        ``DILOCO_SERVER`` env var.
    sync_every : int, optional
        Local optimizer steps between syncs. Falls back to
        ``DILOCO_SYNC_EVERY`` env var. Default ``500``.
    worker_id : str, optional
        Unique worker ID. Falls back to ``DILOCO_WORKER_ID`` env var.
        Auto-generated if unset.
    bf16_comm : bool, optional
        Cast pseudo-gradients to bfloat16. Falls back to
        ``DILOCO_BF16_COMM`` env var. Default ``True``.
    dylu : bool, optional
        Enable Dynamic Local Updates. Falls back to ``DILOCO_DYLU`` env var.
        Default ``False``.
    heartbeat_interval : float, optional
        Seconds between heartbeats. Falls back to
        ``DILOCO_HEARTBEAT_INTERVAL`` env var. Default ``30.0``.
    num_fragments : int, optional
        Number of streaming fragments. Falls back to
        ``DILOCO_NUM_FRAGMENTS`` env var. Default ``1`` (no streaming).
    timeout : float, optional
        Client timeout in seconds. Default ``600``.
    max_sync_retries : int, optional
        Max retries for sync failures. Default ``3``.
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
        work_dispatch: Optional[bool] = None,
    ):
        # Resolve with env var fallbacks
        self.server_addr = server_addr or os.environ.get("DILOCO_SERVER", "")
        self.sync_every = (
            sync_every if sync_every is not None else _env_int("DILOCO_SYNC_EVERY", 500)
        )
        self.worker_id = worker_id or os.environ.get("DILOCO_WORKER_ID", "") or None
        self.bf16_comm = (
            bf16_comm if bf16_comm is not None else _env_bool("DILOCO_BF16_COMM", True)
        )
        self.dylu = dylu if dylu is not None else _env_bool("DILOCO_DYLU", False)
        self.heartbeat_interval = (
            heartbeat_interval
            if heartbeat_interval is not None
            else _env_float("DILOCO_HEARTBEAT_INTERVAL", 30.0)
        )
        self.num_fragments = (
            num_fragments
            if num_fragments is not None
            else _env_int("DILOCO_NUM_FRAGMENTS", 1)
        )
        self.timeout = timeout
        self.max_sync_retries = max_sync_retries
        # Work-unit dispatch opt-in. When True, on_train_begin wraps
        # the train dataloader's dataset with a WorkUnitDataset so the
        # worker pulls per-unit row ranges from the DiLoCo server
        # instead of using a static --num-shards / --shard-index
        # partition (see docs/design/diloco-work-unit-dispatch.md).
        # Defaults to False — preserves the legacy manual-shard path
        # until configs opt in.
        self.work_dispatch = (
            work_dispatch
            if work_dispatch is not None
            else _env_bool("DILOCO_WORK_DISPATCH", False)
        )

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
        """Create and start the DiLoCoWorker; optionally wrap the train dataset.

        Raises
        ------
        DiLoCoRegisterCollisionError
            Propagated from ``DiLoCoWorker.start()`` when the server
            refuses our ``worker_id`` with HTTP 409 (another worker
            with the same id is still registered). The training loop
            sees this as a fatal exception and exits cleanly; the
            server's diagnostic body is in the exception message so
            the operator's TTY pane shows exactly what to do.
        """
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

        from forgather.ml.diloco.client import DiLoCoRegisterCollisionError
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
        try:
            self._worker.start()
        except DiLoCoRegisterCollisionError as exc:
            # Server refused the worker_id (another worker holds it).
            # Log the diagnostic at ERROR so the TTY pane shows it,
            # clear the worker handle so on_train_end / on_save don't
            # try to operate on a half-initialized instance, then
            # re-raise to abort training.
            logger.error(
                "DiLoCoCallback: server refused worker_id=%r — %s",
                self._worker.worker_id,
                exc.diagnostic or str(exc),
            )
            self._worker = None
            raise

        # Apply deferred checkpoint state
        if self._pending_state is not None:
            self._apply_pending_state()
            self._pending_state = None

        # Work-unit dispatch: replace trainer.train_dataloader.dataset
        # with a WorkUnitDataset that pulls per-unit row ranges from
        # the DiLoCo server. Opt-in via the work_dispatch constructor
        # arg / DILOCO_WORK_DISPATCH env var. When off, the dataset is
        # untouched and the worker uses whatever sharding the project
        # template configured (typically --num-shards / --shard-index
        # — legacy path; see design doc for migration).
        if self.work_dispatch:
            train_dataloader = kwargs.get("train_dataloader")
            if train_dataloader is None:
                logger.error(
                    "DiLoCoCallback: work_dispatch=True but no train_dataloader "
                    "in callback kwargs. Falling back to whatever sharding the "
                    "project template configured."
                )
            else:
                self._install_work_dispatch(train_dataloader)

        logger.info(
            f"DiLoCoCallback: worker started "
            f"(server={self.server_addr}, sync_every={self.sync_every}, "
            f"work_dispatch={self.work_dispatch})"
        )

    def _install_work_dispatch(self, train_dataloader) -> None:
        """Replace the train dataloader's dataset with a WorkUnitDataset.

        Reads load_args off the dataset's resilient-remote-backend (the
        regular dataset_server path stores them as ``._load_args``),
        computes the canonical ``dataset_id``, registers the queue,
        and swaps the dataset on the dataloader in place.
        """
        from forgather.ml.datasets.dataset_id import compute_dataset_id
        from forgather.ml.diloco.work_unit_dataset import WorkUnitDataset

        dataset = getattr(train_dataloader, "dataset", None)
        if dataset is None:
            logger.error(
                "DiLoCoCallback: train_dataloader.dataset is None; cannot "
                "install work-unit dispatch."
            )
            return

        # Reach the load_args. The dataset_server-backed path wraps
        # ``ResilientRemoteBackend`` inside ``ComposableIterableDataset``;
        # the backend carries ``_load_args``. Local-path datasets don't
        # have this, and phase 1 of the work-dispatch design requires
        # dataset_server.
        backend = getattr(dataset, "_backend", None)
        load_args = getattr(backend, "_load_args", None)
        if not load_args:
            logger.error(
                "DiLoCoCallback: train dataset has no _load_args (is it a "
                "dataset_server-backed dataset?). work_dispatch requires the "
                "dataset_server path. Skipping wrap; falling back to manual "
                "sharding."
            )
            return

        try:
            dataset_id = compute_dataset_id(
                path=load_args.get("path"),
                name=load_args.get("name"),
                split=load_args.get("split"),
                data_files=load_args.get("data_files"),
                revision=load_args.get("revision"),
            )
        except ValueError as exc:
            logger.error(
                "DiLoCoCallback: could not compute dataset_id from load_args=%r: %s",
                load_args,
                exc,
            )
            return

        try:
            length = len(dataset)
        except TypeError:
            logger.error(
                "DiLoCoCallback: train dataset has no __len__; work_dispatch "
                "needs a fixed dataset length to compute per-unit row ranges. "
                "Skipping wrap."
            )
            return

        # shuffle_seed: same seed across the fleet → same shuffle →
        # matched unit ranges. For phase 1 we fix this at 0 — the
        # operator can override via a future template knob if they
        # want reproducible shuffles. Multi-epoch rotation (new
        # shuffle_seed per epoch) is a follow-up.
        shuffle_seed = 0

        client = self._worker.client
        try:
            reply = client.register_dataset(
                worker_id=self._worker.worker_id,
                dataset_id=dataset_id,
                shuffle_seed=shuffle_seed,
                hint={"length": length},
            )
        except Exception as exc:
            logger.error(
                "DiLoCoCallback: /datasets/register failed for dataset_id=%s: %s. "
                "Skipping work-unit dispatch wrap.",
                dataset_id,
                exc,
            )
            return

        total_units = int(reply["total_units"])
        wrapped = WorkUnitDataset(
            base=dataset,
            client=client,
            worker_id=self._worker.worker_id,
            dataset_id=dataset_id,
            shuffle_seed=shuffle_seed,
            total_units=total_units,
            length=length,
        )
        train_dataloader.dataset = wrapped
        logger.info(
            "DiLoCoCallback: train dataset wrapped with WorkUnitDataset "
            "(dataset_id=%s, shuffle_seed=%d, K=%d, length=%d)",
            dataset_id,
            shuffle_seed,
            total_units,
            length,
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
        logger.debug("DiLoCoCallback: checkpoint state deferred until on_train_begin")
