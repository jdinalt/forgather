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

    **The callback is fail-fast on misconfiguration.** If it's constructed
    (in the trainer's callback list) but ``server_addr`` is unset (and
    ``DILOCO_SERVER`` is also unset in the env), ``on_train_begin``
    raises ``DiLoCoServerUnreachable`` rather than no-op'ing. The
    older "silent no-op when DILOCO_SERVER is unset" path was a
    silent-failure footgun: two workers running with the callback
    present but no reachable server would train islands with no
    coordination and no error. The template gates the callback
    include on ``getenv("DILOCO_SERVER")`` so vanilla finetunes
    never reach this code path at all; if the gate is bypassed and
    we end up here without a server, we fail loudly.

    Likewise, a *configured* but *unreachable* server is fatal at
    startup — the callback does a ``/status`` round-trip before
    proceeding so the operator sees the failure in the TTY pane,
    not five hundred steps later when the first sync fails.

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
    ):
        # Resolve with env var fallbacks. ``diloco_server_addr()`` is
        # the canonical reader — strips whitespace, treats empty /
        # whitespace-only as unset (matches ``diloco_is_enabled()``).
        from forgather.ml.diloco import diloco_server_addr

        self.server_addr = server_addr or diloco_server_addr()
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
        """Create and start the DiLoCoWorker.

        Raises
        ------
        DiLoCoRegisterCollisionError
            Propagated from ``DiLoCoWorker.start()`` when the server
            refuses our ``worker_id`` with HTTP 409 (another worker
            with the same id is still registered). The training loop
            sees this as a fatal exception and exits cleanly; the
            server's diagnostic body is in the exception message so
            the operator's TTY pane shows exactly what to do.

        Note: work-unit dispatch (training-data dispatch via the
        DiLoCo server's work-queue endpoints) is **not** handled
        here — it's a separate, self-configuring concern owned by
        :func:`forgather.ml.datasets.work_unit_dispatch.maybe_enable_work_dispatch`
        which the project's preprocess step calls under
        ``shard_dataset.method: work_units``. The two subsystems
        share env vars (``DILOCO_WORKER_ID``, ``DILOCO_SERVER``) but
        otherwise don't interact — this callback only manages the
        worker's parameter-sync lifecycle.
        """
        from forgather.ml.diloco.client import (
            DiLoCoClient,
            DiLoCoModelMismatchError,
            DiLoCoRegisterCollisionError,
            DiLoCoServerUnreachable,
        )
        from forgather.ml.diloco.worker import DiLoCoWorker

        if not self.active:
            # Fail-fast: the callback being in the trainer's enabled
            # list but having no server_addr is a misconfiguration. The
            # older silent no-op hid two-worker islands-without-sync
            # incidents. The template should have gated this include
            # on ``getenv("DILOCO_SERVER")``; if we reach here that
            # gate is missing.
            raise DiLoCoServerUnreachable(
                "DiLoCoCallback is enabled but neither the ``server_addr`` "
                "constructor arg nor the ``DILOCO_SERVER`` env var is set. "
                "Either gate the callback include in the template on "
                "``getenv('DILOCO_SERVER')`` (vanilla finetune path), or "
                "set DILOCO_SERVER to point at a running diloco server."
            )

        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        trainer = kwargs.get("trainer")
        if model is None or optimizer is None:
            raise RuntimeError(
                "DiLoCoCallback: model or optimizer not provided in "
                "on_train_begin kwargs. Cannot initialize DiLoCoWorker."
            )

        # Pipeline-parallel detection (issue #84). The pipeline trainer
        # stores per-rank stage modules in ``trainer.pipeline_modules``
        # and the meta-device root model in ``trainer.model``. Cloning
        # meta tensors fails, so on pipeline trainers we hand the worker
        # a ``PipelineParamView`` over the on-device stage modules and
        # register as one rank of a ``pp_world_size``-sized group. Every
        # pipeline rank becomes its own DiLoCo worker (worker_id derived
        # from the operator's base id with a ``_pp<rank>`` suffix); the
        # server (commits 1-3) coordinates the per-rank slice
        # submissions into one logical DiLoCo job.
        param_view = None
        group_kwargs: Dict[str, object] = {}
        worker_id = self.worker_id
        if trainer is not None and getattr(trainer, "pipeline_modules", None):
            from forgather.ml.diloco.param_view import PipelineParamView

            pp_rank = trainer.dist.rank
            pp_world_size = trainer.dist.world_size
            param_view = PipelineParamView(
                pipeline_modules=trainer.pipeline_modules,
                sharing_metadata=getattr(trainer, "sharing_metadata", None),
            )
            base_id = worker_id or DiLoCoWorker._generate_worker_id()
            worker_id = f"{base_id}_pp{pp_rank}"
            group_kwargs = dict(
                group_id=base_id,
                pp_rank=pp_rank,
                pp_world_size=pp_world_size,
            )
            logger.info(
                f"DiLoCoCallback: pipeline trainer detected "
                f"(pp_rank={pp_rank}/{pp_world_size}); registering as "
                f"group '{base_id}' member '{worker_id}'."
            )

        # Reachability pre-check: a /status round-trip before we
        # bother building the worker. Surfaces "server URL wrong",
        # "server down", "wrong port", "firewall" while the operator
        # is still watching the TTY, instead of 500 local steps later
        # when the first sync fails.
        #
        # DDP rank 0 only — followers don't talk to the server, so
        # they shouldn't probe it either. They'll still hit the
        # broadcast collective inside DiLoCoWorker.start(), so if the
        # leader fails the probe and aborts, followers will deadlock
        # waiting for the broadcast — but the leader's exception is
        # the actionable signal the operator needs, and the worker's
        # train loop will get torn down by the trainer once the
        # leader's process exits.
        import torch.distributed as dist

        is_leader = (
            not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
        )
        if is_leader:
            probe = DiLoCoClient(self.server_addr, timeout=min(self.timeout, 10.0))
            try:
                probe.get_status()
            except Exception as exc:
                raise DiLoCoServerUnreachable(
                    f"DiLoCoCallback: /status round-trip to "
                    f"{self.server_addr!r} failed at startup: {exc}. "
                    f"The server must be running and reachable before "
                    f"workers can register."
                ) from exc

        self._worker = DiLoCoWorker(
            model=model,
            optimizer=optimizer,
            server_addr=self.server_addr,
            sync_every=self.sync_every,
            worker_id=worker_id,
            bf16_comm=self.bf16_comm,
            timeout=self.timeout,
            dylu=self.dylu,
            heartbeat_interval=self.heartbeat_interval,
            num_fragments=self.num_fragments,
            max_sync_retries=self.max_sync_retries,
            param_view=param_view,
            **group_kwargs,
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
        except DiLoCoModelMismatchError as exc:
            # Server rejected our model fingerprint — operator
            # almost certainly pointed this worker at the wrong
            # --model-id-or-path. Surface the per-param diagnostic
            # so they can spot the divergent dim immediately.
            logger.error(
                "DiLoCoCallback: server rejected worker model " "(worker_id=%r) — %s",
                self._worker.worker_id,
                exc.diagnostic or str(exc),
            )
            self._worker = None
            raise

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
        logger.debug("DiLoCoCallback: checkpoint state deferred until on_train_begin")
