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

# Relayed trainer-control commands -> integer code for the cross-rank
# all_reduce(MAX) in on_step_end. 0 == "no command". Names mirror the
# worker-side trainer-control vocabulary (control_callback.COMMAND_CODES).
_RELAY_CODE = {"save_checkpoint": 1, "save_and_stop": 2, "abort": 3}
_RELAY_NAME = {v: k for k, v in _RELAY_CODE.items()}


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


class DiLoCoCallback(TrainerCallback):
    """
    Trainer callback that manages a DiLoCoWorker for distributed local-SGD training.

    Implements the Stateful protocol for checkpoint persistence. The checkpoint
    manager auto-discovers Stateful callbacks and saves/restores their state.

    **The callback is fail-fast on misconfiguration.** If it's constructed
    (in the trainer's callback list) but ``server_addr`` is unset (and
    ``DILOCO_SERVER`` is also unset in the env), ``on_load_model_weights``
    raises ``DiLoCoServerUnreachable`` rather than no-op'ing. The
    older "silent no-op when DILOCO_SERVER is unset" path was a
    silent-failure footgun: two workers running with the callback
    present but no reachable server would train islands with no
    coordination and no error. The template gates the callback
    include on ``getenv("DILOCO_SERVER")`` so vanilla finetunes
    never reach this code path at all; if the gate is bypassed and
    we end up here without a server, we fail loudly.

    Likewise, a *configured* but *unreachable* server is fatal at
    startup — the callback does a ``/info`` round-trip before
    proceeding so the operator sees the failure in the TTY pane,
    not five hundred steps later when the first sync fails.

    **Server-authoritative settings.** ``sync_every``, ``bf16_comm``,
    ``dylu`` and ``num_fragments`` must match across the whole group for
    the sync barrier / outer step / fragment barriers to be coherent, so
    they are owned by the server: the worker reads them verbatim from the
    server's ``/info`` at startup (the leader fetches and broadcasts to
    DDP followers). There is no client override — a divergent value is
    never useful. Only genuinely client-local knobs remain as constructor
    args / env vars.

    Parameters
    ----------
    server_addr : str, optional
        DiLoCo server address (``"host:port"``). Falls back to
        ``DILOCO_SERVER`` env var.
    worker_id : str, optional
        Unique worker ID. Falls back to ``DILOCO_WORKER_ID`` env var.
        Auto-generated if unset.
    heartbeat_interval : float, optional
        Seconds between heartbeats (a client-local send cadence, validated
        against the server's ``heartbeat_timeout``). Falls back to
        ``DILOCO_HEARTBEAT_INTERVAL`` env var. Default ``30.0``.
    timeout : float, optional
        Client timeout in seconds. Default ``600``.
    max_sync_retries : int, optional
        Max retries for sync failures. Default ``3``.
    """

    def __init__(
        self,
        server_addr: Optional[str] = None,
        worker_id: Optional[str] = None,
        heartbeat_interval: Optional[float] = None,
        timeout: float = 600,
        max_sync_retries: int = 3,
        auth_token: Optional[str] = None,
        verify_tls: Optional[bool] = None,
    ):
        # Resolve with env var fallbacks. ``diloco_server_addr()`` is
        # the canonical reader — strips whitespace, treats empty /
        # whitespace-only as unset (matches ``diloco_is_enabled()``).
        from forgather.ml.diloco import diloco_server_addr

        self.server_addr = server_addr or diloco_server_addr()
        self.worker_id = worker_id or os.environ.get("DILOCO_WORKER_ID", "") or None
        self.heartbeat_interval = (
            heartbeat_interval
            if heartbeat_interval is not None
            else _env_float("DILOCO_HEARTBEAT_INTERVAL", 30.0)
        )
        # Server-authoritative settings: these MUST match across the group
        # for the sync barrier / outer step / fragment barriers to be
        # coherent, so the worker takes them verbatim from the server's
        # /info (resolved in on_load_model_weights) with no client override.
        # They stay None until then.
        self.sync_every: Optional[int] = None
        self.bf16_comm: Optional[bool] = None
        self.dylu: Optional[bool] = None
        self.num_fragments: Optional[int] = None
        self.timeout = timeout
        self.max_sync_retries = max_sync_retries
        # Security (issue #90): bearer token + TLS verification. ``None``
        # delegates token discovery to ``DiLoCoClient`` (explicit arg →
        # env var → loopback per-port file), which covers the
        # locally-spawned case without requiring template changes.
        # ``verify_tls`` defaults to True so misconfigured remotes
        # fail closed.
        self.auth_token = auth_token or os.environ.get("FORGATHER_DILOCO_SERVER_TOKEN")
        if verify_tls is None:
            verify_tls = _env_bool("DILOCO_VERIFY_TLS", True)
        self.verify_tls = verify_tls

        # Worker instance (created in on_load_model_weights)
        self._worker = None

        # Deferred checkpoint state (loaded before on_load_model_weights)
        self._pending_state: Optional[Dict[str, Any]] = None

    @property
    def active(self) -> bool:
        """Whether DiLoCo integration is configured (server_addr is set)."""
        return bool(self.server_addr)

    def _resolve_server_settings(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the server-authoritative settings from an /info payload.

        These four (sync_every, bf16_comm, dylu, num_fragments) must match
        across the group, so the server is the sole authority — there is no
        client override. A server that doesn't advertise them (too old, or
        ``expected_client_settings.sync_every`` is null) is a fatal
        misconfiguration, not something to paper over with a client default
        (no-silent-fallback).
        """
        from forgather.ml.diloco.client import DiLoCoServerUnreachable

        ecs = info.get("expected_client_settings") or {}
        sync_every = ecs.get("sync_every")
        if sync_every is None:
            raise DiLoCoServerUnreachable(
                f"DiLoCoCallback: server at {self.server_addr!r} did not "
                f"advertise a sync_every in /info "
                f"(expected_client_settings={ecs!r}). It is likely an "
                f"older server predating server-authoritative settings; "
                f"upgrade the diloco server."
            )
        return {
            "sync_every": int(sync_every),
            "bf16_comm": bool(ecs.get("bf16_comm", True)),
            "dylu": bool(ecs.get("dylu", False)),
            "num_fragments": int(ecs.get("num_fragments_default", 1)),
            "heartbeat_timeout": ecs.get("heartbeat_timeout"),
        }

    def _validate_heartbeat(self, heartbeat_timeout) -> None:
        """Fail loud if the client's heartbeat cadence can't beat the
        server's death timeout. ``heartbeat_interval`` stays a client knob
        (it's a genuinely local send cadence, not a must-match value), but
        a cadence at or above the server's timeout guarantees spurious
        eviction, so reject it up front. ``heartbeat_timeout <= 0`` means
        death detection is disabled — nothing to validate."""
        if heartbeat_timeout and heartbeat_timeout > 0:
            if self.heartbeat_interval >= heartbeat_timeout:
                raise ValueError(
                    f"DiLoCoCallback: heartbeat_interval="
                    f"{self.heartbeat_interval}s is >= the server's "
                    f"heartbeat_timeout={heartbeat_timeout}s; the worker "
                    f"would be evicted between heartbeats. Set "
                    f"--heartbeat-interval (or DILOCO_HEARTBEAT_INTERVAL) "
                    f"well below {heartbeat_timeout}s."
                )

    def on_load_model_weights(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Load the model weights from the DiLoCo server (trainer hook).

        Dispatched by the trainer during ``_prepare`` at the point a
        checkpoint would load — i.e. when ``checkpoint_components`` excludes
        ``"model"`` (the DiLoCo default). Builds the ``DiLoCoWorker``,
        registers (which returns the server's global params), applies them,
        and flags them ``_is_hf_initialized`` so the trainer's following
        initialize-missing pass fills only the rest (non-persistent buffers
        like RoPE ``inv_freq``). The weights arrive *via* register, so this is
        the whole worker bring-up; the parameter-sync optimizer hook and
        heartbeat it installs then run during training.

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
                "on_load_model_weights kwargs. Cannot initialize DiLoCoWorker."
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

        # Reachability pre-check + settings negotiation in one /info
        # round-trip, before we bother building the worker. Surfaces
        # "server URL wrong", "server down", "wrong port", "firewall"
        # while the operator is still watching the TTY, instead of 500
        # local steps later when the first sync fails. /info also carries
        # the server-authoritative settings (sync_every, bf16_comm, dylu,
        # num_fragments) the worker must adopt verbatim.
        #
        # DDP rank 0 only — followers don't talk to the server. The leader
        # fetches /info and broadcasts the result to followers so every
        # rank syncs in lockstep. Crucially, the leader broadcasts an
        # *error sentinel* on failure (server down, too-old server, etc.)
        # rather than raising before the collective — otherwise followers
        # would block forever inside broadcast_object_list waiting on a
        # rank 0 that already exited. With the sentinel, every rank raises
        # the same actionable error together and the job fails fast.
        import torch.distributed as dist

        ddp = dist.is_available() and dist.is_initialized()
        is_leader = not ddp or dist.get_rank() == 0
        settings: Optional[Dict[str, Any]] = None
        nego_error: Optional[str] = None
        if is_leader:
            probe = DiLoCoClient(
                self.server_addr,
                timeout=min(self.timeout, 10.0),
                token=self.auth_token,
                verify_tls=self.verify_tls,
            )
            try:
                info = probe.get_info()
                settings = self._resolve_server_settings(info)
            except DiLoCoServerUnreachable as exc:
                # _resolve_server_settings already produced an actionable
                # message (e.g. server too old to advertise sync_every).
                nego_error = str(exc)
            except Exception as exc:
                nego_error = (
                    f"DiLoCoCallback: /info round-trip to "
                    f"{self.server_addr!r} failed at startup: {exc}. "
                    f"The server must be running and reachable before "
                    f"workers can register."
                )
        if ddp:
            holder = [(settings, nego_error)]
            dist.broadcast_object_list(holder, src=0)
            settings, nego_error = holder[0]
        if nego_error is not None:
            # Raised on every rank (leader and followers) — no deadlock.
            raise DiLoCoServerUnreachable(nego_error)

        self.sync_every = settings["sync_every"]
        self.bf16_comm = settings["bf16_comm"]
        self.dylu = settings["dylu"]
        self.num_fragments = settings["num_fragments"]
        self._validate_heartbeat(settings["heartbeat_timeout"])
        logger.info(
            "DiLoCoCallback: using server settings sync_every=%s bf16_comm=%s "
            "dylu=%s num_fragments=%s",
            self.sync_every,
            self.bf16_comm,
            self.dylu,
            self.num_fragments,
        )

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
            auth_token=self.auth_token,
            verify_tls=self.verify_tls,
            # Reported to the server only so the webui can correlate this
            # worker to its forgather job by output_dir when the worker-id
            # was renamed away from the job's queue_id (issue #103). MUST
            # match the job's recorded output_dir byte-for-byte: the control
            # callback writes os.path.abspath(args.output_dir) to its
            # endpoint file (control_callback.py), which becomes the job's
            # output_dir, so apply the identical transform here — a raw
            # (possibly relative) args.output_dir would never string-match.
            output_dir=(
                os.path.abspath(args.output_dir)
                if getattr(args, "output_dir", None)
                else None
            ),
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

        # Flag exactly the tensors the worker applied — its PARAMETERS — so the
        # trainer's subsequent initialize-missing pass fills only the rest
        # (non-persistent buffers like RoPE inv_freq). The worker syncs
        # parameters, not buffers (ParamView.apply_global iterates
        # named_parameters), so flag the parameter names only: that way the
        # trainer's _verify_external_weights_loaded still catches a persistent
        # buffer the server never supplied (it stays unflagged) instead of us
        # masking it. Without flagging, the apply-path init would re-randomize
        # and clobber the just-applied global params. Flag the materialized
        # module(s): the pipeline trainer's stages live in pipeline_modules
        # (trainer.model is the meta skeleton), otherwise the model itself.
        from forgather.ml.sharded_checkpoint import flag_loaded_tensors

        pp_modules = getattr(trainer, "pipeline_modules", None) if trainer else None
        for mod in pp_modules or [model]:
            flag_loaded_tensors(mod, {name for name, _ in mod.named_parameters()})

        # Apply deferred checkpoint state
        if self._pending_state is not None:
            self._apply_pending_state()
            self._pending_state = None

        logger.info(
            f"DiLoCoCallback: worker started "
            f"(server={self.server_addr}, sync_every={self.sync_every})"
        )

    def on_train_begin(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Defensive check that the worker was started.

        The worker is brought up earlier, in :meth:`on_load_model_weights`
        (dispatched during ``_prepare`` when model weights are external). This
        callback is forgather-specific, so that hook always fires for a
        correctly configured DiLoCo run. If the worker is still ``None`` here,
        the run is misconfigured — the callback is active but
        ``checkpoint_components`` didn't exclude ``"model"``, so the trainer
        never asked us to load the server's weights. Fail loud rather than
        train a model the server never filled.
        """
        if self._worker is None:
            raise RuntimeError(
                "DiLoCoCallback: worker was not started by on_load_model_weights. "
                "The DiLoCo callback is active but the trainer never dispatched "
                "the weight-load hook — exclude 'model' from checkpoint_components "
                "so the worker loads weights from the server (the DiLoCo template "
                "sets this)."
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

    def on_step_end(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Apply any server-relayed trainer-control command.

        The server queues save / save-and-stop / abort via
        ``/control/command`` and delivers it on the leader's heartbeat; the
        worker stashes it. Here we drain it and set the matching control
        flags so the relay drives the trainer loop exactly like the direct
        trainer-control endpoint would.

        Multi-rank safety: only the leader heartbeats, so only it holds a
        command — but every rank must reach the same stop/save decision or
        the ranks diverge and deadlock at the next collective. We therefore
        ``all_reduce(MAX)`` a one-int command code across the default process
        group: every rank participates every step (no divergence in the
        collective), and MAX carries the leader's non-zero code to all.

        This assumes ranks dispatch ``on_step_end`` in lockstep (equal-length
        iteration) — the standard DDP requirement; uneven dataloader
        exhaustion already deadlocks DDP at the gradient all-reduce
        independent of this collective. ``self._worker`` is created on every
        rank (followers just never heartbeat), so the early return below is
        consistent across ranks and can't cause a partial collective.
        """
        if self._worker is None:
            return control

        local = self._worker.consume_pending_command()
        code = _RELAY_CODE.get(local, 0)

        code = self._sync_command_code(code)
        if code:
            self._apply_relay_command(_RELAY_NAME[code], control)
        return control

    def _sync_command_code(self, code: int) -> int:
        """all_reduce(MAX) the command code across ranks; identity in the
        single-process case. Failures are logged and treated as "no command"
        rather than risking a half-stopped cluster on a transient comm error."""
        try:
            import torch
            import torch.distributed as dist

            if (
                dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            ):
                device = self._command_device()
                t = torch.tensor([code], dtype=torch.long, device=device)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                return int(t.item())
        except Exception as e:  # pragma: no cover - comm failure path
            logger.warning(f"DiLoCoCallback: command sync failed: {e}")
            return 0
        return code

    def _command_device(self):
        """Device for the command all_reduce — a real on-device parameter so
        the tensor rides the same backend (NCCL/gloo) the group uses.

        Read from the worker's ``param_view`` (the actual sharded/on-device
        tensors), NOT ``model.parameters()``: under pipeline parallelism the
        trainer's ``model`` root is the meta-device skeleton, so its params
        report ``device='meta'`` and a tensor built there can't be
        all-reduced. ``param_view`` always points at the live tensors.
        """
        import torch

        try:
            if self._worker is not None:
                for _name, p in self._worker.param_view.named_parameters():
                    if p.device.type != "meta":
                        return p.device
        except Exception:
            pass
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _apply_relay_command(self, command: str, control: TrainerControl):
        """Map a relayed command to TrainerControl flags (mirrors
        control_callback._apply_command for save / save-and-stop / abort)."""
        if command == "save_checkpoint":
            control.should_save = True
        elif command == "save_and_stop":
            control.should_save = True
            control.should_training_stop = True
        elif command == "abort":
            control.should_abort_without_save = True
            control.should_training_stop = True
        logger.info(f"DiLoCoCallback: applied relayed command '{command}'")

    def on_log(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ):
        """Inject DiLoCo sync metrics into the logs dict, and snapshot the
        run's training metrics for the server's unified-stats aggregator."""
        if self._worker is None:
            return
        if logs is not None:
            logs.update(self._worker.sync_metrics)
        # Hand the server a normalized snapshot of this worker's metrics to
        # aggregate. Sourced here (the DiLoCo callback) in parallel to the
        # control callback's relay, so server stats don't depend on it.
        self._worker.set_stats(self._build_stats_snapshot(state, logs))

    def on_evaluate(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[dict] = None,
        **kwargs,
    ):
        """Report eval results to the server's aggregator.

        The control callback only instruments ``on_log`` (no eval), so eval
        loss would otherwise never reach the server. Carries the eval loss
        plus the step it was computed at; the server smooths it with a weak
        EMA across workers (post-sync evals reflect the same global model).
        """
        if self._worker is None or not metrics:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return
        self._worker.set_stats(
            self._build_stats_snapshot(state, None, eval_loss=eval_loss)
        )

    @staticmethod
    def _build_stats_snapshot(
        state: TrainerState,
        logs: Optional[dict],
        eval_loss: Optional[float] = None,
    ) -> dict:
        """Map trainer state + log dict onto the normalized stats schema the
        server aggregator consumes (see ``diloco/stats.py``).

        Cumulative counters come from ``TrainerState`` (authoritative and
        checkpoint-persisted); transient gauges come from the log dict. Keeps
        the schema mapping on the trainer side so the server stays decoupled
        from trainer-specific log-key names.
        """

        def _sum_mem(v):
            if isinstance(v, (list, tuple)):
                return float(sum(x for x in v if isinstance(x, (int, float))))
            return float(v) if isinstance(v, (int, float)) else None

        snap: dict = {}
        tokens = getattr(state, "num_input_tokens_seen", None)
        if tokens is not None:
            snap["tokens_total"] = int(tokens)
        flos = getattr(state, "total_flos", None)
        if flos:
            snap["flos_total"] = float(flos)
        step = getattr(state, "global_step", None)
        if step is not None:
            snap["step_total"] = int(step)
        # Per-worker progress target (this worker's planned optimizer steps).
        # Reported so the server / `forgather diloco status` can show each
        # worker's global-step / max-steps progress; it's a passthrough
        # per-worker value, not aggregated across workers (which may run
        # different budgets). See diloco/stats.py.
        max_steps = getattr(state, "max_steps", None)
        if max_steps is not None and max_steps > 0:
            snap["max_steps"] = int(max_steps)

        if logs:
            if logs.get("tokens") is not None:
                snap["tokens_window"] = logs["tokens"]
            for key in ("loss", "grad_norm", "tok_per_sec", "mfu"):
                if logs.get(key) is not None:
                    snap[key] = logs[key]
            pm = logs.get("peak_mem", logs.get("peak_mem_allocated"))
            mem = _sum_mem(pm) if pm is not None else None
            if mem is not None:
                snap["peak_mem"] = mem

        if eval_loss is not None:
            snap["eval_loss"] = float(eval_loss)
            if step is not None:
                snap["eval_step"] = int(step)
        return snap

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
        """Defer state restoration until the worker is created.

        Checkpoint loading happens during _prepare(); the callback's state
        load_state_dict runs before on_load_model_weights builds the worker,
        so the worker doesn't exist yet. We store the state and apply it once
        the worker is created (in on_load_model_weights).
        """
        if not state_dict:
            return
        self._pending_state = state_dict
        logger.debug(
            "DiLoCoCallback: checkpoint state deferred until on_load_model_weights"
        )
