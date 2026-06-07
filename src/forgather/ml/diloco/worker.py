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
import torch.distributed as dist
import torch.nn as nn

from .client import DiLoCoClient
from .coordinator import CoordinatorClient
from .fragments import FragmentManager
from .param_view import ParamView, SimpleModelParamView
from .sync_backend import HttpStarBackend, OuterSyncBackend

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
        upload_dtype: Optional[str] = None,
        upload_sr: bool = False,
        download_dtype: str = "fp32",
        download_sr: bool = False,
        bf16_comm: Optional[bool] = None,
        timeout: float = 600,
        dylu: bool = False,
        heartbeat_interval: float = 30.0,
        num_fragments: int = 1,
        max_sync_retries: int = 3,
        param_view: Optional[ParamView] = None,
        group_id: Optional[str] = None,
        pp_rank: int = 0,
        pp_world_size: int = 1,
        auth_token: Optional[str] = None,
        verify_tls: bool = True,
        output_dir: Optional[str] = None,
        backend: Optional[OuterSyncBackend] = None,
        coordinator: Optional[CoordinatorClient] = None,
        report_sync_state: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.sync_every = sync_every
        # Local on-disk output dir (logs/checkpoints). Reported to the
        # server purely so the webui's DiLoCo view can correlate a worker
        # back to its forgather job by output_dir when the worker-id was
        # renamed away from the job's queue_id (issue #103). Not used by
        # the sync protocol.
        self.output_dir = output_dir
        self._initial_sync_every = sync_every
        # Wire precision (issue #130). Four server-authoritative knobs
        # advertised via ``/info`` and ratified by the callback; the
        # worker takes them verbatim from its construction kwargs.
        # ``bf16_comm`` is a deprecated alias for ``upload_dtype`` and
        # mutual-exclusive with it. See ``DiLoCoServer.__init__`` for
        # the matching server-side schema and dtype matrix.
        if bf16_comm is not None and upload_dtype is not None:
            raise ValueError(
                "DiLoCoWorker: pass either bf16_comm (deprecated) or "
                "upload_dtype, not both."
            )
        if bf16_comm is not None:
            upload_dtype = "bf16" if bf16_comm else "fp32"
        if upload_dtype is None:
            upload_dtype = "bf16"
        if upload_dtype not in ("fp32", "bf16"):
            raise ValueError(
                f"upload_dtype must be 'fp32' or 'bf16', got {upload_dtype!r}"
            )
        if download_dtype not in ("fp32", "bf16"):
            raise ValueError(
                f"download_dtype must be 'fp32' or 'bf16', got {download_dtype!r}"
            )
        self.upload_dtype = upload_dtype
        self.upload_sr = bool(upload_sr)
        self.download_dtype = download_dtype
        self.download_sr = bool(download_sr)
        # Legacy mirror for log lines / older callers reading
        # ``worker.bf16_comm`` (read-only — write via ``upload_dtype``).
        self.bf16_comm = self.upload_dtype == "bf16"
        self.worker_id = worker_id or self._generate_worker_id()
        self.dylu = dylu
        self.heartbeat_interval = heartbeat_interval
        self.num_fragments = num_fragments
        self.max_sync_retries = max_sync_retries

        # Pipeline-group registration metadata (issue #84). When
        # ``pp_world_size > 1`` this worker is one rank of a
        # ``group_id``; the registration payload carries the ``group``
        # block and the server treats all ranks as a single logical
        # DiLoCo worker that together covers the full model. Solo
        # workers leave defaults (group_id=None, pp_world_size=1).
        if pp_world_size < 1:
            raise ValueError(f"pp_world_size must be >= 1, got {pp_world_size}")
        if pp_rank < 0 or pp_rank >= pp_world_size:
            raise ValueError(
                f"pp_rank must satisfy 0 <= pp_rank < pp_world_size; "
                f"got pp_rank={pp_rank}, pp_world_size={pp_world_size}"
            )
        self.group_id = group_id
        self.pp_rank = pp_rank
        self.pp_world_size = pp_world_size

        # Param-view abstraction (issue #84). Defaults to a single-
        # module view that matches pre-#84 behavior. Pipeline trainers
        # pass a PipelineParamView covering only this rank's slice.
        self.param_view: ParamView = param_view or SimpleModelParamView(model)

        # Security (issue #90): forward bearer + TLS verification to
        # the HTTP client. ``auth_token=None`` lets the client fall
        # back to env-var / loopback-file discovery, which covers the
        # common locally-spawned case where forgather_server wrote the
        # token to the per-port file.
        self.client = DiLoCoClient(
            server_addr,
            timeout=timeout,
            token=auth_token,
            verify_tls=verify_tls,
        )

        # Outer-synchronization backend (issue #154). The bulk tensor legs
        # (join / synchronize / synchronize_fragment / leave) route through a
        # pluggable backend; the coordination plane (heartbeat, /info, control)
        # stays on ``self.client`` directly. Defaults to the HTTP star backend
        # wrapping the client above, so behavior is unchanged. A trainer/config
        # may inject a different backend without the worker knowing the
        # transport. The backend owns the upload wire representation: the
        # default takes the worker's negotiated ``upload_dtype`` / ``upload_sr``,
        # but for an injected backend the backend is authoritative and the
        # worker's copies (used for the /info report and log line) are advisory.
        self.backend: OuterSyncBackend = backend or HttpStarBackend(
            self.client, upload_dtype=self.upload_dtype, upload_sr=self.upload_sr
        )

        # Coordination surface (issue #154): heartbeat / negotiation, distinct
        # from the sync backend. Defaults to a facade over the same client, so
        # behavior is unchanged. A future backend whose parameter authority is
        # not the coordinator (serverless collective, shared-mem) can inject a
        # coordinator pointing at the HTTP server while syncing peer-to-peer.
        self.coordinator: CoordinatorClient = coordinator or CoordinatorClient(
            self.client
        )
        # Report per-worker DiLoCo sync-state on the heartbeat so the
        # coordinator can show this worker's progress in its diagnostics — the
        # only sync-progress signal for an off-server backend (shared-memory).
        self.report_sync_state = bool(report_sync_state)

        # Replicated/collective backends (CollectiveBackend) make every rank an
        # independent DiLoCo replica that participates symmetrically in the outer
        # step: each rank computes its own pseudo-gradient and the backend's
        # all-reduce *is* the cross-rank agreement, so there is no leader/follower
        # split and no post-sync broadcast. Detected from the capability flag and
        # used below to make every rank its own worker (like a pipeline rank).
        self._symmetric = self.backend.runs_outer_optimizer == "replicated"

        # DDP rank-awareness. When running inside a torch-distributed
        # group (e.g. ``torchrun`` with WORLD_SIZE > 1, or a Forgather
        # DDP trainer), the whole DDP job is ONE DiLoCo worker —
        # rank 0 talks to the server (register / sync / heartbeat /
        # deregister) and broadcasts post-sync params to the other
        # ranks via NCCL so DDP stays consistent. Other ranks have no
        # business contacting the server and would 409 on register if
        # they tried.
        #
        # Under pipeline parallel (``pp_world_size > 1``), every rank
        # IS its own DiLoCo worker — there's nothing to broadcast
        # across pipeline ranks because each rank owns a different
        # slice. The DDP leader/follower model only kicks in for
        # within-stage DDP replicas (not currently composed with
        # pipeline in the forgather trainer, but the plumbing leaves
        # the door open).
        if dist.is_available() and dist.is_initialized():
            self._is_dist = True
            self._ddp_rank = dist.get_rank()
            self._ddp_world_size = dist.get_world_size()
        else:
            self._is_dist = False
            self._ddp_rank = 0
            self._ddp_world_size = 1
        if self._symmetric and self.pp_world_size > 1:
            # Collective + pipeline is incompatible: a collective all-reduce
            # covers the *full* model, but each pipeline rank owns only its
            # slice. Fail loud rather than mismatch the collective.
            raise ValueError(
                "DiLoCo collective backend (replicated outer optimizer) is not "
                "compatible with pipeline parallel (pp_world_size="
                f"{self.pp_world_size}): the all-reduce spans the full model "
                "while each pipeline rank holds only its slice. Use the HTTP "
                "backend for pipeline groups."
            )
        if self.pp_world_size > 1:
            # Every pipeline rank is its own DiLoCo worker (each owns
            # one DDP-group-of-one); we are always the leader of that
            # group-of-one. Cross-pipeline-rank broadcast is a no-op
            # (different slices).
            self._is_leader = True
            # Pipeline + within-stage DDP composition isn't supported
            # by the forgather trainer today. Under pure pipeline,
            # the torch.distributed world_size equals pp_world_size
            # (each process IS one pipeline rank). Within-stage DDP
            # would multiply the world_size by the per-stage replica
            # count, so world_size > pp_world_size is the signal. If
            # both were active, the worker would build a
            # PipelineParamView covering this rank's slice while DDP
            # would also try to broadcast post-sync params via NCCL —
            # but each pipeline rank holds different parameter names,
            # so the broadcast collective would mismatch and either
            # deadlock or surface as an opaque NCCL error. When
            # pipeline+DDP composition lands, this gate will check
            # for the ``pp_group`` plumbing instead.
            if self._is_dist and self._ddp_world_size > self.pp_world_size:
                raise ValueError(
                    f"DiLoCo + pipeline-parallel (pp_world_size="
                    f"{self.pp_world_size}) is not yet compatible "
                    f"with within-stage DDP "
                    f"(torch.distributed world_size="
                    f"{self._ddp_world_size}, exceeds pp_world_size). "
                    "Use one process per pipeline rank without "
                    "within-stage DDP, or wait for the pipeline+DDP "
                    "composition to land."
                )
            # DyLU under pipeline groups would push ranks to different
            # ``sync_every`` settings and break the group barrier
            # (every rank must hit the sync boundary at the same
            # local step). The server's ``_compute_dylu_sync_every``
            # also gates on this, but failing at construction time
            # gives the operator a clear single-process error rather
            # than a silent wire-level skip.
            if self.dylu:
                raise ValueError(
                    f"DiLoCo DyLU is not compatible with pipeline "
                    f"groups (pp_world_size={self.pp_world_size}). "
                    "Per-rank sync-every adjustments would desync "
                    "the group's barrier. DyLU is server-controlled — "
                    "restart the diloco server without --dylu, or use a "
                    "non-pipeline trainer."
                )
        elif self._symmetric:
            # Collective backend: every rank is its own independent DiLoCo
            # replica (owns the full model, unlike a pipeline rank's slice).
            # Every rank registers, heartbeats, and syncs; the backend's
            # all-reduce is the barrier, so there is no leader/follower split or
            # post-sync broadcast. Mirrors the pipeline "every rank is its own
            # worker" model above.
            self._is_leader = True
        else:
            self._is_leader = self._ddp_rank == 0

        # Fragment manager (None if num_fragments <= 1)
        self._fragment_manager: Optional[FragmentManager] = None
        if num_fragments > 1:
            # Streaming-fragment sync isn't DDP-rank-aware yet —
            # followers would each spawn a background thread submitting
            # ``submit_fragment_pseudogradients`` against the server
            # with an unregistered worker_id (since followers never
            # registered), racing the leader. Refuse the combo at
            # construction time rather than silently misbehave. The
            # non-streaming path (num_fragments==1) IS DDP-rank-aware
            # via the leader/follower split in start()/_sync().
            #
            # Pipeline+fragments IS supported (issue #84): each
            # pipeline rank fragments its own slice; the server's
            # per-fragment barrier coordinates across ranks. The DDP
            # restriction above is unchanged.
            if self._is_dist and self._ddp_world_size > 1 and self.pp_world_size == 1:
                raise ValueError(
                    f"DiLoCo streaming-fragment sync (num_fragments="
                    f"{num_fragments}) is not yet compatible with DDP "
                    f"(world_size={self._ddp_world_size}). Use "
                    "num_fragments=1 under DDP, or run streaming on a "
                    "single-process worker (no torch.distributed group)."
                )
            # FragmentManager partitions named_parameters of its model
            # argument; under pipeline this needs to be the slice's
            # params, not the meta-device root model. Pass through the
            # ParamView's view instead.
            self._fragment_manager = FragmentManager(self.param_view, num_fragments)

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
        # Relayed trainer-control command (save / save-and-stop / abort),
        # delivered on the heartbeat response and drained by the callback.
        self._pending_command: Optional[str] = None
        self._pending_command_lock = threading.Lock()
        # Latest unified-stats snapshot, set by the DiLoCo callback from the
        # trainer's log dict and shipped on the next heartbeat (consume-once,
        # so the server's loss EMA isn't re-fed the same sample if a log
        # didn't advance between heartbeats). See diloco/stats.py.
        self._pending_stats: Optional[dict] = None
        self._pending_stats_lock = threading.Lock()

        # Streaming state: at most one fragment in-flight at a time.
        # The background thread submits pseudo-gradients and stores the
        # result. The main thread applies the result before starting the
        # next fragment submission.
        self._inflight_thread: Optional[threading.Thread] = None
        self._inflight_result: Optional[
            Tuple[int, Optional[Dict[str, torch.Tensor]]]
        ] = None

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

    def _broadcast_params_from_leader(self):
        """Broadcast model parameters from rank 0 to all other DDP ranks.

        No-op when not in a torch-distributed group. Used to keep DDP
        ranks consistent after the leader applies new global params
        received from the DiLoCo server — without this, the leader's
        post-sync weights would diverge from the followers' stale
        weights, breaking DDP's "all ranks have identical params"
        invariant.

        Collective: every DDP rank in the group must call this at the
        same logical step. The leader sends; the followers receive.

        Under pipeline parallel (``pp_world_size > 1``), this is a
        no-op: every pipeline rank is its own DiLoCo worker and owns a
        different slice, so there's nothing to broadcast across ranks.
        When pipeline + within-stage DDP composition is added later,
        this will broadcast across the within-stage DDP sub-group only.
        """
        if not self._is_dist:
            return
        if self._symmetric:
            # Collective backend: every rank already ran the replicated outer
            # step and holds identical params (and started from the same
            # rank-0-broadcast init), so there is no leader to broadcast from.
            return
        if self.pp_world_size > 1:
            # Each rank IS the leader of its own DDP-group-of-one; no
            # cross-rank broadcast is meaningful with disjoint slices.
            return
        for _name, p in self.param_view.named_parameters():
            dist.broadcast(p.data, src=0)

    def start(self):
        """Register with server, load global params, install optimizer hooks.

        Under DDP, only rank 0 talks to the server. All ranks install
        the optimizer hook (so they participate in the broadcast at
        sync time), but the HTTP round-trip is leader-only.
        """
        if self._active:
            logger.warning("DiLoCoWorker already active")
            return

        if self._is_leader:
            logger.info(
                f"DiLoCoWorker {self.worker_id}: joining sync backend "
                f"({type(self.backend).__name__}, DDP rank "
                f"{self._ddp_rank}/{self._ddp_world_size})"
            )

            # Join the backend and get the initial global params.
            worker_info = self._get_worker_info()
            global_params = self.backend.join(
                worker_id=self.worker_id, worker_info=worker_info
            )

            # If the backend's join didn't register us with the HTTP coordinator
            # (e.g. shared-memory, whose join is a region attach), register
            # separately for coordinator membership so the server tracks us in
            # its diagnostics. The returned params are ignored — the backend is
            # our source of truth.
            if not self.backend.registers_with_coordinator:
                try:
                    self.coordinator.register(self.worker_id, worker_info)
                except Exception as exc:  # best-effort: diagnostics, not sync
                    logger.warning(
                        f"DiLoCoWorker {self.worker_id}: coordinator "
                        f"registration failed (diagnostics only): {exc}"
                    )

            # Load global params into model
            self._apply_global_params(global_params)
            self._save_global_params_snapshot()
        else:
            logger.info(
                f"DiLoCoWorker {self.worker_id}: DDP follower "
                f"(rank {self._ddp_rank}/{self._ddp_world_size}); leader "
                "owns the server connection."
            )

        # Broadcast the leader's (post-register) params to all DDP
        # ranks so everyone starts training from the same checkpoint.
        # Redundant if all ranks loaded from the same --model-id-or-path
        # but cheap insurance — the server may have returned params
        # different from what the leader loaded (other DiLoCo hosts
        # already pushed updates).
        self._broadcast_params_from_leader()
        if not self._is_leader:
            # Followers' global-snapshot tracks the leader's so future
            # sync-time pseudograd math (leader-only) and any
            # diagnostic comparisons stay consistent.
            self._save_global_params_snapshot()

        # Install optimizer hook (on every rank — followers need to
        # be in the broadcast collective at sync time).
        hook = self.optimizer.register_step_post_hook(self._post_step_hook)
        self._hooks.append(hook)

        self._active = True
        self._local_step = 0

        # Start heartbeat thread (leader only — followers don't have
        # a server connection to heartbeat to)
        if self._is_leader and self.heartbeat_interval > 0:
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
            f"up={self.upload_dtype}{'+SR' if self.upload_sr else ''}, "
            f"down={self.download_dtype}{'+SR' if self.download_sr else ''}, "
            f"dylu={self.dylu}{streaming_info}"
        )

    def stop(self):
        """Remove hooks and deregister from server.

        Under DDP, only the leader (rank 0) deregisters; followers
        never registered to begin with. Hook removal is per-rank.
        """
        if not self._active:
            return

        # Wait for any in-flight fragment to complete (leader only —
        # followers have no in-flight server work).
        if self._is_leader:
            self._wait_and_apply_inflight_fragment()

            # Stop heartbeat thread
            if self._heartbeat_thread is not None:
                self._heartbeat_stop.set()
                self._heartbeat_thread.join(timeout=5)
                self._heartbeat_thread = None

        # Remove optimizer hooks (every rank installed one)
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        # Leave the backend (leader only), and the coordinator too if we
        # registered there separately for membership.
        if self._is_leader:
            self.backend.leave(worker_id=self.worker_id)
            if not self.backend.registers_with_coordinator:
                try:
                    self.coordinator.deregister(self.worker_id)
                except Exception:  # best-effort
                    pass

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
        """Gather worker metadata for registration.

        Carries the structural slice fingerprint (``param_shapes`` for
        this rank's slice) and, when ``pp_world_size > 1``, the
        ``group`` block declaring this worker's pipeline-group
        membership. See issue #84.
        """
        info = {
            "hostname": platform.node(),
            "sync_every": self.sync_every,
            # Report the resolved wire-precision settings (issue #130).
            # Kept distinct rather than collapsed into ``bf16_comm`` so
            # the server can log / audit per-direction choices. The
            # legacy ``bf16_comm`` key is included for older servers
            # that grep it.
            "upload_dtype": self.upload_dtype,
            "upload_sr": self.upload_sr,
            "download_dtype": self.download_dtype,
            "download_sr": self.download_sr,
            "bf16_comm": self.bf16_comm,
            "dylu": self.dylu,
            # Optional local output dir, for webui job correlation only
            # (issue #103); omitted when unknown.
            **({"output_dir": self.output_dir} if self.output_dir else {}),
            # Structural slice fingerprint. Server validates per-slice
            # shape consistency at register time and verifies group
            # coverage at seal time. For solo workers the slice IS the
            # full model, and the contract collapses to the pre-#84
            # full-model fingerprint check.
            "param_shapes": self.param_view.param_shapes(),
        }

        # Pipeline-group block (issue #84). Solo workers omit this and
        # the server treats them as a degenerate group of one.
        if self.pp_world_size > 1:
            info["group"] = {
                "group_id": self.group_id,
                "pp_rank": self.pp_rank,
                "pp_world_size": self.pp_world_size,
            }

        # Add GPU info if available
        if torch.cuda.is_available():
            info["num_gpus"] = torch.cuda.device_count()
            info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]

        return info

    def _save_global_params_snapshot(self):
        """Save a CPU copy of this rank's slice as the global reference point.

        The snapshot dtype follows the live model dtype (see
        ``ParamView.snapshot``). Issue #130's original design proposed
        an unconditional fp32 master snapshot on the worker to preserve
        the server's fp32 master across the round trip; after working
        through the arithmetic we dropped that: when the server casts
        to bf16 with SR for download, the "fp32 master" on the worker
        would hold bf16-quantized values padded with mantissa zeros —
        same information at twice the storage. The four-knob
        precision schema + Python's natural promotion in
        ``compute_pseudograds`` covers the experiment matrix correctly
        without a forced cast here. See
        ``docs/design/diloco-pipeline-groups.md`` for the full
        discussion.
        """
        self._global_params = self.param_view.snapshot()

    def _compute_pseudogradients(self) -> Dict[str, torch.Tensor]:
        """
        Compute pseudo-gradients: global_params - local_params.

        This represents the negative of the accumulated local update direction.
        The server's outer optimizer uses these as gradients to update the
        global parameters toward where workers have moved.

        Under pipeline parallel each rank only computes pseudo-gradients
        for its own slice; the server aggregates per-name across the
        contributing slices.
        """
        return self.param_view.compute_pseudograds(self._global_params)

    def _apply_global_params(self, global_params: Dict[str, torch.Tensor]):
        """Load updated global params (or this rank's slice of them) into the model."""
        self.param_view.apply_global(global_params)

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

        Under DDP, only the leader (rank 0) computes pseudogradients,
        sends them to the server, and applies the returned global
        params. All ranks then participate in a broadcast so DDP stays
        consistent. Followers skip the HTTP work but still increment
        their step counters / sync_count to stay in lockstep with the
        leader.

        On connection error, the leader attempts to re-register with
        the server and retry the sync up to max_sync_retries times.
        If all retries fail, logs the error and continues training
        (the sync is skipped; followers fall through to the broadcast
        of the leader's unchanged params, which is a no-op).
        """
        t0 = time.time()

        if self._is_leader:
            logger.info(
                f"DiLoCoWorker {self.worker_id}: starting sync "
                f"(round {self._sync_count + 1}, after {self._local_step} local steps)"
            )

            # Compute raw pseudo-gradients (the backend applies the wire cast).
            pseudograds = self._compute_pseudogradients()

            # Submit with retry on connection failure
            result = None
            retry_delay = 2.0
            for attempt in range(self.max_sync_retries + 1):
                try:
                    result = self.backend.synchronize(
                        worker_id=self.worker_id, pseudograds=pseudograds
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

            # A committed round yields the next global params; a skipped/failed
            # round (or exhausted retries) leaves them unset.
            new_global_params = (
                result.params if result is not None and result.committed else None
            )

            if new_global_params is not None:
                # Wire sizes are reported by the backend, which owns the cast and
                # so is the only place that knows the on-wire (e.g. bf16) size.
                # Fall back to estimating from the tensors for backends that
                # don't report them.
                send_bytes = (
                    result.sent_bytes
                    if result.sent_bytes is not None
                    else sum(p.numel() * p.element_size() for p in pseudograds.values())
                )
                recv_bytes = (
                    result.recv_bytes
                    if result.recv_bytes is not None
                    else sum(
                        p.numel() * p.element_size() for p in new_global_params.values()
                    )
                )
                self._last_sync_send_bytes = send_bytes
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

        # Broadcast post-sync params from leader to all followers so
        # DDP's "all ranks have identical params" invariant holds.
        # Every DDP rank must participate in the collective at the
        # same logical step — _local_step is in lockstep across ranks
        # (DDP's gradient all-reduce keeps optimizer steps synchronized
        # 1:1) so this fires on every rank simultaneously.
        self._broadcast_params_from_leader()
        if not self._is_leader:
            # Followers refresh their snapshot to match the leader's
            # post-sync params so any future diagnostic comparison
            # remains apples-to-apples.
            self._save_global_params_snapshot()

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
            global_params = self.backend.join(
                worker_id=self.worker_id, worker_info=worker_info
            )
            self._apply_global_params(global_params)
            self._save_global_params_snapshot()
            self._reconnections += 1
            logger.info(f"DiLoCoWorker {self.worker_id}: reconnected successfully")
        except Exception as e:
            logger.warning(f"DiLoCoWorker {self.worker_id}: reconnection failed: {e}")

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

        # Compute pseudo-gradients for this fragment from this rank's
        # slice. Under pipeline parallel each rank submits its own
        # slice's portion of fragment_id; the server aggregates per-name
        # across the contributing ranks.
        pseudograds = self._fragment_manager.compute_fragment_pseudogradients(
            fragment_id,
            self._global_params,
            self.param_view,
        )

        logger.info(
            f"DiLoCoWorker {self.worker_id}: submitting fragment {fragment_id} "
            f"(step {self._local_step})"
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
            result = self.backend.synchronize_fragment(
                worker_id=self.worker_id,
                fragment_id=fragment_id,
                pseudograds=pseudograds,
            )
            # Gate on commit, same as the full-model path: a non-committed
            # round contributes no new params (None flows to the apply guard
            # in _wait_and_apply_inflight_fragment).
            params = result.params if result.committed else None
            self._inflight_result = (fragment_id, params)

            elapsed = time.time() - start_time
            self._total_sync_time += elapsed
            if result.sent_bytes is not None:
                self._last_sync_send_bytes = result.sent_bytes

            sent_mb = (
                f"{result.sent_bytes / 1e6:.1f} MB, "
                if result.sent_bytes is not None
                else ""
            )
            logger.debug(
                f"DiLoCoWorker {self.worker_id}: fragment {fragment_id} "
                f"sync complete ({sent_mb}{elapsed:.1f}s)"
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
                    frag_id, new_params, self.param_view, self._global_params
                )
                self._fragment_syncs += 1

    def _heartbeat_loop(self):
        """Background thread that sends periodic heartbeats to the server."""
        while not self._heartbeat_stop.wait(timeout=self.heartbeat_interval):
            if not self._active:
                break
            try:
                speed = self.get_steps_per_second()
                response = self.coordinator.heartbeat(
                    self.worker_id,
                    steps_per_second=speed,
                    stats=self._consume_stats(),
                    sync_state=(
                        self._sync_state_for_heartbeat()
                        if self.report_sync_state
                        else None
                    ),
                )

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

                # Capture any relayed trainer-control command (save /
                # save-and-stop / abort) for the callback to apply on its
                # next step. Last-one-wins if several arrive before pickup
                # (stop/abort are terminal, so coalescing is harmless).
                cmd = response.get("command")
                if cmd is not None:
                    with self._pending_command_lock:
                        self._pending_command = cmd
                    logger.info(
                        f"DiLoCoWorker {self.worker_id}: received control "
                        f"command '{cmd}' from server"
                    )
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

    def consume_pending_command(self) -> Optional[str]:
        """Return and clear any relayed trainer-control command.

        Called by the DiLoCo callback (leader rank) each step. Returns
        ``None`` when nothing is queued. Thread-safe against the heartbeat
        thread that sets it.
        """
        with self._pending_command_lock:
            cmd = self._pending_command
            self._pending_command = None
            return cmd

    def set_stats(self, snap: Optional[dict]) -> None:
        """Merge a unified-stats snapshot into the pending heartbeat payload.

        Called by the DiLoCo callback (leader rank) from ``on_log`` and
        ``on_evaluate``. The two write disjoint fields (train metrics vs
        ``eval_loss``/``eval_step``), so we *merge* rather than replace: a
        merge means an eval reported between heartbeats isn't clobbered by a
        subsequent ``on_log`` (and vice versa) before the heartbeat ships it.
        Per-key last-write-wins keeps the latest value; the whole dict is
        cleared once consumed by a heartbeat.
        """
        if not snap:
            return
        with self._pending_stats_lock:
            if self._pending_stats is None:
                self._pending_stats = dict(snap)
            else:
                self._pending_stats.update(snap)

    def _consume_stats(self) -> Optional[dict]:
        """Return and clear the pending stats snapshot (consume-once)."""
        with self._pending_stats_lock:
            snap = self._pending_stats
            self._pending_stats = None
            return snap

    def get_steps_per_second(self) -> float:
        """Compute current training speed from step timestamps."""
        if len(self._step_timestamps) < 2:
            return 0.0
        duration = self._step_timestamps[-1] - self._step_timestamps[0]
        if duration <= 0:
            return 0.0
        return (len(self._step_timestamps) - 1) / duration

    def _sync_state_for_heartbeat(self) -> dict:
        """Curated per-worker sync metrics for the coordinator's diagnostics.

        Flat numeric dict (the server sanitizes it). For an off-server backend
        (shared-memory) this is the only signal of the worker's sync progress —
        its server-side ``sync_round`` stays 0 since it never submits."""
        return {
            "sync_count": self._sync_count,
            "last_sync_time": self._last_sync_time,
            "total_sync_time": self._total_sync_time,
            "last_send_mb": self._last_sync_send_bytes / 1e6,
            "last_recv_mb": self._last_sync_recv_bytes / 1e6,
            "sync_every": self.sync_every,
        }

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
