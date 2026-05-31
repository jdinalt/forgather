"""
DiLoCo HTTP Parameter Server.

Standalone process that holds global model parameters, receives pseudo-gradients
from workers, and applies an outer optimizer step. Supports two modes:

- **Synchronous** (default): Workers block at a barrier until all have submitted.
  Server averages all pseudo-gradients and applies one outer optimizer step.

- **Asynchronous**: Workers submit and receive updated params immediately without
  waiting. Uses Delayed Nesterov (DN) momentum to avoid momentum amplification
  from stale gradients, and Dynamic Local Updates (DyLU) to adapt sync frequency
  per worker based on relative speed.

Usage:
    # Synchronous (default)
    server = DiLoCoServer(model_state_dict, num_workers=3, port=8512)

    # Asynchronous with DN momentum
    server = DiLoCoServer(model_state_dict, num_workers=3, port=8512,
                          async_mode=True, dn_buffer_size=3)

    server.run()   # Blocking
    server.start() # Non-blocking (background thread)
"""

import base64
import hashlib
import io
import json
import logging
import os
import socket
import ssl
import struct
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import torch

from forgather.ml.diloco.auth import authenticate_request
from forgather.ml.diloco.model_def import (
    MODEL_HASH_HEADER,
    compute_bundle_hash,
    pack_model_def,
)
from forgather.ml.sharded_checkpoint import (
    find_latest_checkpoint,
)
from forgather.ml.sharded_checkpoint import load_checkpoint as load_model_checkpoint
from forgather.ml.sharded_checkpoint import (
    maybe_delete_oldest_checkpoint,
    next_checkpoint_path,
)
from forgather.ml.sharded_checkpoint import save_checkpoint as save_model_checkpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class WorkerInfo:
    """Registered worker metadata."""

    worker_id: str
    hostname: str
    registered_at: float
    last_heartbeat: float
    sync_round: int = 0
    last_sync_server_round: int = 0  # Server round when this worker last synced
    steps_per_second: float = 0.0
    # Worker's local output dir, reported at registration. Used only by
    # the webui to correlate a worker to its forgather job by output_dir
    # when the worker-id was renamed away from the job's queue_id — e.g. a
    # resumable run that must reuse a stable worker name (issue #103).
    output_dir: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerGroup:
    """A set of workers that together cover the server's full parameter set.

    Solo workers form a degenerate group of one with ``pp_world_size=1``
    — the slice they declare must equal the server's full param set,
    matching the pre-#84 fingerprint contract. Pipeline-parallel workers
    register with ``pp_world_size > 1``; each member declares only its
    rank's slice. The group is "sealed" once ``len(members) ==
    pp_world_size``, at which point the union of all member slices must
    exactly cover ``self._param_names`` (modulo tied-parameter aliases).

    Sync barrier semantics: a sync round releases only when every member
    of every group has submitted its slice's pseudo-gradients. The
    server's outer optimizer then aggregates contributions per-name —
    only the workers whose slice contained a given name participate in
    that name's average.

    Worker-death policy (issue #84): on any member's death — heartbeat
    timeout, deregister, or partial-group rollback — every member of the
    group is evicted atomically. The remaining members would only hold
    an incomplete slice of the model and could not produce valid
    pseudo-gradients.
    """

    group_id: str
    pp_world_size: int
    members: Dict[int, str] = field(default_factory=dict)  # pp_rank -> worker_id
    member_param_names: Dict[int, set] = field(default_factory=dict)
    created_at: float = 0.0
    sealed: bool = False


@dataclass
class WorkQueue:
    """One work-unit queue, keyed by ``(dataset_id, shuffle_seed)``.

    A queue is a flat bitmap of K bits. An unset bit means "available";
    a set bit means "issued — consumed from the queue regardless of
    worker fate". Issuance is one-way: a unit is never returned to the
    queue, so no per-unit timeout / heartbeat tracking is needed and no
    row is ever trained on twice within an epoch. Worst case a dying
    worker loses ≤ 1 unit out of K (default 1024).

    ``completed`` is an optional second bitmap tracking units a worker
    confirmed it drained. Not required for correctness — workers MAY
    call ``/work/complete``; the bitmap stays all-zero if they don't.
    Useful only for the diagnostic surface to distinguish
    "issued ∧ completed" from "issued, fate unknown".

    ``hint_length`` is the row count the *first* worker reported when
    registering this dataset_id. Later registrations of the same
    dataset_id must report a matching length, else 409.

    ``by_worker`` accumulates per-worker request/complete counters for
    the diagnostic surface — same rationale as ``completed``: nothing
    in the issuance path depends on it.
    """

    total_units: int  # K
    issued: bytearray
    completed: bytearray
    hint_length: int
    issued_count: int = 0
    completed_count: int = 0
    by_worker: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Optional dataset-identity strings the first registering worker
    # supplies. They're surfaced as-is via ``/work/queues`` and
    # ``/work/queue`` so the webui can show a human-readable label
    # ("roneneldan/TinyStories@train") next to the otherwise-opaque
    # 16-hex ``dataset_id``. Best-effort — workers that don't ship
    # these fields just leave them None.
    dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_split: Optional[str] = None
    dataset_revision: Optional[str] = None
    dataset_data_files: Optional[List[str]] = None

    @classmethod
    def empty(
        cls,
        total_units: int,
        hint_length: int,
        *,
        dataset_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_split: Optional[str] = None,
        dataset_revision: Optional[str] = None,
        dataset_data_files: Optional[List[str]] = None,
    ) -> "WorkQueue":
        nbytes = (total_units + 7) // 8
        return cls(
            total_units=total_units,
            issued=bytearray(nbytes),
            completed=bytearray(nbytes),
            hint_length=hint_length,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            dataset_revision=dataset_revision,
            dataset_data_files=dataset_data_files,
        )


def _queue_summary_dict(
    dataset_id: str, shuffle_seed: int, q: "WorkQueue"
) -> Dict[str, Any]:
    """Shape the JSON-serializable summary the webui consumes for both
    ``/work/queues`` (list) and ``/work/queue`` (detail). The detail
    response adds bitmaps + by_worker on top of this base.
    """
    hint: Dict[str, Any] = {"length": q.hint_length}
    if q.dataset_path is not None:
        hint["path"] = q.dataset_path
    if q.dataset_name is not None:
        hint["name"] = q.dataset_name
    if q.dataset_split is not None:
        hint["split"] = q.dataset_split
    if q.dataset_revision is not None:
        hint["revision"] = q.dataset_revision
    if q.dataset_data_files:
        hint["data_files"] = list(q.dataset_data_files)
    return {
        "dataset_id": dataset_id,
        "shuffle_seed": shuffle_seed,
        "total_units": q.total_units,
        "issued_count": q.issued_count,
        "completed_count": q.completed_count,
        "hint": hint,
    }


def _bit_set(bm: bytearray, i: int) -> None:
    bm[i >> 3] |= 1 << (i & 7)


def _bit_get(bm: bytearray, i: int) -> bool:
    return bool(bm[i >> 3] & (1 << (i & 7)))


def _find_lowest_unset(bm: bytearray, total_bits: int) -> int:
    """Return the index of the lowest unset bit in ``bm`` (within the
    first ``total_bits`` bits), or -1 if every bit is set.

    Scans byte-by-byte (each byte is 8 bits) and falls back to a
    bitwise scan within the first byte that isn't 0xFF. K=1024 means
    128 bytes per queue — even worst case (~half full) is a couple of
    hundred byte comparisons, dwarfed by the HTTP round-trip cost.
    """
    full_bytes = total_bits >> 3
    for byte_idx in range(full_bytes):
        b = bm[byte_idx]
        if b != 0xFF:
            for bit in range(8):
                if not (b & (1 << bit)):
                    return (byte_idx << 3) | bit
    # Handle a possible partial trailing byte (when K isn't a
    # multiple of 8). Rare in practice — default K=1024 — but harmless.
    extra = total_bits & 7
    if extra:
        byte_idx = full_bytes
        b = bm[byte_idx]
        for bit in range(extra):
            if not (b & (1 << bit)):
                return (byte_idx << 3) | bit
    return -1


# Per-control-action allowlist of fields that are safe to audit-log.
# Today's actions only carry intent metadata (no secrets); the allowlist
# guards against a future control endpoint that accepts credential
# material silently landing it in the audit log. Unknown actions log
# no data fields.
_CONTROL_AUDIT_FIELDS: Dict[str, frozenset] = {
    "save_state": frozenset(),
    "kick_worker": frozenset({"worker_id"}),
    "update_optimizer": frozenset({"lr", "momentum", "nesterov"}),
    "update_num_workers": frozenset({"num_workers"}),
    "shutdown": frozenset(),
}


def _audit_control_data(action: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Redact a ``/control/{action}`` payload to its allowlisted fields.

    Unknown actions return an empty dict — surfacing that a payload was
    received without exposing its contents. Unknown fields under a
    known action are silently dropped (logged at DEBUG via the audit
    record's absence of those keys, not as a warning, to keep audit
    output deterministic).
    """
    allowed = _CONTROL_AUDIT_FIELDS.get(action, frozenset())
    if not isinstance(data, dict) or not allowed:
        return {}
    return {k: v for k, v in data.items() if k in allowed}


def _utc_iso_now() -> str:
    """ISO-8601 UTC timestamp suitable for log records.

    Uses ``datetime.now(timezone.utc).isoformat()``; the trailing
    timezone offset is ``+00:00`` so grep + sort work as expected.
    """
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _default_outer_optimizer_factory(params):
    """Default outer optimizer: SGD with Nesterov momentum (DiLoCo paper defaults)."""
    return torch.optim.SGD(params, lr=0.7, momentum=0.9, nesterov=True)


def _serialize_state_dict(state_dict: Dict[str, torch.Tensor]) -> bytes:
    """Serialize a state dict to bytes using torch.save."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


def _deserialize_state_dict(data: bytes) -> Dict[str, torch.Tensor]:
    """Deserialize bytes to a state dict using torch.load."""
    buf = io.BytesIO(data)
    return torch.load(buf, map_location="cpu", weights_only=True)


def _read_request_body(handler: BaseHTTPRequestHandler) -> bytes:
    """Read the full request body from an HTTP handler."""
    content_length = int(handler.headers.get("Content-Length", 0))
    return handler.rfile.read(content_length)


def _request_host(handler: BaseHTTPRequestHandler) -> Optional[str]:
    """Hostname the client used to reach us, from the ``Host`` header.

    Returns just the host (port stripped). Used to advertise a bulk
    listener URL that's routable *from the worker's perspective* when
    the server bound a wildcard address (``0.0.0.0`` / ``::``) and has
    no idea what its own routable address is. ``None`` when the header
    is absent or unparseable.
    """
    raw = handler.headers.get("Host")
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    # Strip the port. IPv6 literals are bracketed: ``[::1]:8512``.
    if raw.startswith("["):
        end = raw.find("]")
        if end != -1:
            return raw[1:end]
        return raw
    return raw.rsplit(":", 1)[0] if ":" in raw else raw


def _send_json_response(handler: BaseHTTPRequestHandler, data: dict, status: int = 200):
    """Send a JSON response."""
    body = json.dumps(data).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_tensor_response(
    handler: BaseHTTPRequestHandler,
    state_dict: Dict[str, torch.Tensor],
    extra_headers: Optional[Dict[str, str]] = None,
):
    """Send a state dict as an octet-stream response.

    ``extra_headers`` (e.g. ``{"X-Forgather-Bulk-Url": "..."}``) ride
    along on the same response so worker registration can learn the
    bulk-listener URL without an extra round-trip.
    """
    data = _serialize_state_dict(state_dict)
    handler.send_response(200)
    handler.send_header("Content-Type", "application/octet-stream")
    handler.send_header("Content-Length", str(len(data)))
    for k, v in (extra_headers or {}).items():
        handler.send_header(k, v)
    handler.end_headers()
    handler.wfile.write(data)


class DiLoCoServer:
    """
    Central DiLoCo parameter server.

    In synchronous mode, holds global model parameters, accepts pseudo-gradient
    submissions from workers, averages them when all workers have submitted
    (barrier), applies the outer optimizer, and returns updated global parameters.

    In asynchronous mode, applies each worker's pseudo-gradients immediately upon
    receipt. Uses Delayed Nesterov (DN) momentum to prevent momentum amplification
    from stale async gradients, and Dynamic Local Updates (DyLU) to recommend
    per-worker sync frequencies based on relative training speeds.

    Args:
        output_dir: Directory in which to save logs and checkpoints.
        num_workers: Expected number of workers.
        from_checkpoint: Path to specific model checkpoint to load. Defaults to searching output_dir.
        port: HTTP server port. If None, auto-selects an available port.
        outer_optimizer_factory: Callable that takes a parameter list and returns
            a torch.optim.Optimizer. Defaults to SGD(lr=0.7, momentum=0.9, nesterov=True).
        host: Host address to bind to. Defaults to "127.0.0.1".
        save_every_n_rounds: Save server state every N sync rounds. Set to = 0 to disable save.
        save_total_limit: Maximum number of checkpoints to keep. Oldest are
            deleted when the limit is exceeded. 0 means keep all.
        async_mode: If True, apply pseudo-gradients immediately without barrier.
        safetensors: Save using saftetensor, else torch.save()
        dn_buffer_size: Delayed Nesterov buffer size. In async mode, buffer this
            many pseudo-gradient submissions before applying the outer optimizer
            with momentum. Between buffered steps, apply simple gradient descent
            (no momentum). Set to 0 to disable DN (apply momentum every step).
            Only used in async mode.
        dylu_enabled: If True, compute per-worker recommended sync_every based
            on relative training speeds (Dynamic Local Updates). Only used in
            async mode.
        dylu_base_sync_every: Base sync_every for the fastest worker (H in paper).
            Slower workers get proportionally smaller values. Only used when
            dylu_enabled=True.
        heartbeat_timeout: Seconds since last heartbeat before a worker is
            considered dead and evicted. Set to 0 to disable health monitoring.
            Default: 120 seconds.
        min_workers: Minimum number of workers required to apply the outer
            optimizer in sync mode. If the number of registered workers drops
            below this value, the sync barrier will not release. Default: 1.

    The server no longer ships a built-in web dashboard; the
    forgather webui's DiLoCo view reproduces and supersedes it
    (see tools/forgather_server/webui/src/components/DiLoCoPanel.tsx).
    All control endpoints (/control/save_state, /kick_worker,
    /update_optimizer, /update_num_workers, /shutdown) are unchanged
    and remain the API the webui talks to.
    """

    def __init__(
        self,
        output_dir: str,
        num_workers: int,
        from_checkpoint: Optional[str] = None,
        port: Optional[int] = None,
        outer_optimizer_factory: Optional[Callable] = None,
        host: str = "127.0.0.1",
        save_every_n_rounds: int = 10,
        save_total_limit: int = 3,
        async_mode: bool = False,
        safetensors: bool = True,
        dn_buffer_size: int = 0,
        dylu_enabled: bool = False,
        dylu_base_sync_every: int = 500,
        sync_every: int = 500,
        bf16_comm: bool = True,
        num_fragments: int = 1,
        heartbeat_timeout: float = 120.0,
        min_workers: int = 1,
        auth_token: Optional[str] = None,
        ssl_context: Optional["ssl.SSLContext"] = None,
        bulk_port: Optional[int] = None,
        bulk_ssl_context: Optional["ssl.SSLContext"] = None,
        bulk_auth_enabled: bool = True,
        default_work_units: int = 1024,
    ):
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        if min_workers < 1:
            raise ValueError(f"min_workers must be >= 1, got {min_workers}")
        if default_work_units < 1:
            raise ValueError(
                f"default_work_units must be >= 1, got {default_work_units}"
            )

        self.num_workers = num_workers
        self.min_workers = min_workers
        self.host = host
        self.port = port or self._find_available_port()
        self.output_dir = output_dir
        self.save_every_n_rounds = save_every_n_rounds
        self.save_total_limit = save_total_limit
        self.async_mode = async_mode
        self.safetensors = safetensors
        self.dn_buffer_size = dn_buffer_size
        self.dylu_enabled = dylu_enabled
        self.dylu_base_sync_every = dylu_base_sync_every
        # Group-wide worker settings the server is authoritative for (issue
        # #53 follow-up). These MUST match across the group for the sync /
        # fragment barriers to be coherent, so the operator sets them on
        # the server and workers adopt them verbatim from /info. ``bf16_comm``
        # is centralized here too (the server doesn't need it to decode an
        # upload, but a single operator-facing knob keeps the group's wire
        # format consistent rather than per-worker).
        self.sync_every = sync_every
        self.bf16_comm = bf16_comm
        self.num_fragments = num_fragments
        self.heartbeat_timeout = heartbeat_timeout
        self.default_work_units = default_work_units
        self.outer_optimizer_factory = (
            outer_optimizer_factory or _default_outer_optimizer_factory
        )
        # Security (issue #90): bearer token for the control plane and
        # optional pre-built SSL context for the listening socket.
        # ``auth_token=None`` keeps backwards-compat for callers that
        # don't authenticate yet (existing tests, in-process workers
        # constructed directly without a token); production CLI/spawn
        # paths always populate it. ``ssl_context=None`` keeps the
        # listener cleartext.
        if auth_token is not None and not auth_token:
            # Empty-string token is treated as "auth disabled" by
            # ``verify_bearer``. That's intentional for the
            # ``auth_token=None`` path, but an explicit empty string
            # is almost always a misconfiguration (e.g. a token file
            # that's whitespace-only). Surface it loudly so the
            # operator notices.
            logger.warning(
                "DiLoCoServer constructed with auth_token='' (empty); "
                "auth is effectively disabled. Pass auth_token=None to "
                "do this intentionally, or supply a real token."
            )
        self.auth_token = auth_token
        self.ssl_context = ssl_context
        # Optional second listener for bulk data transport (pseudo-
        # gradients + global-params). When ``bulk_port`` is set the
        # three bulk endpoints are served *only* on that port; the
        # control port returns 404 with an ``X-Forgather-Bulk-Url``
        # hint header. Operators opt into the second listener for
        # throughput; on a trusted LAN they typically also disable
        # TLS+auth there (``bulk_ssl_context=None``,
        # ``bulk_auth_enabled=False``) — matching torch.distributed's
        # posture. Even with auth off, the per-request torch.load uses
        # ``weights_only=True`` so a malicious peer can only disrupt
        # training, not RCE the host.
        self.bulk_port = bulk_port
        self.bulk_ssl_context = bulk_ssl_context
        self.bulk_auth_enabled = bulk_auth_enabled
        self._bulk_server = None
        self._bulk_server_thread = None
        # Audit log (issue #90). Append-only JSONL records for events
        # worth reconstructing after the fact: registrations, evictions,
        # outer-optimizer steps, and control actions. Best-effort —
        # write errors are logged but don't fail the request that
        # triggered them. Lives next to model checkpoints in the
        # configured output_dir so a post-mortem operator finds it
        # alongside the rest of the run state.
        self._audit_lock = threading.Lock()
        self._audit_path = (
            os.path.join(output_dir, "diloco_audit.log") if output_dir else None
        )
        # Persistent append-mode handle, opened lazily on the first
        # audit record and reused for the server's lifetime (closed in
        # stop()/run()'s finally). Avoids an open/close syscall pair per
        # event.
        self._audit_fh = None
        self._running = False
        # Directory the model was loaded from, captured by load_state. The
        # /model_def endpoint serves the non-weight definition files (config,
        # custom code, tokenizer) from here so DiLoCo workers can construct
        # the model without a shared filesystem (issue #53).
        self._loaded_checkpoint_dir: Optional[str] = None
        # Packed bundle is content-stable for the server's lifetime
        # (_loaded_checkpoint_dir never changes), so build it once on first
        # request and cache it — avoids re-walking + re-reading the dir, and
        # holding N in-memory copies when many workers fetch concurrently.
        self._model_def_bundle: Optional[bytes] = None
        self._model_def_lock = threading.Lock()
        self.load_state(from_checkpoint)

    def _initialize(self, model_state_dict: Dict[str, torch.Tensor]):
        assert self._running == False
        # Model metadata (computed before parameters are wrapped)
        self._model_params = sum(v.numel() for v in model_state_dict.values())
        self._model_size_mb = sum(
            v.numel() * v.element_size() for v in model_state_dict.values()
        ) / (1024 * 1024)

        logger.info(
            f"Model loaded: {self._model_params:,} parameters ({self._model_size_mb:.1f} MB)"
        )

        # Global parameters - stored as nn.Parameters for the optimizer
        self._param_names: List[str] = list(model_state_dict.keys())
        self._param_list = torch.nn.ParameterList(
            [
                torch.nn.Parameter(v.clone().float().cpu(), requires_grad=False)
                for v in model_state_dict.values()
            ]
        )

        # Coarse model fingerprint advertised via /info. Workers use it to
        # decide whether a cached model-definition bundle is still valid
        # (issue #53) and as an early, pre-construction compatibility gate
        # (the per-parameter /register fingerprint stays the fine-grained
        # check). Derived from the parameter (name, shape) set the server
        # holds; deterministic across the server's lifetime.
        self._model_hash = self._compute_model_hash(model_state_dict)

        # Outer optimizer
        self.outer_optimizer = self.outer_optimizer_factory(
            self._param_list.parameters()
        )

        # Extract outer LR for use in DN direct gradient steps
        self._outer_lr = self.outer_optimizer.param_groups[0]["lr"]

        # Worker registry
        self._workers: Dict[str, WorkerInfo] = {}
        self._workers_lock = threading.Lock()

        # Worker-group registry (issue #84). Every registered worker_id
        # belongs to exactly one group. Solo workers form a degenerate
        # group of one (group_id == worker_id, pp_world_size=1). Pipeline-
        # parallel workers form a group of pp_world_size; the group is
        # sealed when all members have registered. ``_worker_to_group``
        # is the inverse index used by submit / death paths to find the
        # owning group from a worker_id. Both dicts are protected by
        # ``_workers_lock`` (same critical section as ``_workers``).
        self._groups: Dict[str, WorkerGroup] = {}
        self._worker_to_group: Dict[str, str] = {}

        # Sync state - uses a Condition for proper barrier synchronization.
        # Each round is tracked by number; completed round results are stored
        # so threads that wake up late still get the correct result.
        self._sync_round = 0
        self._pending_pseudograds: Dict[str, Dict[str, torch.Tensor]] = {}
        self._sync_cond = threading.Condition()
        self._completed_rounds: Dict[int, Dict[str, torch.Tensor]] = {}

        # Async state
        self._async_lock = threading.Lock()
        self._total_submissions = 0  # Total pseudo-gradient submissions received

        # Delayed Nesterov (DN) state - buffer pseudo-gradients, apply momentum
        # only every dn_buffer_size submissions to avoid momentum amplification
        # from stale async gradients.
        self._dn_grad_buffer: List[Dict[str, torch.Tensor]] = []

        # Fragment streaming state - for per-fragment sync.
        # Maps param name -> index in _param_list for fast lookup.
        self._param_name_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(self._param_names)
        }
        # Per-fragment sync tracking (sync mode):
        # (fragment_id, round) -> worker_id -> pseudograds
        self._fragment_pending: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = (
            defaultdict(dict)
        )
        self._fragment_rounds: Dict[int, int] = defaultdict(int)
        # (fragment_id, round) -> {worker_id: {name: tensor}}. Stored
        # per-worker so each rank in a pipeline group receives only the
        # names it submitted for this fragment (its slice's intersection
        # with the fragment), not the union across the group.
        self._completed_fragment_rounds: Dict[
            Tuple[int, int], Dict[str, Dict[str, torch.Tensor]]
        ] = {}
        # Reuse _sync_cond for fragment barrier notifications
        self._fragment_submissions = 0  # Total fragment submissions

        # Fault tolerance: dynamic barrier tracking.
        # _round_expected_workers is the set of worker IDs the sync barrier
        # expects for the current round. It is snapshotted from _workers at
        # the start of each round. Workers that join mid-round are NOT added
        # to the current round's expected set (they participate starting next
        # round). Workers that die are removed from this set, which may
        # cause the barrier to release early.
        self._round_expected_workers: Optional[set] = None

        # Health monitor (created on start/run if heartbeat_timeout > 0)
        self._health_monitor = None

        # Track worker deaths for status reporting
        self._total_worker_deaths = 0

        # Work-unit dispatch state: per (dataset_id, shuffle_seed) queue.
        # Keyed by Tuple[str, int]; in-process only — wire / disk forms use
        # a "dataset_id|seed" string-joined key (see save_state/load_state)
        # to dodge tuple-key serialization quirks.
        # ``dataset_id`` is the worker-computed hash of normalized
        # ``{path, name, split, data_files, revision}`` (see
        # docs/design/diloco-work-unit-dispatch.md).
        self._work_queues: Dict[Tuple[str, int], WorkQueue] = {}
        self._work_queues_lock = threading.Lock()
        # ``_dataset_lengths`` snapshots the first-registered length per
        # dataset_id so a later worker shipping a stale dataset config
        # (different row count) is caught with a 409 at register time.
        self._dataset_lengths: Dict[str, int] = {}

        # Server state
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._started_at: Optional[float] = None
        self._dirty = False

    @staticmethod
    def _find_available_port(start_port: int = 8512, max_attempts: int = 100) -> int:
        """Find an available port.

        Static so callers (e.g. the CLI) can resolve the concrete port
        an ephemeral ``--port 0`` will land on *before* constructing the
        server — the per-port token file must be keyed on the real port,
        not on 0.
        """
        for i in range(max_attempts):
            port = start_port + i
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                continue
        raise RuntimeError(
            f"No available port in range {start_port}-{start_port + max_attempts}"
        )

    @staticmethod
    def _compute_model_hash(model_state_dict: Dict[str, torch.Tensor]) -> str:
        """Coarse, deterministic fingerprint of the model's parameter set.

        Hashes the sorted ``(name, shape)`` pairs. Stable across restarts
        and machines for the same architecture; changes when the parameter
        topology changes. Advertised via /info so workers can validate a
        cached model-definition bundle and gate compatibility before
        constructing the model.
        """
        h = hashlib.sha256()
        for name in sorted(model_state_dict.keys()):
            shape = tuple(model_state_dict[name].shape)
            h.update(name.encode("utf-8"))
            h.update(repr(shape).encode("utf-8"))
        return h.hexdigest()

    def get_global_params(self) -> Dict[str, torch.Tensor]:
        """Get current global parameters as a state dict."""
        return {
            name: param.data.clone()
            for name, param in zip(self._param_names, self._param_list)
        }

    def _apply_outer_optimizer(self, pending_audit=None):
        """Average pending pseudo-gradients and apply the outer optimizer step.

        ``pending_audit``: when a list is passed, the ``outer_step``
        audit record is appended to it instead of written immediately —
        callers hold ``self._sync_cond`` and must flush the batch (via
        :meth:`_audit_many`) only after releasing it, so the disk write
        never stalls the barrier. When ``None`` (direct/test callers not
        under the lock) the record is written inline.

        Per-name aggregation is over **the workers whose slice contained
        that name**, not over the whole worker set. For solo groups the
        contributor set equals the full worker set and behavior collapses
        to the pre-#84 contract. For pipeline-parallel groups, each name
        is held by typically one rank per group (G ranks across G groups
        of pp_world_size members each); tied-parameter aliases may be
        held by multiple ranks and are averaged identically since the
        aliased pseudo-gradients are by construction the same data.
        """
        if not self._pending_pseudograds:
            return

        missing_contributors: List[str] = []
        for i, name in enumerate(self._param_names):
            contributors: List[torch.Tensor] = []
            for worker_pseudograds in self._pending_pseudograds.values():
                pg = worker_pseudograds.get(name)
                if pg is not None:
                    contributors.append(pg.float())
            if not contributors:
                # No worker carries this name — the group-coverage check
                # at registration should have prevented this. Skip the
                # outer optimizer step on this slot (grad=None makes
                # SGD/Adam treat it as a no-op).
                missing_contributors.append(name)
                self._param_list[i].grad = None
                continue
            avg = contributors[0].clone()
            for pg in contributors[1:]:
                avg.add_(pg)
            avg.div_(len(contributors))
            self._param_list[i].grad = avg

        if missing_contributors:
            sample = missing_contributors[:5]
            logger.error(
                f"_apply_outer_optimizer: {len(missing_contributors)} "
                f"param(s) had no contributor in this round: {sample}"
                f"{'...' if len(missing_contributors) > 5 else ''}. "
                f"The group-coverage check should have prevented this; "
                f"check for a stale group registry."
            )

        contributing_workers = list(self._pending_pseudograds.keys())
        self.outer_optimizer.step()
        self.outer_optimizer.zero_grad()

        self._sync_round += 1
        self._pending_pseudograds.clear()
        self._dirty = True

        logger.info(f"Outer optimizer step complete. Sync round: {self._sync_round}")
        outer_step_event = (
            "outer_step",
            {
                "sync_round": self._sync_round,
                "contributors": contributing_workers,
                "missing_contributors": missing_contributors or None,
            },
        )
        if pending_audit is not None:
            pending_audit.append(outer_step_event)
        else:
            self._audit(outer_step_event[0], **outer_step_event[1])

        # Periodic save
        if (
            self.save_every_n_rounds > 0
            and self._sync_round % self.save_every_n_rounds == 0
        ):
            self.save_state()

    def _apply_async_pseudograd(
        self, worker_id: str, pseudograds: Dict[str, torch.Tensor]
    ):
        """
        Apply a single worker's pseudo-gradients in async mode.

        If DN is disabled (dn_buffer_size=0), applies the outer optimizer with
        full momentum on every submission.

        If DN is enabled, buffers pseudo-gradients and alternates between:
        - Direct gradient steps (param -= lr * grad) for intermediate submissions
        - Full outer optimizer steps (with momentum) every dn_buffer_size submissions
        """
        if self.dn_buffer_size <= 0:
            # No DN: apply outer optimizer directly on each submission
            for i, name in enumerate(self._param_names):
                self._param_list[i].grad = pseudograds[name].float()
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
        else:
            # DN: buffer gradients, alternate between direct and momentum steps
            self._dn_grad_buffer.append(pseudograds)

            if len(self._dn_grad_buffer) >= self.dn_buffer_size:
                # Full momentum step: average the buffer, apply outer optimizer
                n = len(self._dn_grad_buffer)
                for i, name in enumerate(self._param_names):
                    avg_grad = None
                    for buffered_pg in self._dn_grad_buffer:
                        pg = buffered_pg[name].float()
                        if avg_grad is None:
                            avg_grad = pg.clone()
                        else:
                            avg_grad.add_(pg)
                    avg_grad.div_(n)
                    self._param_list[i].grad = avg_grad

                self.outer_optimizer.step()
                self.outer_optimizer.zero_grad()
                self._dn_grad_buffer.clear()
            else:
                # Intermediate step: direct gradient descent (no momentum)
                # param -= lr * grad
                with torch.no_grad():
                    for i, name in enumerate(self._param_names):
                        self._param_list[i].data.sub_(
                            self._outer_lr * pseudograds[name].float()
                        )

        self._sync_round += 1
        self._total_submissions += 1
        self._dirty = True

        # Periodic save
        if (
            self.save_every_n_rounds > 0
            and self._sync_round % self.save_every_n_rounds == 0
        ):
            self.save_state()

    def _get_params_by_names(self, param_names) -> Dict[str, torch.Tensor]:
        """Get current global parameters for a subset of parameter names."""
        return {
            name: self._param_list[self._param_name_to_idx[name]].data.clone()
            for name in param_names
            if name in self._param_name_to_idx
        }

    def _apply_fragment_outer_optimizer(
        self, pseudograds_list: List[Dict[str, Dict[str, torch.Tensor]]]
    ):
        """
        Average pseudo-gradients and apply the outer optimizer for a fragment.

        Only sets .grad on parameters present in the pseudo-gradients. The
        optimizer skips parameters with None grad, so only the fragment's
        parameters are updated. Momentum buffers for other parameters remain
        untouched.

        Args:
            pseudograds_list: List of per-worker pseudo-gradient dicts for this
                fragment. Each dict maps param_name -> tensor.

        Per-name aggregation is over **the workers whose submission
        carried that name** — same contributors-only model as the
        full-sync path. Under pipeline groups each rank's fragment_id_k
        covers a different subset of names (the fragment's intersection
        with the rank's slice); the per-name averaging naturally handles
        that.
        """
        if not pseudograds_list:
            return

        # Union of all submitted names; each name is updated using only
        # the workers that carried it.
        frag_param_names: List[str] = []
        seen: set = set()
        for worker_pg in pseudograds_list:
            for name in worker_pg.keys():
                if name not in seen:
                    seen.add(name)
                    frag_param_names.append(name)

        for name in frag_param_names:
            idx = self._param_name_to_idx[name]
            contributors: List[torch.Tensor] = [
                worker_pg[name].float()
                for worker_pg in pseudograds_list
                if name in worker_pg
            ]
            if not contributors:
                continue
            avg = contributors[0].clone()
            for pg in contributors[1:]:
                avg.add_(pg)
            avg.div_(len(contributors))
            self._param_list[idx].grad = avg

        # step() skips params with None grad; only fragment params updated
        self.outer_optimizer.step()
        self.outer_optimizer.zero_grad()
        self._dirty = True

    def _get_expected_worker_count(self) -> int:
        """
        Get the number of workers the sync barrier should wait for.

        In the first round (before any workers have submitted), uses the
        initial num_workers. After that, uses the size of
        _round_expected_workers, which is snapshotted from registered
        workers at the start of each round.
        """
        if self._round_expected_workers is not None:
            return len(self._round_expected_workers)
        return self.num_workers

    def _snapshot_round_expected_workers(self):
        """
        Snapshot the current set of registered workers as the expected
        participants for the next sync round. Called when a round completes
        and when the first submission of a round arrives (lazy init).
        """
        with self._workers_lock:
            self._round_expected_workers = set(self._workers.keys())

    def _handle_worker_death(self, worker_id: str):
        """
        Handle a dead worker: remove from registry, unblock barriers.

        This method is called by the HealthMonitor when a worker's heartbeat
        times out, or during explicit deregistration. It must handle both
        sync and async modes, full-model and fragment barriers.

        **Group eviction (issue #84)**: when the dying worker belongs to a
        pipeline-parallel group (``pp_world_size > 1``), every member of
        the group is evicted atomically. The remaining members would hold
        only a partial slice of the model and could not produce valid
        pseudo-gradients. Solo workers (``pp_world_size == 1``) are the
        common case and behave exactly like pre-#84: just the dying
        worker is removed, and its now-empty group entry is cleaned up.

        Lock ordering: _sync_cond -> _workers_lock (same as submit handlers).
        """
        # Audit records are collected here and flushed *after* the
        # _sync_cond is released — never do blocking disk I/O while
        # holding the barrier (it would stall every parked worker).
        pending_audit: List[Tuple[str, Dict[str, Any]]] = []
        with self._sync_cond:
            with self._workers_lock:
                if worker_id not in self._workers:
                    return

                # Identify the owning group and pick the eviction set.
                group_id = self._worker_to_group.get(worker_id)
                group = self._groups.get(group_id) if group_id else None
                if group is not None and group.pp_world_size > 1:
                    # Whole-group atomic eviction.
                    evict = [
                        wid for wid in group.members.values() if wid in self._workers
                    ]
                else:
                    evict = [worker_id]

                # Clean up the group entry: drop the dying member, remove
                # the group when the last member is gone.
                if group is not None:
                    for wid in evict:
                        for pr, m_wid in list(group.members.items()):
                            if m_wid == wid:
                                del group.members[pr]
                                group.member_param_names.pop(pr, None)
                                break
                    if not group.members:
                        self._groups.pop(group.group_id, None)

                for wid in evict:
                    self._workers.pop(wid, None)
                    self._worker_to_group.pop(wid, None)
                remaining = len(self._workers)

            self._total_worker_deaths += len(evict)

            # Update num_workers (but respect min_workers floor)
            self.num_workers = max(self.min_workers, remaining)

            if len(evict) > 1:
                logger.warning(
                    f"Worker {worker_id} died; evicting whole group "
                    f"'{group_id}' ({len(evict)} member(s)). "
                    f"Remaining: {remaining}, num_workers now {self.num_workers}"
                )
            else:
                logger.warning(
                    f"Worker {worker_id} died. "
                    f"Remaining: {remaining}, num_workers now {self.num_workers}"
                )
            pending_audit.append(
                (
                    "eviction",
                    {
                        "trigger_worker_id": worker_id,
                        "evicted": list(evict),
                        "group_id": group_id,
                        "remaining": remaining,
                    },
                )
            )

            # --- Full-model sync barrier ---
            # Remove every evicted worker's pending submission and update
            # the expected workers set.
            for wid in evict:
                self._pending_pseudograds.pop(wid, None)
                if self._round_expected_workers is not None:
                    self._round_expected_workers.discard(wid)

            expected = self._get_expected_worker_count()

            # Only release the barrier if there's actually something to
            # apply. After a whole-group eviction the expected set and
            # pending dict can both be empty, in which case ``issubset``
            # trivially returns True — we'd stamp a stale-state
            # ``_completed_rounds`` entry that future waiters could
            # block on. Gate on non-empty pending so the eviction path
            # cleanly defers to the next live submission.
            if expected > 0 and self._pending_pseudograds and self._round_complete():
                # Enough workers have submitted - release the barrier
                my_round = self._sync_round
                self._apply_outer_optimizer(pending_audit)
                self._completed_rounds[my_round] = self.get_global_params()
                self._snapshot_round_expected_workers()

            # --- Per-fragment sync barriers ---
            # For each active fragment, remove every evicted worker's pending
            # submission and check if the barrier should release.
            for frag_id in list(self._fragment_pending.keys()):
                for wid in evict:
                    self._fragment_pending[frag_id].pop(wid, None)

                if expected > 0 and self._fragment_round_complete(frag_id):
                    my_frag_round = self._fragment_rounds[frag_id]
                    pending = self._fragment_pending[frag_id]
                    pg_list = list(pending.values())
                    contributing_workers = list(pending.keys())
                    if pg_list:
                        self._apply_fragment_outer_optimizer(pg_list)

                        per_worker: Dict[str, Dict[str, torch.Tensor]] = {
                            wid: self._get_params_by_names(list(pgs.keys()))
                            for wid, pgs in pending.items()
                        }
                        self._completed_fragment_rounds[(frag_id, my_frag_round)] = (
                            per_worker
                        )

                        self._fragment_rounds[frag_id] += 1
                        self._fragment_pending[frag_id].clear()
                        self._fragment_submissions += len(pg_list)
                        self._sync_round += 1
                        pending_audit.append(
                            (
                                "fragment_outer_step",
                                {
                                    "fragment_id": frag_id,
                                    "fragment_round": my_frag_round,
                                    "sync_round": self._sync_round,
                                    "contributors": contributing_workers,
                                    "triggered_by": "eviction",
                                },
                            )
                        )

            # Wake all waiting threads so they re-evaluate their conditions
            self._sync_cond.notify_all()

        # _sync_cond released: now it's safe to do the audit disk I/O.
        self._audit_many(pending_audit)

    def _compute_dylu_sync_every(self, worker_id: str) -> Optional[int]:
        """
        Compute recommended sync_every for a worker using Dynamic Local Updates.

        DyLU adjusts each worker's sync frequency proportional to its speed
        relative to the fastest worker: H_w = floor((v_w / v_max) * H_base).
        This ensures faster workers contribute more updates while slower workers
        don't become bottlenecks.

        **Disabled under pipeline groups (issue #84):** in sync mode all
        ranks of a group must reach the barrier at the same local step.
        Per-rank DyLU adjustments would push ranks to different
        ``sync_every`` values and the barrier would never release. The
        server returns ``None`` for any worker belonging to a
        ``pp_world_size > 1`` group; the worker's heartbeat handler
        then skips the adjustment.

        Returns None if not enough speed data is available, the server
        runs in sync mode, or the worker is in a pipeline group.
        """
        if not self.dylu_enabled:
            return None
        # Sync-mode + group-aware barrier requires lockstep sync_every.
        if not self.async_mode:
            return None
        with self._workers_lock:
            group_id = self._worker_to_group.get(worker_id)
            if group_id is not None:
                group = self._groups.get(group_id)
                if group is not None and group.pp_world_size > 1:
                    return None

        with self._workers_lock:
            speeds = {
                wid: w.steps_per_second
                for wid, w in self._workers.items()
                if w.steps_per_second > 0
            }

        if not speeds or worker_id not in speeds:
            return None

        max_speed = max(speeds.values())
        if max_speed <= 0:
            return None

        worker_speed = speeds[worker_id]
        recommended = max(
            1, int((worker_speed / max_speed) * self.dylu_base_sync_every)
        )
        return recommended

    def _handle_register(self, handler: BaseHTTPRequestHandler):
        """Handle worker registration.

        Supports dynamic joining: new workers can register at any time and
        receive the current global parameters. If the number of registered
        workers exceeds num_workers, num_workers is increased. The new worker
        will NOT be expected for the current sync round (only for the next one).

        **Worker_id uniqueness is enforced**: a second registration of a
        worker_id that's already in the registry is refused with 409. The
        server treats worker_id itself as the uniqueness proxy — it has no
        view into how the worker's config templates use that ID downstream
        (output dir naming, log file paths, etc.), so collision at the
        identity layer is the only honest signal we can act on. Operators
        recovering from a crashed worker either wait for the heartbeat
        eviction (~heartbeat_timeout seconds) or POST /deregister.

        **Group registration (issue #84)**: when the worker sends a ``group``
        block in the registration payload — ``{"group_id": str, "pp_rank":
        int, "pp_world_size": int}`` — it joins (or creates) a worker
        group. The group is sealed when ``pp_world_size`` members have
        registered, at which point the union of their slices is verified
        to cover the server's full param set. Workers without a ``group``
        block form a degenerate group of one (group_id == worker_id,
        pp_world_size=1), preserving the pre-#84 contract for solo
        workers. Async mode rejects ``pp_world_size > 1`` — see commit 2.
        """
        body = _read_request_body(handler)
        info = json.loads(body.decode("utf-8"))
        worker_id = info["worker_id"]

        # Slice-fingerprint pre-check: workers that send ``param_shapes``
        # have their slice validated against the server's master params
        # BEFORE any registry mutation. Missing-on-server (worker has a
        # name the server doesn't) and shape mismatches are hard errors;
        # missing-on-worker is allowed at this point (sliced workers).
        # Workers that omit ``param_shapes`` skip the check — used by
        # test mocks. Production workers always send it.
        slice_shapes = info.get("param_shapes")
        if slice_shapes is not None:
            mismatch = self._diff_slice_fingerprint(slice_shapes)
            if mismatch is not None:
                _send_json_response(
                    handler,
                    {"error": mismatch, "kind": "slice_mismatch"},
                    422,
                )
                return

        # Parse group block (issue #84). Absent → solo group of one;
        # present → join the specified group at the specified rank slot.
        group_info = info.get("group")
        if group_info is None:
            group_id = worker_id
            pp_rank = 0
            pp_world_size = 1
        else:
            try:
                group_id = str(group_info["group_id"])
                pp_rank = int(group_info["pp_rank"])
                pp_world_size = int(group_info["pp_world_size"])
            except (KeyError, TypeError, ValueError) as e:
                _send_json_response(
                    handler,
                    {
                        "error": (
                            f"invalid 'group' block in registration payload: "
                            f"{e!r}. Expected "
                            f"{{group_id: str, pp_rank: int, pp_world_size: int}}."
                        )
                    },
                    400,
                )
                return
            if pp_world_size < 1 or pp_rank < 0 or pp_rank >= pp_world_size:
                _send_json_response(
                    handler,
                    {
                        "error": (
                            f"invalid group geometry: pp_rank={pp_rank}, "
                            f"pp_world_size={pp_world_size}. Required: "
                            f"pp_world_size >= 1, 0 <= pp_rank < pp_world_size."
                        )
                    },
                    400,
                )
                return
            if pp_world_size > 1 and self.async_mode:
                # Async barrier semantics with disjoint slice contributions
                # is fragile (which submissions can be combined? per-rank
                # for one group? across groups?). Out of scope for #84;
                # operators can downgrade to sync mode or use a non-
                # pipeline trainer.
                _send_json_response(
                    handler,
                    {
                        "error": (
                            "pipeline-group registration "
                            f"(pp_world_size={pp_world_size}) is not "
                            "compatible with async_mode. Start the "
                            "DiLoCo server without --async, or use a "
                            "non-pipeline trainer."
                        )
                    },
                    400,
                )
                return

        seal_error: Optional[str] = None
        with self._workers_lock:
            if worker_id in self._workers:
                _send_json_response(
                    handler,
                    {
                        "error": (
                            f"worker_id '{worker_id}' is already registered; "
                            f"if the previous worker is dead, wait for heartbeat "
                            f"eviction (~{self.heartbeat_timeout:.0f}s default) or "
                            f"POST /deregister"
                        )
                    },
                    409,
                )
                return

            # Find or create the group.
            group = self._groups.get(group_id)
            if group is None:
                group = WorkerGroup(
                    group_id=group_id,
                    pp_world_size=pp_world_size,
                    created_at=time.time(),
                )
                self._groups[group_id] = group
            else:
                if group.sealed:
                    _send_json_response(
                        handler,
                        {
                            "error": (
                                f"group '{group_id}' is sealed "
                                f"({group.pp_world_size} member(s) registered); "
                                f"to rejoin, deregister the existing member at "
                                f"pp_rank first or use a different group_id"
                            )
                        },
                        409,
                    )
                    return
                if group.pp_world_size != pp_world_size:
                    _send_json_response(
                        handler,
                        {
                            "error": (
                                f"group '{group_id}' declared pp_world_size="
                                f"{group.pp_world_size} on first member but "
                                f"this member declares pp_world_size="
                                f"{pp_world_size}; all members must agree"
                            )
                        },
                        422,
                    )
                    return
                if pp_rank in group.members:
                    _send_json_response(
                        handler,
                        {
                            "error": (
                                f"group '{group_id}' already has a member at "
                                f"pp_rank={pp_rank}: "
                                f"'{group.members[pp_rank]}'. Each pp_rank slot "
                                f"can hold only one worker."
                            )
                        },
                        409,
                    )
                    return

            # Register the member.
            group.members[pp_rank] = worker_id
            group.member_param_names[pp_rank] = (
                set(slice_shapes) if slice_shapes is not None else set()
            )

            # Seal + coverage check when the last rank slot fills.
            if len(group.members) == group.pp_world_size:
                # Skip coverage when ANY member omitted param_shapes —
                # test-only path; production workers always send shapes.
                have_shapes = all(
                    bool(names) or group.pp_world_size == 1
                    for names in group.member_param_names.values()
                )
                if slice_shapes is not None and have_shapes:
                    seal_error = self._check_group_coverage(group)
                if seal_error is not None:
                    # Atomic rollback: evict every member already registered.
                    self._rollback_group(group)
                else:
                    group.sealed = True

            if seal_error is None:
                self._workers[worker_id] = WorkerInfo(
                    worker_id=worker_id,
                    hostname=info.get("hostname", "unknown"),
                    registered_at=time.time(),
                    last_heartbeat=time.time(),
                    output_dir=info.get("output_dir"),
                    extra=info.get("extra", {}),
                )
                self._worker_to_group[worker_id] = group_id
                num_registered = len(self._workers)

        if seal_error is not None:
            _send_json_response(
                handler,
                {"error": seal_error, "kind": "group_coverage"},
                422,
            )
            return

        # If more workers than expected, grow the expected count.
        if num_registered > self.num_workers:
            self.num_workers = num_registered
            logger.info(
                f"Worker {worker_id} joined dynamically, "
                f"num_workers raised to {self.num_workers}"
            )

        if pp_world_size > 1:
            logger.info(
                f"Worker {worker_id} registered "
                f"(group='{group_id}' pp_rank={pp_rank}/{pp_world_size}, "
                f"{num_registered}/{self.num_workers} total)"
            )
        else:
            logger.info(
                f"Worker {worker_id} registered "
                f"({num_registered}/{self.num_workers})"
            )
        self._audit(
            "register",
            worker_id=worker_id,
            hostname=info.get("hostname"),
            group_id=group_id,
            pp_rank=pp_rank,
            pp_world_size=pp_world_size,
            num_registered=num_registered,
        )

        # Return global params. When a bulk listener is configured,
        # advertise its URL via response header so the worker can
        # route subsequent submit_pseudograd / global_params calls to
        # it directly (issue #90). Pass the worker's own view of our
        # host so a wildcard-bound server advertises a routable URL.
        bulk_url = self.get_bulk_url(_request_host(handler))
        extra = {"X-Forgather-Bulk-Url": bulk_url} if bulk_url else None
        if self.async_mode:
            with self._async_lock:
                _send_tensor_response(handler, self.get_global_params(), extra)
        else:
            _send_tensor_response(handler, self.get_global_params(), extra)

    def _diff_slice_fingerprint(
        self, slice_shapes: Dict[str, List[int]]
    ) -> Optional[str]:
        """Verify a single rank's slice against the server's master params.

        Each name in ``slice_shapes`` must appear in the server with
        matching shape. Extras (names on the worker but not on the
        server) are a hard error — the operator likely pointed this
        worker at the wrong model. Returns ``None`` on a clean match,
        a diagnostic string otherwise.

        Missing names (server names absent from the slice) are
        intentionally *allowed* here: a pipeline-parallel rank holds
        only its slice. Full-model coverage is verified at group-seal
        time by ``_check_group_coverage``. For solo workers (group of
        one, pp_world_size=1) the coverage check at seal time
        immediately enforces equality and the contract collapses to
        the pre-#84 fingerprint check.
        """
        server_shapes: Dict[str, List[int]] = {
            name: list(self._param_list[i].shape)
            for i, name in enumerate(self._param_names)
        }
        worker_set = set(slice_shapes)
        server_set = set(server_shapes)

        missing_on_server = worker_set - server_set
        shape_mismatch = [
            (name, slice_shapes[name], server_shapes[name])
            for name in worker_set & server_set
            if list(slice_shapes[name]) != server_shapes[name]
        ]

        if not (missing_on_server or shape_mismatch):
            return None

        parts = [
            "Worker slice does not match server model. The operator "
            "likely pointed this worker at the wrong "
            "--model-id-or-path (or built it from a different "
            "model project / config than the server is using)."
        ]
        if shape_mismatch:
            sample = shape_mismatch[:5]
            parts.append(f"  Shape mismatch on {len(shape_mismatch)} param(s):")
            for name, wshape, sshape in sample:
                parts.append(f"    {name}: worker={wshape}, server={sshape}")
            if len(shape_mismatch) > 5:
                parts.append(f"    ... and {len(shape_mismatch) - 5} more")
        if missing_on_server:
            sample = sorted(missing_on_server)[:5]
            parts.append(
                f"  {len(missing_on_server)} param(s) on worker but not server: "
                f"{sample}{'...' if len(missing_on_server) > 5 else ''}"
            )
        return "\n".join(parts)

    def _check_group_coverage(self, group: WorkerGroup) -> Optional[str]:
        """Verify a sealed group's slice union exactly covers the server.

        Called at the moment a group becomes sealed (its last member
        registers). Returns ``None`` on success, a diagnostic string
        on failure. Duplicate names across slices are allowed —
        tied-parameter aliases legitimately appear in more than one
        rank's slice (see e.g. weight-tied embed/lm_head with the
        embed on stage 0 and the lm_head's transposed view on the
        final stage); the per-name averaging on the apply path treats
        same-data alias contributions correctly.

        A non-empty ``missing`` set means at least one server param has
        no rank holding it — a real correctness bug in the pipeline
        split or in the operator's model wiring. We refuse to seal the
        group in that case; the registration handler is responsible for
        rolling back any partial-group state.
        """
        union: set = set()
        for names in group.member_param_names.values():
            union |= names
        missing = set(self._param_names) - union
        if not missing:
            return None

        sample = sorted(missing)[:10]
        return (
            f"Group '{group.group_id}' slices do not cover the server's "
            f"full parameter set. {len(missing)} param(s) are not held "
            f"by any rank: {sample}{'...' if len(missing) > 10 else ''}. "
            f"Verify the pipeline split across pp_world_size="
            f"{group.pp_world_size} ranks is consistent with the "
            f"server's model."
        )

    def _rollback_group(self, group: WorkerGroup) -> None:
        """Evict every registered member of a group.

        Used on partial-group registration failures (e.g. coverage
        check failed at seal-time) and on any member's death (the
        atomic-eviction policy from issue #84). Caller MUST hold
        ``_workers_lock``. Does not notify the sync barrier — the
        caller is responsible for that if needed.
        """
        for wid in list(group.members.values()):
            self._workers.pop(wid, None)
            self._worker_to_group.pop(wid, None)
        self._groups.pop(group.group_id, None)

    def _validate_pseudograd_params(
        self, worker_id: str, pseudograds: Dict[str, torch.Tensor]
    ) -> Optional[str]:
        """Validate pseudo-gradient parameter names against the worker's
        registered slice.

        Each submitted name must exist on the server (no "extras"). For a
        solo worker, the expected name set equals the server's full
        ``_param_names``. For a pipeline-parallel rank, the expected set
        is the slice it registered with — looked up via
        ``_worker_to_group`` → ``WorkerGroup.member_param_names``.
        Submissions with extras → 400; submissions missing some of the
        worker's own slice names → 400 (the worker is internally
        inconsistent, e.g. a config edit mid-training).

        Returns an error message string if there's a mismatch, None if valid.
        """
        pg_names = set(pseudograds.keys())
        global_names = set(self._param_names)

        # Always reject names that aren't on the server at all.
        extra = pg_names - global_names
        if extra:
            sample = sorted(extra)[:5]
            parts = [
                f"Parameter name mismatch from worker {worker_id}.",
                f"  Unexpected {len(extra)} params (sent by worker, "
                f"not on server): {sample}"
                f"{'...' if len(extra) > 5 else ''}",
                "  This usually means the worker is using a different "
                "model architecture than the server.",
            ]
            return "\n".join(parts)

        # For sliced workers, also reject submissions missing names from
        # the registered slice (worker is internally inconsistent).
        expected = self._slice_expectation_for(worker_id)
        if expected is not None and not pg_names.issuperset(expected):
            missing = expected - pg_names
            sample = sorted(missing)[:5]
            return (
                f"Parameter name mismatch from worker {worker_id}.\n"
                f"  Worker registered a slice with {len(expected)} "
                f"param(s) but submitted only {len(pg_names)}. "
                f"Missing {len(missing)} from this submission: "
                f"{sample}{'...' if len(missing) > 5 else ''}"
            )
        return None

    def _slice_expectation_for(self, worker_id: str) -> Optional[set]:
        """Return the registered slice's param-name set for this worker.

        ``None`` when the worker has no slice metadata (legacy / test
        path that omitted ``param_shapes`` at register time). Caller
        treats that as "any submission with no extras is acceptable".

        Lock invariant: this helper acquires ``_workers_lock`` itself
        and must NOT be called from inside another ``_workers_lock``
        critical section.
        """
        with self._workers_lock:
            group_id = self._worker_to_group.get(worker_id)
            if group_id is None:
                return None
            group = self._groups.get(group_id)
            if group is None:
                return None
            for pp_rank, wid in group.members.items():
                if wid == worker_id:
                    names = group.member_param_names.get(pp_rank)
                    return set(names) if names else None
        return None

    def _worker_group_sealed(self, worker_id: str) -> Optional[bool]:
        """Return whether the worker's owning group is sealed.

        Returns ``None`` when the worker is unknown (already evicted or
        never registered). Returns ``True`` for solo groups (sealed at
        registration) and for pipeline groups with all members
        registered; ``False`` for partial pipeline groups.

        Used by the submit / heartbeat paths to refuse submissions
        from unsealed groups (correctness — only sealed groups have a
        verified-cover slice union) and from ghost worker_ids (workers
        whose group was atomically evicted but whose HTTP client is
        still alive).
        """
        with self._workers_lock:
            if worker_id not in self._workers:
                return None
            group_id = self._worker_to_group.get(worker_id)
            if group_id is None:
                return None
            group = self._groups.get(group_id)
            return group.sealed if group is not None else None

    def _round_complete(self) -> bool:
        """True iff every expected worker has submitted for this round.

        Solo groups: collapses to the pre-#84 length check (every
        registered worker_id has a pending submission). Pipeline groups:
        every member of every group has submitted its slice.
        """
        if self._round_expected_workers is None:
            return False
        submitted = set(self._pending_pseudograds.keys())
        return self._round_expected_workers.issubset(submitted)

    def _handle_submit_pseudograd(self, handler: BaseHTTPRequestHandler):
        """
        Handle pseudo-gradient submission.

        In sync mode: blocks until all workers submit, then applies the outer
        optimizer and returns updated global params to all workers.

        In async mode: applies the pseudo-gradient immediately and returns
        updated global params without waiting.
        """
        # Read the request: length-prefixed JSON header + tensor payload
        body = _read_request_body(handler)

        header_len = struct.unpack("!I", body[:4])[0]
        header = json.loads(body[4 : 4 + header_len].decode("utf-8"))
        tensor_data = body[4 + header_len :]

        worker_id = header["worker_id"]
        pseudograds = _deserialize_state_dict(tensor_data)

        # Reject ghost worker_ids: a worker whose group was atomically
        # evicted but whose HTTP client is still alive could otherwise
        # silently re-inject pseudograds into the next round
        # (``_round_complete`` uses ``issubset`` which tolerates extras,
        # and per-name aggregation picks them up). Pre-#89 a missing
        # worker_id was silently accepted into ``_pending_pseudograds``.
        sealed = self._worker_group_sealed(worker_id)
        if sealed is None:
            _send_json_response(
                handler,
                {
                    "error": (
                        f"worker_id '{worker_id}' is not registered; "
                        f"if its group was atomically evicted, the "
                        f"worker should re-register or exit."
                    ),
                    "kind": "unknown_worker",
                },
                404,
            )
            return
        if not sealed:
            _send_json_response(
                handler,
                {
                    "error": (
                        f"worker_id '{worker_id}' belongs to an unsealed "
                        f"group; submissions are accepted only after all "
                        f"pp_world_size members have registered."
                    ),
                    "kind": "group_unsealed",
                },
                409,
            )
            return

        # Validate parameter names match
        error = self._validate_pseudograd_params(worker_id, pseudograds)
        if error:
            logger.error(error)
            _send_json_response(handler, {"error": error}, 400)
            return

        if self.async_mode:
            self._handle_submit_async(handler, worker_id, pseudograds)
        else:
            self._handle_submit_sync(handler, worker_id, pseudograds)

    def _handle_submit_sync(self, handler, worker_id, pseudograds):
        """Synchronous pseudo-gradient submission with fault-tolerant barrier.

        The barrier target is _round_expected_workers (snapshotted at round
        start) rather than the fixed num_workers. If a worker dies mid-round,
        _handle_worker_death removes it from the expected set and may release
        the barrier early.
        """
        # Collected under the lock, flushed after release (see _audit).
        pending_audit: List[Tuple[str, Dict[str, Any]]] = []
        with self._sync_cond:
            my_round = self._sync_round

            # Lazy-init expected workers for the first round
            if self._round_expected_workers is None:
                self._snapshot_round_expected_workers()

            self._pending_pseudograds[worker_id] = pseudograds

            with self._workers_lock:
                if worker_id in self._workers:
                    self._workers[worker_id].last_heartbeat = time.time()
                    self._workers[worker_id].sync_round += 1
                    self._workers[worker_id].last_sync_server_round = my_round + 1

            expected = self._get_expected_worker_count()
            submitted = len(self._pending_pseudograds)
            logger.info(
                f"Worker {worker_id} submitted pseudograds "
                f"({submitted}/{expected}) for round {my_round}"
            )

            if self._round_complete():
                self._apply_outer_optimizer(pending_audit)
                self._completed_rounds[my_round] = self.get_global_params()

                # Snapshot expected workers for the next round
                self._snapshot_round_expected_workers()

                old_rounds = [r for r in self._completed_rounds if r < my_round - 1]
                for r in old_rounds:
                    del self._completed_rounds[r]

                self._sync_cond.notify_all()

            while my_round not in self._completed_rounds:
                if not self._sync_cond.wait(timeout=600):
                    _send_json_response(handler, {"error": "Sync timeout"}, 504)
                    return

            global_params = self._completed_rounds[my_round]

        # _sync_cond released: flush audit records off the barrier path.
        self._audit_many(pending_audit)
        _send_tensor_response(handler, global_params)

    def _handle_submit_async(self, handler, worker_id, pseudograds):
        """Asynchronous pseudo-gradient submission - apply immediately."""
        with self._async_lock:
            # Compute staleness: how many server updates since this worker last synced
            staleness = 0
            with self._workers_lock:
                if worker_id in self._workers:
                    w = self._workers[worker_id]
                    staleness = self._sync_round - w.last_sync_server_round
                    w.last_heartbeat = time.time()
                    w.sync_round += 1
                    w.last_sync_server_round = self._sync_round + 1

            logger.info(
                f"Worker {worker_id} submitted pseudograds (async), "
                f"staleness={staleness}, server_round={self._sync_round}"
            )

            # Apply this worker's pseudo-gradients immediately
            self._apply_async_pseudograd(worker_id, pseudograds)

            global_params = self.get_global_params()

        _send_tensor_response(handler, global_params)

    def _handle_submit_fragment_pseudograd(self, handler: BaseHTTPRequestHandler):
        """
        Handle fragment pseudo-gradient submission.

        Used by streaming DiLoCo workers that split the model into fragments
        for staggered sync. Each submission contains pseudo-gradients for only
        one fragment's parameters.

        In sync mode: per-fragment barrier (all workers must submit the same
        fragment before the outer optimizer applies).
        In async mode: applies immediately like full-model async.
        """
        body = _read_request_body(handler)

        header_len = struct.unpack("!I", body[:4])[0]
        header = json.loads(body[4 : 4 + header_len].decode("utf-8"))
        tensor_data = body[4 + header_len :]

        worker_id = header["worker_id"]
        fragment_id = header["fragment_id"]
        pseudograds = _deserialize_state_dict(tensor_data)

        # Ghost / unsealed-group rejection (same gate as the full-sync
        # submit path).
        sealed = self._worker_group_sealed(worker_id)
        if sealed is None:
            _send_json_response(
                handler,
                {
                    "error": (f"worker_id '{worker_id}' is not registered"),
                    "kind": "unknown_worker",
                },
                404,
            )
            return
        if not sealed:
            _send_json_response(
                handler,
                {
                    "error": (f"worker_id '{worker_id}' belongs to an unsealed group"),
                    "kind": "group_unsealed",
                },
                409,
            )
            return

        # Validate all submitted param names exist in the global model
        unknown = set(pseudograds.keys()) - set(self._param_names)
        if unknown:
            sample = sorted(unknown)[:5]
            error = (
                f"Fragment {fragment_id} from worker {worker_id} contains "
                f"{len(unknown)} unknown parameter names: {sample}"
                f"{'...' if len(unknown) > 5 else ''}. "
                f"This usually means the worker is using a different model."
            )
            logger.error(error)
            _send_json_response(handler, {"error": error}, 400)
            return

        # Slice-membership validation (consistent with full-sync path):
        # the fragment's submitted names must be a subset of the
        # worker's registered slice. Without this a rank could spoof
        # updates to names outside its slice and silently corrupt
        # another rank's parameters.
        expected = self._slice_expectation_for(worker_id)
        if expected is not None:
            outside_slice = set(pseudograds.keys()) - expected
            if outside_slice:
                sample = sorted(outside_slice)[:5]
                _send_json_response(
                    handler,
                    {
                        "error": (
                            f"Fragment {fragment_id} from worker "
                            f"{worker_id} contains {len(outside_slice)} "
                            f"name(s) outside this rank's registered "
                            f"slice: {sample}"
                            f"{'...' if len(outside_slice) > 5 else ''}"
                        )
                    },
                    400,
                )
                return

        if self.async_mode:
            self._handle_submit_fragment_async(
                handler, worker_id, fragment_id, pseudograds
            )
        else:
            self._handle_submit_fragment_sync(
                handler, worker_id, fragment_id, pseudograds
            )

    def _fragment_round_complete(self, fragment_id: int) -> bool:
        """True iff every expected worker has submitted for this fragment.

        Uses ``_round_expected_workers`` as the membership snapshot
        (same set the full-sync barrier uses — every worker_id from
        every active group). For solo groups this collapses to the
        pre-#84 length check. For pipeline groups, the fragment
        releases only when each rank has submitted its slice's portion
        of ``fragment_id``.
        """
        if self._round_expected_workers is None:
            return False
        submitted = set(self._fragment_pending[fragment_id].keys())
        return self._round_expected_workers.issubset(submitted)

    def _handle_submit_fragment_sync(
        self, handler, worker_id, fragment_id, pseudograds
    ):
        """Per-fragment synchronous submission with fault-tolerant barrier."""
        # Collected under the lock, flushed to disk after release so the
        # audit write never stalls the barrier (see _audit).
        pending_audit: List[Tuple[str, Dict[str, Any]]] = []
        with self._sync_cond:
            my_round = self._fragment_rounds[fragment_id]

            # Lazy-init expected workers (same as full-sync path).
            if self._round_expected_workers is None:
                self._snapshot_round_expected_workers()

            self._fragment_pending[fragment_id][worker_id] = pseudograds

            with self._workers_lock:
                if worker_id in self._workers:
                    self._workers[worker_id].last_heartbeat = time.time()

            expected = self._get_expected_worker_count()
            submitted = len(self._fragment_pending[fragment_id])
            logger.info(
                f"Worker {worker_id} submitted fragment {fragment_id} pseudograds "
                f"({submitted}/{expected}) for fragment round {my_round}"
            )

            if self._fragment_round_complete(fragment_id):
                # All expected workers submitted for this fragment.
                # Aggregate using contributors-only per-name; then build
                # the per-worker response (each worker receives only the
                # names it submitted — important under pipeline groups
                # where different ranks of the same fragment_id carry
                # different name subsets).
                pending = self._fragment_pending[fragment_id]
                pg_list = list(pending.values())
                contributing_workers = list(pending.keys())
                self._apply_fragment_outer_optimizer(pg_list)

                per_worker: Dict[str, Dict[str, torch.Tensor]] = {
                    wid: self._get_params_by_names(list(pgs.keys()))
                    for wid, pgs in pending.items()
                }
                self._completed_fragment_rounds[(fragment_id, my_round)] = per_worker

                self._fragment_rounds[fragment_id] += 1
                self._fragment_pending[fragment_id].clear()
                self._fragment_submissions += len(pg_list)
                self._sync_round += 1
                pending_audit.append(
                    (
                        "fragment_outer_step",
                        {
                            "fragment_id": fragment_id,
                            "fragment_round": my_round,
                            "sync_round": self._sync_round,
                            "contributors": contributing_workers,
                        },
                    )
                )

                # Clean up old fragment rounds
                old = [
                    k
                    for k in self._completed_fragment_rounds
                    if k[0] == fragment_id and k[1] < my_round - 1
                ]
                for k in old:
                    del self._completed_fragment_rounds[k]

                self._sync_cond.notify_all()

            # Wait for this fragment round's result.
            key = (fragment_id, my_round)
            while key not in self._completed_fragment_rounds:
                if not self._sync_cond.wait(timeout=600):
                    _send_json_response(
                        handler, {"error": "Fragment sync timeout"}, 504
                    )
                    return

            per_worker_results = self._completed_fragment_rounds[key]
            result = per_worker_results.get(worker_id)
            if result is None:
                # Worker died mid-round and was removed from pending; if
                # the barrier still released for the survivors, this
                # worker's slot is gone. Fall back to whatever was
                # computed for any rank (legacy / solo-group safety).
                # Pipeline workers never hit this since whole-group
                # eviction takes the survivor out before the barrier.
                if per_worker_results:
                    result = next(iter(per_worker_results.values()))
                else:
                    _send_json_response(
                        handler,
                        {"error": "fragment round completed without our submission"},
                        500,
                    )
                    return

        # _sync_cond released: flush audit records off the barrier path.
        self._audit_many(pending_audit)
        _send_tensor_response(handler, result)

    def _handle_submit_fragment_async(
        self, handler, worker_id, fragment_id, pseudograds
    ):
        """Asynchronous fragment submission - apply immediately."""
        with self._async_lock:
            with self._workers_lock:
                if worker_id in self._workers:
                    self._workers[worker_id].last_heartbeat = time.time()

            logger.info(
                f"Worker {worker_id} submitted fragment {fragment_id} (async), "
                f"server_round={self._sync_round}"
            )

            # Apply fragment's pseudo-gradients to the outer optimizer
            frag_param_names = list(pseudograds.keys())
            for name in frag_param_names:
                idx = self._param_name_to_idx[name]
                self._param_list[idx].grad = pseudograds[name].float()

            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()

            self._sync_round += 1
            self._total_submissions += 1
            self._fragment_submissions += 1
            self._dirty = True

            result = self._get_params_by_names(frag_param_names)

        _send_tensor_response(handler, result)

    def _handle_get_global_params(self, handler: BaseHTTPRequestHandler):
        """Handle request for current global parameters."""
        if self.async_mode:
            with self._async_lock:
                _send_tensor_response(handler, self.get_global_params())
        else:
            _send_tensor_response(handler, self.get_global_params())

    def _handle_heartbeat(self, handler: BaseHTTPRequestHandler):
        """Handle worker heartbeat."""
        body = _read_request_body(handler)
        info = json.loads(body.decode("utf-8"))
        worker_id = info["worker_id"]

        with self._workers_lock:
            if worker_id not in self._workers:
                # Ghost worker: its group was atomically evicted but
                # this HTTP client is still alive. Tell it explicitly
                # so it stops sending heartbeats (and stops re-injecting
                # pseudograds — see the submit handlers' same guard).
                _send_json_response(
                    handler,
                    {
                        "error": (
                            f"worker_id '{worker_id}' is not registered; "
                            f"re-register or exit."
                        ),
                        "kind": "unknown_worker",
                    },
                    404,
                )
                return
            self._workers[worker_id].last_heartbeat = time.time()
            if "steps_per_second" in info:
                self._workers[worker_id].steps_per_second = info["steps_per_second"]

        # Compute DyLU recommendation if enabled
        recommended_sync_every = self._compute_dylu_sync_every(worker_id)

        response = {
            "status": "ok",
            "sync_round": self._sync_round,
            "num_workers": self.num_workers,
            "num_registered": len(self._workers),
        }
        if recommended_sync_every is not None:
            response["recommended_sync_every"] = recommended_sync_every

        _send_json_response(handler, response)

    def _handle_deregister(self, handler: BaseHTTPRequestHandler):
        """Handle worker deregistration.

        Uses _handle_worker_death for proper barrier cleanup so that
        remaining workers are not blocked waiting for the departed worker.
        """
        body = _read_request_body(handler)
        info = json.loads(body.decode("utf-8"))
        worker_id = info["worker_id"]

        logger.info(f"Worker {worker_id} deregistering")
        self._audit("deregister", worker_id=worker_id)
        self._handle_worker_death(worker_id)

        _send_json_response(handler, {"status": "ok"})

    def _handle_status(self, handler: BaseHTTPRequestHandler):
        """Handle status request."""
        with self._workers_lock:
            workers = {
                wid: {
                    "hostname": w.hostname,
                    "registered_at": w.registered_at,
                    "last_heartbeat": w.last_heartbeat,
                    "sync_round": w.sync_round,
                    "last_sync_server_round": w.last_sync_server_round,
                    "steps_per_second": w.steps_per_second,
                    "output_dir": w.output_dir,
                }
                for wid, w in self._workers.items()
            }

        if self.async_mode:
            with self._async_lock:
                pending = []
                dn_buffered = len(self._dn_grad_buffer)
        else:
            with self._sync_cond:
                pending = list(self._pending_pseudograds.keys())
            dn_buffered = 0

        response = {
            "status": "running",
            "mode": "async" if self.async_mode else "sync",
            "sync_round": self._sync_round,
            "num_workers": self.num_workers,
            "num_registered": len(workers),
            "workers": workers,
            "pending_submissions": pending,
            "started_at": self._started_at,
            "uptime_seconds": time.time() - self._started_at if self._started_at else 0,
        }

        if self.async_mode:
            response["total_submissions"] = self._total_submissions
            response["dn_buffer_size"] = self.dn_buffer_size
            response["dn_buffered"] = dn_buffered
            response["dylu_enabled"] = self.dylu_enabled
            if self.dylu_enabled:
                response["dylu_base_sync_every"] = self.dylu_base_sync_every

        if self._fragment_submissions > 0:
            response["fragment_submissions"] = self._fragment_submissions

        if self._total_worker_deaths > 0:
            response["total_worker_deaths"] = self._total_worker_deaths

        response["heartbeat_timeout"] = self.heartbeat_timeout
        response["min_workers"] = self.min_workers

        # Dashboard-related fields
        pg = self.outer_optimizer.param_groups[0]
        response["outer_lr"] = pg.get("lr", 0)
        response["outer_momentum"] = pg.get("momentum", 0)
        response["save_dir"] = self.output_dir
        response["model_params"] = self._model_params
        response["model_size_mb"] = round(self._model_size_mb, 2)

        _send_json_response(handler, response)

    def _handle_info(self, handler: BaseHTTPRequestHandler):
        """Handle info request.

        Returns the static-ish facts a client needs to negotiate compatible
        settings before submitting pseudo-gradients: which checkpoint the
        server was started from, the parameter count, and recommended
        client-side defaults (``expected_client_settings``). Distinct from
        ``/status`` which is the live, rapidly-changing snapshot.
        """
        response = {
            "output_dir": self.output_dir,
            "mode": "async" if self.async_mode else "sync",
            "async_mode": self.async_mode,
            "num_workers": self.num_workers,
            "num_parameters": self._model_params,
            "model_size_mb": round(self._model_size_mb, 2),
            "dylu_enabled": self.dylu_enabled,
            "dylu_base_sync_every": self.dylu_base_sync_every,
            # Coarse model fingerprint (issue #53). Workers validate a
            # cached model-definition bundle against this and use it as an
            # early compatibility gate before constructing the model.
            "model_hash": self._model_hash,
            # The server is the sole authority for these settings: they must
            # match across the group for the sync barrier / outer step /
            # fragment barriers to be coherent, so the worker takes them
            # verbatim (no client override). ``settings_authority`` signals
            # that intent to clients and tooling.
            "settings_authority": "server",
            "expected_client_settings": {
                # The group's inner-step cadence. Under DyLU every worker
                # ramps to the base rate (the per-worker scaling anchor);
                # otherwise it's the operator-set server ``sync_every``.
                # Always non-null so the worker can drop its own knob.
                "sync_every": (
                    self.dylu_base_sync_every if self.dylu_enabled else self.sync_every
                ),
                "dylu": self.dylu_enabled,
                "bf16_comm": self.bf16_comm,
                "num_fragments_min": 1,
                "num_fragments_default": self.num_fragments,
                # Exposed so the worker can validate its (client-local)
                # heartbeat send cadence against the server's death timeout.
                "heartbeat_timeout": self.heartbeat_timeout,
            },
        }
        _send_json_response(handler, response)

    def _handle_model_def(self, handler: BaseHTTPRequestHandler):
        """Serve the model-definition bundle (issue #53).

        Streams an uncompressed tar of the non-weight files from the
        checkpoint dir the server was started from — ``config.json``, the
        custom modeling/configuration ``.py`` closure, and the tokenizer —
        so a DiLoCo worker can construct the model without a shared
        filesystem and without ever fetching weights (those arrive via the
        parameter sync). Weights, shard indices, server state, and the
        audit log are excluded by ``model_def`` policy.

        Control-plane only and bearer-required: ``do_GET`` runs the auth
        check before routing here, and the endpoint is never added to the
        bulk listener's allow-set — the custom model code is not
        world-readable. The ``X-Forgather-Model-Hash`` header lets the
        worker pair the bundle with the server's advertised ``model_hash``.
        """
        if not self._loaded_checkpoint_dir or not os.path.isdir(
            self._loaded_checkpoint_dir
        ):
            _send_json_response(
                handler,
                {"error": "server has no model-definition directory to serve"},
                503,
            )
            return
        try:
            with self._model_def_lock:
                if self._model_def_bundle is None:
                    self._model_def_bundle = pack_model_def(self._loaded_checkpoint_dir)
                payload = self._model_def_bundle
        except OSError as exc:
            _send_json_response(
                handler, {"error": f"could not read model definition: {exc}"}, 500
            )
            return
        handler.send_response(200)
        handler.send_header("Content-Type", "application/x-tar")
        handler.send_header("Content-Length", str(len(payload)))
        handler.send_header(MODEL_HASH_HEADER, self._model_hash)
        handler.end_headers()
        handler.wfile.write(payload)

    # ------------------------------------------------------------------
    # Work-unit dispatch (see docs/design/diloco-work-unit-dispatch.md)
    # ------------------------------------------------------------------

    def _handle_register_dataset(self, handler: BaseHTTPRequestHandler):
        """Register a ``(dataset_id, shuffle_seed)`` queue, or confirm an
        existing one.

        First registration of a ``dataset_id`` snapshots its length; later
        registrations of the *same* dataset_id must report a matching
        length or get a 409. The queue itself is per
        ``(dataset_id, shuffle_seed)`` — same dataset under a new seed
        (new epoch) gets a fresh queue.
        """
        try:
            body = json.loads(_read_request_body(handler).decode("utf-8"))
        except (ValueError, json.JSONDecodeError) as exc:
            _send_json_response(handler, {"error": f"bad JSON body: {exc}"}, 400)
            return

        worker_id = body.get("worker_id") or ""
        dataset_id = body.get("dataset_id")
        seed_raw = body.get("shuffle_seed")
        hint = body.get("hint") or {}

        if not isinstance(dataset_id, str) or not dataset_id:
            _send_json_response(handler, {"error": "dataset_id required"}, 400)
            return
        try:
            shuffle_seed = int(seed_raw)
        except (TypeError, ValueError):
            _send_json_response(handler, {"error": "shuffle_seed (int) required"}, 400)
            return
        try:
            hint_length = int(hint.get("length"))
        except (TypeError, ValueError):
            _send_json_response(handler, {"error": "hint.length (int) required"}, 400)
            return
        if hint_length < 1:
            _send_json_response(
                handler, {"error": f"hint.length must be >= 1, got {hint_length}"}, 400
            )
            return

        with self._work_queues_lock:
            # Length-mismatch detection: a later worker shipping a stale
            # dataset config (different row count) is caught here rather
            # than allowed to silently mis-window.
            prior_length = self._dataset_lengths.get(dataset_id)
            if prior_length is not None and prior_length != hint_length:
                _send_json_response(
                    handler,
                    {
                        "error": (
                            f"dataset_id '{dataset_id}' was previously registered "
                            f"with length={prior_length}; new hint.length="
                            f"{hint_length} disagrees"
                        )
                    },
                    409,
                )
                return
            self._dataset_lengths[dataset_id] = hint_length

            key = (dataset_id, shuffle_seed)
            queue = self._work_queues.get(key)
            if queue is None:
                # Pluck optional dataset-identity strings from the hint.
                # The first worker to register the queue snapshots
                # these; later workers' values are ignored (they
                # should be identical anyway since dataset_id is a
                # hash of exactly this set of fields).
                hint_path = (
                    hint.get("path") if isinstance(hint.get("path"), str) else None
                )
                hint_name = (
                    hint.get("name") if isinstance(hint.get("name"), str) else None
                )
                hint_split = (
                    hint.get("split") if isinstance(hint.get("split"), str) else None
                )
                hint_revision = (
                    hint.get("revision")
                    if isinstance(hint.get("revision"), str)
                    else None
                )
                hint_data_files = hint.get("data_files")
                if isinstance(hint_data_files, str):
                    hint_data_files = [hint_data_files]
                elif isinstance(hint_data_files, list):
                    hint_data_files = [x for x in hint_data_files if isinstance(x, str)]
                else:
                    hint_data_files = None
                queue = WorkQueue.empty(
                    self.default_work_units,
                    hint_length,
                    dataset_path=hint_path,
                    dataset_name=hint_name,
                    dataset_split=hint_split,
                    dataset_revision=hint_revision,
                    dataset_data_files=hint_data_files,
                )
                self._work_queues[key] = queue
                label = hint_path or dataset_id
                if hint_split:
                    label = f"{label}@{hint_split}"
                logger.info(
                    f"Registered work queue {dataset_id}@{shuffle_seed} "
                    f"(K={self.default_work_units}, length={hint_length}, "
                    f"label={label!r}, worker={worker_id!r})"
                )
            self._dirty = True

        _send_json_response(handler, {"total_units": queue.total_units})

    def _handle_request_work(self, handler: BaseHTTPRequestHandler):
        """Issue the next available work unit from a queue.

        One-way issuance: the returned unit is consumed from the queue
        regardless of worker fate (no reissue, no per-unit timeout).
        Worst case a dying worker loses one unit out of K. The benefit
        is that no row is ever trained twice within an epoch — see
        design doc §"Issuance is one-way".
        """
        try:
            body = json.loads(_read_request_body(handler).decode("utf-8"))
        except (ValueError, json.JSONDecodeError) as exc:
            _send_json_response(handler, {"error": f"bad JSON body: {exc}"}, 400)
            return

        worker_id = body.get("worker_id") or ""
        dataset_id = body.get("dataset_id")
        seed_raw = body.get("shuffle_seed")
        if not isinstance(dataset_id, str) or not dataset_id:
            _send_json_response(handler, {"error": "dataset_id required"}, 400)
            return
        try:
            shuffle_seed = int(seed_raw)
        except (TypeError, ValueError):
            _send_json_response(handler, {"error": "shuffle_seed (int) required"}, 400)
            return

        with self._work_queues_lock:
            queue = self._work_queues.get((dataset_id, shuffle_seed))
            if queue is None:
                _send_json_response(
                    handler,
                    {
                        "error": (
                            f"no queue for ({dataset_id}, {shuffle_seed}); "
                            f"call /datasets/register first"
                        )
                    },
                    404,
                )
                return

            unit_id = _find_lowest_unset(queue.issued, queue.total_units)
            if unit_id < 0:
                _send_json_response(handler, {"exhausted": True})
                return

            _bit_set(queue.issued, unit_id)
            queue.issued_count += 1
            counters = queue.by_worker.setdefault(
                worker_id, {"units_issued": 0, "units_completed": 0}
            )
            counters["units_issued"] += 1
            self._dirty = True

        _send_json_response(handler, {"unit_id": unit_id})

    def _handle_complete_work(self, handler: BaseHTTPRequestHandler):
        """Mark a unit as confirmed-completed (diagnostic only).

        Workers MAY call this on successful drain. Nothing about
        issuance state changes if it's omitted; the completed bitmap
        just stays zero. The diagnostic surface uses it to distinguish
        "issued ∧ completed" from "issued, fate unknown".
        """
        try:
            body = json.loads(_read_request_body(handler).decode("utf-8"))
        except (ValueError, json.JSONDecodeError) as exc:
            _send_json_response(handler, {"error": f"bad JSON body: {exc}"}, 400)
            return

        worker_id = body.get("worker_id") or ""
        dataset_id = body.get("dataset_id")
        seed_raw = body.get("shuffle_seed")
        unit_raw = body.get("unit_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            _send_json_response(handler, {"error": "dataset_id required"}, 400)
            return
        try:
            shuffle_seed = int(seed_raw)
            unit_id = int(unit_raw)
        except (TypeError, ValueError):
            _send_json_response(
                handler, {"error": "shuffle_seed and unit_id (int) required"}, 400
            )
            return

        with self._work_queues_lock:
            queue = self._work_queues.get((dataset_id, shuffle_seed))
            if queue is None:
                _send_json_response(
                    handler,
                    {"error": f"no queue for ({dataset_id}, {shuffle_seed})"},
                    404,
                )
                return
            if not (0 <= unit_id < queue.total_units):
                _send_json_response(
                    handler,
                    {
                        "error": f"unit_id {unit_id} out of range [0, {queue.total_units})"
                    },
                    400,
                )
                return
            # Idempotent: completing an already-completed unit is a no-op.
            if not _bit_get(queue.completed, unit_id):
                _bit_set(queue.completed, unit_id)
                queue.completed_count += 1
                counters = queue.by_worker.setdefault(
                    worker_id, {"units_issued": 0, "units_completed": 0}
                )
                counters["units_completed"] += 1
                self._dirty = True

        _send_json_response(handler, {"ack": True})

    def _handle_get_queues(self, handler: BaseHTTPRequestHandler):
        """List all active work queues (summary only, no bitmaps)."""
        with self._work_queues_lock:
            queues = [
                _queue_summary_dict(dsid, seed, q)
                for (dsid, seed), q in self._work_queues.items()
            ]
        _send_json_response(handler, queues)

    def _handle_get_queue(self, handler: BaseHTTPRequestHandler):
        """Single-queue detail with bitmaps (base64-encoded) and per-worker counts.

        Bitmaps are K bits packed little-endian within each byte. K=1024
        → 128 bytes per bitmap; cheap. The diagnostic UI decodes these
        client-side to render a per-unit heatmap.
        """
        # The do_GET dispatcher passes self.path verbatim (no query
        # parsing). Strip the path off and parse the query here.
        parsed = urlparse(handler.path)
        params = parse_qs(parsed.query)
        dataset_id = (params.get("dataset_id") or [None])[0]
        seed_raw = (params.get("shuffle_seed") or [None])[0]
        if not dataset_id:
            _send_json_response(handler, {"error": "dataset_id required"}, 400)
            return
        try:
            shuffle_seed = int(seed_raw)
        except (TypeError, ValueError):
            _send_json_response(handler, {"error": "shuffle_seed (int) required"}, 400)
            return

        with self._work_queues_lock:
            queue = self._work_queues.get((dataset_id, shuffle_seed))
            if queue is None:
                _send_json_response(
                    handler,
                    {"error": f"no queue for ({dataset_id}, {shuffle_seed})"},
                    404,
                )
                return
            response = _queue_summary_dict(dataset_id, shuffle_seed, queue)
            response.update(
                issued_bitmap_b64=base64.b64encode(bytes(queue.issued)).decode("ascii"),
                completed_bitmap_b64=base64.b64encode(bytes(queue.completed)).decode(
                    "ascii"
                ),
                # Copy so the response isn't mutated by concurrent
                # request/complete calls after we drop the lock.
                by_worker={k: dict(v) for k, v in queue.by_worker.items()},
            )
        _send_json_response(handler, response)

    def _handle_control(self, handler: BaseHTTPRequestHandler, action: str):
        """Dispatch control actions."""
        try:
            body = _read_request_body(handler)
            try:
                data = json.loads(body.decode("utf-8")) if body else {}
            except json.JSONDecodeError:
                _send_json_response(handler, {"error": "Invalid JSON"}, 400)
                return

            # Audit every control invocation. Phase 1 has no per-caller
            # identity binding (issue #90 deferred); the bearer that
            # got past the auth gate is the only identity we have.
            #
            # Per-action allowlist for ``data`` fields so the audit log
            # captures the *intent* of the action without becoming a
            # future-compat hazard for new control endpoints that may
            # accept secret material (tokens, credentials, etc.).
            # Unknown actions log no data; unknown fields under a known
            # action are dropped.
            self._audit(
                "control",
                action=action,
                data=_audit_control_data(action, data),
            )

            if action == "save_state":
                self._handle_control_save(handler, data)
            elif action == "kick_worker":
                self._handle_control_kick(handler, data)
            elif action == "update_optimizer":
                self._handle_control_update_optimizer(handler, data)
            elif action == "update_num_workers":
                self._handle_control_update_num_workers(handler, data)
            elif action == "shutdown":
                self._handle_control_shutdown(handler, data)
            else:
                _send_json_response(
                    handler, {"error": f"Unknown control action: {action}"}, 404
                )
        except Exception as e:
            logger.error(f"Error handling control/{action}: {e}", exc_info=True)
            _send_json_response(handler, {"error": str(e)}, 500)

    def _handle_control_save(self, handler, data):
        """Save server state on demand."""
        # Unconditional save
        self.save_state()
        _send_json_response(handler, {"status": "ok", "sync_round": self._sync_round})

    def _handle_control_kick(self, handler, data):
        """Evict a worker."""
        worker_id = data.get("worker_id")
        if not worker_id:
            _send_json_response(handler, {"error": "worker_id required"}, 400)
            return
        with self._workers_lock:
            if worker_id not in self._workers:
                _send_json_response(
                    handler, {"error": f"Worker {worker_id} not found"}, 404
                )
                return
        self._handle_worker_death(worker_id)
        logger.info(f"Worker {worker_id} kicked via control endpoint")
        _send_json_response(handler, {"status": "ok", "worker_id": worker_id})

    def _handle_control_update_optimizer(self, handler, data):
        """Update outer optimizer hyperparameters in-place."""
        pg = self.outer_optimizer.param_groups[0]
        updated = {}
        if "lr" in data:
            pg["lr"] = float(data["lr"])
            self._outer_lr = pg["lr"]
            updated["lr"] = pg["lr"]
        if "momentum" in data:
            pg["momentum"] = float(data["momentum"])
            updated["momentum"] = pg["momentum"]
        if not updated:
            _send_json_response(
                handler,
                {"error": "No parameters to update (provide lr and/or momentum)"},
                400,
            )
            return
        logger.info(f"Outer optimizer updated via control: {updated}")
        _send_json_response(handler, {"status": "ok", **updated})

    def _handle_control_update_num_workers(self, handler, data):
        """Update the expected number of workers."""
        num_workers = data.get("num_workers")
        if num_workers is None:
            _send_json_response(handler, {"error": "num_workers required"}, 400)
            return
        num_workers = int(num_workers)
        if num_workers < self.min_workers:
            _send_json_response(
                handler,
                {"error": f"num_workers must be >= min_workers ({self.min_workers})"},
                400,
            )
            return
        self.num_workers = num_workers
        logger.info(f"num_workers updated to {num_workers} via control")
        _send_json_response(handler, {"status": "ok", "num_workers": self.num_workers})

    def _handle_control_shutdown(self, handler, data):
        """Gracefully shut down the server."""
        logger.info("Shutdown requested via control endpoint")
        if self.save_every_n_rounds > 0:
            self.save_state()
        _send_json_response(handler, {"status": "ok", "message": "Shutting down"})
        # Stop in a separate thread so the response can be sent first
        threading.Thread(target=self.stop, daemon=True).start()

    def _audit(self, event: str, **fields: Any) -> None:
        """Append a JSONL event to ``<output_dir>/diloco_audit.log``.

        Best-effort: if the write fails (disk full, permissions, etc.)
        we log a warning and move on rather than failing the operation
        that triggered the event. The audit log is a *record*, not a
        guard — security checks must not depend on it landing on
        disk. Records carry a UTC timestamp and arbitrary kwargs.

        Holds a single append-mode file handle for the server's
        lifetime (opened lazily on first record) so the common path is
        one ``write`` + ``flush`` rather than an ``open``/``close`` pair
        per event. **Never call this while holding ``self._sync_cond``**
        — a slow disk would otherwise stall the sync barrier and every
        worker parked on it. The barrier paths accumulate events in a
        local list and flush via :meth:`_audit` only after releasing the
        condition (see ``_handle_worker_death`` /
        ``_handle_submit_fragment_pseudograd``).

        No-op when ``output_dir`` is empty (in-process tests that
        construct DiLoCoServer without a real directory).
        """
        if not self._audit_path:
            return
        record: Dict[str, Any] = {
            "ts": _utc_iso_now(),
            "event": event,
            **fields,
        }
        line = json.dumps(record, default=str) + "\n"
        try:
            with self._audit_lock:
                if self._audit_fh is None:
                    self._audit_fh = open(self._audit_path, "a", encoding="utf-8")
                self._audit_fh.write(line)
                self._audit_fh.flush()
        except OSError as exc:
            logger.warning(
                "audit-log write failed (event=%s, path=%s): %s",
                event,
                self._audit_path,
                exc,
            )

    def _audit_many(self, events: "List[Tuple[str, Dict[str, Any]]]") -> None:
        """Flush a batch of ``(event, fields)`` records collected while a
        lock was held. Call *after* releasing ``self._sync_cond`` so the
        file I/O never blocks the barrier."""
        for event, fields in events:
            self._audit(event, **fields)

    def _close_audit(self) -> None:
        """Close the persistent audit handle (idempotent)."""
        with self._audit_lock:
            if self._audit_fh is not None:
                try:
                    self._audit_fh.close()
                except OSError:
                    pass
                self._audit_fh = None

    # Three "bulk" endpoints: large tensor transfers. When a separate
    # bulk port is configured these are removed from the control port
    # and only served on the bulk listener. Centralized here so the
    # routing tables and the off-port 404 hint stay in sync.
    _BULK_PATHS = frozenset(
        {"/submit_pseudograd", "/submit_fragment_pseudograd", "/global_params"}
    )

    #: Bind hosts that are not routable as a connect target. When the
    #: server binds one of these we can't put it in the advertised bulk
    #: URL — the worker would dial a wildcard / its own loopback.
    _WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", ""})

    def get_bulk_url(self, request_host: Optional[str] = None) -> Optional[str]:
        """Public URL workers should use for the bulk endpoints.

        ``None`` when no separate bulk listener is configured (bulk
        endpoints are served on the control port). The ``/register``
        response includes this string under the
        ``X-Forgather-Bulk-Url`` header so workers learn about it
        without an extra round-trip.

        When the server bound a wildcard address (``0.0.0.0`` / ``::``)
        it has no reliable view of its own routable address, so
        advertising ``self.host`` verbatim would hand remote workers an
        unroutable ``http://0.0.0.0:<port>``. In that case we prefer
        ``request_host`` — the host the registering worker actually used
        to reach the control port, which is routable from that worker by
        construction — and fall back to loopback only as a last resort.
        """
        if self.bulk_port is None:
            return None
        scheme = "https" if self.bulk_ssl_context is not None else "http"
        host = self.host
        if host in self._WILDCARD_HOSTS or host is None:
            host = request_host or "127.0.0.1"
        return f"{scheme}://{host}:{self.bulk_port}"

    def _create_handler(self, role: str = "control"):
        """Create a request handler bound to this server.

        ``role`` selects which set of endpoints + which auth token the
        handler enforces:

        * ``"control"`` — full route table. When ``bulk_port`` is set
          the three bulk paths are intentionally absent (a 404 with an
          ``X-Forgather-Bulk-Url`` hint is returned instead).
        * ``"bulk"`` — only the three bulk paths plus ``/health``;
          auth check uses ``bulk_auth_enabled`` + the bearer
          (defaults to the control token when ``bulk_auth_enabled``).
        """
        server_ref = self

        class DiLoCoRequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Route HTTP logs through our logger
                logger.debug(format, *args)

            def _expected_token(self) -> Optional[str]:
                """Token this listener checks against. The bulk
                listener can opt out entirely via
                ``bulk_auth_enabled=False`` — in that case bulk
                endpoints are unauthenticated."""
                if role == "bulk" and not server_ref.bulk_auth_enabled:
                    return None
                # Both roles share the same bearer when auth is on —
                # there's one secret per server, two listeners.
                return server_ref.auth_token

            def _authenticated(self, path: str) -> bool:
                """Request auth: mTLS peer cert OR bearer token.

                ``/health`` is intentionally exempt for liveness probes.
                """
                if path == "/health":
                    return True
                return authenticate_request(self, self._expected_token())

            def _bulk_offloaded(self, path: str) -> bool:
                """If this is a bulk endpoint and we have a separate
                bulk listener, return 404 + hint and tell the caller
                to stop. Avoids two ways into the bulk plane (a
                slow-but-secure path on the control listener and a
                fast-but-cleartext path on the bulk listener) which
                would let an attacker pick whichever is convenient."""
                if (
                    role != "control"
                    or server_ref.bulk_port is None
                    or path not in DiLoCoServer._BULK_PATHS
                ):
                    return False
                bulk_url = server_ref.get_bulk_url(_request_host(self)) or ""
                self.send_response(404)
                self.send_header("X-Forgather-Bulk-Url", bulk_url)
                self.send_header("Content-Type", "application/json")
                msg = json.dumps(
                    {
                        "error": (
                            f"Bulk endpoint {path} is served on the "
                            f"bulk listener at {bulk_url}; control "
                            f"port refuses it to keep the security "
                            f"profile unambiguous."
                        ),
                        "bulk_url": bulk_url,
                    }
                ).encode("utf-8")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                return True

            def _allowed_in_role(self, path: str) -> bool:
                """Bulk listener only serves bulk endpoints + /health."""
                if role == "control":
                    return True
                return path in DiLoCoServer._BULK_PATHS or path == "/health"

            def do_POST(self):
                try:
                    path = self.path.rstrip("/")
                    if not self._allowed_in_role(path):
                        _send_json_response(
                            self,
                            {"error": f"Endpoint {path} not served on this port"},
                            404,
                        )
                        return
                    # Auth check before the bulk-URL hint (issue #90
                    # review L1): an unauthenticated caller doesn't get
                    # to learn the bulk-listener topology.
                    if not self._authenticated(path):
                        return
                    if self._bulk_offloaded(path):
                        return
                    if path == "/register":
                        server_ref._handle_register(self)
                    elif path == "/submit_pseudograd":
                        server_ref._handle_submit_pseudograd(self)
                    elif path == "/submit_fragment_pseudograd":
                        server_ref._handle_submit_fragment_pseudograd(self)
                    elif path == "/heartbeat":
                        server_ref._handle_heartbeat(self)
                    elif path == "/deregister":
                        server_ref._handle_deregister(self)
                    elif path == "/datasets/register":
                        server_ref._handle_register_dataset(self)
                    elif path == "/work/request":
                        server_ref._handle_request_work(self)
                    elif path == "/work/complete":
                        server_ref._handle_complete_work(self)
                    elif path.startswith("/control/"):
                        action = path[len("/control/") :]
                        server_ref._handle_control(self, action)
                    else:
                        _send_json_response(
                            self, {"error": f"Unknown endpoint: {path}"}, 404
                        )
                except Exception as e:
                    logger.error(f"Error handling POST {self.path}: {e}", exc_info=True)
                    _send_json_response(self, {"error": str(e)}, 500)

            def do_GET(self):
                try:
                    # urlparse separates the path from the query string;
                    # rstrip("/") here would mangle "/work/queue?…" so
                    # apply it after parsing.
                    parsed = urlparse(self.path)
                    path = parsed.path.rstrip("/")
                    if not self._allowed_in_role(path):
                        _send_json_response(
                            self,
                            {"error": f"Endpoint {path} not served on this port"},
                            404,
                        )
                        return
                    # Auth check before the bulk-URL hint (issue #90
                    # review L1): unauthenticated callers don't learn
                    # the bulk-listener topology.
                    if not self._authenticated(path):
                        return
                    if self._bulk_offloaded(path):
                        return
                    if path == "/health":
                        _send_json_response(self, {"status": "ok"})
                    elif path == "/global_params":
                        server_ref._handle_get_global_params(self)
                    elif path == "/status":
                        server_ref._handle_status(self)
                    elif path == "/info":
                        server_ref._handle_info(self)
                    elif path == "/model_def":
                        server_ref._handle_model_def(self)
                    elif path == "/work/queues":
                        server_ref._handle_get_queues(self)
                    elif path == "/work/queue":
                        server_ref._handle_get_queue(self)
                    else:
                        _send_json_response(
                            self, {"error": f"Unknown endpoint: {path}"}, 404
                        )
                except Exception as e:
                    logger.error(f"Error handling GET {self.path}: {e}", exc_info=True)
                    _send_json_response(self, {"error": str(e)}, 500)

        return DiLoCoRequestHandler

    def save_state(self, path: Optional[str] = None):
        """Save server state (global params + outer optimizer) to disk.

        Uses HF-compatible sharded checkpoint format for model weights and a
        separate server_state.pt for optimizer, round counter, and metadata.

        Args:
            path: Explicit directory to save into. When provided, saves directly
                there without checkpoint rotation. When None, uses save_dir with
                automatic checkpoint-{round} naming and rotation.
        """
        if not self._running:
            raise RuntimeError("Server is not running.")

        if not self._dirty:
            logger.info("State is clean. Skipping save.")
            return

        if path is not None:
            checkpoint_path = path
        else:
            checkpoint_path = next_checkpoint_path(self.output_dir, self._sync_round)

        os.makedirs(checkpoint_path, exist_ok=True)

        # Save model weights as HF-compatible sharded checkpoint
        save_model_checkpoint(
            checkpoint_path, self.get_global_params(), safetensors=self.safetensors
        )

        # Work-queue state is intentionally NOT persisted (#46). Earlier
        # versions rode the per-queue bitmap into server_state.pt so a
        # server restart preserved issuance — the rationale was crash
        # recovery within a run. In practice the more common pattern
        # was "operator changed the dataset, output_dir stayed the
        # same" → workers register a fresh dataset_id, but the old
        # queue lingers in /work/queues with a stale hint.length and
        # an unrecognized dataset_id key. The 2026-05-26 bringup
        # chased exactly this confusion.
        #
        # Trade: on a server crash, in-flight units (≤ N_workers, one
        # per worker) become re-issuable. That matches the design's
        # accepted worker-death budget. Workers re-register their
        # datasets on startup anyway, so the server reconstructs the
        # queue map on demand. Operators who really want mid-epoch
        # resume can keep their own queue snapshot.
        server_state = {
            "outer_optimizer": self.outer_optimizer.state_dict(),
            "sync_round": self._sync_round,
            "num_workers": self.num_workers,
            "param_names": self._param_names,
            "async_mode": self.async_mode,
            "total_submissions": self._total_submissions,
        }
        torch.save(server_state, os.path.join(checkpoint_path, "server_state.pt"))

        logger.info(f"Server state saved to {checkpoint_path}")

        # Rotate checkpoints (only in standard mode, not explicit path)
        if path is None and self.output_dir and self.save_total_limit > 0:
            maybe_delete_oldest_checkpoint(self.output_dir, self.save_total_limit)
        self._dirty = False

    @staticmethod
    def _reorder_state_dict(
        state_dict: Dict[str, torch.Tensor], saved_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Rebuild ``state_dict`` in the order given by ``saved_names``.

        Validates the key sets match (loud failure on architecture
        drift). Extra keys in ``state_dict`` not in ``saved_names`` are
        dropped — the saved order is authoritative; if the live model
        gained a param the operator should treat that as a model-arch
        change and not resume.
        """
        sd_keys = set(state_dict.keys())
        saved_keys = set(saved_names)
        missing = saved_keys - sd_keys
        extra = sd_keys - saved_keys
        if missing or extra:
            raise RuntimeError(
                f"Model param keys don't match the saved checkpoint's "
                f"param_names (model arch drift?). Missing: {sorted(missing)[:5]}"
                f"{'...' if len(missing) > 5 else ''}. "
                f"Extra: {sorted(extra)[:5]}"
                f"{'...' if len(extra) > 5 else ''}."
            )
        return {name: state_dict[name] for name in saved_names}

    def load_state(self, checkpoint_path: Optional[str] = None):
        """Load server state from a checkpoint directory and reset internal
        Args:
            checkpoint_path: Path to a checkpoint directory; defaults to searching output_dir.

        Param ordering is taken from ``server_state.pt``'s ``param_names``
        list (the canonical order at save time), not from the on-disk
        model state_dict iteration order. The two can disagree — e.g.
        ``save_model_checkpoint`` currently writes safetensors index
        keys in arbitrary hash order — and the SGD optimizer's
        ``state_dict()`` keys momentum buffers by integer slot, so a
        slot-vs-slot mismatch on reload would silently apply a momentum
        buffer of one shape to a param of another. Caught the bug
        cold on the May 26 bringup: hidden_size=512 param paired with
        intermediate_size=1280 momentum → ``buf.add_(grad)`` crash on
        the first sync after restart. See #45 for the trace.
        """
        if self._running:
            raise RuntimeError(
                "Server can't load a checkpoint while running. Stop it first!"
            )

        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(self.output_dir)
            if not checkpoint_path:
                raise ValueError(
                    f"No checkpoints found in {self.output_dir}. Please provide a valid model directory."
                )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"The checkpoint path, {checkpoint_path}, does not exist"
            )
        if not os.path.isdir(checkpoint_path):
            raise NotADirectoryError(
                f"The checkpoint path, {checkpoint_path}, is not a directory"
            )

        # Load model weights via sharded checkpoint API
        logger.info(f"Loading model from checkpoint at {checkpoint_path}")
        state_dict = load_model_checkpoint(checkpoint_path, module=None, device="cpu")

        # Peek at server_state.pt BEFORE initializing so we can reorder
        # the state_dict to match the canonical save-time order. The
        # outer-optimizer state uses integer-keyed slots so the param
        # list at slot i must hold the same param it did at save time.
        server_state_path = os.path.join(checkpoint_path, "server_state.pt")
        server_state: Optional[Dict[str, Any]] = None
        if os.path.exists(server_state_path):
            server_state = torch.load(
                server_state_path, map_location="cpu", weights_only=False
            )
            saved_names = server_state.get("param_names")
            if saved_names:
                state_dict = self._reorder_state_dict(state_dict, saved_names)
            else:
                # Legacy server_state.pt without param_names: nothing we
                # can do to canonicalize. The optimizer reload may
                # still misalign — log loudly so the operator can spot
                # a downstream crash and reach for a clean restart.
                logger.warning(
                    "server_state.pt has no 'param_names' entry — using "
                    "state_dict iteration order, which may not match the "
                    "saved optimizer state. Pre-#45 checkpoint."
                )

        # Initialize from state-dictionary
        self._initialize(state_dict)

        # Remember where we loaded from so /model_def can serve the
        # definition bundle, and fold the bundle's content hash into the
        # advertised model_hash. _initialize() set model_hash from the
        # parameter (name, shape) set alone; folding in the on-disk config /
        # custom-code / tokenizer contents means a worker's cached bundle
        # stamp invalidates on *any* definition change (a config tweak or an
        # edited modeling .py), not only a parameter-shape change.
        self._loaded_checkpoint_dir = os.path.realpath(checkpoint_path)
        try:
            bundle_hash = compute_bundle_hash(self._loaded_checkpoint_dir)
            self._model_hash = hashlib.sha256(
                (self._model_hash + ":" + bundle_hash).encode("utf-8")
            ).hexdigest()
        except OSError as exc:
            # A hash over the definition files is best-effort: if the dir is
            # unreadable we keep the parameter-only hash rather than fail the
            # load. /model_def will surface the real error if hit.
            logger.warning("Could not hash model-definition bundle: %s", exc)

        # Load server state if present
        if server_state is not None:
            self.outer_optimizer.load_state_dict(server_state["outer_optimizer"])
            self._sync_round = server_state["sync_round"]
            self._total_submissions = server_state.get("total_submissions", 0)

            # Work-queue state is no longer persisted (#46). For
            # backward-compat with pre-#46 checkpoints, surface a
            # warning if any work-queue entries are present — they're
            # being silently ignored, and the operator should know in
            # case they were expecting mid-epoch resume from this
            # checkpoint.
            legacy_queues = server_state.get("work_queues") or {}
            if legacy_queues:
                logger.warning(
                    "Ignoring %d work-queue entr%s in legacy server_state.pt — "
                    "work-queue persistence was removed in #46. Workers will "
                    "re-register their datasets on connect; any in-flight "
                    "units from the prior run will be re-issued.",
                    len(legacy_queues),
                    "y" if len(legacy_queues) == 1 else "ies",
                )

            logger.info(
                f"Server state loaded from {checkpoint_path}, at round {self._sync_round}"
            )
        else:
            logger.warning(
                f"No server_state.pt found in {checkpoint_path}, loaded model weights only (starting fresh)"
            )

    def _start_health_monitor(self):
        """Start the health monitor if heartbeat_timeout > 0."""
        if self.heartbeat_timeout > 0:
            from .health import HealthMonitor

            self._health_monitor = HealthMonitor(
                self,
                heartbeat_timeout=self.heartbeat_timeout,
            )
            self._health_monitor.start()

    def _stop_health_monitor(self):
        """Stop the health monitor if running."""
        if self._health_monitor is not None:
            self._health_monitor.stop()
            self._health_monitor = None

    def _wrap_tls(self) -> None:
        """If an SSL context is configured, wrap the listening socket.

        The stdlib pattern: replace ``server.socket`` with the wrapped
        version *before* serve_forever runs. Each accepted connection
        then negotiates TLS on accept. ``CERT_OPTIONAL`` on the context
        means client certs are accepted but not required — bearer-only
        clients keep working; mTLS callers get cluster-identity proof.
        """
        if self.ssl_context is None:
            return
        self._server.socket = self.ssl_context.wrap_socket(
            self._server.socket, server_side=True
        )

    def _start_bulk_listener(self) -> None:
        """Spawn the bulk-port listener when configured.

        Runs in its own daemon thread with its own handler class
        (``role="bulk"``). The bulk handler enforces the bulk-port's
        own auth-token / TLS settings; cleartext + unauthenticated is
        explicitly allowed for trusted-LAN throughput.
        """
        if self.bulk_port is None:
            return
        bulk_handler_class = self._create_handler(role="bulk")
        self._bulk_server = ThreadingHTTPServer(
            (self.host, self.bulk_port), bulk_handler_class
        )
        self._bulk_server.daemon_threads = True
        if self.bulk_ssl_context is not None:
            self._bulk_server.socket = self.bulk_ssl_context.wrap_socket(
                self._bulk_server.socket, server_side=True
            )
        self._bulk_server_thread = threading.Thread(
            target=self._bulk_server.serve_forever, daemon=True
        )
        self._bulk_server_thread.start()
        bulk_scheme = "https" if self.bulk_ssl_context is not None else "http"
        bulk_auth = (
            "bearer-required"
            if self.bulk_auth_enabled and self.auth_token
            else "no-auth"
        )
        logger.info(
            f"DiLoCo bulk listener on {bulk_scheme}://{self.host}:{self.bulk_port} "
            f"(auth={bulk_auth})"
        )

    def _stop_bulk_listener(self) -> None:
        """Counterpart to ``_start_bulk_listener``."""
        if self._bulk_server is not None:
            self._bulk_server.shutdown()
            if self._bulk_server_thread is not None:
                self._bulk_server_thread.join(timeout=5)
            self._bulk_server.server_close()
            self._bulk_server = None
            self._bulk_server_thread = None

    def run(self):
        """Run the server (blocking). Call this from the main process."""
        if self._running:
            raise RuntimeError("Run cannot be called, if already running.")
        handler_class = self._create_handler()
        self._server = ThreadingHTTPServer((self.host, self.port), handler_class)
        self._server.daemon_threads = True
        self._wrap_tls()
        self._running = True
        self._started_at = time.time()

        mode = "async" if self.async_mode else "sync"
        scheme = "https" if self.ssl_context is not None else "http"
        auth_state = "bearer-required" if self.auth_token else "no-auth"
        logger.info(
            f"DiLoCo server starting on {scheme}://{self.host}:{self.port} "
            f"(mode={mode}, auth={auth_state})"
        )
        logger.info(
            f"Expecting {self.num_workers} worker(s), min_workers={self.min_workers}"
        )
        if self.heartbeat_timeout > 0:
            logger.info(f"Health monitoring: timeout={self.heartbeat_timeout}s")
        if self.async_mode and self.dn_buffer_size > 0:
            logger.info(f"Delayed Nesterov: buffer_size={self.dn_buffer_size}")
        if self.dylu_enabled:
            logger.info(f"DyLU enabled: base_sync_every={self.dylu_base_sync_every}")

        self._start_health_monitor()
        self._start_bulk_listener()

        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server interrupted by Ctrl-C")
            if self.save_every_n_rounds > 0:
                logger.info("Saving server state before shutdown...")
                self.save_state()
        finally:
            self._stop_health_monitor()
            self._stop_bulk_listener()
            self._running = False
            self._server.server_close()
            self._close_audit()
            logger.info("Server stopped")

    def start(self):
        """Start the server in a background thread (non-blocking)."""
        if self._running:
            raise RuntimeError("Start cannot be called, if already running.")
        handler_class = self._create_handler()
        self._server = ThreadingHTTPServer((self.host, self.port), handler_class)
        self._server.daemon_threads = True
        self._wrap_tls()
        self._running = True
        self._started_at = time.time()

        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._server_thread.start()

        self._start_health_monitor()
        self._start_bulk_listener()

        mode = "async" if self.async_mode else "sync"
        scheme = "https" if self.ssl_context is not None else "http"
        auth_state = "bearer-required" if self.auth_token else "no-auth"
        logger.info(
            f"DiLoCo server started on {scheme}://{self.host}:{self.port} "
            f"(mode={mode}, auth={auth_state}, background)"
        )
        logger.info(
            f"Expecting {self.num_workers} worker(s), min_workers={self.min_workers}"
        )

    def stop(self):
        """Stop the background server."""
        if not self._running:
            raise RuntimeError("Stop cannot be called, unless we are already running.")
        self._stop_health_monitor()
        self._stop_bulk_listener()
        if self._server:
            self._server.shutdown()
            self._running = False
            if self._server_thread:
                self._server_thread.join(timeout=5)
            self._server.server_close()
            self._close_audit()
            logger.info("Server stopped")

    @property
    def address(self) -> str:
        """Return the server address as host:port."""
        return f"{self.host}:{self.port}"
