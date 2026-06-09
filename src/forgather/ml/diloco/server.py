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
import json
import logging
import math
import os
import platform
import re
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
    enumerate_model_def_files,
    pack_model_def,
)
from forgather.ml.diloco.wire_serialize import (
    WIRE_FORMATS,
    deserialize_state_dict,
    serialize_state_dict,
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
    # Latest unified-stats snapshot this worker reported on its heartbeat
    # (normalized schema, see diloco/stats.py). Surfaced per-worker in
    # /status; folded into the server's StatsAggregator for the aggregate view.
    stats: Dict[str, Any] = field(default_factory=dict)
    # Latest per-worker DiLoCo sync-state from the heartbeat (sync_count,
    # last_sync_time, last send/recv MB). For an off-server backend
    # (shared-memory) this is the only progress signal — the server-side
    # ``sync_round`` above stays 0 since the worker never submits. Surfaced in
    # /status.
    sync_state: Dict[str, Any] = field(default_factory=dict)
    # Trainer-control command queued for this worker, delivered on its next
    # heartbeat and cleared on delivery (the relay channel for collective
    # save / save-and-stop / abort issued via /control/command). One of
    # ``"save_checkpoint"``, ``"save_and_stop"``, ``"abort"``, or ``None``.
    pending_command: Optional[str] = None


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
    "command": frozenset({"command", "worker_id"}),
    "shutdown": frozenset(),
}

# Trainer-control commands the relay (/control/command) accepts and queues
# for delivery on a worker's next heartbeat. These mirror the worker-side
# trainer-control vocabulary (see control_callback.COMMAND_CODES) minus the
# server-irrelevant ones: the relay only drives save / stop / abort.
_RELAY_COMMANDS = frozenset({"save_checkpoint", "save_and_stop", "abort"})


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


def _describe_optimizer(optimizer) -> str:
    """One-line description of an optimizer for status display.

    Class name + the full hyperparameter set of its (single) param group,
    e.g. ``SGD(lr=0.7, momentum=0.9, dampening=0, weight_decay=0,
    nesterov=True)``. This is more informative than reconstructing
    ``lr``/``momentum`` client-side — it shows ``nesterov`` and every other
    knob — and it generalizes to optimizers other than SGD without the CLI
    having to know their shape. The outer optimizer always has a single param
    group (one ParameterList); if that ever changes, fall back to a count.
    """
    name = type(optimizer).__name__
    groups = optimizer.param_groups
    if len(groups) != 1:
        return f"{name}({len(groups)} param groups)"
    items = ", ".join(f"{k}={v}" for k, v in groups[0].items() if k != "params")
    return f"{name}({items})"


def _serialize_state_dict(
    state_dict: Dict[str, torch.Tensor], fmt: str = "pickle"
) -> bytes:
    """Serialize a state dict to bytes using the named wire format."""
    return serialize_state_dict(state_dict, fmt)


def _deserialize_state_dict(
    data: bytes, fmt: str = "pickle"
) -> Dict[str, torch.Tensor]:
    """Deserialize bytes to a state dict using the named wire format."""
    return deserialize_state_dict(data, fmt)


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
    fmt: str = "pickle",
):
    """Send a state dict as an octet-stream response.

    ``extra_headers`` (e.g. ``{"X-Forgather-Bulk-Url": "..."}``) ride
    along on the same response so worker registration can learn the
    bulk-listener URL without an extra round-trip.

    ``fmt`` selects the wire codec. The response carries no header, so the
    worker decodes it with the format it adopted from /info — pass the server's
    authoritative ``wire_format`` so the two ends agree.
    """
    data = _serialize_state_dict(state_dict, fmt)
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
        upload_dtype: Optional[str] = None,
        upload_sr: bool = False,
        download_dtype: str = "fp32",
        download_sr: bool = False,
        wire_format: str = "pickle",
        backend: str = "http",
        bf16_comm: Optional[bool] = None,
        num_fragments: int = 1,
        heartbeat_timeout: float = 120.0,
        min_workers: int = 1,
        auth_token: Optional[str] = None,
        ssl_context: Optional["ssl.SSLContext"] = None,
        tls_cert_file: Optional[str] = None,
        tls_key_file: Optional[str] = None,
        tls_ca_file: Optional[str] = None,
        bulk_cleartext: bool = False,
        grpc_enabled: bool = False,
        default_work_units: int = 1024,
        run_name: Optional[str] = None,
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
        # The stable, operator-configured worker count (the launch ``-n``).
        # Unlike ``self.num_workers`` — which drifts at runtime: raised on
        # over-registration, lowered to ``max(min_workers, remaining)`` on
        # worker death — this never changes after construction. It is the
        # single source of truth for the shared-memory group size (a
        # co-located group is a fixed N) and is advertised via /info so a
        # follower sizes its region correctly regardless of who has registered
        # yet.
        self._configured_num_workers = num_workers
        self.min_workers = min_workers
        self.host = host
        self.port = port or self._find_available_port()
        self.output_dir = output_dir
        # Operator label for this run's stats log dir; consumed in
        # _initialize (which also runs on resume, so it must not own this).
        self._run_name = run_name
        self.save_every_n_rounds = save_every_n_rounds
        self.save_total_limit = save_total_limit
        self.async_mode = async_mode
        self.safetensors = safetensors
        # Bulk-tensor wire codec (issue #154), advertised via /info and
        # authoritative for both legs. "pickle" (default) keeps back-compat with
        # an older worker; "safetensors" drops pickle for an explicit typed,
        # zero-copy frame. Validated against the known codecs.
        if wire_format not in WIRE_FORMATS:
            raise ValueError(
                f"wire_format must be one of {WIRE_FORMATS}, got {wire_format!r}"
            )
        self.wire_format = wire_format
        # Sync backend the group must use (issue #154). The backend is a
        # group-wide invariant — a collective group is one all-reduce world, a
        # shared-memory group is co-located, an HTTP group is independent
        # workers; there is no valid mixed group. The server *declares* it here
        # and advertises it via /info so every worker validates its own launched
        # backend against it (fail loud on disagreement). For ``http`` and
        # ``collective`` this is pure declaration metadata; for ``shared_memory``
        # the server additionally *is* the aggregator (set up below).
        if backend not in ("http", "shared_memory", "collective"):
            raise ValueError(
                f"backend must be 'http', 'shared_memory', or 'collective', "
                f"got {backend!r}"
            )
        self.backend = backend
        # Shared-memory (Flavor 2, issue #154): shared-memory DiLoCo is
        # single-host and the server is co-located, so the server maps the same
        # region and runs the outer step itself — owning the master + outer
        # momentum and checkpointing them through its normal save_state /
        # load_state — instead of electing an aggregator worker whose trained
        # state was never persisted (the root cause of the checkpoint-0 /
        # resume-divergence bugs). The region is created in start() (after the
        # master is loaded) and torn down in stop(). The dir is host-local and
        # stable per port so a restart reclaims/rebuilds the same region via the
        # ownership lease; it is advertised in /info as ``shm_group_dir`` so a
        # follower attaches without re-deriving it.
        self._shm_agg = None
        self._shm_agg_thread: Optional[threading.Thread] = None
        self._shm_stop = threading.Event()
        self._shm_group_dir: Optional[str] = None
        if self.backend == "shared_memory":
            import tempfile

            self._shm_group_dir = os.path.join(
                tempfile.gettempdir(), f"diloco_shm_p{self.port}"
            )
        self.dn_buffer_size = dn_buffer_size
        self.dylu_enabled = dylu_enabled
        self.dylu_base_sync_every = dylu_base_sync_every
        # Group-wide worker settings the server is authoritative for (issue
        # #53 follow-up). These MUST match across the group for the sync /
        # fragment barriers to be coherent, so the operator sets them on
        # the server and workers adopt them verbatim from /info.
        #
        # Wire precision (issue #130). Four independent server-authoritative
        # knobs, one per (direction × dtype-cast-vs-SR) cell:
        #   - ``upload_dtype``  / ``upload_sr``:  worker → server pseudo-grads
        #   - ``download_dtype`` / ``download_sr``: server → worker params
        # ``upload_dtype`` defaults to bf16 (today's default) and
        # ``download_dtype`` to fp32 (today's default); SR is off in both
        # directions until the operator opts in for the convergence-
        # compression experiment. The legacy ``bf16_comm`` boolean is
        # accepted as a deprecated alias for ``upload_dtype`` (passing both
        # raises). The dtype enum is ``{"fp32","bf16"}``; ``"fp8_e4m3"`` /
        # ``"fp8_e5m2"`` slot in as a future pure-addition.
        self.sync_every = sync_every
        if bf16_comm is not None and upload_dtype is not None:
            raise ValueError(
                "DiLoCoServer: pass either bf16_comm (deprecated) or "
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
        # Legacy mirror — read by log lines, old tests, the deprecated
        # ``/info`` key. True iff the upload leg is bf16, matching the
        # pre-refactor semantics of the single ``bf16_comm`` flag.
        self.bf16_comm = self.upload_dtype == "bf16"
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
        # The control-plane TLS material as file paths (issue #154). gRPC needs
        # PEM bytes, not a Python SSLContext, so _grpc_security reads these to
        # build matching ssl_server_credentials. Present iff TLS is on.
        self.tls_cert_file = tls_cert_file
        self.tls_key_file = tls_key_file
        self.tls_ca_file = tls_ca_file
        # Optional second listener for bulk data transport (pseudo-
        # gradients + global-params). When ``bulk_cleartext`` is set the
        # three bulk endpoints are served *only* on this listener and the
        # control port returns 404 with an ``X-Forgather-Bulk-Url`` hint
        # header. The bulk listener is always cleartext + unauthenticated
        # on a server-picked ephemeral port: its entire purpose is to
        # bypass TLS for throughput on a trusted LAN, and a bearer token
        # over a sniffable cleartext socket is security theater (anyone on
        # the wire reads the tensors anyway). The port is ephemeral because
        # workers learn it over the TLS-protected control plane (the
        # ``/register`` response header), so there's nothing to gain from
        # pinning a fixed number. ``weights_only=True`` on every bulk
        # ``torch.load`` keeps a malicious peer to disrupting training, not
        # RCE-ing the host. ``self.bulk_port`` is filled in with the actual
        # OS-assigned port once the listener binds (see
        # ``_start_bulk_listener``); it is ``None`` until then / when off.
        self._bulk_enabled = bool(bulk_cleartext)
        self.bulk_port: Optional[int] = None
        self._bulk_server = None
        self._bulk_server_thread = None
        # Optional gRPC bulk transport (issue #154). A streaming HTTP/2 listener
        # serving the same three bulk legs, advertised via /info so a worker can
        # negotiate it. It SUPERSEDES the cleartext bulk listener: gRPC is
        # TLS-capable without urllib's overhead, so when enabled the cleartext
        # listener is not also started (two fast paths into the bulk plane is the
        # exact smell the control/bulk role-split exists to prevent). ``grpc_port``
        # is the OS-assigned ephemeral port, filled once the listener binds.
        self._grpc_enabled = bool(grpc_enabled)
        if self._grpc_enabled:
            self._bulk_enabled = False
        self.grpc_port: Optional[int] = None
        self._grpc_server = None
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
        # Directory the /model_def bundle is actually served from. The model
        # definition (config + custom code + tokenizer) lives with the model,
        # not in each rotated checkpoint: save_state writes only weights +
        # server_state.pt into checkpoint-N, so a server restarted off such a
        # checkpoint has no definition there. Resolved in load_state to a dir
        # that actually carries it (the loaded checkpoint if self-contained,
        # else output_dir — the model's home). None when neither has it, in
        # which case /model_def fails loudly instead of serving an empty
        # bundle. See _resolve_model_def_dir.
        self._model_def_dir: Optional[str] = None
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

        # Known-worker roster (issue #103 follow-up). Every worker_id that
        # has ever registered, mapped to its last-reported output_dir and
        # registration time. Unlike ``_workers`` (live registrations) this
        # is NOT cleared on deregistration/death and IS persisted with the
        # server's checkpoints (server_state.pt), so a server restart
        # remembers the names. The webui offers the not-currently-running
        # entries as a menu so an operator can relaunch a worker under its
        # old id — the only way to resume that worker from its own
        # checkpoint, since the checkpoint path is the worker-id-suffixed
        # output_dir. Guarded by ``_workers_lock``.
        self._known_workers: Dict[str, Dict[str, Any]] = {}

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

        # Unified statistics: aggregate per-worker training metrics reported
        # on heartbeats into a server-level view (total tokens/flos/steps,
        # aggregate throughput/mfu/memory, smoothed train/eval loss). Lifetime
        # counters + EMA state persist in the server checkpoint; the JSON log
        # stream (when output_dir is set) resumes like a worker's logger.
        from .stats import StatsAggregator

        self._stats = StatsAggregator()
        # Per-run log directory (mirrors the trainer: <output_dir>/runs/
        # <time_ns>_<name>), holding both the JSONL stats stream and a
        # TensorBoard event file. A fresh start gets a new dir; the chosen
        # subdir is persisted in the checkpoint so a resume continues into the
        # same dir (TensorBoard resumes via purge_step). Distinct runs are
        # retained for comparison/overlay. _maybe_log_stats runs on concurrent
        # heartbeat threads, so its IO + bookkeeping are guarded by this lock.
        # (``self._run_name`` is set in __init__ — _initialize also runs on
        # resume and must not clobber it.)
        self._stats_log_lock = threading.Lock()
        self._stats_run_dir = None  # resolved absolute run dir (lazy)
        self._stats_run_subdir = None  # relative "runs/..." for the checkpoint
        self._resume_run_subdir = None  # prior run subdir, from load_state
        self._tb_writer = None  # torch TensorBoard SummaryWriter (lazy)
        self._tb_purge_step = None  # discard TB events after this step on resume
        self._stats_log_step = -1  # last total_steps a stats record was logged at
        self._last_logged_eval_step = None  # eval_step of the last eval logged

        # Server state
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._started_at: Optional[float] = None
        self._dirty = False

        # Coordinated graceful shutdown (relay save_and_stop to workers, let
        # them drain while we keep serving, then save + stop). Re-entrancy
        # guard so the signal path and /control/shutdown converge on one run.
        self._shutdown_lock = threading.Lock()
        self._shutting_down = False
        self._shutdown_event = threading.Event()  # "shutdown requested"
        self._shutdown_done = threading.Event()  # "drain + stop finished"

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

    def _worker_owned_param_names(self, worker_id: str) -> Optional[set]:
        """Names this worker's slice owns; ``None`` for a solo worker.

        Pipeline-parallel workers register a per-rank slice
        (``slice_shapes`` at register time, stored in
        ``WorkerGroup.member_param_names[pp_rank]``) — they only need
        the parameters their stage holds. Solo workers (degenerate
        groups of one with ``pp_world_size=1``) cover the full model
        by construction, so this returns ``None`` for them and the
        caller skips filtering.

        Returns ``None`` (no filtering) when:
          - the worker isn't in a group (untracked / pre-#84 client),
          - the group has ``pp_world_size <= 1`` (solo),
          - the per-rank ownership set is empty (test path that
            omitted ``param_shapes`` at register time — the seal-time
            coverage check is skipped there too).

        Otherwise returns the set of parameter names the worker owns.

        Reads ``_worker_to_group`` and ``_groups`` without holding
        ``_workers_lock``. The dicts are concurrent-safe at the
        per-key level (CPython dict ops are GIL-atomic), and a stale
        read — e.g. the worker was just evicted — falls back to
        ``None`` → full state, which is safe (the response is moot
        if the worker is gone). Returning ``None`` on a transient
        race is preferable to acquiring the lock from inside helper
        callers that may already hold it (``_workers_lock`` is a
        non-reentrant ``threading.Lock``).
        """
        group_id = self._worker_to_group.get(worker_id)
        if not group_id:
            return None
        group = self._groups.get(group_id)
        if group is None or group.pp_world_size <= 1:
            return None
        # Reverse-lookup pp_rank from members. O(pp_world_size); small.
        for rank, wid in group.members.items():
            if wid == worker_id:
                owned = group.member_param_names.get(rank)
                if owned:
                    return owned
                return None
        return None

    def _params_for_worker(
        self,
        worker_id: str,
        source: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return the global parameter state the worker actually needs.

        For a PP-group rank that owns only its slice (issue #84
        groups), filter the response down to that rank's parameter
        names — sending the full averaged model to every rank wastes
        roughly ``(pp_world_size - 1) / pp_world_size`` of the
        per-rank download bandwidth and the worker discards what it
        doesn't own anyway (see ``PipelineParamView.apply_global``).

        For solo workers and untracked clients the response is the
        full model state (unchanged from pre-PP behavior), preserving
        backward-compat with the pre-#84 ParamView contract.

        ``source`` lets the caller pass an already-built state dict
        (e.g. ``_completed_rounds[my_round]`` shared across barrier
        waiters) so we don't re-clone the full model when the source
        is already cached. ``None`` falls back to a fresh
        ``get_global_params()`` clone.
        """
        owned = self._worker_owned_param_names(worker_id)
        if owned is None:
            result = source if source is not None else self.get_global_params()
        elif source is not None:
            result = {name: source[name] for name in owned if name in source}
        else:
            result = self._get_params_by_names(owned)
        return self._cast_for_download(result)

    def _cast_for_download(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Cast a response state dict to the configured ``download_dtype``.

        Three callers — ``_params_for_worker`` (the sync/register hot
        path) and the two fragment response builders
        (``_handle_submit_fragment_sync``, ``_handle_submit_fragment_async``).
        Shared helper so a future fp8 enum extension lands in exactly
        one place.

        When ``download_dtype == "fp32"`` this is the identity (the
        server stores params as fp32, so no cast happens). When
        ``download_dtype == "bf16"`` and ``download_sr`` is set, the
        cast goes through ``fp32_to_bf16_stochastic_round`` — the same
        SR helper the optimizers use. Otherwise it's a round-to-
        nearest cast via ``.bfloat16()``.

        Note on the apply-side counterpart: ``ParamView.apply_global``
        widens the wire dtype back to the live model dtype with a
        plain RNE cast. So a true-bf16 worker paired with an
        fp32-download server still loses sub-ULP signal on every
        round, with no SR opportunity at the apply step. To capture
        that signal the operator should set ``download_dtype="bf16"``
        + ``download_sr=True`` so the SR happens here, on the
        server, before transmit.

        The cast is applied per-tensor on CPU (the server's master
        params live on CPU). The returned dict has the same key set
        as the input.
        """
        if self.download_dtype == "fp32":
            return d
        if self.download_dtype == "bf16":
            if self.download_sr:
                from forgather.ml.optim.rounding_utils import (
                    fp32_to_bf16_stochastic_round,
                )

                # SR is fp32-input-only by construction; promote any
                # non-fp32 tensors to fp32 first so SR is well-defined.
                # (Currently all server-side tensors are fp32 — see
                # ``server.py:625`` where they're ``.float()``ed at
                # load — but the upcast is cheap and future-proofs
                # against a server-side dtype change.)
                return {
                    name: fp32_to_bf16_stochastic_round(t.float())
                    for name, t in d.items()
                }
            return {name: t.bfloat16() for name, t in d.items()}
        raise ValueError(f"unsupported download_dtype: {self.download_dtype!r}")

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

            # Drop evicted workers from the live-gauge contributions (their
            # delta baselines are kept, so a resumed worker reusing the id
            # continues its accounting rather than re-adding from zero).
            for wid in evict:
                self._stats.drop_worker(wid)

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
                # Remember this worker for the webui's restart menu, even
                # after it later deregisters (issue #103 follow-up). Upsert
                # so a re-registration refreshes the output_dir / timestamp.
                self._known_workers[worker_id] = {
                    "output_dir": info.get("output_dir"),
                    "last_registered": time.time(),
                }
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

        # Return global params, filtered to the worker's slice for
        # PP-group members (the rank only needs its stage's params,
        # not the full averaged model). Solo workers and untracked
        # clients still get the full state. See
        # ``_params_for_worker`` for the load-bearing reason.
        #
        # When a bulk listener is configured, advertise its URL via
        # response header so the worker can route subsequent
        # submit_pseudograd / global_params calls to it directly
        # (issue #90). Pass the worker's own view of our host so a
        # wildcard-bound server advertises a routable URL.
        bulk_url = self.get_bulk_url(_request_host(handler))
        extra = {"X-Forgather-Bulk-Url": bulk_url} if bulk_url else None
        if self.async_mode:
            with self._async_lock:
                _send_tensor_response(
                    handler,
                    self._params_for_worker(worker_id),
                    extra,
                    fmt=self.wire_format,
                )
        else:
            _send_tensor_response(
                handler,
                self._params_for_worker(worker_id),
                extra,
                fmt=self.wire_format,
            )

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
        # ``fmt`` is stamped by the client; absent ⇒ "pickle" (an older worker).
        pseudograds = _deserialize_state_dict(tensor_data, header.get("fmt", "pickle"))

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

            # Filter the cached completed-round state down to this
            # worker's slice for PP-group members; solo workers and
            # untracked clients get the full state. The barrier
            # caches one full-model snapshot per round (line 1848 /
            # line 1162) which is shared across waiters; we slice
            # per-worker on the way out to avoid duplicating the
            # full clone N times.
            response_params = self._params_for_worker(
                worker_id, source=self._completed_rounds[my_round]
            )

        # _sync_cond released: flush audit records off the barrier path.
        self._audit_many(pending_audit)
        _send_tensor_response(handler, response_params, fmt=self.wire_format)

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

            # Filter to this worker's slice for PP-group members;
            # solo workers get the full state.
            global_params = self._params_for_worker(worker_id)

        _send_tensor_response(handler, global_params, fmt=self.wire_format)

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
        # ``fmt`` is stamped by the client; absent ⇒ "pickle" (an older worker).
        pseudograds = _deserialize_state_dict(tensor_data, header.get("fmt", "pickle"))

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
        _send_tensor_response(
            handler, self._cast_for_download(result), fmt=self.wire_format
        )

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

        _send_tensor_response(
            handler, self._cast_for_download(result), fmt=self.wire_format
        )

    def _handle_get_global_params(self, handler: BaseHTTPRequestHandler):
        """Handle request for current global parameters.

        Respects ``download_dtype`` / ``download_sr`` like the sync /
        register response paths — a late joiner or recovery client
        wants the same wire format the rest of the group sees.
        """
        if self.async_mode:
            with self._async_lock:
                _send_tensor_response(
                    handler,
                    self._cast_for_download(self.get_global_params()),
                    fmt=self.wire_format,
                )
        else:
            _send_tensor_response(
                handler,
                self._cast_for_download(self.get_global_params()),
                fmt=self.wire_format,
            )

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
            # Unified-stats snapshot (optional): store the latest per-worker
            # view and fold it into the aggregate. Sanitized to the known
            # numeric schema first — the body is worker-supplied, so this
            # bounds the retained footprint and what /status echoes, and keeps
            # non-finite / non-numeric values out of the aggregate and the
            # wire JSON. Done under the workers lock so a concurrent status
            # read sees a consistent worker record.
            from .stats import sanitize_stats, sanitize_sync_state

            worker_stats = sanitize_stats(info.get("stats"))
            if worker_stats:
                self._workers[worker_id].stats = worker_stats
            # Per-worker DiLoCo sync-state (issue #154): the worker's own view of
            # its sync progress, so the server can show it even for an off-server
            # backend (shared-memory) that never submits. Its own whitelist (not
            # the training-stats schema) — worker-supplied, bound to finite numbers.
            worker_sync_state = sanitize_sync_state(info.get("sync_state"))
            if worker_sync_state:
                self._workers[worker_id].sync_state = worker_sync_state
            # Read (don't yet clear) any queued trainer-control command. We
            # clear it only *after* the response is successfully sent below,
            # so a dropped/failed heartbeat response doesn't silently lose the
            # command — it's redelivered on the next heartbeat. Heartbeats are
            # serial per worker (only the leader sends them), so there's no
            # concurrent-delivery race for the same worker_id.
            command = self._workers[worker_id].pending_command

        # Fold the worker's snapshot into the aggregate (StatsAggregator has
        # its own lock; kept out of the workers lock to avoid holding it over
        # the EMA math).
        if worker_stats:
            self._stats.update(worker_id, worker_stats)

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
        if command is not None:
            response["command"] = command

        _send_json_response(handler, response)

        # Delivered: clear the command now that the response went out. Guard
        # against a newer command queued between the read and here (leave it
        # for the next heartbeat) so we don't drop it.
        if command is not None:
            with self._workers_lock:
                w = self._workers.get(worker_id)
                if w is not None and w.pending_command == command:
                    w.pending_command = None

        # Append an aggregate-stats record (throttled to one per sync round) —
        # done after the response so heartbeat latency isn't tied to file IO.
        if worker_stats:
            self._maybe_log_stats()

    def _maybe_log_stats(self):
        """Append one aggregate-stats record to the server's stats log when
        training has advanced (keyed by total optimizer steps).

        The log is **JSONL** (one JSON object per line), opened in append mode
        — deliberately not the JSON-array writer the trainer uses. A
        long-running server is restarted often and reuses its ``output_dir``;
        append-only is robust to a pre-existing log from any prior run (no
        exclusive-create FileExistsError, no truncate-to-empty), whereas the
        array format needs bracket/close management that proved fragile across
        restarts. The first write of each process truncates a stale log so a
        run starts a clean stream. Best-effort: a logging failure must never
        break the heartbeat path.
        """
        if not self.output_dir:
            return
        # Serialize concurrent heartbeat threads: the step check and the
        # appends must be atomic or two workers race on the same files.
        with self._stats_log_lock:
            snap = self._stats.snapshot()
            steps = int(snap.get("total_steps", 0) or 0)
            if steps <= self._stats_log_step:
                return
            run_dir = self._ensure_run_dir()
            if run_dir is None:
                return
            data = {k: v for k, v in snap.items() if k != "total_steps"}
            data["global_step"] = steps
            data["sync_round"] = self._sync_round
            data["timestamp"] = time.time()
            # Emit eval_loss only on records where a *new* eval arrived (the
            # eval_step advanced); null it otherwise. eval is sparse relative
            # to the per-step stats stream, so repeating the held EMA value on
            # every record would draw a flat staircase — nulling between evals
            # lets the plot connect the real eval points into a curve (the
            # chart spans gaps), matching a worker's TensorBoard eval line.
            eval_step = snap.get("eval_step")
            eval_fresh = (
                eval_step is not None and eval_step != self._last_logged_eval_step
            )
            if not eval_fresh:
                data["eval_loss"] = None
            else:
                self._last_logged_eval_step = eval_step
            # JSONL stream (append-only; the run dir is unique per fresh run so
            # a new run starts an empty file, while a resumed run continues its
            # own file).
            try:
                with open(os.path.join(run_dir, self._STATS_LOG_FILENAME), "a") as f:
                    f.write(json.dumps(data) + "\n")
            except Exception as e:  # pragma: no cover - must not break HB
                logger.warning("Failed to write server stats log: %s", e)
            # TensorBoard mirror, for overlay/comparison across runs.
            self._log_to_tensorboard(steps, snap, eval_fresh)
            self._stats_log_step = steps

    def _ensure_run_dir(self) -> Optional[str]:
        """Resolve (and create) this run's log directory under
        ``<output_dir>/runs/``. Caller holds ``_stats_log_lock``.

        Reuses the prior run's dir on resume (``_resume_run_subdir`` from the
        checkpoint) for continuity; otherwise creates a fresh
        ``runs/<time_ns>_<name>`` (matching the trainer's convention, so common
        tools / TensorBoard can read them together). Returns None when there's
        no ``output_dir`` to write under.
        """
        if self._stats_run_dir is not None:
            return self._stats_run_dir
        if not self.output_dir:
            return None
        if self._resume_run_subdir:
            candidate = os.path.join(self.output_dir, self._resume_run_subdir)
            if os.path.isdir(candidate):
                self._stats_run_dir = candidate
                self._stats_run_subdir = self._resume_run_subdir
                logger.info("Stats: resuming run log dir %s", candidate)
                return candidate
        name = self._run_name or platform.node() or "diloco"
        # Sanitize the (operator-supplied) name to a safe path component —
        # no separators / traversal, conservative character set.
        name = re.sub(r"[^A-Za-z0-9._-]", "-", name)[:64] or "diloco"
        subdir = os.path.join("runs", f"{time.time_ns()}_{name}")
        self._stats_run_dir = os.path.join(self.output_dir, subdir)
        self._stats_run_subdir = subdir
        os.makedirs(self._stats_run_dir, exist_ok=True)
        logger.info("Stats: logging this run to %s", self._stats_run_dir)
        return self._stats_run_dir

    def _log_to_tensorboard(self, step: int, snap: dict, eval_fresh: bool) -> None:
        """Mirror the aggregate snapshot to a TensorBoard event file in the run
        dir. Caller holds ``_stats_log_lock``. Best-effort. Scalar tags match
        the trainer's (``train-loss``/``eval-loss``/``grad-norm``) so a worker's
        own TB run and the server aggregate overlay in TensorBoard."""
        try:
            if self._tb_writer is None:
                from torch.utils.tensorboard import SummaryWriter

                kwargs = {}
                if self._tb_purge_step is not None:
                    kwargs["purge_step"] = self._tb_purge_step
                self._tb_writer = SummaryWriter(self._stats_run_dir, **kwargs)
            w = self._tb_writer

            def add(tag, key):
                v = snap.get(key)
                if isinstance(v, (int, float)):
                    w.add_scalar(tag, v, global_step=step)

            add("train-loss", "train_loss")
            add("grad-norm", "grad_norm")
            add("tokens-per-sec", "tok_per_sec")
            add("mfu", "mfu")
            add("total-tokens", "total_tokens")
            add("total-flos", "total_flos")
            add("peak-memory", "peak_memory")
            if eval_fresh:
                add("eval-loss", "eval_loss")
            w.flush()
        except Exception as e:  # pragma: no cover - must not break HB
            logger.warning("Failed to write server TensorBoard stats: %s", e)

    def _stats_log_path(self) -> Optional[str]:
        """Path to the current run's aggregate-stats JSONL log, or None when no
        run dir has been resolved yet (no ``output_dir`` / nothing logged)."""
        if not self._stats_run_dir:
            return None
        return os.path.join(self._stats_run_dir, self._STATS_LOG_FILENAME)

    def _handle_stats_history(self, handler: BaseHTTPRequestHandler):
        """Serve the aggregate-stats history (the JSONL log the server writes)
        for the webui's loss-curve plot.

        Reads the JSONL log line by line (skipping any blank/partial line),
        returning the records downsampled to ``max_points`` (the latest point
        is always kept). Empty when no ``output_dir`` / nothing logged yet.
        Control-plane, bearer-authenticated like ``/status`` (do_GET runs the
        auth gate first).
        """
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(handler.path).query)
        try:
            max_points = int(qs.get("max_points", ["2000"])[0])
        except (ValueError, TypeError, IndexError):
            max_points = 2000
        max_points = max(1, min(max_points, 20000))

        records: List[dict] = []
        total = 0
        path = self._stats_log_path()
        if path and os.path.isfile(path):
            try:
                # Two passes so peak memory scales with max_points, not the
                # (training-length) log size: pass 1 counts records to pick a
                # stride, pass 2 parses only the kept lines (every stride-th,
                # plus the last). The webui polls this on a refresh interval,
                # so re-parsing the whole growing file each time would be
                # needlessly expensive.
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            total += 1
                stride = 1 if total <= max_points else math.ceil(total / max_points)
                with open(path) as f:
                    idx = -1
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        idx += 1
                        if idx % stride == 0 or idx == total - 1:
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue  # skip a partial/torn line
            except Exception as e:
                logger.warning("stats_history: failed to read %s: %s", path, e)
                records = []
                total = 0

        _send_json_response(
            handler,
            {
                "records": records,
                "count": total,
                "downsampled": total > max_points,
            },
        )

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
                    "stats": w.stats,
                    "sync_state": w.sync_state,
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
        # Full one-line description (class + all hyperparameters, incl.
        # nesterov) — generalizes beyond SGD; the lr/momentum fields above
        # stay for the webui's separate metric cells + back-compat.
        response["outer_optimizer"] = _describe_optimizer(self.outer_optimizer)
        response["save_dir"] = self.output_dir
        response["model_params"] = self._model_params
        response["model_size_mb"] = round(self._model_size_mb, 2)

        # Unified aggregate training statistics (total tokens/flos/steps,
        # aggregate throughput/mfu/memory, smoothed train/eval loss).
        response["aggregate_stats"] = self._stats.snapshot()

        _send_json_response(handler, response)

    def _handle_known_workers(self, handler: BaseHTTPRequestHandler):
        """Return the roster of every worker_id the server has ever seen.

        Each entry carries the worker's last-reported ``output_dir``, its
        last registration time, and a ``running`` flag (true iff it's
        currently registered). The webui offers the not-running entries as
        a menu so an operator can relaunch a worker under its old id and
        thereby resume from that worker's own checkpoint (issue #103
        follow-up). The roster is persisted with the server's checkpoints,
        so it survives a server restart.
        """
        with self._workers_lock:
            live = set(self._workers.keys())
            workers = [
                {
                    "worker_id": wid,
                    "output_dir": rec.get("output_dir"),
                    "last_registered": rec.get("last_registered"),
                    "running": wid in live,
                }
                for wid, rec in self._known_workers.items()
            ]
        # Stable order: running first, then most-recently-registered.
        workers.sort(key=lambda w: (not w["running"], -(w["last_registered"] or 0)))
        _send_json_response(handler, {"workers": workers})

    def _outer_optimizer_info(self) -> dict:
        """The live outer optimizer's class + SGD hyperparameters, so a backend
        that runs the outer step itself can reproduce it exactly."""
        opt = self.outer_optimizer
        pg = opt.param_groups[0]
        return {
            "name": type(opt).__name__,
            "lr": pg.get("lr"),
            "momentum": pg.get("momentum", 0.0),
            "nesterov": pg.get("nesterov", False),
            "dampening": pg.get("dampening", 0.0),
            "weight_decay": pg.get("weight_decay", 0.0),
        }

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
            # The checkpoint dir the server loaded its master weights from. A
            # non-HTTP backend (e.g. shared-memory, on the same host) can seed
            # its region from this same init reference instead of receiving the
            # weights over the wire (issue #154 "join returns a reference").
            "model_checkpoint_dir": self._loaded_checkpoint_dir,
            # The outer-optimizer config, so a backend that runs the outer step
            # itself (shared-memory) reproduces the server's exactly instead of
            # silently defaulting (issue #154). SGD hyperparameters from the live
            # optimizer's param group + class name.
            "outer_optimizer": self._outer_optimizer_info(),
            "mode": "async" if self.async_mode else "sync",
            "async_mode": self.async_mode,
            "num_workers": self.num_workers,
            # Shared-memory rendezvous (Flavor 2, issue #154). When the server
            # is the shared-memory aggregator it owns the region; a follower
            # reads the dir + group size from here rather than re-deriving them.
            # ``shm_group_size`` is the STABLE configured worker count (the
            # launch ``-n``), not the mutable ``num_workers``, so a follower
            # sizes its region correctly no matter who has registered yet. Both
            # are null for non-shared-memory backends.
            "shm_group_dir": self._shm_group_dir,
            "shm_group_size": (
                self._configured_num_workers
                if self.backend == "shared_memory"
                else None
            ),
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
                # Wire precision (issue #130). Four server-authoritative
                # knobs covering each direction × dtype-vs-SR. Workers
                # adopt these verbatim — the group must agree on the
                # wire format for sync barriers to be coherent.
                "upload_dtype": self.upload_dtype,
                "upload_sr": self.upload_sr,
                "download_dtype": self.download_dtype,
                "download_sr": self.download_sr,
                # Bulk-tensor wire codec (issue #154). Authoritative for both
                # legs: the upload also stamps it per-frame, but the download
                # carries no header, so the worker must adopt this. "pickle"
                # (default) keeps an older worker interoperable.
                "wire_format": self.wire_format,
                # Sync backend the group must use (issue #154). The worker
                # validates its own launched backend against this and fails loud
                # on disagreement — it cannot *adopt* it (the backend fixes the
                # launch topology), only agree. Absent ⇒ an older server ⇒ the
                # worker skips the check.
                "backend": self.backend,
                # Deprecated alias kept so pre-#130 workers parsing
                # ``bf16_comm`` from ``expected_client_settings`` still
                # negotiate a compatible upload format. True iff the
                # current ``upload_dtype`` is bf16 — semantically
                # identical to the pre-refactor flag.
                "bf16_comm": self.bf16_comm,
                "num_fragments_min": 1,
                "num_fragments_default": self.num_fragments,
                # Exposed so the worker can validate its (client-local)
                # heartbeat send cadence against the server's death timeout.
                "heartbeat_timeout": self.heartbeat_timeout,
            },
            # Bulk transport negotiation (issue #154). "grpc" when the gRPC
            # listener is up, else "http" (the default + universal fallback).
            # ``grpc_endpoint`` is the host:port a worker dials for the gRPC bulk
            # legs (None for http). An older server omits both ⇒ the worker
            # defaults to http, so negotiation is purely for peer back-compat.
            "transport": "grpc" if self._grpc_enabled else "http",
            "grpc_endpoint": self.get_grpc_url(_request_host(handler)),
        }
        _send_json_response(handler, response)

    def _resolve_model_def_dir(self) -> Optional[str]:
        """Pick the directory to serve the model-definition bundle from.

        The definition (config.json + custom modeling/configuration ``.py`` +
        tokenizer) belongs to the model's home directory, not to each rotated
        checkpoint: ``save_state`` writes only weights + ``server_state.pt``
        into ``checkpoint-N``, so a server restarted off such a checkpoint has
        no definition there. Prefer the dir we loaded from when it actually
        carries the definition (e.g. a self-contained ``--from-checkpoint``
        model dir), else fall back to ``output_dir`` — the model's home, where
        the original config/code/tokenizer live alongside the ``checkpoints/``
        subtree. Returns None when neither has it, so ``/model_def`` can fail
        loudly rather than ship a worker an empty bundle (issue #53 / #103).
        """
        for cand in (self._loaded_checkpoint_dir, self.output_dir):
            if cand and os.path.isdir(cand) and enumerate_model_def_files(cand):
                return os.path.realpath(cand)
        return None

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
        if not self._model_def_dir or not os.path.isdir(self._model_def_dir):
            _send_json_response(
                handler,
                {
                    "error": (
                        "server has no model-definition directory to serve: "
                        "neither the loaded checkpoint nor output_dir contains "
                        "config.json / modeling code / tokenizer. Start the "
                        "server from a self-contained model dir, or place the "
                        "definition at the output_dir top level."
                    )
                },
                503,
            )
            return
        try:
            with self._model_def_lock:
                if self._model_def_bundle is None:
                    self._model_def_bundle = pack_model_def(self._model_def_dir)
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
            elif action == "command":
                self._handle_control_command(handler, data)
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

    def _handle_control_command(self, handler, data):
        """Queue a trainer-control command for one or all workers.

        Body: ``{"command": "save_checkpoint"|"save_and_stop"|"abort",
        "worker_id": "..."}``. ``worker_id`` is optional — omitted (or
        ``"*"``) queues the command for every currently-registered worker.

        This is the relay that lets the CLI (and the webui) drive the
        per-worker controls without reaching each worker's trainer-control
        HTTP endpoint directly: the command rides the worker's next
        heartbeat response, and the DiLoCo callback applies it to the
        trainer loop. Latency is bounded by ``heartbeat_interval``.
        """
        command = data.get("command")
        if command not in _RELAY_COMMANDS:
            _send_json_response(
                handler,
                {
                    "error": (
                        f"unknown command {command!r}; expected one of "
                        f"{sorted(_RELAY_COMMANDS)}"
                    )
                },
                400,
            )
            return
        worker_id = data.get("worker_id")
        with self._workers_lock:
            if worker_id in (None, "", "*"):
                targets = list(self._workers.keys())
            elif worker_id in self._workers:
                targets = [worker_id]
            else:
                _send_json_response(
                    handler, {"error": f"Worker {worker_id} not found"}, 404
                )
                return
            for wid in targets:
                self._workers[wid].pending_command = command
        logger.info(
            "Queued control command %r for %d worker(s): %s",
            command,
            len(targets),
            targets,
        )
        _send_json_response(
            handler,
            {"status": "ok", "command": command, "workers": targets},
        )

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
        """Gracefully shut down the server (webui "Shutdown server" button /
        ``forgather diloco shutdown``).

        Runs the coordinated drain: relay ``save_and_stop`` to all workers,
        keep serving while they checkpoint + deregister (so no worker deadlocks
        on the sync barrier), then save server state and stop. Done in a
        background thread so the HTTP response goes out first and the server
        keeps serving the workers' final syncs during the drain.
        """
        logger.info("Shutdown requested via control endpoint")
        _send_json_response(handler, {"status": "ok", "message": "Shutting down"})
        threading.Thread(target=self.graceful_shutdown, daemon=True).start()

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

    # Three "bulk" endpoints: large tensor transfers. When the cleartext
    # bulk plane is enabled these are removed from the control port and
    # only served on the bulk listener. Centralized here so the routing
    # tables and the off-port 404 hint stay in sync.
    _BULK_PATHS = frozenset(
        {"/submit_pseudograd", "/submit_fragment_pseudograd", "/global_params"}
    )

    #: Filename of the aggregate-stats JSONL log, under ``<output_dir>/logs/``.
    _STATS_LOG_FILENAME = "diloco_server_stats.jsonl"

    #: Bind hosts that are not routable as a connect target. When the
    #: server binds one of these we can't put it in the advertised bulk
    #: URL — the worker would dial a wildcard / its own loopback.
    _WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", ""})

    def get_bulk_url(self, request_host: Optional[str] = None) -> Optional[str]:
        """Public URL workers should use for the bulk endpoints.

        ``None`` when the cleartext bulk plane is disabled (bulk endpoints
        are served on the control port) or before the listener has bound.
        The ``/register`` response includes this string under the
        ``X-Forgather-Bulk-Url`` header so workers learn the
        server-assigned ephemeral port without an extra round-trip.

        When the server bound a wildcard address (``0.0.0.0`` / ``::``)
        it has no reliable view of its own routable address, so
        advertising ``self.host`` verbatim would hand remote workers an
        unroutable ``http://0.0.0.0:<port>``. In that case we prefer
        ``request_host`` — the host the registering worker actually used
        to reach the control port, which is routable from that worker by
        construction — and fall back to loopback only as a last resort.
        """
        if not self._bulk_enabled or self.bulk_port is None:
            return None
        # The bulk plane is always cleartext (it exists to bypass TLS).
        host = self.host
        if host in self._WILDCARD_HOSTS or host is None:
            host = request_host or "127.0.0.1"
        return f"http://{host}:{self.bulk_port}"

    def get_grpc_url(self, request_host: Optional[str] = None) -> Optional[str]:
        """The ``host:port`` of the gRPC bulk listener, for /info advertisement.

        ``None`` when gRPC is disabled or before the listener has bound. Like
        ``get_bulk_url``, a wildcard bind prefers the worker's ``request_host``
        (routable from that worker by construction) over an unroutable
        ``0.0.0.0``. No scheme — gRPC channel security is set by the worker from
        the negotiated TLS posture, not encoded in the address.
        """
        if not self._grpc_enabled or self.grpc_port is None:
            return None
        host = self.host
        if host in self._WILDCARD_HOSTS or host is None:
            host = request_host or "127.0.0.1"
        return f"{host}:{self.grpc_port}"

    def _create_handler(self, role: str = "control"):
        """Create a request handler bound to this server.

        ``role`` selects which set of endpoints + which auth token the
        handler enforces:

        * ``"control"`` — full route table. When the cleartext bulk
          plane is enabled the three bulk paths are intentionally absent
          (a 404 with an ``X-Forgather-Bulk-Url`` hint is returned).
        * ``"bulk"`` — only the three bulk paths plus ``/health``,
          always unauthenticated (the cleartext bulk plane never checks
          the bearer; see ``_expected_token``).
        """
        server_ref = self

        class DiLoCoRequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Route HTTP logs through our logger
                logger.debug(format, *args)

            def _expected_token(self) -> Optional[str]:
                """Token this listener checks against. The cleartext bulk
                listener is always unauthenticated — sending the bearer
                over a sniffable socket would leak the control-plane
                credential to anyone on the wire, so we never check it
                there."""
                if role == "bulk":
                    return None
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
                    or not server_ref._bulk_enabled
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
                    elif path == "/known_workers":
                        server_ref._handle_known_workers(self)
                    elif path == "/info":
                        server_ref._handle_info(self)
                    elif path == "/model_def":
                        server_ref._handle_model_def(self)
                    elif path == "/work/queues":
                        server_ref._handle_get_queues(self)
                    elif path == "/work/queue":
                        server_ref._handle_get_queue(self)
                    elif path == "/stats_history":
                        server_ref._handle_stats_history(self)
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

        # Snapshot the model + outer optimizer + round together under the sync
        # condition so a concurrent outer step (the shared-memory aggregation
        # thread, or an HTTP sync barrier) can't tear the master across the read
        # — a control-save racing a step would otherwise persist some params
        # from before the step and some from after. The outer step mutates these
        # three under the same condition, so this captures a consistent set; the
        # disk I/O below runs off the lock on these immutable snapshots.
        with self._sync_cond:
            global_params = self.get_global_params()
            outer_opt_state = self.outer_optimizer.state_dict()
            sync_round = self._sync_round
            total_submissions = self._total_submissions

        if path is not None:
            checkpoint_path = path
        else:
            checkpoint_path = next_checkpoint_path(self.output_dir, sync_round)

        os.makedirs(checkpoint_path, exist_ok=True)

        # Save model weights as HF-compatible sharded checkpoint
        save_model_checkpoint(
            checkpoint_path, global_params, safetensors=self.safetensors
        )

        # Work-unit dispatch state IS persisted: the server is the authority
        # for which rows each worker has consumed, so a restart must not
        # re-issue already-trained units within an epoch (#105). Snapshot the
        # per-(dataset_id, shuffle_seed) issued/completed bitmaps under the
        # lock (concurrent issuance must not mutate them mid-serialization),
        # plus the per-dataset length snapshot used for the registration
        # integrity check. Disk form keys queues by a "dataset_id|seed"
        # string (tuple keys don't round-trip cleanly) and stores bytes, not
        # bytearray. A dataset change hashes to a different dataset_id → a
        # fresh key → any stale queue is simply never matched and sits inert
        # (an operator wanting a hard reset restarts from the model weights
        # and purges the rest).
        with self._work_queues_lock:
            work_queues = {
                f"{ds}|{seed}": {
                    "total_units": q.total_units,
                    "issued": bytes(q.issued),
                    "completed": bytes(q.completed),
                    "hint_length": q.hint_length,
                    "issued_count": q.issued_count,
                    "completed_count": q.completed_count,
                    "by_worker": {w: dict(c) for w, c in q.by_worker.items()},
                    "dataset_path": q.dataset_path,
                    "dataset_name": q.dataset_name,
                    "dataset_split": q.dataset_split,
                    "dataset_revision": q.dataset_revision,
                    "dataset_data_files": (
                        list(q.dataset_data_files) if q.dataset_data_files else None
                    ),
                }
                for (ds, seed), q in self._work_queues.items()
            }
            dataset_lengths = dict(self._dataset_lengths)

        # Snapshot the known-worker roster under the lock so a concurrent
        # registration can't mutate the dict mid-serialization (issue #103
        # follow-up). Persisting it here lets a restarted server offer the
        # previous workers' names for checkpoint-resuming relaunch.
        with self._workers_lock:
            known_workers = {k: dict(v) for k, v in self._known_workers.items()}

        server_state = {
            "outer_optimizer": outer_opt_state,
            "sync_round": sync_round,
            "num_workers": self.num_workers,
            "param_names": self._param_names,
            "async_mode": self.async_mode,
            "total_submissions": total_submissions,
            "known_workers": known_workers,
            "work_queues": work_queues,
            "dataset_lengths": dataset_lengths,
            # Unified-stats lifetime counters + EMA state, so total
            # tokens/flos/steps and the smoothed loss survive a restart, plus
            # the run's log subdir so a resume continues logging into the same
            # runs/<...> dir (JSONL appends, TensorBoard resumes via purge_step)
            # rather than fragmenting across a new dir each restart.
            "stats": self._stats.state_dict(),
            "stats_run_subdir": self._stats_run_subdir,
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
            # Distinguish "directory doesn't exist" from "directory exists but
            # holds no checkpoints" — the former was previously reported as the
            # latter, which is misleading (especially with a relative path that
            # resolved against an unexpected CWD). Show the absolute path so the
            # caller can see exactly where we looked.
            if not os.path.isdir(self.output_dir):
                raise FileNotFoundError(
                    f"Model directory does not exist: {self.output_dir} "
                    f"(resolved to {os.path.abspath(self.output_dir)}; "
                    f"cwd={os.getcwd()})"
                )
            checkpoint_path = find_latest_checkpoint(self.output_dir)
            if not checkpoint_path:
                raise ValueError(
                    f"No checkpoints found in {self.output_dir} "
                    f"({os.path.abspath(self.output_dir)}). "
                    f"Please provide a valid model directory."
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

        # Remember where we loaded from, then resolve where the model
        # *definition* bundle is served from — the loaded checkpoint if it
        # carries the definition, else output_dir (the model's home). A
        # rotated server checkpoint holds only weights + server_state.pt, so
        # after a restart-from-checkpoint the definition is found at
        # output_dir, not in checkpoint-N (issue #103). Fold the bundle's
        # content hash into the advertised model_hash so a worker's cached
        # bundle stamp invalidates on *any* definition change (a config tweak
        # or an edited modeling .py), not only a parameter-shape change.
        self._loaded_checkpoint_dir = os.path.realpath(checkpoint_path)
        self._model_def_dir = self._resolve_model_def_dir()
        if self._model_def_dir is None:
            logger.warning(
                "No model-definition files (config.json / modeling .py / "
                "tokenizer) found in the loaded checkpoint (%s) or output_dir "
                "(%s). /model_def will return 503 and DiLoCo workers cannot "
                "stage the model. Start from a self-contained model dir or "
                "place the definition at the output_dir top level.",
                self._loaded_checkpoint_dir,
                self.output_dir,
            )
        else:
            try:
                bundle_hash = compute_bundle_hash(self._model_def_dir)
                self._model_hash = hashlib.sha256(
                    (self._model_hash + ":" + bundle_hash).encode("utf-8")
                ).hexdigest()
            except OSError as exc:
                # A hash over the definition files is best-effort: if the dir
                # is unreadable we keep the parameter-only hash rather than
                # fail the load. /model_def will surface the real error if hit.
                logger.warning("Could not hash model-definition bundle: %s", exc)

        # Load server state if present
        if server_state is not None:
            self.outer_optimizer.load_state_dict(server_state["outer_optimizer"])
            self._sync_round = server_state["sync_round"]
            self._total_submissions = server_state.get("total_submissions", 0)
            # Restore the known-worker roster so the webui can offer the
            # previous run's workers for checkpoint-resuming relaunch after
            # a server restart (issue #103 follow-up). Absent on pre-feature
            # checkpoints, in which case the roster simply starts empty and
            # repopulates as workers register.
            self._known_workers = server_state.get("known_workers", {}) or {}

            # Restore unified-stats lifetime state (counters, per-worker
            # last-seen baselines, loss EMA). Absent on pre-feature
            # checkpoints → start fresh. Reuse the prior run's log subdir for
            # continuity (JSONL append + TensorBoard purge_step past the
            # restored step); the throttle starts below total_steps so the
            # first post-restart record writes.
            self._stats.load_state_dict(server_state.get("stats") or {})
            self._resume_run_subdir = server_state.get("stats_run_subdir")
            if self._resume_run_subdir:
                self._tb_purge_step = self._stats.total_steps
            self._stats_log_step = -1
            # Seed the last-logged eval step from the restored aggregator so a
            # resume doesn't re-log the same eval point that was already
            # recorded before the checkpoint.
            self._last_logged_eval_step = self._stats.snapshot().get("eval_step")

            # Restore work-unit dispatch state (#105): the per-(dataset_id,
            # shuffle_seed) issued/completed bitmaps and the per-dataset
            # length snapshot. A worker re-registering its dataset hits the
            # restored queue (``_handle_register_dataset`` reuses any existing
            # queue for the key), so already-issued units stay issued and are
            # not re-handed-out — the server is the authority for consumed
            # rows. Absent on pre-feature checkpoints → start empty (today's
            # rebuild-on-reregister behavior, which re-issues from unit 0).
            self._dataset_lengths = server_state.get("dataset_lengths", {}) or {}
            restored = server_state.get("work_queues") or {}
            self._work_queues = {}
            for qkey, d in restored.items():
                # Skip-and-warn on ANY bad entry (malformed key, missing/typed
                # field, or a bitmap whose length disagrees with total_units)
                # rather than letting one corrupt/partial entry abort the whole
                # load and brick a restart. A length mismatch would otherwise
                # surface much later as an IndexError during issuance.
                try:
                    ds, seed_str = qkey.rsplit("|", 1)
                    seed = int(seed_str)
                    total_units = int(d["total_units"])
                    nbytes = (total_units + 7) // 8
                    issued = bytearray(d["issued"])
                    completed = bytearray(d.get("completed") or bytes(nbytes))
                    if len(issued) != nbytes or len(completed) != nbytes:
                        raise ValueError(
                            f"bitmap length {len(issued)}/{len(completed)} "
                            f"!= expected {nbytes} for total_units={total_units}"
                        )
                    queue = WorkQueue(
                        total_units=total_units,
                        issued=issued,
                        completed=completed,
                        hint_length=int(d["hint_length"]),
                        issued_count=int(d.get("issued_count", 0)),
                        completed_count=int(d.get("completed_count", 0)),
                        by_worker=d.get("by_worker") or {},
                        dataset_path=d.get("dataset_path"),
                        dataset_name=d.get("dataset_name"),
                        dataset_split=d.get("dataset_split"),
                        dataset_revision=d.get("dataset_revision"),
                        dataset_data_files=d.get("dataset_data_files"),
                    )
                except (ValueError, AttributeError, KeyError, TypeError) as exc:
                    logger.warning(
                        "Skipping malformed work-queue entry %r in "
                        "server_state.pt: %s",
                        qkey,
                        exc,
                    )
                    continue
                self._work_queues[(ds, seed)] = queue

            if self._work_queues:
                issued_total = sum(q.issued_count for q in self._work_queues.values())
                logger.info(
                    "Restored %d work queue(s) from checkpoint (%d unit(s) "
                    "already issued); re-registering workers resume mid-epoch.",
                    len(self._work_queues),
                    issued_total,
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
        """Spawn the cleartext bulk listener when enabled.

        Runs in its own daemon thread with its own handler class
        (``role="bulk"``). Always cleartext + unauthenticated, bound to an
        OS-assigned ephemeral port — the listener exists to bypass TLS for
        throughput, so encryption/auth on it would defeat the purpose. The
        bound port is read back from the socket and stored on
        ``self.bulk_port`` so ``get_bulk_url`` can advertise it to workers
        over the control plane.
        """
        if not self._bulk_enabled:
            return
        bulk_handler_class = self._create_handler(role="bulk")
        # Bind port 0: the OS picks a free ephemeral port, guaranteeing it
        # isn't already in use. Workers receive the real port via the
        # ``X-Forgather-Bulk-Url`` header on the (TLS-protected) control
        # plane, so it never needs to be a stable, operator-chosen number.
        self._bulk_server = ThreadingHTTPServer((self.host, 0), bulk_handler_class)
        self._bulk_server.daemon_threads = True
        self.bulk_port = self._bulk_server.socket.getsockname()[1]
        self._bulk_server_thread = threading.Thread(
            target=self._bulk_server.serve_forever, daemon=True
        )
        self._bulk_server_thread.start()
        logger.info(
            f"DiLoCo bulk listener on http://{self.host}:{self.bulk_port} "
            f"(cleartext, no-auth — TLS bypassed for throughput)"
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
            self.bulk_port = None

    def _grpc_security(self):
        """Resolve ``(server_credentials, authenticate)`` for the gRPC listener,
        following the control plane's TLS posture (issue #154).

        - **TLS** iff the control plane has TLS (``ssl_context`` set) and the
          cert/key paths were plumbed in: build ``ssl_server_credentials`` from
          the same cert/key (encryption + server authentication). Else cleartext.
        - **Auth gate** iff TLS is on *and* a bearer token is configured: an
          ``authenticate`` callable that requires a matching bearer over the
          secure channel (``authenticate_grpc_context``), aborting UNAUTHENTICATED
          otherwise. No token ⇒ open (like the HTTP no-auth path); cleartext ⇒
          no bearer gate (a bearer over a sniffable socket is theater, matching
          the cleartext bulk listener). gRPC authenticates the worker by bearer,
          not mTLS — see ``authenticate_grpc_context``.
        """
        creds = None
        if self.ssl_context is not None and self.tls_cert_file and self.tls_key_file:
            from forgather.ml.diloco import grpc_bulk

            creds = grpc_bulk.make_server_credentials(
                self.tls_cert_file, self.tls_key_file, self.tls_ca_file
            )

        authenticate = None
        if creds is not None and self.auth_token:
            import grpc

            from forgather.ml.diloco.auth import authenticate_grpc_context

            token = self.auth_token

            def authenticate(context):
                if not authenticate_grpc_context(context, token):
                    context.abort(
                        grpc.StatusCode.UNAUTHENTICATED,
                        "missing or invalid bearer token",
                    )

        return creds, authenticate

    def _start_grpc_listener(self) -> None:
        """Spawn the gRPC bulk listener when enabled (issue #154).

        The bound ephemeral port is read back into ``self.grpc_port`` and
        advertised via /info. ``grpc`` is imported lazily here so a server with
        gRPC off pays no import cost. Security posture comes from
        ``_grpc_security`` (TLS following the control-plane posture, with an
        optional bearer gate; or cleartext/open).
        """
        if not self._grpc_enabled:
            return
        from forgather.ml.diloco import grpc_bulk

        creds, authenticate = self._grpc_security()
        self._grpc_server, self.grpc_port = grpc_bulk.make_grpc_server(
            self,
            self.host,
            authenticate=authenticate,
            server_credentials=creds,
            max_workers=max(self.num_workers + 4, 8),
        )
        self._grpc_server.start()
        posture = "TLS" if creds is not None else "cleartext"
        auth = "bearer" if authenticate is not None else "open"
        logger.info(
            f"DiLoCo gRPC bulk listener on {self.host}:{self.grpc_port} "
            f"({posture}, {auth})"
        )

    def _stop_grpc_listener(self) -> None:
        """Counterpart to ``_start_grpc_listener``."""
        if self._grpc_server is not None:
            self._grpc_server.stop(grace=5).wait(timeout=6)
            self._grpc_server = None
            self.grpc_port = None

    def run(self):
        """Run the server (blocking). Call this from the main process.

        Serves on a background thread and blocks the main thread until a
        shutdown is requested (SIGTERM — how the forgather_server scheduler
        stops a server job — SIGINT/Ctrl-C, or the ``/control/shutdown``
        endpoint). All three converge on :meth:`graceful_shutdown`, which
        relays ``save_and_stop`` to the workers and lets them drain (while the
        background thread keeps serving so they don't deadlock on the sync
        barrier) before saving state and stopping.
        """
        if self._running:
            raise RuntimeError("Run cannot be called, if already running.")
        # Serve in a background thread (start() also logs the banner + brings
        # up the health monitor and bulk listener). The main thread then
        # blocks until shutdown so the coordinated drain can keep serving.
        self.start()

        import signal as _signal

        def _request_shutdown(signum, frame):
            # Don't tear down from the handler — just wake the main thread,
            # which runs the coordinated shutdown while the server keeps
            # serving the workers' final syncs.
            self._shutdown_event.set()

        prev_sigterm = prev_sigint = None
        try:
            prev_sigterm = _signal.signal(_signal.SIGTERM, _request_shutdown)
            prev_sigint = _signal.signal(_signal.SIGINT, _request_shutdown)
        except (ValueError, OSError):
            pass  # not on the main thread — signals stay default

        try:
            self._shutdown_event.wait()
        finally:
            if prev_sigterm is not None:
                _signal.signal(_signal.SIGTERM, prev_sigterm)
            if prev_sigint is not None:
                _signal.signal(_signal.SIGINT, prev_sigint)
            # Coordinated drain + save + stop. No-ops if a /control/shutdown
            # already ran it (re-entrancy guard); otherwise this is the run.
            self.graceful_shutdown()

    def _start_shm_aggregator(self):
        """Create the shared-memory region and start the server-side aggregation
        loop (Flavor 2, issue #154).

        The server seeds the region from its current master — the loaded weights
        on a fresh start, or the restored *trained* master + outer momentum on a
        resume — and is the sole writer of the master / generation. Workers
        attach as followers and contribute pseudo-gradients into the region's
        accumulator; this loop runs the outer step and publishes.
        """
        from .shared_memory_aggregator import SharedMemoryAggregator

        self._shm_stop.clear()
        self._shm_agg = SharedMemoryAggregator(self._shm_group_dir)
        self._shm_agg.start(self.get_global_params(), self._configured_num_workers)
        self._shm_agg_thread = threading.Thread(
            target=self._shm_aggregation_loop, name="shm-aggregator", daemon=True
        )
        self._shm_agg_thread.start()
        logger.info(
            "Shared-memory aggregator started: region=%s group_size=%d",
            self._shm_group_dir,
            self._configured_num_workers,
        )

    def _stop_shm_aggregator(self):
        """Stop the aggregation loop and release/clean up the region.
        Idempotent."""
        if self._shm_agg is None:
            return
        self._shm_stop.set()
        wedged = False
        if self._shm_agg_thread is not None:
            self._shm_agg_thread.join(timeout=10)
            if self._shm_agg_thread.is_alive():
                # The loop didn't exit (e.g. wedged holding the region flock
                # while blocked on _sync_cond). Calling stop() would re-take the
                # flock and block shutdown indefinitely; skip the lock-taking
                # cleanup and let the OS free the lease + region on process exit.
                wedged = True
                logger.error(
                    "shm aggregator thread did not exit within 10s; skipping "
                    "region cleanup to avoid blocking shutdown (the OS reclaims "
                    "the lease + region on process exit)."
                )
            self._shm_agg_thread = None
        try:
            if not wedged:
                self._shm_agg.stop()
        finally:
            self._shm_agg = None

    def _abort_shm_group(self):
        """The aggregation loop died: mark the region dead so parked followers
        fail loud immediately (rather than each blocking out its lock_timeout on
        a generation that will never advance), and trip the server's shutdown so
        it doesn't keep serving a group that can no longer make progress."""
        try:
            if self._shm_agg is not None:
                self._shm_agg.abort()
        except Exception:
            logger.exception("shm aggregator: abort failed")
        self._shutdown_event.set()

    def _shm_aggregation_loop(self):
        """Poll the region; each time every follower has contributed, apply one
        outer step and publish, then handle audit + periodic save off the region
        lock. Runs until stop() signals teardown.

        The barrier is dynamic (see ``SharedMemoryAggregator.wait_for_round``):
        a follower that ``leave()``s — the clean shutdown / drain path — shrinks
        the live count so the round releases on the survivors. The remaining gap
        is a follower that *crashes* without leave()ing: its attach slot lingers,
        the live count never drops, and this loop waits (shared-memory is
        ``fault_tolerant=False`` — a dead co-located process kills the group).
        TODO(shm-fault-tolerance): tie the health monitor's worker-death signal
        to the region so a crashed follower's slot is reclaimed and the round
        fails loud instead of stalling.
        """
        while not self._shm_stop.is_set() and self._running:
            try:
                ready = self._shm_agg.wait_for_round(timeout=0.5)
            except Exception:
                logger.exception("shm aggregator: error waiting for round")
                self._abort_shm_group()
                return
            if not ready:
                continue  # no full round yet (workers still training)
            try:
                self._shm_agg.aggregate(self._shm_outer_step)
            except Exception:
                logger.exception("shm aggregator: outer step failed; aborting group")
                self._abort_shm_group()
                return
            logger.info(
                "Shared-memory outer step complete. Sync round: %d", self._sync_round
            )
            self._audit(
                "outer_step", sync_round=self._sync_round, backend="shared_memory"
            )
            if (
                self.save_every_n_rounds > 0
                and self._sync_round % self.save_every_n_rounds == 0
            ):
                try:
                    self.save_state()
                except Exception:
                    logger.exception("shm aggregator: periodic save_state failed")

    def _shm_outer_step(self, avg_grads):
        """Apply one outer optimizer step over the averaged pseudo-gradient and
        return the new master (the publish payload).

        Mirrors :meth:`_apply_outer_optimizer`'s core: the HTTP path averages
        ``_pending_pseudograds`` itself, whereas here the region already summed
        the followers' contributions and the aggregator divided by the
        contributor count, so ``avg_grads`` is the per-name mean. The model,
        optimizer, and round are mutated together under ``_sync_cond``;
        ``save_state`` snapshots the same three under that condition, so a
        concurrent control-save never captures a torn master. The optimizer
        state (momentum) lives in this process and is checkpointed by
        ``save_state`` — which is what makes a shared-memory run resume cleanly.
        """
        with self._sync_cond:
            for i, name in enumerate(self._param_names):
                self._param_list[i].grad = avg_grads[name].float()
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            self._sync_round += 1
            self._dirty = True
            return self.get_global_params()

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
        self._start_grpc_listener()
        if self.backend == "shared_memory":
            self._start_shm_aggregator()

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
        # Stop the shared-memory aggregation loop first (before save_state
        # below) so the master isn't being concurrently stepped while we
        # snapshot it, and the region is released/cleaned up.
        if self.backend == "shared_memory":
            self._stop_shm_aggregator()
        # Flush state before teardown so a graceful stop doesn't lose the
        # rounds / issued work-units accumulated since the last autosave
        # (#105). save_state no-ops when clean; never let a save failure
        # block shutdown. Mirrors the run() Ctrl-C path.
        if self.save_every_n_rounds > 0:
            try:
                logger.info("Saving server state before shutdown...")
                self.save_state()
            except Exception as exc:
                logger.error("Failed to save server state on stop: %s", exc)
        self._stop_health_monitor()
        self._stop_bulk_listener()
        self._stop_grpc_listener()
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass
        if self._server:
            self._server.shutdown()
            self._running = False
            if self._server_thread:
                self._server_thread.join(timeout=5)
            self._server.server_close()
            self._close_audit()
            logger.info("Server stopped")

    def _relay_command_all(self, command: str) -> List[str]:
        """Queue a trainer-control command (e.g. ``save_and_stop``) for every
        registered worker; it rides each worker's next heartbeat. Returns the
        targeted worker ids."""
        with self._workers_lock:
            targets = list(self._workers.keys())
            for wid in targets:
                self._workers[wid].pending_command = command
        return targets

    def graceful_shutdown(self, timeout: float = 300.0, poll: float = 0.5) -> None:
        """Coordinated cluster shutdown.

        Relays ``save_and_stop`` to every worker (delivered on their next
        heartbeat), then keeps the server **serving** while they finish their
        current step — including any in-flight sync round — checkpoint, and
        deregister. Serving during the drain is essential: a worker parked on
        the sync barrier would deadlock if the server stopped accepting
        submissions. As each worker leaves, the barrier's expected set shrinks
        (the normal worker-death path), so the remaining workers release. Once
        all workers have drained (or ``timeout`` elapses), saves server state
        and stops.

        Idempotent and safe to call from any thread: the signal path (SIGTERM/
        SIGINT in :meth:`run`) and the ``/control/shutdown`` endpoint both
        funnel here, and the first caller wins.
        """
        with self._shutdown_lock:
            first = not self._shutting_down
            self._shutting_down = True
        if not first:
            # Another caller (e.g. a /control/shutdown daemon) is already
            # draining. Block until it finishes rather than returning — a
            # signal-triggered caller in run() would otherwise let the process
            # exit and kill the in-flight drain before save_state completes.
            self._shutdown_done.wait(timeout + 30.0)
            return
        try:
            if self._running:
                targets = self._relay_command_all("save_and_stop")
                logger.info(
                    "Graceful shutdown: relayed save_and_stop to %d worker(s): %s",
                    len(targets),
                    targets,
                )
                deadline = time.time() + timeout
                while time.time() < deadline:
                    with self._workers_lock:
                        if not self._workers:
                            break
                    time.sleep(poll)
                with self._workers_lock:
                    remaining = list(self._workers.keys())
                if remaining:
                    logger.warning(
                        "Graceful shutdown: %d worker(s) still registered after "
                        "%.0fs (%s); saving and stopping anyway.",
                        len(remaining),
                        timeout,
                        remaining,
                    )
                else:
                    logger.info("Graceful shutdown: all workers drained.")
        finally:
            try:
                if self._running:
                    self.stop()
            finally:
                self._shutdown_done.set()
                self._shutdown_event.set()

    @property
    def address(self) -> str:
        """Return the server address as host:port."""
        return f"{self.host}:{self.port}"
