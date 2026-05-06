"""Cluster identity / membership endpoints.

The peer-pull membership task in ``cluster_membership.py`` calls
``GET /api/cluster/members`` on every known peer once per tick. The
auth middleware allows that call from a known-peer source IP without
the bearer token (see ``auth._PEER_ALLOWED_PATHS``).

A browser session calls these endpoints to render the Nodes view.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import cluster, cluster_jobs
from .gpus import GpuInfoModel, GpuPolicyModel, SetGpuPolicyRequest, _to_model

log = logging.getLogger("forgather_server.routes.cluster")

router = APIRouter(prefix="/cluster", tags=["cluster"])

# Timeout for master->peer GPU snapshot fetches. Tighter than the
# membership pull timeout because the Nodes view should feel snappy;
# a slow peer simply shows up empty in this round and refreshes next
# request.
PEER_GPU_TIMEOUT_SECONDS = 2.0


class MemberModel(BaseModel):
    node_id: str
    hostname: str
    address: str
    port: int
    cluster_name: str
    forgather_version: str
    first_seen: float
    last_seen: float
    reachable: bool
    last_source: str
    # Probe payload — versions, interfaces, CPU summary. Loose dict
    # rather than a strict schema because adding a new probe field
    # should not require synchronized backend + frontend rollout
    # across the cluster (a node still on Phase 1 sends None, a node
    # on a later Phase 2 revision may add a key the frontend hasn't
    # seen yet).
    probe: Optional[Dict[str, Any]] = None


class SelfModel(BaseModel):
    node_id: str
    hostname: str
    cluster_name: str
    port: int
    forgather_version: str
    started_at: float
    is_master: bool


class MembersResponse(BaseModel):
    cluster_name: Optional[str] = None
    self_node_id: Optional[str] = None
    master_node_id: Optional[str] = None
    members: List[MemberModel] = []
    # Server-side timestamp lets clients sanity-check clock skew when
    # diagnosing stale-membership reports.
    server_time: float


class MasterResponse(BaseModel):
    master_node_id: Optional[str] = None
    is_self_master: bool = False
    cluster_active: bool = False


def _to_member_model(m: cluster.MemberInfo) -> MemberModel:
    return MemberModel(
        node_id=m.node_id,
        hostname=m.hostname,
        address=m.address,
        port=m.port,
        cluster_name=m.cluster_name,
        forgather_version=m.forgather_version,
        first_seen=m.first_seen,
        last_seen=m.last_seen,
        reachable=m.reachable,
        last_source=m.last_source,
        probe=m.probe,
    )


@router.get("/members", response_model=MembersResponse)
def list_members():
    ident = cluster.self_identity()
    return MembersResponse(
        cluster_name=ident.cluster_name if ident else None,
        self_node_id=ident.node_id if ident else None,
        master_node_id=cluster.master_node_id(),
        members=[_to_member_model(m) for m in cluster.members()],
        server_time=time.time(),
    )


@router.get("/self", response_model=Optional[SelfModel])
def get_self():
    """Return this node's identity, or ``None`` when cluster is inactive.

    Returning ``None`` (rather than 404) keeps the webui's flow simple:
    "ask once, render Nodes view if non-null." A 404 here would be
    indistinguishable from a routing bug.
    """
    ident = cluster.self_identity()
    if ident is None:
        return None
    return SelfModel(
        node_id=ident.node_id,
        hostname=ident.hostname,
        cluster_name=ident.cluster_name,
        port=ident.port,
        forgather_version=ident.forgather_version,
        started_at=ident.started_at,
        is_master=cluster.is_self_master(),
    )


@router.get("/master", response_model=MasterResponse)
def get_master():
    return MasterResponse(
        master_node_id=cluster.master_node_id(),
        is_self_master=cluster.is_self_master(),
        cluster_active=cluster.is_active(),
    )


# ---------------------------------------------------------------------------
# GPU aggregation
# ---------------------------------------------------------------------------


class ClusterGpusEntry(BaseModel):
    """Per-node bucket in the cluster GPU response.

    ``error`` holds a short string when a fetch fails, so the webui
    can show "node B unreachable" without confusing it with "node B
    has zero GPUs". ``stale`` is set if the entry comes from an
    unreachable member's last-known snapshot — Phase 2 may add a
    cache; v1 just leaves it False.
    """

    node_id: str
    hostname: str
    address: str
    reachable: bool
    gpus: List[GpuInfoModel] = []
    error: Optional[str] = None


class ClusterGpusResponse(BaseModel):
    nodes: List[ClusterGpusEntry] = []
    server_time: float


@router.get("/gpus_local", response_model=List[GpuInfoModel])
def gpus_local(response: Response):
    """Local GPU snapshot — counterpart of ``/api/gpus`` carved out for
    peer-pull. Identical payload; the alternate path lets the auth
    carve-out target only this cluster-scoped surface.

    The node identity is returned as the ``X-Forgather-Node-Id``
    header so the master-side aggregator can sanity-check that the
    response actually came from the node it intended to call. Caught
    a real bug where both nodes advertised loopback over mDNS and the
    master ended up calling itself when fetching peer GPUs.
    """
    from .. import gpu_monitor

    ident = cluster.self_identity()
    if ident is not None:
        response.headers["X-Forgather-Node-Id"] = ident.node_id
    return [_to_model(g) for g in gpu_monitor.snapshot()]


async def _fetch_peer_gpus(
    client: httpx.AsyncClient, member: cluster.MemberInfo
) -> ClusterGpusEntry:
    self_id = cluster.self_identity()
    if self_id is not None and member.node_id == self_id.node_id:
        # Same-process: skip the network round-trip and call the
        # snapshot path directly. Avoids surprising failure modes
        # when the server is bound to localhost only and a peer
        # entry happens to list its public address.
        from .. import gpu_monitor

        return ClusterGpusEntry(
            node_id=member.node_id,
            hostname=member.hostname,
            address=member.address,
            reachable=True,
            gpus=[_to_model(g) for g in gpu_monitor.snapshot()],
        )
    if not member.reachable:
        return ClusterGpusEntry(
            node_id=member.node_id,
            hostname=member.hostname,
            address=member.address,
            reachable=False,
            error="member unreachable",
        )
    url = f"http://{member.address}:{member.port}/api/cluster/gpus_local"
    try:
        r = await client.get(url, timeout=PEER_GPU_TIMEOUT_SECONDS)
    except (httpx.HTTPError, OSError) as e:
        return ClusterGpusEntry(
            node_id=member.node_id,
            hostname=member.hostname,
            address=member.address,
            reachable=False,
            error=f"fetch failed: {e.__class__.__name__}",
        )
    if r.status_code != 200:
        return ClusterGpusEntry(
            node_id=member.node_id,
            hostname=member.hostname,
            address=member.address,
            reachable=False,
            error=f"http {r.status_code}",
        )
    # Verify the response actually came from the node we expected.
    # If mDNS misadvertised an address (loopback artifact, NAT, etc.)
    # the request can land on a different node and we'd silently
    # display the wrong GPUs against the wrong hostname.
    served_by = r.headers.get("x-forgather-node-id") or r.headers.get(
        "X-Forgather-Node-Id"
    )
    if served_by and served_by != member.node_id:
        return ClusterGpusEntry(
            node_id=member.node_id,
            hostname=member.hostname,
            address=member.address,
            reachable=False,
            error=(
                f"address {member.address}:{member.port} "
                f"served by node {served_by[:8]} (misadvertised?)"
            ),
        )
    try:
        items = r.json()
    except ValueError:
        return ClusterGpusEntry(
            node_id=member.node_id,
            hostname=member.hostname,
            address=member.address,
            reachable=True,
            error="non-JSON response",
        )
    gpus: List[GpuInfoModel] = []
    if isinstance(items, list):
        for raw in items:
            try:
                gpus.append(GpuInfoModel(**raw))
            except Exception:
                log.debug(
                    "skipping malformed GPU entry from %s: %r", url, raw
                )
    return ClusterGpusEntry(
        node_id=member.node_id,
        hostname=member.hostname,
        address=member.address,
        reachable=True,
        gpus=gpus,
    )


class GpuPolicyLocalRequest(BaseModel):
    """Body of POST /api/cluster/gpu_policy_local — same shape as
    ``SetGpuPolicyRequest`` but with the GPU index inline so the
    endpoint can sit at a fixed path (the auth carve-out keys on the
    exact path string)."""

    gpu_index: int
    disabled: Optional[bool] = None
    min_priority: Optional[int] = None


@router.post("/gpu_policy_local", response_model=GpuPolicyModel)
def gpu_policy_local(req: GpuPolicyLocalRequest, response: Response):
    """Apply a GPU policy update on this node.

    Counterpart of ``POST /api/gpus/{idx}/policy`` carved out for
    inter-node mutation. The auth middleware allows POST on this
    exact path from a known peer IP without the bearer token; see
    ``auth._PEER_ALLOWED_MUTATIONS``.

    Returns the same ``X-Forgather-Node-Id`` header as ``gpus_local``
    so the master can sanity-check that the request landed on the
    intended node.
    """
    from .. import gpu_policy as gpu_policy_module

    ident = cluster.self_identity()
    if ident is not None:
        response.headers["X-Forgather-Node-Id"] = ident.node_id
    result = gpu_policy_module.set_policy(
        req.gpu_index,
        disabled=req.disabled,
        min_priority=req.min_priority,
    )
    return GpuPolicyModel(
        disabled=result.disabled, min_priority=result.min_priority
    )


@router.post(
    "/nodes/{node_id}/gpus/{gpu_index}/policy", response_model=GpuPolicyModel
)
async def set_node_gpu_policy(
    node_id: str, gpu_index: int, req: SetGpuPolicyRequest
):
    """Master-side proxy: forward a GPU policy change to the named node.

    Looks up ``node_id`` in the cluster member table, POSTs the
    payload to that node's ``/api/cluster/gpu_policy_local`` (auth
    bypassed by the peer-call carve-out — both ends agree the
    request originates from a cluster peer), and returns the peer's
    response. If the target is the local node, short-circuits the
    network and applies the policy in-process.
    """
    target = next((m for m in cluster.members() if m.node_id == node_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"unknown node {node_id}")
    self_id = cluster.self_identity()
    if self_id is not None and node_id == self_id.node_id:
        from .. import gpu_policy as gpu_policy_module

        result = gpu_policy_module.set_policy(
            gpu_index,
            disabled=req.disabled,
            min_priority=req.min_priority,
        )
        return GpuPolicyModel(
            disabled=result.disabled, min_priority=result.min_priority
        )
    if not target.reachable:
        raise HTTPException(
            status_code=503,
            detail=f"node {target.hostname} is currently unreachable",
        )
    url = f"http://{target.address}:{target.port}/api/cluster/gpu_policy_local"
    payload = {
        "gpu_index": gpu_index,
        "disabled": req.disabled,
        "min_priority": req.min_priority,
    }
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                url, json=payload, timeout=PEER_GPU_TIMEOUT_SECONDS
            )
        except (httpx.HTTPError, OSError) as e:
            raise HTTPException(
                status_code=502,
                detail=f"forward to {target.hostname} failed: {e}",
            )
    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=(
                f"node {target.hostname} returned {r.status_code}: "
                f"{r.text[:200]}"
            ),
        )
    served_by = r.headers.get("x-forgather-node-id") or r.headers.get(
        "X-Forgather-Node-Id"
    )
    if served_by and served_by != node_id:
        raise HTTPException(
            status_code=502,
            detail=(
                f"address {target.address}:{target.port} answered as node "
                f"{served_by[:8]}; refusing to apply policy"
            ),
        )
    body = r.json()
    return GpuPolicyModel(**body)


@router.get("/gpus", response_model=ClusterGpusResponse)
async def cluster_gpus():
    """Aggregate GPU snapshots across the cluster.

    Fetches in parallel with ``asyncio.gather``; one slow peer does
    not block the rest. When the cluster is inactive, returns an
    empty list — the webui handles that as "single-node mode, use
    /api/gpus instead".
    """
    if not cluster.is_active():
        return ClusterGpusResponse(nodes=[], server_time=time.time())
    targets = cluster.members()
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[_fetch_peer_gpus(client, m) for m in targets],
            return_exceptions=False,
        )
    return ClusterGpusResponse(nodes=list(results), server_time=time.time())


# ---------------------------------------------------------------------------
# Bandwidth probe
# ---------------------------------------------------------------------------

# Default payload size for a bandwidth measurement. 32 MiB is enough
# to take a measurable fraction of a second on a gigabit link
# (~0.3 s) and a few seconds on a 100 Mbps WAN bridge — well above
# the noise floor of HTTP overhead. Capped at 256 MiB so a malformed
# query parameter can't make the server allocate a comically large
# response.
DEFAULT_BANDWIDTH_BYTES = 32 * 1024 * 1024
MAX_BANDWIDTH_BYTES = 256 * 1024 * 1024
_BANDWIDTH_CHUNK = b"X" * 65536  # 64 KiB chunks

# Cache TTL: bandwidth doesn't change minute-to-minute, but a node
# coming back online or a network reroute should refresh on the
# next user-triggered probe. The UI exposes a refresh button; the
# cache exists to keep the first paint of the Nodes view from
# blocking on a multi-second probe.
BANDWIDTH_CACHE_TTL_SECONDS = 60 * 60  # 1 hour
PEER_BANDWIDTH_TIMEOUT_SECONDS = 30.0


class BandwidthEntry(BaseModel):
    peer_node_id: str
    peer_hostname: str
    peer_address: str
    bytes_transferred: int
    elapsed_seconds: float
    mbps: float
    timestamp: float
    error: Optional[str] = None


class BandwidthResponse(BaseModel):
    measurements: List[BandwidthEntry] = []
    server_time: float


# Module-level cache. Keyed by peer node_id. Populated by
# /api/cluster/bandwidth/refresh; read by /api/cluster/bandwidth.
_bandwidth_cache: Dict[str, BandwidthEntry] = {}


def _stream_bytes(total: int):
    """Generator producing exactly ``total`` bytes in 64 KiB chunks."""
    sent = 0
    while sent < total:
        size = min(len(_BANDWIDTH_CHUNK), total - sent)
        if size == len(_BANDWIDTH_CHUNK):
            yield _BANDWIDTH_CHUNK
        else:
            yield _BANDWIDTH_CHUNK[:size]
        sent += size


@router.get("/bandwidth_local")
def bandwidth_local(
    bytes: int = Query(
        default=DEFAULT_BANDWIDTH_BYTES,
        ge=4096,
        le=MAX_BANDWIDTH_BYTES,
        description="Number of bytes to stream back; clamped to the server limit.",
    ),
):
    """Stream a deterministic byte blob for the caller to time.

    The peer-allowed bandwidth target. The caller times their own
    receive; we just feed bytes as fast as the kernel will accept
    them. Streamed in chunks so a multi-MiB body doesn't sit in
    memory all at once.
    """
    # StreamingResponse builds its own headers; setting headers on
    # the dependency-injected ``response`` is silently dropped. Build
    # the full header dict here instead.
    headers = {
        "Content-Length": str(bytes),
        "Cache-Control": "no-store",
    }
    ident = cluster.self_identity()
    if ident is not None:
        headers["X-Forgather-Node-Id"] = ident.node_id
    return StreamingResponse(
        _stream_bytes(bytes),
        media_type="application/octet-stream",
        headers=headers,
    )


async def _measure_one_peer(
    client: httpx.AsyncClient,
    member: cluster.MemberInfo,
    bytes_to_pull: int,
) -> BandwidthEntry:
    self_id = cluster.self_identity()
    if self_id is not None and member.node_id == self_id.node_id:
        # Measuring against self is meaningless (loopback throughput
        # is dominated by memcpy, not the network). Return a
        # deliberate zero so the UI can render "self" specially
        # rather than a misleading multi-Gbps loopback number.
        return BandwidthEntry(
            peer_node_id=member.node_id,
            peer_hostname=member.hostname,
            peer_address=member.address,
            bytes_transferred=0,
            elapsed_seconds=0.0,
            mbps=0.0,
            timestamp=time.time(),
            error="self",
        )
    if not member.reachable:
        return BandwidthEntry(
            peer_node_id=member.node_id,
            peer_hostname=member.hostname,
            peer_address=member.address,
            bytes_transferred=0,
            elapsed_seconds=0.0,
            mbps=0.0,
            timestamp=time.time(),
            error="member unreachable",
        )
    url = (
        f"http://{member.address}:{member.port}"
        f"/api/cluster/bandwidth_local?bytes={bytes_to_pull}"
    )
    received = 0
    start = time.monotonic()
    try:
        async with client.stream(
            "GET", url, timeout=PEER_BANDWIDTH_TIMEOUT_SECONDS
        ) as r:
            if r.status_code != 200:
                return BandwidthEntry(
                    peer_node_id=member.node_id,
                    peer_hostname=member.hostname,
                    peer_address=member.address,
                    bytes_transferred=0,
                    elapsed_seconds=0.0,
                    mbps=0.0,
                    timestamp=time.time(),
                    error=f"http {r.status_code}",
                )
            served_by = r.headers.get(
                "x-forgather-node-id"
            ) or r.headers.get("X-Forgather-Node-Id")
            if served_by and served_by != member.node_id:
                return BandwidthEntry(
                    peer_node_id=member.node_id,
                    peer_hostname=member.hostname,
                    peer_address=member.address,
                    bytes_transferred=0,
                    elapsed_seconds=0.0,
                    mbps=0.0,
                    timestamp=time.time(),
                    error=(
                        f"address {member.address}:{member.port} "
                        f"served by node {served_by[:8]}"
                    ),
                )
            async for chunk in r.aiter_bytes():
                received += len(chunk)
    except (httpx.HTTPError, OSError) as e:
        return BandwidthEntry(
            peer_node_id=member.node_id,
            peer_hostname=member.hostname,
            peer_address=member.address,
            bytes_transferred=received,
            elapsed_seconds=time.monotonic() - start,
            mbps=0.0,
            timestamp=time.time(),
            error=f"fetch failed: {e.__class__.__name__}",
        )
    elapsed = max(time.monotonic() - start, 1e-6)
    # Convert bytes/sec to Mbits/sec — the operator-facing unit on
    # any network spec sheet, and what the Samantha tutorial uses.
    mbps = (received * 8) / elapsed / 1_000_000
    return BandwidthEntry(
        peer_node_id=member.node_id,
        peer_hostname=member.hostname,
        peer_address=member.address,
        bytes_transferred=received,
        elapsed_seconds=elapsed,
        mbps=mbps,
        timestamp=time.time(),
        error=None,
    )


@router.get("/bandwidth", response_model=BandwidthResponse)
def get_bandwidth():
    """Return cached bandwidth measurements, dropping stale entries."""
    now = time.time()
    fresh = [
        e
        for e in _bandwidth_cache.values()
        if now - e.timestamp <= BANDWIDTH_CACHE_TTL_SECONDS
    ]
    return BandwidthResponse(measurements=fresh, server_time=now)


@router.post("/bandwidth/refresh", response_model=BandwidthResponse)
async def refresh_bandwidth(
    bytes: int = Query(
        default=DEFAULT_BANDWIDTH_BYTES,
        ge=4096,
        le=MAX_BANDWIDTH_BYTES,
    ),
):
    """Run a fresh measurement against every reachable peer.

    Sequential (not parallel): two simultaneous bulk transfers would
    saturate the local NIC and produce numbers that under-report
    each link's actual throughput. The serial total is N peers ×
    ~few seconds on a gigabit LAN — fine for a user-initiated probe.
    """
    if not cluster.is_active():
        return BandwidthResponse(measurements=[], server_time=time.time())
    targets = [
        m
        for m in cluster.members()
        if cluster.self_identity() is None
        or m.node_id != cluster.self_identity().node_id
    ]
    results: List[BandwidthEntry] = []
    async with httpx.AsyncClient() as client:
        for member in targets:
            entry = await _measure_one_peer(client, member, bytes)
            _bandwidth_cache[member.node_id] = entry
            results.append(entry)
    return BandwidthResponse(measurements=results, server_time=time.time())


def _reset_bandwidth_cache_for_tests() -> None:
    _bandwidth_cache.clear()


# ---------------------------------------------------------------------------
# Cluster-coordinator submit fanout (Phase 3)
# ---------------------------------------------------------------------------

# rdzv args structure mirrored on both ends. Kept loose (Dict[str, Any])
# rather than a strict Pydantic schema because the launcher passes it
# straight to torchrun and a future torch release may add knobs we
# don't want to gate the cluster on.


class TrainingLocalRequest(BaseModel):
    """Payload for POST /api/cluster/training_local.

    The master constructs one of these per participating peer and POSTs
    to that peer's local server. The handler enqueues a training job
    with the supplied rdzv args; the local scheduler picks it up
    normally.
    """

    project_dir: str
    config: str
    dynamic_args: Dict[str, Any] = {}
    requested_gpus: int = 1
    priority: int = 0
    rdzv_args: Dict[str, Any]
    extra_env: Dict[str, str] = {}
    # Cluster bundle id this enqueue is part of. Stored on the queue
    # item's job_params so the master can correlate per-peer queue ids
    # back to the bundle when listing or cancelling.
    cluster_job_id: Optional[str] = None


class TrainingLocalResponse(BaseModel):
    queue_id: str
    node_id: Optional[str] = None


@router.post("/training_local", response_model=TrainingLocalResponse)
def training_local(req: TrainingLocalRequest, response: Response):
    """Peer-side handler: enqueue a multi-node training job locally.

    Same shape as the master's existing /api/queue path but pinned to
    job_type='training' and with the rdzv args spread into job_params
    so the scheduler's _build_training picks them up. Auth is allowed
    for known cluster peers without a bearer token (see auth carve-out).
    """
    from .. import queue_store

    item = queue_store.QueueItem.new(
        project_dir=req.project_dir,
        config=req.config,
        dynamic_args=req.dynamic_args,
        requested_gpus=req.requested_gpus,
        priority=req.priority,
        job_type="training",
        job_params={
            "rdzv_args": dict(req.rdzv_args),
            "extra_env": dict(req.extra_env),
            "cluster_job_id": req.cluster_job_id,
        },
    )
    queue_store.add_item(item)
    ident = cluster.self_identity()
    if ident is not None:
        response.headers["X-Forgather-Node-Id"] = ident.node_id
    log.info(
        "cluster fanout: enqueued training job %s for cluster_job_id=%s "
        "(rank=%s/%s)",
        item.queue_id,
        req.cluster_job_id,
        req.rdzv_args.get("node_rank"),
        req.rdzv_args.get("nnodes"),
    )
    return TrainingLocalResponse(
        queue_id=item.queue_id,
        node_id=ident.node_id if ident is not None else None,
    )


class TrainingCancelLocalRequest(BaseModel):
    queue_id: str


class TrainingCancelLocalResponse(BaseModel):
    queue_id: str
    cancelled: bool
    detail: str = ""


@router.post(
    "/training_cancel_local", response_model=TrainingCancelLocalResponse
)
def training_cancel_local(
    req: TrainingCancelLocalRequest, response: Response
):
    """Peer-side cancel: try to remove the queue item, abort if running.

    Returns ``cancelled=True`` if the item was waiting (and thus removed)
    or running (in which case the local scheduler aborts it). The master's
    bundle-cancel sums these into the bundle status.
    """
    from .. import scheduler

    ident = cluster.self_identity()
    if ident is not None:
        response.headers["X-Forgather-Node-Id"] = ident.node_id
    # ``abort_or_cancel`` handles both the queued and running cases.
    # Returns False for unknown queue ids and for items already in a
    # terminal status — the master's bundle-cancel logs that and moves
    # on rather than treating it as a hard error.
    try:
        ok = scheduler.abort_or_cancel(req.queue_id)
    except Exception as e:
        return TrainingCancelLocalResponse(
            queue_id=req.queue_id,
            cancelled=False,
            detail=f"abort failed: {e}",
        )
    return TrainingCancelLocalResponse(
        queue_id=req.queue_id,
        cancelled=ok,
        detail="" if ok else "unknown queue_id or already terminal",
    )


# ---------------------------------------------------------------------------
# Cluster-coordinator submit (Phase 3)
# ---------------------------------------------------------------------------

# Default rendezvous port. torchrun's c10d backend picks an ephemeral
# one if the user doesn't specify; we pin a known value so the master
# can include it in the rdzv-endpoint without a separate negotiation.
DEFAULT_RDZV_PORT = 29400
PEER_TRAINING_TIMEOUT_SECONDS = 10.0


class MemberSubmitSpec(BaseModel):
    node_id: str
    nproc_per_node: int
    nccl_socket_ifname: Optional[str] = None


class ClusterJobSubmitRequest(BaseModel):
    project_dir: str
    config: str
    dynamic_args: Dict[str, Any] = {}
    priority: int = 0
    members: List[MemberSubmitSpec]
    # node_id whose advertised address torchrun will use as rdzv host.
    # Defaults to the master if not provided.
    rdzv_node_id: Optional[str] = None
    rdzv_port: int = DEFAULT_RDZV_PORT
    # When True, ignore version-mismatch warnings. The UI surfaces them
    # before submit and asks the operator to acknowledge — so by the
    # time we get here the operator has already eyeballed the diff.
    allow_version_mismatch: bool = False


class MemberAssignmentModel(BaseModel):
    node_id: str
    hostname: str
    address: str
    port: int
    queue_id: str
    nproc_per_node: int
    node_rank: int
    nccl_socket_ifname: Optional[str] = None


class ClusterJobModel(BaseModel):
    cluster_job_id: str
    project_dir: str
    config: str
    submitted_at: float
    rdzv_endpoint: str
    rdzv_id: str
    rdzv_node_id: str
    members: List[MemberAssignmentModel]
    status: str
    cancelled_at: Optional[float] = None


class ClusterJobSubmitResponse(BaseModel):
    cluster_job: ClusterJobModel
    warnings: List[str] = []


def _to_cluster_job_model(job: cluster_jobs.ClusterJob) -> ClusterJobModel:
    return ClusterJobModel(
        cluster_job_id=job.cluster_job_id,
        project_dir=job.project_dir,
        config=job.config,
        submitted_at=job.submitted_at,
        rdzv_endpoint=job.rdzv_endpoint,
        rdzv_id=job.rdzv_id,
        rdzv_node_id=job.rdzv_node_id,
        members=[
            MemberAssignmentModel(
                node_id=m.node_id,
                hostname=m.hostname,
                address=m.address,
                port=m.port,
                queue_id=m.queue_id,
                nproc_per_node=m.nproc_per_node,
                node_rank=m.node_rank,
                nccl_socket_ifname=m.nccl_socket_ifname,
            )
            for m in job.members
        ],
        status=job.status,
        cancelled_at=job.cancelled_at,
    )


def _check_version_mismatch(
    members: List[cluster.MemberInfo],
) -> List[str]:
    """Return human-readable warnings for any version key that differs
    across the participating members. Returns an empty list when all
    participants agree (or when probe data is missing — that's a
    different signal, surfaced separately).
    """
    counts: Dict[str, Dict[str, int]] = {}
    for m in members:
        versions = (m.probe or {}).get("versions") or {}
        for key, val in versions.items():
            if key not in {"forgather", "torch", "nccl", "transformers"}:
                continue
            counts.setdefault(key, {})[val] = counts.get(key, {}).get(val, 0) + 1
    warnings: List[str] = []
    for key, vals in counts.items():
        if len(vals) > 1:
            warnings.append(
                f"{key} differs across the cluster: "
                + ", ".join(
                    f"{val} ({n})" for val, n in sorted(vals.items())
                )
            )
    return warnings


async def _fanout_training(
    client: httpx.AsyncClient,
    target: cluster.MemberInfo,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """POST a training_local enqueue to one peer; returns the response.

    The local-node case short-circuits to the in-process handler so a
    single-host cluster (or a self-rank in a multi-node submit) doesn't
    need a network round-trip.
    """
    self_id = cluster.self_identity()
    if self_id is not None and target.node_id == self_id.node_id:
        from .. import queue_store

        item = queue_store.QueueItem.new(
            project_dir=payload["project_dir"],
            config=payload["config"],
            dynamic_args=payload.get("dynamic_args") or {},
            requested_gpus=int(payload.get("requested_gpus", 1)),
            priority=int(payload.get("priority", 0)),
            job_type="training",
            job_params={
                "rdzv_args": dict(payload["rdzv_args"]),
                "extra_env": dict(payload.get("extra_env") or {}),
                "cluster_job_id": payload.get("cluster_job_id"),
            },
        )
        queue_store.add_item(item)
        return {"queue_id": item.queue_id, "node_id": self_id.node_id}
    url = f"http://{target.address}:{target.port}/api/cluster/training_local"
    try:
        r = await client.post(
            url, json=payload, timeout=PEER_TRAINING_TIMEOUT_SECONDS
        )
    except (httpx.HTTPError, OSError) as e:
        raise HTTPException(
            status_code=502,
            detail=f"forward to {target.hostname} failed: {e}",
        )
    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=(
                f"node {target.hostname} returned {r.status_code}: "
                f"{r.text[:200]}"
            ),
        )
    served_by = r.headers.get("x-forgather-node-id") or r.headers.get(
        "X-Forgather-Node-Id"
    )
    if served_by and served_by != target.node_id:
        raise HTTPException(
            status_code=502,
            detail=(
                f"address {target.address}:{target.port} answered as node "
                f"{served_by[:8]}; refusing to enqueue"
            ),
        )
    return r.json()


@router.post("/jobs/submit", response_model=ClusterJobSubmitResponse)
async def submit_cluster_job(req: ClusterJobSubmitRequest):
    if not cluster.is_active():
        raise HTTPException(
            status_code=400, detail="cluster mode is not active"
        )
    if not req.members:
        raise HTTPException(status_code=400, detail="members list is empty")
    by_id = {m.node_id: m for m in cluster.members()}
    participating: List[cluster.MemberInfo] = []
    for spec in req.members:
        m = by_id.get(spec.node_id)
        if m is None:
            raise HTTPException(
                status_code=400,
                detail=f"unknown node_id: {spec.node_id}",
            )
        if not m.reachable:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"node {m.hostname} ({spec.node_id[:8]}) is currently "
                    "unreachable"
                ),
            )
        if spec.nproc_per_node < 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"nproc_per_node must be >= 1 for {m.hostname} "
                    f"(got {spec.nproc_per_node})"
                ),
            )
        participating.append(m)

    warnings = _check_version_mismatch(participating)
    if warnings and not req.allow_version_mismatch:
        raise HTTPException(
            status_code=409,
            detail=(
                "version mismatch across participants — pass "
                "allow_version_mismatch=true to override. Differences:\n"
                + "\n".join(warnings)
            ),
        )

    # Pick the rendezvous host. Default = master. Validate it's in the
    # participating set so the rdzv_endpoint is reachable from every
    # peer that needs it.
    rdzv_node_id = req.rdzv_node_id or cluster.master_node_id()
    if rdzv_node_id is None:
        raise HTTPException(
            status_code=400, detail="cluster has no master right now"
        )
    rdzv_member = by_id.get(rdzv_node_id)
    if rdzv_member is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown rdzv_node_id: {rdzv_node_id}",
        )
    rdzv_endpoint = f"{rdzv_member.address}:{req.rdzv_port}"
    rdzv_id = cluster_jobs.new_rdzv_id()
    cluster_job_id = cluster_jobs.new_cluster_job_id()

    # Build per-peer payloads. ``node_rank`` is assigned by index in
    # the request order — the operator picks the order in the submit
    # modal, which usually means master = rank 0.
    fanout_payloads: List[Dict[str, Any]] = []
    for idx, (spec, member) in enumerate(zip(req.members, participating)):
        extra_env: Dict[str, str] = {}
        if spec.nccl_socket_ifname:
            # Pin every transport — not just NCCL — to the operator's
            # chosen interface. Gloo (CPU collectives) and tensorpipe
            # (RPC) each derive their advertised address from
            # socket.gethostname() by default, which resolves to
            # 127.0.0.1/127.0.1.1 on Debian/Ubuntu via /etc/hosts.
            # Without GLOO_SOCKET_IFNAME / TP_SOCKET_IFNAME the rank
            # publishes a loopback address to its peers and Gloo
            # connectFullMesh fails before the trainer ever runs a
            # step. Same value covers all three because the operator
            # already picked the routable LAN interface.
            extra_env["NCCL_SOCKET_IFNAME"] = spec.nccl_socket_ifname
            extra_env["GLOO_SOCKET_IFNAME"] = spec.nccl_socket_ifname
            extra_env["TP_SOCKET_IFNAME"] = spec.nccl_socket_ifname
        fanout_payloads.append(
            {
                "project_dir": req.project_dir,
                "config": req.config,
                "dynamic_args": dict(req.dynamic_args),
                "requested_gpus": spec.nproc_per_node,
                "priority": req.priority,
                "rdzv_args": {
                    "nnodes": len(req.members),
                    "node_rank": idx,
                    "rdzv_backend": "c10d",
                    "rdzv_endpoint": rdzv_endpoint,
                    "rdzv_id": rdzv_id,
                    "nproc_per_node": spec.nproc_per_node,
                    # Skip torch's broken hostname-based host autodetection
                    # (socket.gethostname() resolves to 127.0.1.1 on
                    # Debian/Ubuntu via /etc/hosts, so neither node would
                    # ever recognise itself as the rdzv host and the c10d
                    # store would never bind). Set explicitly per-peer.
                    "is_host": member.node_id == rdzv_node_id,
                },
                "extra_env": extra_env,
                "cluster_job_id": cluster_job_id,
            }
        )

    # Fanout. Sequential — N is small (usually 2-3) and serial keeps
    # the diagnostic story simple if one peer's enqueue fails.
    assignments: List[cluster_jobs.MemberAssignment] = []
    async with httpx.AsyncClient() as client:
        for idx, (spec, member, payload) in enumerate(
            zip(req.members, participating, fanout_payloads)
        ):
            try:
                result = await _fanout_training(client, member, payload)
            except HTTPException:
                # Best-effort cleanup: cancel anything we already enqueued.
                await _cancel_fanout(client, assignments, by_id)
                raise
            queue_id = str(result.get("queue_id") or "")
            if not queue_id:
                await _cancel_fanout(client, assignments, by_id)
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"node {member.hostname} returned no queue_id"
                    ),
                )
            assignments.append(
                cluster_jobs.MemberAssignment(
                    node_id=member.node_id,
                    hostname=member.hostname,
                    address=member.address,
                    port=member.port,
                    queue_id=queue_id,
                    nproc_per_node=spec.nproc_per_node,
                    node_rank=idx,
                    nccl_socket_ifname=spec.nccl_socket_ifname,
                )
            )

    job = cluster_jobs.ClusterJob(
        cluster_job_id=cluster_job_id,
        project_dir=req.project_dir,
        config=req.config,
        submitted_at=time.time(),
        rdzv_endpoint=rdzv_endpoint,
        rdzv_id=rdzv_id,
        rdzv_node_id=rdzv_node_id,
        members=assignments,
    )
    cluster_jobs.add_job(job)
    return ClusterJobSubmitResponse(
        cluster_job=_to_cluster_job_model(job),
        warnings=warnings,
    )


async def _cancel_fanout(
    client: httpx.AsyncClient,
    assignments: List[cluster_jobs.MemberAssignment],
    by_id: Dict[str, cluster.MemberInfo],
) -> None:
    """Best-effort rollback when a partial fanout fails. We've already
    enqueued on some peers; cancel those before propagating the error
    to the operator."""
    for a in assignments:
        member = by_id.get(a.node_id)
        if member is None:
            continue
        await _cancel_one_peer_training(client, member, a.queue_id)


async def _cancel_one_peer_training(
    client: httpx.AsyncClient,
    target: cluster.MemberInfo,
    queue_id: str,
) -> Optional[Dict[str, Any]]:
    self_id = cluster.self_identity()
    if self_id is not None and target.node_id == self_id.node_id:
        from .. import scheduler as _sched

        ok = _sched.abort_or_cancel(queue_id)
        return {"queue_id": queue_id, "cancelled": ok}
    url = f"http://{target.address}:{target.port}/api/cluster/training_cancel_local"
    try:
        r = await client.post(
            url,
            json={"queue_id": queue_id},
            timeout=PEER_TRAINING_TIMEOUT_SECONDS,
        )
    except (httpx.HTTPError, OSError) as e:
        log.warning(
            "cancel forward to %s failed: %s", target.hostname, e
        )
        return None
    if r.status_code != 200:
        log.warning(
            "cancel non-200 from %s: %d", target.hostname, r.status_code
        )
        return None
    return r.json()


@router.get("/jobs", response_model=List[ClusterJobModel])
def list_cluster_jobs():
    return [_to_cluster_job_model(j) for j in cluster_jobs.list_jobs()]


@router.get(
    "/jobs/{cluster_job_id}", response_model=Optional[ClusterJobModel]
)
def get_cluster_job(cluster_job_id: str):
    job = cluster_jobs.get_job(cluster_job_id)
    return _to_cluster_job_model(job) if job is not None else None


class ClusterJobCancelResponse(BaseModel):
    cluster_job_id: str
    cancelled: bool
    per_member: List[Dict[str, Any]]


@router.post(
    "/jobs/{cluster_job_id}/cancel", response_model=ClusterJobCancelResponse
)
async def cancel_cluster_job(cluster_job_id: str):
    job = cluster_jobs.get_job(cluster_job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail=f"unknown cluster_job_id: {cluster_job_id}"
        )
    by_id = {m.node_id: m for m in cluster.members()}
    per_member: List[Dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        for a in job.members:
            member = by_id.get(a.node_id)
            if member is None:
                per_member.append(
                    {
                        "node_id": a.node_id,
                        "queue_id": a.queue_id,
                        "result": "unknown member",
                    }
                )
                continue
            res = await _cancel_one_peer_training(client, member, a.queue_id)
            per_member.append(
                {
                    "node_id": a.node_id,
                    "queue_id": a.queue_id,
                    "result": res or "fanout failed",
                }
            )
    cluster_jobs.mark_cancelled(cluster_job_id)
    cancelled = all(
        isinstance(p["result"], dict) and p["result"].get("cancelled")
        for p in per_member
    )
    return ClusterJobCancelResponse(
        cluster_job_id=cluster_job_id,
        cancelled=cancelled,
        per_member=per_member,
    )
