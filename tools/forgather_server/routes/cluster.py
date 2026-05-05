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
from typing import Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from .. import cluster
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
