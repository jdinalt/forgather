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
from fastapi import APIRouter
from pydantic import BaseModel

from .. import cluster
from .gpus import GpuInfoModel, _to_model

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
def gpus_local():
    """Local GPU snapshot — counterpart of ``/api/gpus`` carved out for
    peer-pull. Identical payload; the alternate path lets the auth
    carve-out target only this cluster-scoped surface.
    """
    from .. import gpu_monitor

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
