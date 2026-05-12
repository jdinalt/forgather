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
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from forgather.tls import httpx_verify

from .. import cluster, cluster_dataset_inventory, cluster_jobs, dataset_source
from ..dataset_source import DatasetSourceError
from .gpus import GpuInfoModel, GpuPolicyModel, SetGpuPolicyRequest, _to_model

log = logging.getLogger("forgather_server.routes.cluster")

router = APIRouter(prefix="/cluster", tags=["cluster"])


def _peer_base(member) -> str:
    """``https://host:port`` if peer advertised TLS, else ``http://``."""
    scheme = "https" if getattr(member, "tls", False) else "http"
    return f"{scheme}://{member.address}:{member.port}"


def _peer_url(member, path: str) -> str:
    return _peer_base(member) + path


def _peer_client(**kwargs) -> httpx.AsyncClient:
    """``httpx.AsyncClient`` with the shared CA bundle pre-wired."""
    kwargs.setdefault("verify", httpx_verify())
    return httpx.AsyncClient(**kwargs)


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
    # Whether this peer serves HTTPS. Drives the scheme used by
    # peer-pull and inter-node HTTP calls.
    tls: bool = False


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
        tls=getattr(m, "tls", False),
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


# ---------------------------------------------------------------------------
# Dataset-server inventory (peer-local)
# ---------------------------------------------------------------------------


class LocalDatasetServerModel(BaseModel):
    """A dataset_server this peer attests to.

    Includes ``auth_token`` — the cluster carve-out gates this surface
    to known cluster peers, so the master's aggregator can poll
    upstream dataset_servers without an extra credential exchange.
    Anything that surfaces this to a browser must strip the token
    first (handled by the master-side webui endpoints in Phase 3 / 6).
    """

    server_id: str
    base_url: str
    auth_token: str
    label: str
    source: str  # "local" or "user"
    peer_node_id: Optional[str] = None


class LocalDatasetServersResponse(BaseModel):
    self_node_id: Optional[str] = None
    servers: List[LocalDatasetServerModel] = []


@router.get(
    "/dataset_servers_local", response_model=LocalDatasetServersResponse
)
def dataset_servers_local(response: Response):
    """Per-peer dataset_server inventory.

    Counterpart of ``/api/cluster/gpus_local`` for the dataset-server
    routing infrastructure: the master fans GETs to this endpoint
    every aggregation tick (Phase 3) to build the cluster-wide server
    set. Carved out of the bearer-token gate for known cluster peers.

    Tokens are returned in the body — the carve-out is peer-only and
    the cluster bearer protects the rest of the surface. See
    ``auth._PEER_ALLOWED_PATHS`` for the trust assumptions.
    """
    ident = cluster.self_identity()
    if ident is not None:
        response.headers["X-Forgather-Node-Id"] = ident.node_id
    servers = cluster_dataset_inventory.local_servers()
    return LocalDatasetServersResponse(
        self_node_id=ident.node_id if ident is not None else None,
        servers=[
            LocalDatasetServerModel(
                server_id=s.server_id,
                base_url=s.base_url,
                auth_token=s.auth_token,
                label=s.label,
                source=s.source,
                peer_node_id=s.peer_node_id,
            )
            for s in servers
        ],
    )


# ---------------------------------------------------------------------------
# Dataset-server inventory (master-side aggregation + router)
# ---------------------------------------------------------------------------


class ClusterDatasetServerModel(BaseModel):
    """Master-aggregated server entry, token stripped.

    Token-free shape is the only one we ever return to a browser; the
    cluster carve-out (``/dataset_servers_local``) is the *only*
    surface that ships bearer tokens, and only to known cluster peers.
    """

    server_id: str
    base_url: str
    label: str
    source: str
    peer_node_id: Optional[str] = None
    healthy: bool
    last_health_check: float
    last_health_error: str
    last_dataset_refresh: float
    last_dataset_error: str
    # Polling counters. ``consecutive_*_failures`` is 0 on a current-
    # success / never-polled server and non-zero exactly when the
    # server is currently in trouble — useful for the webui's "is this
    # stuck or transient" decision.
    total_health_polls: int = 0
    health_failures: int = 0
    consecutive_health_failures: int = 0
    total_dataset_polls: int = 0
    dataset_failures: int = 0
    consecutive_dataset_failures: int = 0


class ClusterDatasetEntryModel(BaseModel):
    """One unique dataset in the cluster (deduped across servers).

    ``dataset_id`` is the canonical key:
      - ``local/<name>`` for entries from ``/v1/local`` (one global
        key per local name — two servers advertising the same name
        are treated as interchangeable replicas);
      - the server-side handle hash (``sha256(canonical(resolved_args))[:16]``)
        for HF / path datasets already loaded somewhere in the
        cluster.
    """

    dataset_id: str
    source: str  # "local" | "hf" | "path"
    name: Optional[str] = None  # local name, when applicable
    load_args: Optional[Dict[str, Any]] = None
    length: Optional[int] = None
    column_names: Optional[List[str]] = None
    server_ids: List[str] = []


class ClusterDatasetInventoryMetrics(BaseModel):
    """Aggregate counters across the master's collect / health /
    dataset-refresh loops. Useful for catching "everything's quiet
    but no data is flowing" failure modes from a single GET."""

    healthy_servers: int = 0
    unhealthy_servers: int = 0
    total_servers: int = 0
    total_datasets: int = 0
    total_health_polls: int = 0
    total_health_failures: int = 0
    total_dataset_polls: int = 0
    total_dataset_failures: int = 0
    # How long ago this node became master, in seconds. None when the
    # node isn't master.
    master_age_seconds: Optional[float] = None


class ClusterDatasetInventoryResponse(BaseModel):
    is_master: bool
    master_become_ts: Optional[float] = None
    last_servers_collect_ts: Optional[float] = None
    last_health_pass_ts: Optional[float] = None
    last_dataset_pass_ts: Optional[float] = None
    servers: List[ClusterDatasetServerModel] = []
    datasets: List[ClusterDatasetEntryModel] = []
    metrics: ClusterDatasetInventoryMetrics = ClusterDatasetInventoryMetrics()


class DatasetRouterResolveResponse(BaseModel):
    base_url: str
    auth_token: str = ""
    server_id: str


def _master_member() -> Optional[cluster.MemberInfo]:
    """Return the cluster member that's currently master, or None.

    ``None`` means either cluster is inactive, no master has been
    elected yet (empty/unreachable cluster), or the master entry has
    no peer-routable address.
    """
    master_id = cluster.master_node_id()
    if master_id is None:
        return None
    self_ident = cluster.self_identity()
    if self_ident is not None and master_id == self_ident.node_id:
        return None
    for m in cluster.members():
        if m.node_id == master_id and m.reachable and m.address:
            return m
    return None


def _self_is_master() -> bool:
    if not cluster.is_active():
        return False
    return cluster.is_self_master()


def _to_dataset_server_model(
    e: "cluster_dataset_inventory.MasterServerEntry",
) -> ClusterDatasetServerModel:
    return ClusterDatasetServerModel(
        server_id=e.server_id,
        base_url=e.base_url,
        label=e.label,
        source=e.source,
        peer_node_id=e.peer_node_id,
        healthy=e.healthy,
        last_health_check=e.last_health_check,
        last_health_error=e.last_health_error,
        last_dataset_refresh=e.last_dataset_refresh,
        last_dataset_error=e.last_dataset_error,
        total_health_polls=e.total_health_polls,
        health_failures=e.health_failures,
        consecutive_health_failures=e.consecutive_health_failures,
        total_dataset_polls=e.total_dataset_polls,
        dataset_failures=e.dataset_failures,
        consecutive_dataset_failures=e.consecutive_dataset_failures,
    )


def _build_inventory_response() -> ClusterDatasetInventoryResponse:
    """Read the master singleton and shape it for the webui.

    Datasets are deduped across servers by ``dataset_id``:
      - Local entries (``/v1/local`` results) key on ``local/<name>``.
      - Handle entries (``/v1/datasets`` results) key on the handle
        hash; entries whose source is ``"local"`` are skipped because
        they already appear under their local name (avoids double-
        counting the same logical dataset).
    """
    servers = cluster_dataset_inventory.master_inventory.servers_snapshot()
    status = cluster_dataset_inventory.master_inventory.status()

    by_id: Dict[str, ClusterDatasetEntryModel] = {}
    for s in servers:
        # Local entries — `/v1/local` shape varies by server build; we
        # tolerate ``[{"name": "stories", "length": ...}, ...]`` and
        # similar.
        for li in s.locals_info:
            if not isinstance(li, dict):
                continue
            name = li.get("name")
            if not isinstance(name, str) or not name:
                continue
            dataset_id = f"local/{name}"
            entry = by_id.get(dataset_id)
            if entry is None:
                entry = ClusterDatasetEntryModel(
                    dataset_id=dataset_id,
                    source="local",
                    name=name,
                    length=(
                        int(li["length"])
                        if isinstance(li.get("length"), (int, float))
                        else None
                    ),
                    column_names=(
                        list(li["column_names"])
                        if isinstance(li.get("column_names"), list)
                        else None
                    ),
                    server_ids=[],
                )
                by_id[dataset_id] = entry
            if s.server_id not in entry.server_ids:
                entry.server_ids.append(s.server_id)

        # Already-loaded handles (HF / path). Skip "local" handles —
        # they're already represented under their local name above.
        for h in s.handles:
            if not isinstance(h, dict):
                continue
            handle_id = h.get("handle")
            if not isinstance(handle_id, str) or not handle_id:
                continue
            src = str(h.get("source") or "hf")
            if src == "local":
                continue
            entry = by_id.get(handle_id)
            if entry is None:
                load_args = h.get("load_args")
                entry = ClusterDatasetEntryModel(
                    dataset_id=handle_id,
                    source=src,
                    name=(
                        load_args.get("name")
                        if isinstance(load_args, dict)
                        else None
                    ),
                    load_args=(
                        load_args if isinstance(load_args, dict) else None
                    ),
                    length=(
                        int(h["length"])
                        if isinstance(h.get("length"), (int, float))
                        else None
                    ),
                    column_names=(
                        list(h["column_names"])
                        if isinstance(h.get("column_names"), list)
                        else None
                    ),
                    server_ids=[],
                )
                by_id[handle_id] = entry
            if s.server_id not in entry.server_ids:
                entry.server_ids.append(s.server_id)

    healthy = sum(1 for s in servers if s.healthy)
    master_age = (
        time.time() - status["master_become_ts"]
        if status["master_become_ts"] is not None
        else None
    )
    metrics = ClusterDatasetInventoryMetrics(
        total_servers=len(servers),
        healthy_servers=healthy,
        unhealthy_servers=len(servers) - healthy,
        total_datasets=len(by_id),
        total_health_polls=sum(s.total_health_polls for s in servers),
        total_health_failures=sum(s.health_failures for s in servers),
        total_dataset_polls=sum(s.total_dataset_polls for s in servers),
        total_dataset_failures=sum(s.dataset_failures for s in servers),
        master_age_seconds=master_age,
    )

    return ClusterDatasetInventoryResponse(
        is_master=status["is_master"],
        master_become_ts=status["master_become_ts"],
        last_servers_collect_ts=status["last_servers_collect_ts"],
        last_health_pass_ts=status["last_health_pass_ts"],
        last_dataset_pass_ts=status["last_dataset_pass_ts"],
        servers=[_to_dataset_server_model(s) for s in servers],
        datasets=list(by_id.values()),
        metrics=metrics,
    )


async def _proxy_inventory_to_master() -> Optional[Dict[str, Any]]:
    """Non-master nodes proxy ``/dataset_inventory`` to the master so
    every webui sees the same view. Returns ``None`` when there's no
    master to proxy to (caller surfaces an "uninitialized" payload)."""
    master = _master_member()
    if master is None:
        return None
    url = _peer_url(master, "/api/cluster/dataset_inventory")
    try:
        async with _peer_client(timeout=5.0) as client:
            r = await client.get(url)
    except (httpx.HTTPError, OSError) as e:
        log.warning("dataset_inventory proxy: %s -> %s", master.hostname, e)
        return None
    if r.status_code != 200:
        log.warning(
            "dataset_inventory proxy non-200: %s status=%d",
            master.hostname,
            r.status_code,
        )
        return None
    try:
        return r.json()
    except ValueError:
        return None


@router.get(
    "/dataset_inventory", response_model=ClusterDatasetInventoryResponse
)
async def dataset_inventory() -> ClusterDatasetInventoryResponse:
    """Master-aggregated dataset inventory.

    Webui-facing: tokens are stripped, the response carries the
    deduped dataset list + the per-server health/refresh state.
    Non-master nodes proxy to the master so every webui instance
    sees the same view.
    """
    if _self_is_master():
        return _build_inventory_response()
    proxied = await _proxy_inventory_to_master()
    if proxied is not None:
        # Re-validate through the pydantic model so the wire schema
        # stays consistent regardless of mixed-version clusters.
        return ClusterDatasetInventoryResponse(**proxied)
    # No master reachable — emit an empty "uninitialized" shape so the
    # webui can render the cold-start hint rather than a 5xx.
    return ClusterDatasetInventoryResponse(is_master=False)


@router.get(
    "/dataset_servers", response_model=List[ClusterDatasetServerModel]
)
async def dataset_servers() -> List[ClusterDatasetServerModel]:
    """Master-aggregated server list (token-stripped).

    Same provenance as ``/dataset_inventory.servers`` but a smaller
    payload — the Explore tab + Servers tab use this directly without
    pulling the full dataset listing.
    """
    if _self_is_master():
        return [
            _to_dataset_server_model(s)
            for s in cluster_dataset_inventory.master_inventory.servers_snapshot()
        ]
    inv = await _proxy_inventory_to_master()
    if inv is None:
        return []
    raw = inv.get("servers") if isinstance(inv, dict) else None
    if not isinstance(raw, list):
        return []
    return [ClusterDatasetServerModel(**item) for item in raw]


@router.get(
    "/dataset_router/resolve", response_model=DatasetRouterResolveResponse
)
async def dataset_router_resolve(
    response: Response,
    path: str = Query(..., description="Dataset path the client wants to load."),
) -> DatasetRouterResolveResponse:
    """Pick a healthy server for the given dataset request.

    Cluster ``auto`` routing (Phase 4) calls this from the training
    process: the client passes its ``path`` (``local/<name>``, an HF
    id, or a filesystem path) and the master returns the URL + token
    of a healthy server.

    Non-master nodes proxy to the master so the call works from any
    cluster member — the training container only ever talks to its
    local forgather_server.

    Returns:
      * **200** with ``{base_url, auth_token, server_id}`` on success.
      * **503** ``Retry-After: 5`` when the inventory is still warming
        up (no completed dataset-refresh pass yet) — fresh master,
        loops haven't finished their first cycle.
      * **410** when warmed but no healthy server can serve ``path``
        (operator config issue — retrying won't help).
    """
    if _self_is_master():
        inv = cluster_dataset_inventory.master_inventory
        if not inv.is_warmed_up():
            response.status_code = 503
            response.headers["Retry-After"] = "5"
            raise HTTPException(
                status_code=503,
                detail=(
                    "Dataset-server inventory is still warming up; "
                    "retry shortly."
                ),
                headers={"Retry-After": "5"},
            )
        pick = inv.resolve(path)
        if pick is None:
            raise HTTPException(
                status_code=410,
                detail=(
                    f"No healthy dataset_server can serve {path!r} in the "
                    "current cluster."
                ),
            )
        base_url, token = pick
        # Find the server_id matching the chosen base_url for client
        # diagnostics (logged when training resumes).
        sid = ""
        for s in inv.servers_snapshot():
            if s.base_url == base_url and s.auth_token == token:
                sid = s.server_id
                break
        return DatasetRouterResolveResponse(
            base_url=base_url, auth_token=token, server_id=sid
        )

    # Not master — proxy to master.
    master = _master_member()
    if master is None:
        raise HTTPException(
            status_code=503,
            detail="No cluster master is currently reachable.",
            headers={"Retry-After": "5"},
        )
    url = _peer_url(master, "/api/cluster/dataset_router/resolve")
    try:
        async with _peer_client(timeout=5.0) as client:
            r = await client.get(url, params={"path": path})
    except (httpx.HTTPError, OSError) as e:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach cluster master: {e}",
        )
    if r.status_code == 200:
        body = r.json()
        return DatasetRouterResolveResponse(**body)
    # Forward the master's error verbatim (including 503 Retry-After).
    detail = r.text
    try:
        detail = r.json().get("detail", detail)
    except ValueError:
        pass
    raise HTTPException(
        status_code=r.status_code,
        detail=detail,
        headers=(
            {"Retry-After": r.headers["Retry-After"]}
            if "Retry-After" in r.headers
            else None
        ),
    )


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
    url = _peer_url(member, "/api/cluster/gpus_local")
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
                log.debug("skipping malformed GPU entry from %s: %r", url, raw)
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
    return GpuPolicyModel(disabled=result.disabled, min_priority=result.min_priority)


@router.post("/nodes/{node_id}/gpus/{gpu_index}/policy", response_model=GpuPolicyModel)
async def set_node_gpu_policy(node_id: str, gpu_index: int, req: SetGpuPolicyRequest):
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
    url = _peer_url(target, "/api/cluster/gpu_policy_local")
    payload = {
        "gpu_index": gpu_index,
        "disabled": req.disabled,
        "min_priority": req.min_priority,
    }
    async with _peer_client() as client:
        try:
            r = await client.post(url, json=payload, timeout=PEER_GPU_TIMEOUT_SECONDS)
        except (httpx.HTTPError, OSError) as e:
            raise HTTPException(
                status_code=502,
                detail=f"forward to {target.hostname} failed: {e}",
            )
    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=(
                f"node {target.hostname} returned {r.status_code}: " f"{r.text[:200]}"
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
    async with _peer_client() as client:
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
    url = _peer_url(member, f"/api/cluster/bandwidth_local?bytes={bytes_to_pull}")
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
            served_by = r.headers.get("x-forgather-node-id") or r.headers.get(
                "X-Forgather-Node-Id"
            )
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
    async with _peer_client() as client:
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


class TrainingStatusLocalResponse(BaseModel):
    queue_id: str
    # Mirrors job_records.JobRecord.status: queued/starting/running/done/
    # failed/cancelled, plus "unknown" when the queue_id isn't found
    # locally (record GC'd, or never reached the peer).
    status: str
    exit_code: Optional[int] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None


@router.get("/training_status_local", response_model=TrainingStatusLocalResponse)
def training_status_local(queue_id: str, response: Response):
    """Peer-side status snapshot for one local queue item.

    Used by the master to roll up cluster-job status without
    proxying the full /api/jobs surface to peers (that would widen
    the trusted-peer auth carve-out unnecessarily). Read-only and
    bounded to the single queue_id in the query string.
    """
    from .. import job_records

    ident = cluster.self_identity()
    if ident is not None:
        response.headers["X-Forgather-Node-Id"] = ident.node_id
    rec = job_records.get_record(queue_id)
    if rec is None:
        return TrainingStatusLocalResponse(queue_id=queue_id, status="unknown")
    return TrainingStatusLocalResponse(
        queue_id=queue_id,
        status=rec.status,
        exit_code=rec.exit_code,
        started_at=rec.started_at,
        finished_at=rec.finished_at,
        error=rec.error,
    )


class TrainingCancelLocalRequest(BaseModel):
    queue_id: str


class TrainingCancelLocalResponse(BaseModel):
    queue_id: str
    cancelled: bool
    detail: str = ""


@router.post("/training_cancel_local", response_model=TrainingCancelLocalResponse)
def training_cancel_local(req: TrainingCancelLocalRequest, response: Response):
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
    # See routes/queue.py::EnqueueRequest. Resolved here on the master
    # and merged into every peer's extra_env so the whole cluster pulls
    # examples from the same dataset_server (typically a dedicated data
    # host the entire LAN can reach).
    dataset_source: Optional[Dict[str, Any]] = None


class MemberAssignmentModel(BaseModel):
    node_id: str
    hostname: str
    address: str
    port: int
    queue_id: str
    nproc_per_node: int
    node_rank: int
    nccl_socket_ifname: Optional[str] = None
    # Live status of this rank's queue item, fetched via per-peer
    # status lookup at read time. None when the master couldn't
    # reach the peer (UI shows it as a question mark).
    current_status: Optional[str] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None


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
    # Roll-up of per-member statuses, computed at read time.
    # cancelled > failed > running > done > submitted (priority order
    # — see _rollup_cluster_status). UI shows this; the bundle's own
    # ``status`` only flips to "cancelled" when the master fans out
    # a cancel.
    rolled_up_status: str = "submitted"
    # Dataset-source choice from the submit request — None /
    # ``{"kind": "local"}`` for the in-process loader, ``{"kind":
    # "auto"}`` for cluster auto-routing, ``{"kind": "server",
    # "server_id": ...}`` for a pinned URL. Surfaced verbatim so the
    # operator can see "did this bundle actually use auto-routing"
    # without consulting per-rank env logs.
    dataset_source: Optional[Dict[str, Any]] = None


class ClusterJobSubmitResponse(BaseModel):
    cluster_job: ClusterJobModel
    warnings: List[str] = []


def _to_cluster_job_model(
    job: cluster_jobs.ClusterJob,
    member_statuses: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ClusterJobModel:
    """Project a bundle record onto its API model.

    ``member_statuses`` maps queue_id → status dict (status, exit_code,
    error, etc). When omitted, members render with ``current_status``
    = None — used in tests and in the submit response, where the
    statuses haven't been polled yet.
    """
    statuses = member_statuses or {}
    member_models = []
    for m in job.members:
        st = statuses.get(m.queue_id) or {}
        member_models.append(
            MemberAssignmentModel(
                node_id=m.node_id,
                hostname=m.hostname,
                address=m.address,
                port=m.port,
                queue_id=m.queue_id,
                nproc_per_node=m.nproc_per_node,
                node_rank=m.node_rank,
                nccl_socket_ifname=m.nccl_socket_ifname,
                current_status=st.get("status"),
                exit_code=st.get("exit_code"),
                error=st.get("error"),
            )
        )
    return ClusterJobModel(
        cluster_job_id=job.cluster_job_id,
        project_dir=job.project_dir,
        config=job.config,
        submitted_at=job.submitted_at,
        rdzv_endpoint=job.rdzv_endpoint,
        rdzv_id=job.rdzv_id,
        rdzv_node_id=job.rdzv_node_id,
        members=member_models,
        status=job.status,
        dataset_source=job.dataset_source,
        cancelled_at=job.cancelled_at,
        rolled_up_status=_rollup_cluster_status(
            job, [m.current_status for m in member_models]
        ),
    )


# Per-member status priority for the bundle roll-up. Higher index wins
# when members disagree — e.g. one rank "running" + one "failed"
# rolls up to "failed" so the UI surfaces the bad news first. "done"
# only wins when *every* member is done (handled in
# _rollup_cluster_status, not here).
_STATUS_PRIORITY = {
    "unknown": 0,
    "queued": 1,
    "submitted": 1,
    "starting": 2,
    "running": 3,
    "done": 4,
    "cancelled": 5,
    "failed": 6,
}


def _rollup_cluster_status(
    job: cluster_jobs.ClusterJob, member_statuses: List[Optional[str]]
) -> str:
    """Aggregate per-member statuses into a single bundle status.

    Priority: cancelled-by-master overrides everything (the operator
    asked for the bundle to stop). Otherwise failed > running >
    cancelled > queued > done. "done" only returns when *all*
    participants finished cleanly — partial completion is ambiguous,
    not done.
    """
    if job.status in ("cancelled", "done", "failed"):
        return job.status
    statuses = [s for s in member_statuses if s]
    if not statuses:
        return job.status
    if all(s == "done" for s in statuses) and len(statuses) == len(job.members):
        return "done"
    return max(statuses, key=lambda s: _STATUS_PRIORITY.get(s, 0))


async def _gather_member_statuses(
    job: cluster_jobs.ClusterJob,
) -> Dict[str, Dict[str, Any]]:
    """Fan out per-member status lookups, return queue_id → status dict.

    Uses the local job_records lookup for the master's own assignment
    (no HTTP round-trip), and ``GET /api/cluster/training_status_local``
    on each remote peer. Each peer gets a short timeout — a slow or
    unresponsive peer must not block the UI list. Unreachable peers
    contribute ``status="unknown"`` rather than an exception.
    """
    from .. import job_records

    self_ident = cluster.self_identity()
    self_node_id = self_ident.node_id if self_ident else None
    by_id = {m.node_id: m for m in cluster.members()}
    out: Dict[str, Dict[str, Any]] = {}

    async def _one(member_assignment: cluster_jobs.MemberAssignment):
        if member_assignment.node_id == self_node_id:
            rec = job_records.get_record(member_assignment.queue_id)
            if rec is None:
                out[member_assignment.queue_id] = {"status": "unknown"}
            else:
                out[member_assignment.queue_id] = {
                    "status": rec.status,
                    "exit_code": rec.exit_code,
                    "started_at": rec.started_at,
                    "finished_at": rec.finished_at,
                    "error": rec.error,
                }
            return
        peer = by_id.get(member_assignment.node_id)
        if peer is None or not peer.reachable:
            out[member_assignment.queue_id] = {"status": "unknown"}
            return
        url = _peer_url(peer, "/api/cluster/training_status_local")
        try:
            async with _peer_client(timeout=2.0) as client:
                r = await client.get(
                    url, params={"queue_id": member_assignment.queue_id}
                )
                if r.status_code == 200:
                    out[member_assignment.queue_id] = r.json()
                else:
                    out[member_assignment.queue_id] = {"status": "unknown"}
        except Exception:
            out[member_assignment.queue_id] = {"status": "unknown"}

    await asyncio.gather(*(_one(m) for m in job.members))
    return out


def _maybe_promote_terminal(job: cluster_jobs.ClusterJob, rolled_up: str) -> None:
    """Stick a terminal roll-up status onto the bundle record itself.

    Once a cluster job is fully done/failed, we don't need to keep
    fanning out status checks for it. Writing the terminal value back
    to the bundle's own ``status`` lets future read-paths short-circuit
    via _rollup_cluster_status's first branch (which returns
    job.status when it's "cancelled" — extending the same idea to
    "done" / "failed").
    """
    if job.status in ("done", "failed", "cancelled"):
        return
    if rolled_up in ("done", "failed"):
        cluster_jobs.set_terminal_status(job.cluster_job_id, rolled_up)


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
                + ", ".join(f"{val} ({n})" for val, n in sorted(vals.items()))
            )
    return warnings


def _derive_iface_from_member(member: cluster.MemberInfo) -> Optional[str]:
    """Pick the interface name whose IP matches ``member.address``.

    Used when the operator leaves the iface picker on "(auto)" in the
    submit modal. The cluster has already chosen ``member.address`` as
    the routable LAN IP (mDNS / peer-pull), and the probe gives us the
    full interface table — matching one back to the other yields the
    name that NCCL/Gloo/TP need to bind.

    Returns ``None`` when no probe data is available or no interface
    matches the advertised address; the caller surfaces that as a 422
    so the operator picks one explicitly.
    """
    probe = member.probe or {}
    interfaces = probe.get("interfaces") or []
    for entry in interfaces:
        if entry.get("address") == member.address:
            name = entry.get("name")
            if name:
                return str(name)
    return None


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
    url = _peer_url(target, "/api/cluster/training_local")
    try:
        r = await client.post(url, json=payload, timeout=PEER_TRAINING_TIMEOUT_SECONDS)
    except (httpx.HTTPError, OSError) as e:
        raise HTTPException(
            status_code=502,
            detail=f"forward to {target.hostname} failed: {e}",
        )
    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=(
                f"node {target.hostname} returned {r.status_code}: " f"{r.text[:200]}"
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
        raise HTTPException(status_code=400, detail="cluster mode is not active")
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
        raise HTTPException(status_code=400, detail="cluster has no master right now")
    rdzv_member = by_id.get(rdzv_node_id)
    if rdzv_member is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown rdzv_node_id: {rdzv_node_id}",
        )
    rdzv_endpoint = f"{rdzv_member.address}:{req.rdzv_port}"
    rdzv_id = cluster_jobs.new_rdzv_id()
    cluster_job_id = cluster_jobs.new_cluster_job_id()

    # Resolve the dataset-source choice once on the master — the URL
    # + token are reachable from every peer (that's the whole point of
    # the central data host), so the same env vars get merged into
    # every per-peer extra_env below. Resolution errors become 400 so
    # the operator sees "your saved server is gone" rather than a
    # cluster of training processes all silently falling back to
    # in-process loading.
    #
    # remote_host_override: for a *cluster* submit the resolved env is
    # shipped to peer training processes, so loopback would route to
    # the peer's own 127.0.0.1 (no dataset server there). Use the
    # master's cluster-routable address — the same one peers already
    # use to peer-pull. cluster.self_identity() gives the local node;
    # its member entry carries the post-discovery address.
    self_ident = cluster.self_identity()
    self_member = (
        next(
            (m for m in cluster.members() if m.node_id == self_ident.node_id),
            None,
        )
        if self_ident
        else None
    )
    master_cluster_addr = self_member.address if self_member else None
    if master_cluster_addr in (None, "", "127.0.0.1", "::1"):
        # update_self_address hasn't run yet (or the cluster module
        # never got a real interface address). Don't ship loopback to
        # peers — that's the exact bug we're fixing.
        if req.dataset_source and req.dataset_source.get("kind") == "server":
            raise HTTPException(
                status_code=503,
                detail=(
                    "cluster master has no routable cluster address yet — "
                    "dataset_server cannot be shared with peers. Wait for "
                    "mDNS discovery to complete and retry."
                ),
            )
    try:
        _dataset_env = dataset_source.resolve_to_env(
            req.dataset_source,
            remote_host_override=master_cluster_addr,
        )
    except DatasetSourceError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Build per-peer payloads. ``node_rank`` is assigned by index in
    # the request order — the operator picks the order in the submit
    # modal, which usually means master = rank 0.
    fanout_payloads: List[Dict[str, Any]] = []
    for idx, (spec, member) in enumerate(zip(req.members, participating)):
        # Pin every transport — not just NCCL — to a specific interface.
        # Gloo (CPU collectives) and tensorpipe (RPC) each derive their
        # advertised address from socket.gethostname() by default, which
        # resolves to 127.0.0.1/127.0.1.1 on Debian/Ubuntu via /etc/hosts.
        # Without GLOO_SOCKET_IFNAME / TP_SOCKET_IFNAME the rank publishes
        # a loopback address to its peers and Gloo connectFullMesh fails
        # before the trainer ever runs a step. Same value covers all
        # three because all of NCCL/Gloo/TP need a routable LAN
        # interface to bind.
        iface = spec.nccl_socket_ifname or _derive_iface_from_member(member)
        if not iface:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"could not auto-derive a network interface for "
                    f"{member.hostname} (advertised address "
                    f"{member.address} did not match any interface in "
                    f"its probe). Pick an interface explicitly in the "
                    f"submit modal — Gloo will publish loopback "
                    f"addresses otherwise and connectFullMesh will fail."
                ),
            )
        extra_env: Dict[str, str] = {
            "NCCL_SOCKET_IFNAME": iface,
            "GLOO_SOCKET_IFNAME": iface,
            "TP_SOCKET_IFNAME": iface,
        }
        # The dataset_source choice is identical for every peer (the
        # whole cluster pulls from the same data host), so resolve
        # once outside the loop — but the easier thing here is to
        # resolve once before the loop and inject inside. Done above;
        # see ``_dataset_env`` set just before the fanout payloads.
        if _dataset_env:
            for k, v in _dataset_env.items():
                extra_env.setdefault(k, v)
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
    async with _peer_client() as client:
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
                    detail=(f"node {member.hostname} returned no queue_id"),
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
        dataset_source=req.dataset_source,
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
    url = _peer_url(target, "/api/cluster/training_cancel_local")
    try:
        r = await client.post(
            url,
            json={"queue_id": queue_id},
            timeout=PEER_TRAINING_TIMEOUT_SECONDS,
        )
    except (httpx.HTTPError, OSError) as e:
        log.warning("cancel forward to %s failed: %s", target.hostname, e)
        return None
    if r.status_code != 200:
        log.warning("cancel non-200 from %s: %d", target.hostname, r.status_code)
        return None
    return r.json()


async def _maybe_proxy_to_master() -> Optional[List[Dict[str, Any]]]:
    """If we're not the master, fetch the cluster-jobs list from it.

    The bundle record lives only on the master. Without this proxy a
    non-master webui shows zero cluster jobs even when one is running
    cluster-wide — the user observed exactly that on muthur while the
    bundle was held on wopr. The master's response already includes
    the rolled-up status fanout, so the non-master returns it
    verbatim. Returns None when:
      - we are the master (caller falls through to local computation)
      - master is unknown / unreachable (caller falls back to local
        empty list rather than failing the page)
    """
    master_id = cluster.master_node_id()
    self_ident = cluster.self_identity()
    if master_id is None or self_ident is None:
        return None
    if master_id == self_ident.node_id:
        return None
    by_id = {m.node_id: m for m in cluster.members()}
    master = by_id.get(master_id)
    if master is None or not master.reachable:
        log.warning(
            "cluster jobs proxy: master %s not reachable from %s, "
            "falling through to local empty view",
            master_id[:8],
            self_ident.node_id[:8],
        )
        return None
    url = _peer_url(master, "/api/cluster/jobs")
    try:
        async with _peer_client(timeout=5.0) as client:
            r = await client.get(url)
            if r.status_code != 200:
                log.warning("cluster jobs proxy: master returned %d", r.status_code)
                return None
            data = r.json()
            if isinstance(data, list):
                return data
            return None
    except Exception:
        log.exception("cluster jobs proxy: GET %s failed", url)
        return None


@router.get("/jobs", response_model=List[ClusterJobModel])
async def list_cluster_jobs():
    """List all cluster jobs with rolled-up status.

    Status comes from per-rank queue items via fanout — each member's
    peer is queried for its local job_record state, and the results
    are aggregated. Already-terminal bundles short-circuit (their
    status is sticky), so the fanout cost only applies to in-flight
    jobs. Slow / unreachable peers don't block the list — they
    contribute "unknown" for that member.

    Non-master nodes proxy to the master (which holds the bundle
    records) so any cluster-mode webui shows the same job list. If
    master is unreachable we return our own (empty) view rather than
    erroring — the page should keep rendering even during a master
    failover.
    """
    proxied = await _maybe_proxy_to_master()
    if proxied is not None:
        return proxied
    jobs = cluster_jobs.list_jobs()
    # Skip the fanout for bundles that are already in a sticky
    # terminal state — their status doesn't change again.
    statuses_by_job: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for j in jobs:
        if j.status in ("done", "failed", "cancelled"):
            statuses_by_job[j.cluster_job_id] = {}
        else:
            statuses_by_job[j.cluster_job_id] = await _gather_member_statuses(j)
    out: List[ClusterJobModel] = []
    for j in jobs:
        ms = statuses_by_job[j.cluster_job_id]
        model = _to_cluster_job_model(j, ms)
        _maybe_promote_terminal(j, model.rolled_up_status)
        out.append(model)
    return out


@router.get("/jobs/{cluster_job_id}", response_model=Optional[ClusterJobModel])
async def get_cluster_job(cluster_job_id: str):
    job = cluster_jobs.get_job(cluster_job_id)
    if job is None:
        return None
    if job.status in ("done", "failed", "cancelled"):
        return _to_cluster_job_model(job, {})
    member_statuses = await _gather_member_statuses(job)
    model = _to_cluster_job_model(job, member_statuses)
    _maybe_promote_terminal(job, model.rolled_up_status)
    return model


class ClusterJobCancelResponse(BaseModel):
    cluster_job_id: str
    cancelled: bool
    per_member: List[Dict[str, Any]]


@router.post("/jobs/{cluster_job_id}/cancel", response_model=ClusterJobCancelResponse)
async def cancel_cluster_job(cluster_job_id: str):
    job = cluster_jobs.get_job(cluster_job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail=f"unknown cluster_job_id: {cluster_job_id}"
        )
    by_id = {m.node_id: m for m in cluster.members()}
    per_member: List[Dict[str, Any]] = []
    async with _peer_client() as client:
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
    cancelled = all(
        isinstance(p["result"], dict) and p["result"].get("cancelled")
        for p in per_member
    )
    # Only stamp the bundle as cancelled when every peer actually
    # cancelled. A partial fan-out leaves some queue items still
    # running on peers; promoting the bundle to "cancelled" here
    # would short-circuit the rollup's per-peer fanout (terminal
    # statuses are sticky) and the operator would see a
    # "cancelled" bundle while the underlying ranks are still
    # consuming GPUs. Leave the bundle non-terminal so the rollup
    # keeps reflecting reality; the response's per-member detail
    # tells the operator which peers failed to cancel.
    if cancelled:
        cluster_jobs.mark_cancelled(cluster_job_id)
    else:
        log.warning(
            "cluster job %s cancel was partial: %s",
            cluster_job_id,
            [
                f"{p['node_id'][:8]}={p['result']}"
                for p in per_member
                if not (isinstance(p["result"], dict) and p["result"].get("cancelled"))
            ],
        )
    return ClusterJobCancelResponse(
        cluster_job_id=cluster_job_id,
        cancelled=cancelled,
        per_member=per_member,
    )


# ---------------------------------------------------------------------------
# Master-side dataset-server proxy
# ---------------------------------------------------------------------------
#
# The webui's Explore + Cluster tabs need to reach dataset_servers
# known anywhere in the cluster — not just the ones registered on
# the local node. ``/api/cluster/dataset_server_proxy/{server_id}/{op}``
# is the cluster-wide equivalent of ``/api/dataset-server/proxy/*``:
#
#   * the master looks up the ``server_id`` in its inventory (built
#     by the Phase 3 loops), pulls the matching ``(base_url, token)``,
#     and calls the upstream dataset_server directly;
#   * non-master nodes forward the same path to the master so every
#     webui sees the same surface regardless of which node serves it.
#
# The inventory itself is the SSRF allowlist — only servers that
# survived the master's collect + health gates appear there, and the
# auth token never crosses out to the browser.

_PROXY_OP_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)

# Map op name (URL segment) -> (upstream method, upstream path template
# or builder, body-passthrough?). Listed explicitly so an unknown op
# returns 404 immediately rather than smuggling traffic through.
_OP_HEALTH = "health"
_OP_AUTH_STATUS = "auth-status"
_OP_DATASETS = "datasets"
_OP_CACHE = "cache"
_OP_LOCAL = "local"
_OP_LOAD = "load"
_OP_LENGTH = "length"
_OP_ITER = "iter"
_ALLOWED_PROXY_OPS = frozenset(
    {
        _OP_HEALTH,
        _OP_AUTH_STATUS,
        _OP_DATASETS,
        _OP_CACHE,
        _OP_LOCAL,
        _OP_LOAD,
        _OP_LENGTH,
        _OP_ITER,
    }
)


def _proxy_auth_headers(token: str) -> Dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _verify_for_proxy(target: str) -> object:
    try:
        from forgather.tls import httpx_verify_for_url

        return httpx_verify_for_url(target)
    except Exception:
        return True


def _safe_json_proxy(r: httpx.Response) -> Any:
    try:
        return r.json()
    except ValueError:
        return {"error": "non-json response from upstream", "body": r.text}


def _upstream_failed_headers(status: int) -> Dict[str, str]:
    """Forward an upstream-auth-failure marker so the webui surfaces a
    clear "your saved token is wrong" message rather than the
    generic forgather-server 401 it would otherwise see."""
    if status in (401, 403):
        return {"x-upstream-auth-failed": "1"}
    return {}


async def _forward_get_to_upstream(
    target: str, token: str
) -> JSONResponse:
    headers = _proxy_auth_headers(token)
    async with httpx.AsyncClient(
        timeout=_PROXY_OP_TIMEOUT, verify=_verify_for_proxy(target)
    ) as client:
        try:
            r = await client.get(target, headers=headers or None)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return JSONResponse(
        status_code=r.status_code,
        content=_safe_json_proxy(r),
        headers=_upstream_failed_headers(r.status_code),
    )


async def _forward_post_to_upstream(
    target: str, token: str, body: bytes, content_type: str
) -> JSONResponse:
    headers = _proxy_auth_headers(token)
    headers["content-type"] = content_type
    async with httpx.AsyncClient(
        timeout=_PROXY_OP_TIMEOUT, verify=_verify_for_proxy(target)
    ) as client:
        try:
            r = await client.post(target, content=body, headers=headers)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return JSONResponse(
        status_code=r.status_code,
        content=_safe_json_proxy(r),
        headers=_upstream_failed_headers(r.status_code),
    )


async def _stream_iter_window(
    target: str, token: str, limit: int
) -> JSONResponse:
    """Materialize a bounded ``/v1/datasets/{handle}/iter`` NDJSON
    window into ``{"rows": [...]}``.

    Same bounded-buffering approach as
    :func:`routes.dataset_server.proxy_iter` — the upstream is asked
    for ``limit`` rows but we stop reading at exactly ``limit`` in case
    a misbehaving server ignores the cap.
    """
    headers = _proxy_auth_headers(token)
    rows: List[Any] = []
    import json as _json

    async with httpx.AsyncClient(
        timeout=_PROXY_OP_TIMEOUT, verify=_verify_for_proxy(target)
    ) as client:
        try:
            async with client.stream("GET", target, headers=headers or None) as r:
                if r.status_code >= 400:
                    body = await r.aread()
                    detail = body.decode("utf-8", errors="replace")
                    return JSONResponse(
                        status_code=r.status_code,
                        content={"detail": detail},
                        headers=_upstream_failed_headers(r.status_code),
                    )
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        rows.append(_json.loads(line))
                    except ValueError:
                        rows.append({"_parse_error": line})
                    if len(rows) >= limit:
                        break
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return JSONResponse({"rows": rows})


def _lookup_proxy_target(server_id: str) -> Tuple[str, str]:
    """Resolve ``server_id`` to ``(base_url, auth_token)`` from the
    master inventory. Raises 404 if unknown — keeps the master from
    being turned into an open relay by a fabricated server_id."""
    entry = cluster_dataset_inventory.master_inventory.get_server(server_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown cluster dataset_server: {server_id!r}",
        )
    return entry.base_url, entry.auth_token


async def _proxy_via_master(
    server_id: str, op: str, request: Request
) -> Response:
    """Forward a proxy request to the master node, preserving method,
    query params, and body."""
    master = _master_member()
    if master is None:
        raise HTTPException(
            status_code=503,
            detail="No cluster master is currently reachable.",
            headers={"Retry-After": "5"},
        )
    base = _peer_base(master)
    path = f"/api/cluster/dataset_server_proxy/{server_id}/{op}"
    target = f"{base}{path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"
    body = await request.body() if request.method == "POST" else None
    async with _peer_client(timeout=_PROXY_OP_TIMEOUT) as client:
        try:
            if request.method == "GET":
                r = await client.get(target)
            else:
                content_type = request.headers.get(
                    "content-type", "application/json"
                )
                r = await client.post(
                    target,
                    content=body,
                    headers={"content-type": content_type},
                )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502,
                detail=f"master proxy: {type(e).__name__}: {e}",
            )
    return JSONResponse(
        status_code=r.status_code,
        content=_safe_json_proxy(r),
        headers=_upstream_failed_headers(r.status_code),
    )


@router.api_route(
    "/dataset_server_proxy/{server_id}/{op}", methods=["GET", "POST"]
)
async def dataset_server_proxy(
    server_id: str, op: str, request: Request
) -> Response:
    """Cluster-wide proxy to a dataset_server known anywhere in the
    cluster.

    Resolves ``server_id`` against the master's inventory and forwards
    the call to the upstream dataset_server with the inventory's
    bearer token. Non-master nodes forward the call to the master so
    every webui instance can use a single path.

    Supported ops (mirrors the per-node ``/api/dataset-server/proxy``):
      * ``health``, ``auth-status``, ``datasets``, ``cache``, ``local`` (GET)
      * ``load`` (POST)
      * ``length`` (GET, ``handle`` query)
      * ``iter`` (GET, ``handle``/``position``/``limit``/``seed`` query;
        the NDJSON stream is materialized into ``{"rows": [...]}``)
    """
    if op not in _ALLOWED_PROXY_OPS:
        raise HTTPException(status_code=404, detail=f"unknown op: {op!r}")

    if not _self_is_master():
        return await _proxy_via_master(server_id, op, request)

    base_url, token = _lookup_proxy_target(server_id)
    base = base_url.rstrip("/")

    if op == _OP_HEALTH:
        return await _forward_get_to_upstream(base + "/v1/health", token)
    if op == _OP_AUTH_STATUS:
        return await _forward_get_to_upstream(base + "/v1/auth/status", token)
    if op == _OP_DATASETS:
        return await _forward_get_to_upstream(base + "/v1/datasets", token)
    if op == _OP_CACHE:
        return await _forward_get_to_upstream(base + "/v1/cache/hf", token)
    if op == _OP_LOCAL:
        return await _forward_get_to_upstream(base + "/v1/local", token)

    if op == _OP_LOAD:
        if request.method != "POST":
            raise HTTPException(status_code=405, detail="load requires POST")
        body = await request.body()
        content_type = request.headers.get("content-type", "application/json")
        return await _forward_post_to_upstream(
            base + "/v1/load", token, body, content_type
        )

    if op == _OP_LENGTH:
        handle = request.query_params.get("handle")
        if not handle:
            raise HTTPException(
                status_code=400, detail="handle query parameter required"
            )
        from urllib.parse import quote as _quote

        return await _forward_get_to_upstream(
            base + f"/v1/datasets/{_quote(handle, safe='')}/length", token
        )

    if op == _OP_ITER:
        handle = request.query_params.get("handle")
        if not handle:
            raise HTTPException(
                status_code=400, detail="handle query parameter required"
            )
        try:
            position = int(request.query_params.get("position", "0"))
            limit = int(request.query_params.get("limit", "25"))
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="position and limit must be integers",
            )
        if position < 0:
            raise HTTPException(
                status_code=400, detail="position must be >= 0"
            )
        if limit < 1 or limit > 500:
            raise HTTPException(
                status_code=400, detail="limit must be in [1, 500]"
            )
        seed = request.query_params.get("seed")
        qs = f"?position={position}&limit={limit}"
        if seed is not None:
            try:
                qs += f"&seed={int(seed)}"
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400, detail="seed must be an integer"
                )
        from urllib.parse import quote as _quote

        return await _stream_iter_window(
            base + f"/v1/datasets/{_quote(handle, safe='')}/iter" + qs,
            token,
            limit,
        )

    raise HTTPException(status_code=500, detail="unreachable")
