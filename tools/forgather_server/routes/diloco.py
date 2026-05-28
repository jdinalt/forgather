"""DiLoCo server discovery + status / control proxy.

Backs the (in-flight) DiLoCo view in the webui. Surface:

  GET    /api/diloco/servers          Unified list (local + registered).
  GET    /api/diloco/server-status    Proxy to an upstream DiLoCo /status.
  GET    /api/diloco/server-info      Proxy to an upstream DiLoCo /info.
  GET    /api/diloco/work-queues      Proxy to upstream /work/queues.
  GET    /api/diloco/work-queue       Proxy to upstream /work/queue.
  POST   /api/diloco/server-control/{action}
                                      Proxy to upstream /control/{action}.
  GET    /api/diloco/registry         List user-added external entries.
  POST   /api/diloco/registry         Add an external entry.
  DELETE /api/diloco/registry/{id}    Remove an external entry.

SSRF policy mirrors :mod:`routes.dataset_server`: loopback is always
allowed, registered URLs are allowed (the act of registering is the
authorization), running locally-spawned diloco_server jobs are allowed
(their URL is derived from job_params), everything else is refused.

Auth / TLS are deliberately out of scope for the initial cut — we
neither send bearer headers nor validate certificates. ``auth_token``
and ``verify_tls`` live on the registry schema so the wire format
doesn't need to change when TLS lands; this proxy layer is the single
chokepoint that will need to learn to honor them.

Cluster aggregation is deferred to a follow-up slice: the unified list
sources are local + registered for now.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .. import diloco_server_registry, job_records
from ..job_records import RUNNING_STATUSES

log = logging.getLogger("forgather_server.routes.diloco")
router = APIRouter(tags=["diloco"])


_PROXY_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)

_DILOCO_JOB_TYPE = "diloco_server"


# ---------------------------------------------------------------------------
# Server discovery
# ---------------------------------------------------------------------------


class DiLoCoServerModel(BaseModel):
    """A DiLoCo server visible to this forgather_server.

    ``source`` is one of:
      ``local``      — spawned by us; lookup keyed by ``queue_id``.
      ``registered`` — user-added external entry; lookup keyed by
                       registry ``id``.
      ``cluster``    — known to a peer; reserved for the follow-up
                       slice (always absent today).
    """

    id: str
    label: str
    base_url: str
    source: str
    host: Optional[str] = None
    port: Optional[int] = None
    # Local-only fields:
    queue_id: Optional[str] = None
    alive: Optional[bool] = None
    # Registered-only fields:
    has_auth_token: Optional[bool] = None
    verify_tls: Optional[bool] = None


def _browser_host(host: Optional[str], routable_host: Optional[str] = None) -> str:
    """Translate a server's bind host into something a browser can hit.

    ``0.0.0.0`` binds every interface but isn't a routable target.
    When the scheduler stamped a ``routable_host`` (a cluster-routable
    address or the first non-loopback psutil-detected IP), prefer it
    so cross-machine browsers can reach the server. Fall back to
    ``localhost`` for same-host operators when no routable address
    was discovered.
    """
    if not host or host == "0.0.0.0":
        if routable_host:
            return routable_host
        return "localhost"
    return host


def _local_servers() -> List[DiLoCoServerModel]:
    """Running JobRecord-derived DiLoCo servers spawned by this forgather_server.

    Terminal-status records (done / failed / aborted) are filtered out:
    a dead server can't be selected for training and can't be inspected
    from the DiLoCo view, so listing it here is pure clutter. The Jobs
    view still shows them for diagnostics.
    """
    out: List[DiLoCoServerModel] = []
    for r in job_records.list_records():
        if r.job_type != _DILOCO_JOB_TYPE:
            continue
        if r.status not in RUNNING_STATUSES:
            continue
        params = r.job_params or {}
        try:
            port = int(params.get("port")) if params.get("port") is not None else None
        except (TypeError, ValueError):
            port = None
        if port is None:
            continue
        host = params.get("host") or "127.0.0.1"
        routable = params.get("routable_host")
        base_url = f"http://{_browser_host(host, routable)}:{port}"
        out.append(
            DiLoCoServerModel(
                id=f"local:{r.queue_id}",
                label=f"{r.config or 'diloco_server'}",
                base_url=base_url,
                source="local",
                host=str(host),
                port=port,
                queue_id=r.queue_id,
                alive=True,
            )
        )
    # Newest first — matches the Jobs view's implicit ordering.
    out.sort(key=lambda s: s.queue_id or "", reverse=True)
    return out


def _ever_local_base_urls() -> List[str]:
    """Base URLs of every DiLoCo server this forgather_server has ever
    spawned, regardless of current job status.

    Used only by the SSRF allowlist. A URL that was once legitimately
    spawned by us (and shown to the user as a server to inspect)
    remains a safe proxy target after the upstream process exits:
    the host:port pair is consented-to. Without this, a user who
    shuts down a local DiLoCo server while the webui panel is still
    polling sees the next request rejected as an SSRF violation
    (403) instead of the more accurate connection-refused (502) that
    would otherwise come from the upstream attempt.
    """
    out: List[str] = []
    for r in job_records.list_records():
        if r.job_type != _DILOCO_JOB_TYPE:
            continue
        params = r.job_params or {}
        try:
            port = int(params.get("port")) if params.get("port") is not None else None
        except (TypeError, ValueError):
            port = None
        if port is None:
            continue
        host = params.get("host") or "127.0.0.1"
        routable = params.get("routable_host")
        out.append(f"http://{_browser_host(host, routable)}:{port}")
    return out


def _registered_servers() -> List[DiLoCoServerModel]:
    return [
        DiLoCoServerModel(
            id=f"registered:{e.id}",
            label=e.label,
            base_url=e.base_url,
            source="registered",
            has_auth_token=bool(e.auth_token),
            verify_tls=e.verify_tls,
        )
        for e in diloco_server_registry.list_entries()
    ]


@router.get("/diloco/servers", response_model=List[DiLoCoServerModel])
def list_servers():
    """Unified list of DiLoCo servers known to this node.

    Returns local (forgather_server-spawned) and user-registered
    entries. Cluster propagation is added in the cluster slice.
    """
    return _local_servers() + _registered_servers()


# ---------------------------------------------------------------------------
# Outbound proxy — status / info / control
# ---------------------------------------------------------------------------


def _validate_base(base: str) -> str:
    """Parse + normalize a base URL; raise 400 on anything we won't proxy to."""
    base = (base or "").strip().rstrip("/")
    if not base:
        raise HTTPException(status_code=400, detail="base is required")
    try:
        parsed = urlparse(base)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad base: {e}")
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            status_code=400, detail=f"unsupported scheme: {parsed.scheme!r}"
        )
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="bad base: missing host")
    return base


def _is_loopback(host: str) -> bool:
    h = (host or "").lower()
    return h in ("127.0.0.1", "localhost", "::1") or h.startswith("127.")


def _known_base_urls() -> List[str]:
    """Bases this server is willing to reach (loopback aside).

    Ever-spawned local URLs (running or terminated) and registered
    URLs together form the SSRF allowlist: the act of registering or
    spawning is the consent. Terminated-but-still-allowlisted local
    URLs hit the actual upstream attempt downstream of this check,
    which produces a 502 on connection-refused — a far more accurate
    signal than the SSRF guard's 403.
    """
    bases: List[str] = list(_ever_local_base_urls())
    for e in diloco_server_registry.list_entries():
        bases.append(e.base_url)
    return bases


def _check_ssrf(base: str) -> None:
    parsed = urlparse(base)
    if _is_loopback(parsed.hostname or ""):
        return
    if base.rstrip("/") in (b.rstrip("/") for b in _known_base_urls()):
        return
    raise HTTPException(
        status_code=403,
        detail=(
            f"refusing to proxy to {base!r}: not loopback and not in the "
            f"DiLoCo registry. Register the URL first via POST "
            f"/api/diloco/registry."
        ),
    )


def _safe_json(resp: httpx.Response) -> Any:
    """Decode upstream JSON; fall back to a 502-shaped envelope on failure."""
    try:
        return resp.json()
    except Exception as e:
        return {"error": f"upstream returned non-JSON: {type(e).__name__}: {e}"}


async def _proxy_get(base: str, path: str) -> JSONResponse:
    base = _validate_base(base)
    _check_ssrf(base)
    target = base + path
    async with httpx.AsyncClient(timeout=_PROXY_TIMEOUT, verify=True) as client:
        try:
            r = await client.get(target)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return JSONResponse(status_code=r.status_code, content=_safe_json(r))


@router.get("/diloco/server-status")
async def proxy_status(base: str) -> JSONResponse:
    """Forward ``GET <base>/status`` and return the upstream JSON verbatim."""
    return await _proxy_get(base, "/status")


@router.get("/diloco/server-info")
async def proxy_info(base: str) -> JSONResponse:
    """Forward ``GET <base>/info`` for client-settings negotiation."""
    return await _proxy_get(base, "/info")


@router.get("/diloco/work-queues")
async def proxy_work_queues(base: str) -> JSONResponse:
    """Forward ``GET <base>/work/queues`` — summary list of active queues."""
    return await _proxy_get(base, "/work/queues")


@router.get("/diloco/work-queue")
async def proxy_work_queue(
    base: str, dataset_id: str, shuffle_seed: int
) -> JSONResponse:
    """Forward ``GET <base>/work/queue?dataset_id=&shuffle_seed=`` —
    single-queue detail with base64 bitmaps and per-worker counts.

    ``dataset_id`` + ``shuffle_seed`` are passed straight through to
    the upstream; ``base`` goes through the standard SSRF allowlist.
    """
    from urllib.parse import quote, urlencode

    qs = urlencode(
        {"dataset_id": dataset_id, "shuffle_seed": int(shuffle_seed)},
        quote_via=quote,
    )
    return await _proxy_get(base, f"/work/queue?{qs}")


# Set of control actions the DiLoCo server itself recognises. Kept here
# so the proxy refuses unknown actions up-front instead of bouncing them
# off the upstream with a confusing 404.
_CONTROL_ACTIONS = frozenset(
    {"save_state", "kick_worker", "update_optimizer", "update_num_workers", "shutdown"}
)


@router.post("/diloco/server-control/{action}")
async def proxy_control(
    action: str, base: str, body: Dict[str, Any] = None
) -> JSONResponse:
    """Forward a control action to the upstream DiLoCo server.

    Body is opaque JSON the upstream interprets (e.g.
    ``{"worker_id": ...}`` for ``kick_worker``,
    ``{"lr": 0.5, "momentum": 0.8}`` for ``update_optimizer``).
    """
    if action not in _CONTROL_ACTIONS:
        raise HTTPException(
            status_code=400, detail=f"unknown control action: {action!r}"
        )
    base = _validate_base(base)
    _check_ssrf(base)
    target = f"{base}/control/{action}"
    async with httpx.AsyncClient(timeout=_PROXY_TIMEOUT, verify=True) as client:
        try:
            r = await client.post(target, json=body or {})
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return JSONResponse(status_code=r.status_code, content=_safe_json(r))


# ---------------------------------------------------------------------------
# User-defined external registry
# ---------------------------------------------------------------------------


class RegistryEntryModel(BaseModel):
    id: str
    label: str
    base_url: str
    has_auth_token: bool
    verify_tls: bool = True


class AddRegistryEntryRequest(BaseModel):
    label: str = ""
    base_url: str
    # Reserved for the TLS / auth follow-up; the proxy ignores them today.
    auth_token: str = ""
    verify_tls: bool = True


def _entry_to_model(e: diloco_server_registry.RegistryEntry) -> RegistryEntryModel:
    return RegistryEntryModel(
        id=e.id,
        label=e.label,
        base_url=e.base_url,
        has_auth_token=bool(e.auth_token),
        verify_tls=e.verify_tls,
    )


@router.get("/diloco/registry", response_model=List[RegistryEntryModel])
def list_registry():
    return [_entry_to_model(e) for e in diloco_server_registry.list_entries()]


@router.post("/diloco/registry", response_model=RegistryEntryModel)
def add_registry_entry(req: AddRegistryEntryRequest):
    base_url = (req.base_url or "").strip()
    if not base_url:
        raise HTTPException(status_code=400, detail="base_url is required")
    try:
        parsed = urlparse(base_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad base_url: {e}")
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            status_code=400, detail=f"unsupported scheme: {parsed.scheme!r}"
        )
    if not parsed.netloc:
        raise HTTPException(status_code=400, detail="bad base_url: missing host")
    try:
        entry = diloco_server_registry.add_entry(
            label=req.label,
            base_url=base_url,
            auth_token=(req.auth_token or "").strip(),
            verify_tls=bool(req.verify_tls),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _entry_to_model(entry)


@router.delete("/diloco/registry/{entry_id}")
def delete_registry_entry(entry_id: str):
    removed = diloco_server_registry.remove_entry(entry_id)
    if removed is None:
        raise HTTPException(
            status_code=404, detail=f"no registry entry with id {entry_id!r}"
        )
    return {"deleted": entry_id}
