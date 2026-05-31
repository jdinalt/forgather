"""DiLoCo server discovery + status / control proxy.

Backs the DiLoCo view in the webui. Surface:

  GET    /api/diloco/servers          Unified list (local + registered).
  GET    /api/diloco/server-status    Proxy to an upstream DiLoCo /status.
  GET    /api/diloco/server-info      Proxy to an upstream DiLoCo /info.
  GET    /api/diloco/known-workers    Proxy to upstream /known_workers.
  GET    /api/diloco/work-queues      Proxy to upstream /work/queues.
  GET    /api/diloco/work-queue       Proxy to upstream /work/queue.
  POST   /api/diloco/server-control/{action}
                                      Proxy to upstream /control/{action}.
  GET    /api/diloco/registry         List user-added external entries.
  POST   /api/diloco/registry         Add an external entry.
  DELETE /api/diloco/registry/{id}    Remove an external entry.
  POST   /api/diloco/generate-worker-names
                                      Mint N unique memorable worker names
                                      (submit-modal batch pool). Local, no
                                      proxy.

SSRF policy mirrors :mod:`routes.dataset_server`: loopback is always
allowed, registered URLs are allowed (the act of registering is the
authorization), running locally-spawned diloco_server jobs are allowed
(their URL is derived from job_params), everything else is refused.

Auth / TLS (issue #90): the proxy now attaches an ``Authorization:
Bearer <token>`` header (resolved per the precedence below) and honors
each registry entry's ``verify_tls`` opt-out:

  1. Explicit ``X-Diloco-Auth-Token`` request header (operator override).
  2. JobRecord auto-lookup for locally-spawned servers — the scheduler
     persisted the token on the record when spawning.
  3. ``diloco_server_registry.find_token(base)`` for user-added remotes.
  4. Empty (server is running ``--no-auth``).

Cluster aggregation is deferred to a follow-up slice: the unified list
sources are local + registered for now.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from forgather.utils import generate_name

from .. import diloco_server_registry, job_records
from ..job_records import RUNNING_STATUSES

log = logging.getLogger("forgather_server.routes.diloco")
router = APIRouter(tags=["diloco"])


_PROXY_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)

_DILOCO_JOB_TYPE = "diloco_server"

_LOCALHOST_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Operator-supplied override header. Mirrors dataset_server's
# ``X-Dataset-Auth-Token`` — lets the webui pass through a per-row
# user-managed token without depending on the registry's stored
# token (which is convenient for one-off "try this server" flows).
_TOKEN_OVERRIDE_HEADER = "X-Diloco-Auth-Token"

# Tag set on the proxy response when the upstream DiLoCo server
# returned 401/403. The webui's fetch wrapper (auth.ts) suppresses
# the auth-required event when this header is present — a 401 from
# *our* session would have been intercepted by AuthMiddleware before
# any proxy code ran, so a 401 surfacing through this proxy is
# always the upstream's auth state, not the operator's. Closes #94.
_UPSTREAM_AUTH_FAILED_HEADER = "x-upstream-auth-failed"


def _upstream_auth_headers(status: int) -> Dict[str, str]:
    """Tag headers when upstream returned an auth failure.

    Mirrors the existing pattern in ``routes/dataset_server.py`` and
    ``routes/inference_proxy.py``. The webui consumer is at
    ``webui/src/auth.ts:82-83``.
    """
    if status in (401, 403):
        return {_UPSTREAM_AUTH_FAILED_HEADER: "1"}
    return {}


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
        # Scheme is stamped by the scheduler at dispatch time so the
        # Job card and DiLoCo view both reflect the actual TLS posture
        # of the spawned child (issue #90). Falls back to ``http`` for
        # records spawned before that stamping landed.
        scheme = params.get("scheme") or "http"
        base_url = f"{scheme}://{_browser_host(host, routable)}:{port}"
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
                # Locally-spawned servers default to bearer auth ON
                # (the scheduler resolves + persists a per-port token
                # unless --no-auth). Surface the lock indicator so the
                # webui doesn't show auth'd local servers as open while
                # showing the lock only on registered remotes.
                has_auth_token=bool(getattr(r, "auth_token", None)),
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

    Caveat: this is "ever-consented-to", not "still-valid". A
    terminated job's host:port pair can later be reused by an
    unrelated process (OS ephemeral-port reuse, DHCP reassignment
    of a LAN address). The threat is bounded — the operator already
    has full RCE on the box, and the proxy only exposes GETs / a
    controlled set of POST control actions — but a long-running
    forgather_server's allowlist will accumulate stale entries. A
    TTL or "prune on job-record removal" hook is a reasonable
    follow-up; tracked separately from this fix.
    """
    seen: Dict[str, None] = {}
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
        scheme = params.get("scheme") or "http"
        seen[f"{scheme}://{_browser_host(host, routable)}:{port}"] = None
    return list(seen)


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


def _token_for_local(base: str) -> Optional[str]:
    """Auto-lookup auth token from JobRecords for locally-spawned servers.

    Walks running ``diloco_server`` JobRecords looking for one whose
    bind port matches ``base``'s port AND whose host translates to
    ``base``'s hostname. A record matches when *any* of these hold:

    * The URL is loopback and the record's bind is loopback or
      ``0.0.0.0`` (the original case).
    * The URL hostname equals the record's stamped ``routable_host``
      — i.e. we synthesized the URL ourselves from the JobRecord, so
      a webui pointing at it via the LAN address is talking to our
      own spawn.
    * The URL hostname equals the record's bind ``host`` (explicit
      LAN bind that the operator chose).

    The webui builds URLs from ``_local_servers``'s synthesis, so the
    middle case is the one that matters when the operator spawns
    DiLoCo on a non-loopback host (binding ``0.0.0.0`` or a specific
    LAN IP). Without it, the proxy can't tie the URL back to the
    JobRecord and can't attach the bearer — the operator sees a 401.

    Returns the persisted bearer token from the matching record, or
    ``None`` when no match is found (server hasn't started, or the
    URL belongs to an external server).
    """
    try:
        parsed = urlparse(base)
    except Exception:
        return None
    host = (parsed.hostname or "").lower()
    if parsed.port is None:
        return None
    host_is_loopback = host in _LOCALHOST_HOSTS
    for r in job_records.list_records():
        if r.job_type != _DILOCO_JOB_TYPE:
            continue
        if r.status not in {"starting", "running"}:
            continue
        params = r.job_params or {}
        rec_port = params.get("port")
        try:
            rec_port = int(rec_port) if rec_port is not None else None
        except (TypeError, ValueError):
            continue
        if rec_port != parsed.port:
            continue
        rec_host = (params.get("host") or "127.0.0.1").lower()
        rec_routable = (params.get("routable_host") or "").lower()
        # Match: loopback URL against a loopback / 0.0.0.0 bind …
        if host_is_loopback and (rec_host in _LOCALHOST_HOSTS or rec_host == "0.0.0.0"):
            return r.auth_token
        # … or the URL hostname equals the synthesized routable host …
        if rec_routable and host == rec_routable:
            return r.auth_token
        # … or the URL hostname equals the record's explicit bind.
        if rec_host == host:
            return r.auth_token
        # Diagnostic for the LAN-browse case: we found a port-matching
        # record but couldn't tie the URL back to its host. Logged at
        # INFO once per call so operators can see exactly what fields
        # the matcher saw without ever logging the token itself.
        log.info(
            "diloco-proxy: port %d matched record %r but host did "
            "not — url_host=%r rec_host=%r rec_routable=%r "
            "rec_auth_token_present=%s",
            parsed.port,
            r.queue_id,
            host,
            rec_host,
            rec_routable,
            bool(r.auth_token),
        )
    return None


def _auth_headers_for(base: str, request: Request) -> Dict[str, str]:
    """Build the upstream Authorization header dict.

    Precedence: explicit ``X-Diloco-Auth-Token`` (operator override),
    JobRecord auto-lookup, registry lookup, empty.
    """
    override = request.headers.get(_TOKEN_OVERRIDE_HEADER)
    if override:
        return {"authorization": f"Bearer {override}"}
    token = _token_for_local(base)
    if token:
        return {"authorization": f"Bearer {token}"}
    saved = diloco_server_registry.find_token(base)
    if saved:
        return {"authorization": f"Bearer {saved}"}
    return {}


def _verify_for(target: str, base: Optional[str] = None) -> object:
    """Pick the right ``verify=`` for an upstream URL.

    Registry entries with ``verify_tls=False`` short-circuit chain
    validation (used for SSH-tunneled remotes where the upstream cert
    doesn't match the tunnel hostname). Otherwise we defer to
    ``httpx_verify_for_url`` which builds an SSLContext from the
    cluster's CA bundle.
    """
    if base is not None and not diloco_server_registry.find_verify_tls(base):
        return False
    try:
        from forgather.tls import httpx_verify_for_url

        return httpx_verify_for_url(target)
    except ImportError:
        return True


async def _proxy_get(base: str, path: str, request: Request) -> JSONResponse:
    base = _validate_base(base)
    _check_ssrf(base)
    target = base + path
    headers = _auth_headers_for(base, request)
    verify = _verify_for(target, base)
    async with httpx.AsyncClient(timeout=_PROXY_TIMEOUT, verify=verify) as client:
        try:
            r = await client.get(target, headers=headers)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    if r.status_code in (401, 403):
        # Diagnostic: a 401/403 here means the bearer we attached (or
        # didn't) was rejected by upstream. Logging the WHAT (header
        # presence, base URL) helps the operator correlate against
        # the JobRecord state without ever logging the token itself.
        log.info(
            "diloco-proxy upstream %s for %s (bearer attached: %s)",
            r.status_code,
            target,
            "yes" if "authorization" in {k.lower() for k in headers} else "no",
        )
    return JSONResponse(
        status_code=r.status_code,
        content=_safe_json(r),
        headers=_upstream_auth_headers(r.status_code),
    )


@router.get("/diloco/server-status")
async def proxy_status(base: str, request: Request) -> JSONResponse:
    """Forward ``GET <base>/status`` and return the upstream JSON verbatim."""
    return await _proxy_get(base, "/status", request)


@router.get("/diloco/server-info")
async def proxy_info(base: str, request: Request) -> JSONResponse:
    """Forward ``GET <base>/info`` for client-settings negotiation."""
    return await _proxy_get(base, "/info", request)


@router.get("/diloco/known-workers")
async def proxy_known_workers(base: str, request: Request) -> JSONResponse:
    """Forward ``GET <base>/known_workers`` — the roster of every worker
    the server has ever seen (with a per-worker ``running`` flag), so the
    submit UI can offer not-running names for checkpoint-resuming relaunch
    (issue #103)."""
    return await _proxy_get(base, "/known_workers", request)


@router.get("/diloco/work-queues")
async def proxy_work_queues(base: str, request: Request) -> JSONResponse:
    """Forward ``GET <base>/work/queues`` — summary list of active queues."""
    return await _proxy_get(base, "/work/queues", request)


@router.get("/diloco/stats-history")
async def proxy_stats_history(
    base: str, request: Request, max_points: int = 2000
) -> JSONResponse:
    """Forward ``GET <base>/stats_history`` — the aggregate-stats history the
    server logs, for the webui's loss-curve plot. ``max_points`` is passed
    straight through (the upstream downsamples, keeping the latest point)."""
    from urllib.parse import quote, urlencode

    qs = urlencode({"max_points": int(max_points)}, quote_via=quote)
    return await _proxy_get(base, f"/stats_history?{qs}", request)


@router.get("/diloco/work-queue")
async def proxy_work_queue(
    base: str, dataset_id: str, shuffle_seed: int, request: Request
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
    return await _proxy_get(base, f"/work/queue?{qs}", request)


# Set of control actions the DiLoCo server itself recognises. Kept here
# so the proxy refuses unknown actions up-front instead of bouncing them
# off the upstream with a confusing 404.
_CONTROL_ACTIONS = frozenset(
    {
        "save_state",
        "kick_worker",
        "update_optimizer",
        "update_num_workers",
        "command",
        "shutdown",
    }
)


@router.post("/diloco/server-control/{action}")
async def proxy_control(
    action: str,
    base: str,
    request: Request,
    body: Dict[str, Any] = None,
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
    headers = _auth_headers_for(base, request)
    verify = _verify_for(target, base)
    async with httpx.AsyncClient(timeout=_PROXY_TIMEOUT, verify=verify) as client:
        try:
            r = await client.post(target, json=body or {}, headers=headers)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return JSONResponse(
        status_code=r.status_code,
        content=_safe_json(r),
        headers=_upstream_auth_headers(r.status_code),
    )


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
    # Bearer token used by the proxy when forwarding requests upstream.
    # Empty string == server is running --no-auth.
    auth_token: str = ""
    # When False, the proxy skips TLS chain validation for this entry —
    # the escape hatch for SSH-tunneled remotes where the upstream cert
    # doesn't match the tunnel hostname.
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


# ---------------------------------------------------------------------------
# Worker-name generation
# ---------------------------------------------------------------------------


class GenerateWorkerNamesRequest(BaseModel):
    # Number of names to return. Bounded below to reject nonsense and above
    # so a typo ("10000") can't spin the rejection-sampling loop pointlessly.
    count: int = 1
    # Names the caller already has in its pool (stopped workers + already-added
    # new ones). The generated batch is guaranteed disjoint from this set so a
    # second "Generate 4" can't collide with the first.
    exclude: List[str] = []


class GenerateWorkerNamesResponse(BaseModel):
    names: List[str]


# Upper bound on a single batch. Far above any realistic worker count; exists
# only to keep the rejection-sampling loop bounded.
_MAX_GENERATE = 256


@router.post(
    "/diloco/generate-worker-names", response_model=GenerateWorkerNamesResponse
)
def generate_worker_names(req: GenerateWorkerNamesRequest):
    """Return ``count`` memorable, mutually-unique worker names.

    Backs the submit modal's "Generate N workers" control. Names come from
    :func:`forgather.utils.generate_name` (adjective-species, ~100K
    permutations). The returned batch is internally unique and disjoint from
    ``exclude``; rejection sampling is bounded so a request that cannot be
    satisfied (pool exhausted by ``exclude``) fails loudly rather than hanging.
    """
    if req.count < 1 or req.count > _MAX_GENERATE:
        raise HTTPException(
            status_code=400,
            detail=f"count must be between 1 and {_MAX_GENERATE}",
        )
    seen = {s.strip() for s in req.exclude if s and s.strip()}
    names: List[str] = []
    # Generous attempt budget: collisions are rare against a 100K-name pool,
    # but a large ``exclude`` shrinks the effective space, so scale with count.
    max_attempts = max(req.count * 100, 2000)
    for _ in range(max_attempts):
        if len(names) >= req.count:
            break
        candidate = generate_name()
        if candidate in seen:
            continue
        seen.add(candidate)
        names.append(candidate)
    if len(names) < req.count:
        raise HTTPException(
            status_code=409,
            detail=(
                f"could only generate {len(names)} unique name(s) of "
                f"{req.count} requested — the name pool may be exhausted"
            ),
        )
    return GenerateWorkerNamesResponse(names=names)
