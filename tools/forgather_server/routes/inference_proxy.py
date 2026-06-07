"""Browser → inference-server proxy.

The webui can't talk to spawned inference servers directly without
running into the browser's cross-origin policy. Rather than try to fix
every possible failure mode at the browser layer (CORS, Private Network
Access, extension blocks, mixed content), we add a tiny same-origin
proxy here. The frontend makes every call to ``/api/inference/*`` on the
forgather-server; this module forwards to whichever upstream URL the
caller names.

Kept simple on purpose: one route per upstream endpoint we surface
(``models`` / ``completions`` / ``chat/completions`` / ``health``), each
with a dedicated response shape so the routing table documents the
feature. Streaming is byte-for-byte passthrough so the Server-Sent
Events framing the client expects flows through unchanged.

SSRF policy
-----------
Default: any URL the operator types into the panel is allowed.
forgather is a single-user research tool; the same auth token that
gates this endpoint also gates training-job submission, which is
already arbitrary code execution on the host. An "SSRF guard"
layered on top of that adds friction without adding security — an
authenticated attacker who could exploit this proxy could just as
easily exfiltrate cloud-metadata creds (or anything else) by
submitting a training job that shells out.

Operators who genuinely want stricter posture (e.g. running
forgather in an environment with non-operator-controlled clients)
pass ``--lock-inference-proxy`` to ``forgather server``;
``_validate_base`` then rejects any non-localhost upstream.

The scheme allow-list stays unconditionally: only ``http`` / ``https``
through this proxy, so ``file://`` and similar exfiltration vectors
are off the table regardless of the lock setting.
"""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .. import auth as auth_mod
from .. import cluster_inference_inventory, inference_server_registry, job_records

log = logging.getLogger("forgather_server.inference_proxy")

router = APIRouter(tags=["inference-proxy"])

# Upstream connects are local by design (inference servers are all on the
# same host). 10s is plenty for /models and /health; completions responses
# stream — httpx holds the connection open for the duration automatically.
_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)


def _verify_for(target: str, base: Optional[str] = None) -> object:
    """Pick ``verify=`` for an upstream URL.

    When the inference server runs with TLS (auto-on from the shared
    config), the upstream URL is ``https://`` and httpx must validate
    against the shared CA bundle — otherwise it falls back to the
    system trust store and rejects our self-signed certs with
    ``CERTIFICATE_VERIFY_FAILED``. For plain ``http://`` upstreams we
    short-circuit to ``True`` (no-op).

    If ``base`` matches a user-registered entry whose ``verify_tls``
    is ``False``, returns ``False`` — chain validation is off and
    the upstream cert is trusted purely on the operator's say-so
    (used for SSH-tunneled remotes where the upstream cert doesn't
    match the tunnel's local hostname).
    """
    if base is not None and not inference_server_registry.find_verify_tls(base):
        return False
    try:
        from forgather.tls import httpx_verify_for_url

        return httpx_verify_for_url(target)
    except ImportError:
        return True


# Completion responses can be large. Use a small chunk size so tokens
# reach the browser promptly rather than sitting in an HTTP buffer.
_STREAM_CHUNK = 1024

_LOCALHOST_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

# Set to True by `forgather server --lock-inference-proxy` to restrict
# the proxy to localhost upstreams. Default off: forgather is a
# single-user research tool and the operator already has full RCE via
# training-job submission, so SSRF adds no real capability. The flag
# exists for the (rare) case of running forgather in an environment
# with non-operator-controlled clients.
LOCK_TO_LOCALHOST = False


def _validate_base(base: str) -> str:
    """Reject obviously-unsafe values before connecting upstream.

    Scheme allow-list (http/https only — no ``file://`` / ``gopher://``
    exfiltration tricks). Host allow-list is empty by default: the
    operator types the URL into the panel, the operator is the one
    using forgather, the operator already has full RCE on the host
    via training-job submission — an "SSRF guard" on top of that adds
    friction without security. Pass ``--lock-inference-proxy`` to the
    server to switch to strict-localhost-only mode.
    """
    try:
        parsed = urlparse(base)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad base url: {e}")
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            status_code=400,
            detail=f"unsupported scheme: {parsed.scheme!r}",
        )
    if not parsed.netloc:
        raise HTTPException(status_code=400, detail="missing host")

    if LOCK_TO_LOCALHOST:
        # Lowercased for case-insensitive match. parsed.hostname strips
        # brackets from IPv6 literals, so "[::1]" arrives here as "::1".
        host = (parsed.hostname or "").lower()
        if host not in _LOCALHOST_HOSTS:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"inference proxy is locked to localhost; "
                    f"refusing to proxy to {host!r}. Restart the server "
                    "without --lock-inference-proxy to allow remote "
                    "upstreams."
                ),
            )

    return base.rstrip("/")


# JobRecord lookup is hit on every proxy request — cache the (host, port)
# -> token index for a few seconds to avoid re-reading job_records.json on
# the hot path. Stale entries are tolerable: a stopped inference job's
# token doesn't help an attacker (the upstream is gone), and a freshly
# spawned job's token is still discoverable on the next miss.
_TOKEN_CACHE_TTL_S = 5.0
_token_cache: Dict[tuple, Optional[str]] = {}
_token_cache_built_at: float = 0.0
_token_cache_lock = Lock()


def _build_token_index() -> Dict[tuple, Optional[str]]:
    """Build (host_str, port) -> auth_token map from running inference records.

    ``host_str`` is normalized to the same loopback aliases the SSRF guard
    accepts so a base URL of ``http://127.0.0.1:8137`` finds a record that
    was spawned on ``localhost``. We don't resolve DNS — string equality
    only — and we deliberately only consider loopback hosts so an off-host
    record can't poison the index.
    """
    index: Dict[tuple, Optional[str]] = {}
    for r in job_records.list_records():
        if r.job_type != "inference":
            continue
        if r.status not in {"starting", "running"}:
            continue
        port = r.job_params.get("port") if r.job_params else None
        if port is None:
            continue
        try:
            port = int(port)
        except (TypeError, ValueError):
            continue
        host = (r.job_params.get("host") or "127.0.0.1").lower()
        if host not in _LOCALHOST_HOSTS:
            continue
        # Mirror the token under every loopback alias so any of them maps
        # to the same record.
        for alias in _LOCALHOST_HOSTS:
            index[(alias, port)] = r.auth_token
    return index


def _token_for(base: str) -> Optional[str]:
    """Look up the bearer token of the inference job listening on ``base``.

    Two-tier lookup:

      1. **Local loopback fast path** — when ``base`` is a localhost
         URL, scan this peer's JobRecords. Cheap, cached, and the only
         path needed on single-node setups.
      2. **Master inventory** — for off-host URLs, consult the cluster
         inference inventory. This path only finds a token on the
         master node (which holds the aggregated snapshot); non-master
         peers return ``None`` and rely on the webui to pass the token
         via the ``X-Inference-Auth-Token`` header (the picker has it,
         since :class:`ClusterInferenceServerModel` includes the
         token). Future work could add a non-master pull loop so the
         proxy auto-attaches on every node, but the current shape
         matches today's local-jobs behavior and keeps the surface
         narrow.

    The ``X-Inference-Auth-Token`` header takes precedence over both —
    see :func:`_auth_headers_for`.
    """
    try:
        parsed = urlparse(base)
    except Exception:
        return None
    if parsed.port is None:
        return None
    host = (parsed.hostname or "").lower()
    if host in _LOCALHOST_HOSTS:
        now = time.monotonic()
        global _token_cache, _token_cache_built_at
        with _token_cache_lock:
            if now - _token_cache_built_at > _TOKEN_CACHE_TTL_S:
                _token_cache = _build_token_index()
                _token_cache_built_at = now
            return _token_cache.get((host, parsed.port))
    # Off-host: consult the cluster inventory. On master nodes this
    # has the full picture; on non-master nodes it's empty and the
    # caller must rely on the ``X-Inference-Auth-Token`` header.
    return cluster_inference_inventory.master_inventory.token_for_url(base)


def _root_of(base: str) -> str:
    """Health endpoint is mounted at the server root, not under ``/v1`` —
    strip a trailing ``/v1`` segment if present."""
    base = base.rstrip("/")
    if base.endswith("/v1"):
        return base[: -len("/v1")]
    return base


# Header the webui sends to pin the upstream token explicitly. Two
# reasons: (a) external upstreams (vLLM, remote inference) aren't in
# the auto-lookup index but still need a token, and (b) the webui
# already shows the token in its Server-URL panel for local servers
# so it's the natural source of truth for both cases. We use a
# dedicated header rather than ``Authorization`` because the user's
# Authorization header is the *forgather-server's* bearer and must
# not leak past the proxy.
_TOKEN_OVERRIDE_HEADER = "x-inference-auth-token"

# Header naming the exact user-registry entry the webui selected. When
# present, the proxy attaches *that entry's* token (or none) and does NOT
# fall back to URL-based lookup — see _auth_headers_for. This keeps the
# token server-side (the browser only holds the id) while letting two
# entries for the same base_url carry independent auth, and makes the UI's
# auth indicator authoritative. See issue #158.
_SERVER_ID_HEADER = "x-inference-server-id"

# Tag we attach to upstream auth failures so the webui's global 401
# handler can distinguish "upstream rejected the inference token" from
# "your forgather-server session expired." Without this, a wrong
# inference token would bounce the user to the server-login screen.
_UPSTREAM_AUTH_FAILED_HEADER = "x-upstream-auth-failed"


def _upstream_auth_headers(status: int) -> Dict[str, str]:
    """Tag 401/403 from upstream so clients can distinguish from a
    same-origin session 401. Empty dict on success / non-auth errors.

    The 403 case is harmless overhead now that the webui no longer
    treats 403 as a reauth signal (see ``webui/src/auth.ts``); kept
    for symmetry in case a future client wants to reuse this header
    for upstream-403 detection.
    """
    if status in (401, 403):
        return {_UPSTREAM_AUTH_FAILED_HEADER: "1"}
    return {}


def _auth_headers_for(base: str, request: Optional[Request] = None) -> Dict[str, str]:
    """Build the upstream auth header dict.

    Precedence:
      1. explicit ``X-Inference-Auth-Token`` from the caller (the token
         itself — used by the cluster picker and CLI clients that hold it);
      2. ``X-Inference-Server-Id`` — entry-bound: attach exactly that
         registry entry's token (or none), with NO URL fallback, so two
         entries sharing a base_url stay independent and a "no auth" entry
         never inherits another entry's token (issue #158);
      3. JobRecord / cluster auto-lookup by URL (locally-spawned + cluster
         servers that weren't named by id);
      4. user-registry lookup by exact URL (a raw URL the user typed);
      5. empty (server is running --no-auth).
    """
    if request is not None:
        override = request.headers.get(_TOKEN_OVERRIDE_HEADER)
        if override:
            return {"authorization": f"Bearer {override}"}
        server_id = request.headers.get(_SERVER_ID_HEADER)
        if server_id:
            entry = inference_server_registry.find_by_id(server_id)
            if entry is not None and entry.auth_token:
                return {"authorization": f"Bearer {entry.auth_token}"}
            # Selected entry exists with no token, or was removed: send
            # nothing. The selection is authoritative — do not guess a token
            # from the URL.
            return {}
    token = _token_for(base)
    if token:
        return {"authorization": f"Bearer {token}"}
    saved = inference_server_registry.find_token(base)
    if saved:
        return {"authorization": f"Bearer {saved}"}
    return {}


@router.get("/inference/health")
async def proxy_health(base: str, request: Request) -> JSONResponse:
    """Forward GET ``<base-root>/health``. Returns the upstream JSON as-is.

    Error handling is deliberately two-tiered: upstream reachability
    errors (connection refused, DNS, timeout) map to 502 so the browser
    can distinguish them from upstream application errors (non-2xx
    status), which pass through with their original status.

    Health is open on the inference server (no auth required) so we don't
    bother adding the bearer header here — but we do anyway for
    consistency in case future health endpoints become auth-gated.
    """
    target = _root_of(_validate_base(base)) + "/health"
    headers = _auth_headers_for(base, request)
    async with httpx.AsyncClient(
        timeout=_TIMEOUT, verify=_verify_for(target, base=base)
    ) as client:
        try:
            r = await client.get(target, headers=headers or None)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return JSONResponse(
        status_code=r.status_code,
        content=_safe_json(r),
        headers=_upstream_auth_headers(r.status_code),
    )


@router.get("/inference/models")
async def proxy_models(base: str, request: Request) -> JSONResponse:
    """Forward GET ``<base>/models``."""
    target = _validate_base(base) + "/models"
    headers = _auth_headers_for(base, request)
    async with httpx.AsyncClient(
        timeout=_TIMEOUT, verify=_verify_for(target, base=base)
    ) as client:
        try:
            r = await client.get(target, headers=headers or None)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return JSONResponse(
        status_code=r.status_code,
        content=_safe_json(r),
        headers=_upstream_auth_headers(r.status_code),
    )


async def _proxy_streaming_post(
    base: str, upstream_path: str, request: Request
) -> StreamingResponse:
    """Forward POST ``<base><upstream_path>`` with streaming passthrough.

    The caller submits the OpenAI-compatible JSON body; we forward it
    verbatim. The upstream response is streamed byte-for-byte so the
    Server-Sent Events framing (``data: {...}\\n\\n``, ``data: [DONE]``)
    reaches the browser exactly as the inference server emits it.

    Shared by ``/inference/completions`` and ``/inference/chat/completions``
    — both upstream endpoints have identical request/response transport
    semantics.
    """
    target = _validate_base(base) + upstream_path
    body = await request.body()

    client = httpx.AsyncClient(timeout=_TIMEOUT, verify=_verify_for(target, base=base))
    # Send our own Content-Type; drop hop-by-hop and origin headers that
    # would confuse the upstream or reflect browser trust scope. We also
    # drop the user's Authorization (if any) and re-add a per-job token
    # the proxy looked up — see _token_for. The browser is auth'd to the
    # forgather-server, not the inference upstream; a user-supplied token
    # should never leak past us.
    upstream_headers = {"content-type": "application/json"}
    accept = request.headers.get("accept")
    if accept:
        upstream_headers["accept"] = accept
    upstream_headers.update(_auth_headers_for(base, request))

    try:
        req = client.build_request(
            "POST", target, content=body, headers=upstream_headers
        )
        response = await client.send(req, stream=True)
    except httpx.RequestError as e:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    except Exception:
        # Anything else (bad URL parsing, runtime errors, etc.) still has
        # to release the client's connection pool — without this the
        # AsyncClient lingers until GC, leaking sockets per request.
        await client.aclose()
        raise

    if response.status_code >= 400:
        # Surface the upstream error body in one shot — no streaming
        # needed for error responses.
        try:
            text = await response.aread()
        finally:
            await response.aclose()
            await client.aclose()
        return StreamingResponse(
            iter([text]),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
            headers=_upstream_auth_headers(response.status_code),
        )

    async def body_iter():
        try:
            async for chunk in response.aiter_raw(chunk_size=_STREAM_CHUNK):
                if chunk:
                    yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    media_type = response.headers.get("content-type", "text/event-stream")
    return StreamingResponse(body_iter(), media_type=media_type)


@router.post("/inference/completions")
async def proxy_completions(base: str, request: Request) -> StreamingResponse:
    """Forward POST ``<base>/completions``."""
    return await _proxy_streaming_post(base, "/completions", request)


@router.post("/inference/chat/completions")
async def proxy_chat_completions(base: str, request: Request) -> StreamingResponse:
    """Forward POST ``<base>/chat/completions``."""
    return await _proxy_streaming_post(base, "/chat/completions", request)


@router.post("/inference/tokenize")
async def proxy_tokenize(base: str, request: Request) -> JSONResponse:
    """Forward POST ``<base-root>/tokenize`` (vLLM-compatible).

    vLLM serves /tokenize at the server root rather than under /v1, so
    strip a trailing /v1 from the configured base before appending the
    path. Non-streaming JSON pass-through; this endpoint never returns
    SSE.
    """
    target = _root_of(_validate_base(base)) + "/tokenize"
    body = await request.body()
    upstream_headers = {"content-type": "application/json"}
    upstream_headers.update(_auth_headers_for(base, request))
    async with httpx.AsyncClient(
        timeout=_TIMEOUT, verify=_verify_for(target, base=base)
    ) as client:
        try:
            r = await client.post(target, content=body, headers=upstream_headers)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return JSONResponse(
        status_code=r.status_code,
        content=_safe_json(r),
        headers=_upstream_auth_headers(r.status_code),
    )


@router.post("/inference/detokenize")
async def proxy_detokenize(base: str, request: Request) -> JSONResponse:
    """Forward POST ``<base-root>/detokenize`` (vLLM-compatible).

    Used by the chat panel's "To completion" button to recover a
    byte-accurate rendered prompt against vLLM, whose /tokenize does
    not include the rendered text. Webui calls /tokenize to get token
    ids, then /detokenize to turn those back into a string. Same
    /v1-stripping rule as /tokenize.
    """
    target = _root_of(_validate_base(base)) + "/detokenize"
    body = await request.body()
    upstream_headers = {"content-type": "application/json"}
    upstream_headers.update(_auth_headers_for(base, request))
    async with httpx.AsyncClient(
        timeout=_TIMEOUT, verify=_verify_for(target, base=base)
    ) as client:
        try:
            r = await client.post(target, content=body, headers=upstream_headers)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return JSONResponse(
        status_code=r.status_code,
        content=_safe_json(r),
        headers=_upstream_auth_headers(r.status_code),
    )


# ---------------------------------------------------------------------------
# User-added inference-server registry
#
# The Inference → Model picker lists *spawned* and *cluster* inference jobs,
# but operators often also dial external OpenAI-compatible servers (vLLM, a
# teammate's box, an external provider) that aren't in any cluster
# inventory. This small registry mirrors the dataset_server one: a JSON-
# persisted list of (label, base_url, auth_token, verify_tls) entries the
# webui can CRUD via these routes. Token resolution wires into the proxy's
# existing chain — see ``_auth_headers_for``.
#
# Intentionally node-local: entries do NOT aggregate across cluster peers,
# unlike the dataset_server registry which feeds into the cluster dataset
# inventory. Hence no ``_wake_cluster_inventory()`` calls on mutation — the
# inference cluster inventory only tracks spawned jobs, not external URLs
# any one operator has bookmarked on their own node.
# ---------------------------------------------------------------------------


class UserEntryModel(BaseModel):
    id: str
    label: str
    base_url: str
    has_auth_token: bool
    # ``False`` means outbound calls to this URL skip TLS chain +
    # hostname validation. Default ``True`` (secure-by-default).
    verify_tls: bool = True


class AddUserEntryRequest(BaseModel):
    label: str = ""
    base_url: str
    auth_token: str = ""
    # Operator-asserted "I trust this channel for other reasons" —
    # used for SSH-tunneled or otherwise out-of-band-secured
    # upstreams whose cert won't validate against the local CA.
    verify_tls: bool = True


@router.get("/inference-servers/user", response_model=List[UserEntryModel])
def list_user_entries():
    return [
        UserEntryModel(
            id=e.id,
            label=e.label,
            base_url=e.base_url,
            has_auth_token=bool(e.auth_token),
            verify_tls=e.verify_tls,
        )
        for e in inference_server_registry.list_entries()
    ]


@router.post("/inference-servers/user", response_model=UserEntryModel)
def add_user_entry(req: AddUserEntryRequest):
    """Add an external inference-server URL to this node's user registry.

    **Pure database operation** — this handler validates only the URL
    format and persists the entry. It does NOT probe the target. The
    operator validates with the Server-URL panel's "Test" button after
    the fact.
    """
    base_url = (req.base_url or "").strip()
    if not base_url:
        raise HTTPException(status_code=400, detail="base_url is required")
    try:
        parsed = urlparse(base_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad base_url: {e}")
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            status_code=400,
            detail=f"unsupported scheme: {parsed.scheme!r}",
        )
    if not parsed.netloc:
        raise HTTPException(status_code=400, detail="bad base_url: missing host")
    try:
        entry = inference_server_registry.add_entry(
            label=req.label,
            base_url=base_url,
            auth_token=(req.auth_token or "").strip(),
            verify_tls=bool(req.verify_tls),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return UserEntryModel(
        id=entry.id,
        label=entry.label,
        base_url=entry.base_url,
        has_auth_token=bool(entry.auth_token),
        verify_tls=entry.verify_tls,
    )


@router.delete("/inference-servers/user/{entry_id}")
def delete_user_entry(entry_id: str):
    removed = inference_server_registry.remove_entry(entry_id)
    if removed is None:
        raise HTTPException(status_code=404, detail=f"no entry: {entry_id}")
    return {"removed": removed.id}


class UserEntryTokenResponse(BaseModel):
    """Raw bearer token for a user-added inference-server registry entry.

    The point of this endpoint is to let the operator lift the token
    out for use by external clients (curl, opencode, OpenAI SDKs, etc.)
    without dropping into the on-disk registry file by hand. The
    listing endpoint deliberately reports ``has_auth_token`` only;
    callers that actually need the secret must hit this dedicated
    route, which mirrors the dataset-server's ``/bundle`` shape and
    its demo-mode gate.
    """

    auth_token: str


@router.get(
    "/inference-servers/user/{entry_id}/token",
    response_model=UserEntryTokenResponse,
)
def get_user_entry_token(entry_id: str):
    """Reveal the stored bearer token for a user-added entry.

    Refused in demo mode: the response body is the bare token, which
    ``redact_sensitive_in_demo`` doesn't catch (the key is
    ``auth_token`` but the field carries the raw secret), so the
    only correct demo behaviour is to refuse the endpoint entirely.
    """
    if auth_mod.demo_mode_enabled():
        raise HTTPException(
            status_code=403,
            detail="token reveal is disabled in read-only demo mode",
        )
    for e in inference_server_registry.list_entries():
        if e.id == entry_id:
            return UserEntryTokenResponse(auth_token=e.auth_token or "")
    raise HTTPException(status_code=404, detail=f"no entry: {entry_id}")


def _safe_json(r: httpx.Response) -> Dict[str, Any]:
    """Return parsed JSON, or an error envelope preserving the raw body.

    Upstream error responses are often plain text (e.g. a uvicorn 500
    page), and the caller can still display that usefully — so we don't
    fail hard on non-JSON.
    """
    try:
        return r.json()
    except ValueError:
        return {"error": "non-json response from upstream", "body": r.text}
