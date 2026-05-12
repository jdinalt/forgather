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
Even though every route is auth-gated, the auth token is still a
confused-deputy risk: a stolen / phished token, or an XSS payload running
in an authenticated tab, could otherwise direct this proxy at internal
hosts (cloud metadata services like 169.254.169.254, other LAN boxes,
etc.). To make that exploit useless on a default install, ``_validate_base``
rejects any host that is not literal localhost (``127.0.0.1`` /
``localhost`` / ``::1``). Single-user secure-LAN deployments that
legitimately need a remote vLLM box can opt back in with the
``FORGATHER_INFERENCE_PROXY_ALLOW_REMOTE`` env var (truthy values
``1``/``true``/``yes``); a WARNING is logged for each non-localhost
target so the choice is visible. The check is purely string-based — we
do not resolve DNS — so a hostname that resolves to loopback still
fails. Use the literal addresses if you mean loopback.
"""

from __future__ import annotations

import logging
import os
import time
from threading import Lock
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import job_records

log = logging.getLogger("forgather_server.inference_proxy")

router = APIRouter(tags=["inference-proxy"])

# Upstream connects are local by design (inference servers are all on the
# same host). 10s is plenty for /models and /health; completions responses
# stream — httpx holds the connection open for the duration automatically.
_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)


def _verify_for(target: str) -> object:
    """Pick ``verify=`` for an upstream URL.

    When the inference server runs with TLS (auto-on from the shared
    config), the upstream URL is ``https://`` and httpx must validate
    against the shared CA bundle — otherwise it falls back to the
    system trust store and rejects our self-signed certs with
    ``CERTIFICATE_VERIFY_FAILED``. For plain ``http://`` upstreams we
    short-circuit to ``True`` (no-op).
    """
    try:
        from forgather.tls import httpx_verify_for_url

        return httpx_verify_for_url(target)
    except ImportError:
        return True

# Completion responses can be large. Use a small chunk size so tokens
# reach the browser promptly rather than sitting in an HTTP buffer.
_STREAM_CHUNK = 1024

# SSRF guard: localhost-only by default, opt-in for remote bases. See
# the module docstring for the policy.
_REMOTE_ALLOW_ENV = "FORGATHER_INFERENCE_PROXY_ALLOW_REMOTE"
_LOCALHOST_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


def _remote_allowed() -> bool:
    """Return True iff the operator opted into non-localhost upstreams."""
    return os.environ.get(_REMOTE_ALLOW_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _validate_base(base: str) -> str:
    """Reject obviously-unsafe values before connecting upstream.

    Two layers: scheme allow-list (http/https only — no ``file://`` /
    ``gopher://`` exfiltration tricks) plus an SSRF host allow-list
    pinned to literal localhost. Hostname comparison is string-based;
    DNS is not resolved (see module docstring).
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
    # parsed.hostname returns the bare host with brackets stripped from
    # IPv6 literals ("::1", not "[::1]"). Lowercased for case-insensitive
    # match against the localhost set.
    host = (parsed.hostname or "").lower()
    if host not in _LOCALHOST_HOSTS:
        if _remote_allowed():
            log.warning(
                "inference proxy forwarding to non-localhost host %r "
                "(opt-in via %s)",
                host,
                _REMOTE_ALLOW_ENV,
            )
        else:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"refusing to proxy to non-localhost host: {host!r} "
                    f"(set {_REMOTE_ALLOW_ENV}=1 to allow)"
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

    Returns ``None`` when no JobRecord matches (user pointed the proxy at
    something we didn't spawn) or when the matching record was started
    with ``--no-auth`` (auth_token is None on that record).
    """
    try:
        parsed = urlparse(base)
    except Exception:
        return None
    host = (parsed.hostname or "").lower()
    if host not in _LOCALHOST_HOSTS or parsed.port is None:
        return None
    now = time.monotonic()
    global _token_cache, _token_cache_built_at
    with _token_cache_lock:
        if now - _token_cache_built_at > _TOKEN_CACHE_TTL_S:
            _token_cache = _build_token_index()
            _token_cache_built_at = now
        return _token_cache.get((host, parsed.port))


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

# Tag we attach to upstream auth failures so the webui's global 401
# handler can distinguish "upstream rejected the inference token" from
# "your forgather-server session expired." Without this, a wrong
# inference token would bounce the user to the server-login screen.
_UPSTREAM_AUTH_FAILED_HEADER = "x-upstream-auth-failed"


def _upstream_auth_headers(status: int) -> Dict[str, str]:
    """Tag 401/403 from upstream so clients can distinguish from a
    same-origin session 401. Empty dict on success / non-auth errors."""
    if status in (401, 403):
        return {_UPSTREAM_AUTH_FAILED_HEADER: "1"}
    return {}


def _auth_headers_for(base: str, request: Optional[Request] = None) -> Dict[str, str]:
    """Build the upstream auth header dict.

    Precedence: explicit ``X-Inference-Auth-Token`` from the caller
    (used by the webui's Server-URL panel and any CLI client that
    knows the token), then fall back to JobRecord auto-lookup. Empty
    when neither path produces a token (no record matches and the
    caller didn't pass one — typical for a server running --no-auth).
    """
    if request is not None:
        override = request.headers.get(_TOKEN_OVERRIDE_HEADER)
        if override:
            return {"authorization": f"Bearer {override}"}
    token = _token_for(base)
    if token:
        return {"authorization": f"Bearer {token}"}
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
    async with httpx.AsyncClient(timeout=_TIMEOUT, verify=_verify_for(target)) as client:
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
    async with httpx.AsyncClient(timeout=_TIMEOUT, verify=_verify_for(target)) as client:
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

    client = httpx.AsyncClient(timeout=_TIMEOUT, verify=_verify_for(target))
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
    async with httpx.AsyncClient(timeout=_TIMEOUT, verify=_verify_for(target)) as client:
        try:
            r = await client.post(target, content=body, headers=upstream_headers)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return JSONResponse(
        status_code=r.status_code,
        content=_safe_json(r),
        headers=_upstream_auth_headers(r.status_code),
    )


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
