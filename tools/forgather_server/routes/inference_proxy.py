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
"""

from __future__ import annotations

import logging
from typing import Any, Dict
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger("forgather_server.inference_proxy")

router = APIRouter(tags=["inference-proxy"])

# Upstream connects are local by design (inference servers are all on the
# same host). 10s is plenty for /models and /health; completions responses
# stream — httpx holds the connection open for the duration automatically.
_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)

# Completion responses can be large. Use a small chunk size so tokens
# reach the browser promptly rather than sitting in an HTTP buffer.
_STREAM_CHUNK = 1024


def _validate_base(base: str) -> str:
    """Reject obviously-unsafe values before connecting upstream.

    The proxy is trusted for http/https schemes only — no ``file://`` or
    ``gopher://`` exfiltration tricks. The single-user localhost-first
    deployment doesn't need tighter host restrictions; external vLLM
    instances are a legitimate future use case.
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
    return base.rstrip("/")


def _root_of(base: str) -> str:
    """Health endpoint is mounted at the server root, not under ``/v1`` —
    strip a trailing ``/v1`` segment if present."""
    base = base.rstrip("/")
    if base.endswith("/v1"):
        return base[: -len("/v1")]
    return base


@router.get("/inference/health")
async def proxy_health(base: str) -> JSONResponse:
    """Forward GET ``<base-root>/health``. Returns the upstream JSON as-is.

    Error handling is deliberately two-tiered: upstream reachability
    errors (connection refused, DNS, timeout) map to 502 so the browser
    can distinguish them from upstream application errors (non-2xx
    status), which pass through with their original status.
    """
    target = _root_of(_validate_base(base)) + "/health"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            r = await client.get(target)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return JSONResponse(
        status_code=r.status_code,
        content=_safe_json(r),
    )


@router.get("/inference/models")
async def proxy_models(base: str) -> JSONResponse:
    """Forward GET ``<base>/models``."""
    target = _validate_base(base) + "/models"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            r = await client.get(target)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return JSONResponse(
        status_code=r.status_code,
        content=_safe_json(r),
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

    client = httpx.AsyncClient(timeout=_TIMEOUT)
    # Send our own Content-Type; drop hop-by-hop and origin headers that
    # would confuse the upstream or reflect browser trust scope.
    upstream_headers = {"content-type": "application/json"}
    accept = request.headers.get("accept")
    if accept:
        upstream_headers["accept"] = accept

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
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            r = await client.post(target, content=body, headers=upstream_headers)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return JSONResponse(status_code=r.status_code, content=_safe_json(r))


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
