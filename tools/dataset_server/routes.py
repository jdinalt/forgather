"""
FastAPI route handlers for the forgather dataset server.

All read/write endpoints are gated by an optional bearer-token
dependency built in :mod:`tools.dataset_server.auth`. Two endpoints
are deliberately open:

- ``GET /v1/health`` — for liveness probes.
- ``GET /v1/auth/status`` — so a client can detect whether auth is
  required before attempting a request.

Streaming is plain newline-delimited JSON. Each example becomes one
``json.dumps(...)`` line; the stream ends with an empty chunk and
the connection closes.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import make_verify_bearer
from .hf_cache import list_hf_cache
from .state import LOAD_FIELDS, PolicyError, ServerState
from .wire import to_jsonable

logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"


def build_router(state: ServerState, auth_token: Optional[str]) -> APIRouter:
    """Create the API router. Auth-gated endpoints only get the dep
    when ``auth_token`` is set; with ``--no-auth`` the dep is omitted
    entirely (matches inference_server behavior)."""
    router = APIRouter()
    deps = [Depends(make_verify_bearer(auth_token))] if auth_token else []

    # ----- open endpoints -----

    @router.get("/v1/health")
    async def health():
        return {
            "status": "ok",
            "service": "forgather-dataset-server",
            "version": API_VERSION,
            "policy": {
                "auth_required": state.auth_required,
                "hf_cache_enabled": state.hf_cache_enabled,
                "allow_paths": state.allow_paths,
                "allow_downloads": state.allow_downloads,
                "local_count": len(state.local_datasets),
            },
        }

    @router.get("/v1/auth/status")
    async def auth_status():
        return {"auth_required": state.auth_required}

    # ----- gated endpoints -----

    @router.get("/v1/datasets", dependencies=deps)
    async def list_datasets():
        handles = []
        for h in state.list_handles():
            entry = state.get_entry(h)
            if entry is None:
                continue
            handles.append(
                {
                    "handle": h,
                    "length": _safe_len(entry.backend),
                    "load_args": entry.load_args,
                    "source": entry.source,
                }
            )
        return {"handles": handles}

    @router.get("/v1/datasets/{handle}", dependencies=deps)
    async def get_dataset(handle: str):
        entry = state.get_entry(handle)
        if entry is None:
            raise HTTPException(404, f"Unknown handle: {handle}")
        return {
            "handle": handle,
            "length": _safe_len(entry.backend),
            "load_args": entry.load_args,
            "source": entry.source,
        }

    @router.get("/v1/datasets/{handle}/length", dependencies=deps)
    async def dataset_length(handle: str):
        backend = state.get(handle)
        if backend is None:
            raise HTTPException(404, f"Unknown handle: {handle}")
        length = _safe_len(backend)
        logger.info("GET length handle=%s -> %s", handle, length)
        return {"length": length}

    @router.get("/v1/datasets/{handle}/iter", dependencies=deps)
    async def dataset_iter(
        handle: str,
        seed: Optional[int] = None,
        position: int = 0,
        limit: Optional[int] = None,
    ):
        backend = state.get(handle)
        if backend is None:
            raise HTTPException(404, f"Unknown handle: {handle}")
        logger.info(
            "GET iter handle=%s seed=%s position=%d limit=%s",
            handle,
            seed,
            position,
            limit,
        )
        view = backend
        if seed is not None:
            view = view.shuffle(seed)
        if position:
            view = view.seek(position)

        return StreamingResponse(
            _stream_examples(view, handle, limit),
            media_type="application/x-ndjson",
        )

    @router.post("/v1/load", dependencies=deps)
    async def load_dataset(request: Request):
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"Invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise HTTPException(400, "Request body must be a JSON object")

        load_args: Dict[str, Any] = {k: body.get(k) for k in LOAD_FIELDS if k in body}
        if not load_args.get("path"):
            raise HTTPException(400, "Missing required field 'path'")

        try:
            handle = state.load_on_demand(load_args)
        except PolicyError as exc:
            logger.info(
                "POST load denied (status=%d) args=%s msg=%s",
                exc.status,
                load_args,
                exc.message,
            )
            raise HTTPException(exc.status, exc.message) from exc

        entry = state.get_entry(handle)
        length = _safe_len(entry.backend) if entry is not None else 0
        logger.info(
            "POST load -> handle=%s length=%d source=%s args=%s",
            handle,
            length,
            entry.source if entry else "?",
            load_args,
        )
        return {
            "handle": handle,
            "length": length,
            "load_args": entry.load_args if entry else load_args,
            "source": entry.source if entry else None,
        }

    @router.get("/v1/cache/hf", dependencies=deps)
    async def cache_hf():
        return list_hf_cache()

    @router.get("/v1/local", dependencies=deps)
    async def list_local():
        return {
            "local": [
                {"name": name, "path": path}
                for name, path in sorted(state.local_datasets.items())
            ]
        }

    return router


def _safe_len(backend) -> int:
    try:
        return len(backend)
    except Exception as exc:  # pragma: no cover
        logger.warning("len(backend) failed: %s", exc)
        return -1


def _stream_examples(view, handle: str, limit: Optional[int]):
    """Generator that yields NDJSON-encoded bytes. Caller wraps in
    ``StreamingResponse``."""
    count = 0
    disconnected = False
    try:
        for example in view:
            if limit is not None and count >= limit:
                break
            yield json.dumps(to_jsonable(example)).encode("utf-8") + b"\n"
            count += 1
    except (BrokenPipeError, ConnectionResetError):
        disconnected = True
    if disconnected:
        logger.info(
            "iter handle=%s done: client disconnected after %d examples",
            handle,
            count,
        )
    else:
        logger.info("iter handle=%s done: streamed %d examples", handle, count)


def error_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    """Match the existing ``{"error": "..."}`` body that the PoC server
    used, so existing test assertions and CLI error parsing still work."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
        headers=getattr(exc, "headers", None) or {},
    )
