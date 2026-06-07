"""Agent endpoints: streamed chat + the propose/approve/reject gate.

Transport mirrors the inference proxy: a plain ``StreamingResponse`` with
``media_type="text/event-stream"`` whose body yields ``data: {json}\\n\\n``
frames. The frontend consumes it with ``fetch`` + ``ReadableStream``
(not ``EventSource``), so the session cookie authenticates the request
automatically — no query-string token needed.

Each frame is one *agent event* dict from ``loop.py`` (text / tool_use /
tool_result / action_card / awaiting_approval / usage / done / error /
action_resolved / recorded), plus a leading ``session`` frame so the
client learns its session id before anything else streams.

Endpoints:
- ``POST /api/agent/message``      — send a user message, stream the turn.
- ``POST /api/agent/approve``      — approve a pending action, stream resume.
- ``POST /api/agent/reject``       — reject a pending action, stream resume.
- ``GET  /api/agent/status``       — whether the agent is configured (no secrets).
- ``GET  /api/agent/sessions/{id}``— conversation history.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import agent_profiles_store as profiles_store
from .. import agent_tls
from ..agent import runtime, session as agent_session

log = logging.getLogger("forgather_server.agent")
router = APIRouter(tags=["agent"])


def _sse(event: Dict) -> bytes:
    return f"data: {json.dumps(event)}\n\n".encode("utf-8")


async def _stream(events: AsyncIterator[Dict], *, session_id: Optional[str] = None) -> StreamingResponse:
    async def body() -> AsyncIterator[bytes]:
        if session_id is not None:
            yield _sse({"type": "session", "session_id": session_id})
        try:
            async for ev in events:
                yield _sse(ev)
        except Exception as e:  # surface as a final error frame, never 500 mid-stream
            log.exception("agent stream failed")
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(body(), media_type="text/event-stream")


class MessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class DecisionRequest(BaseModel):
    action_id: str


@router.get("/agent/status")
def agent_status():
    return runtime.status()


@router.post("/agent/message")
async def agent_message(req: MessageRequest):
    if not runtime.is_enabled():
        raise HTTPException(status_code=503, detail="agent is not configured")
    loop = runtime.get_loop()
    conv = agent_session.get_or_create(req.session_id)
    return await _stream(
        loop.run_user_message(conv, req.message), session_id=conv.session_id
    )


@router.post("/agent/approve")
async def agent_approve(req: DecisionRequest):
    if not runtime.is_enabled():
        raise HTTPException(status_code=503, detail="agent is not configured")
    loop = runtime.get_loop()
    return await _stream(loop.apply_decision(req.action_id, approve=True))


@router.post("/agent/reject")
async def agent_reject(req: DecisionRequest):
    if not runtime.is_enabled():
        raise HTTPException(status_code=503, detail="agent is not configured")
    loop = runtime.get_loop()
    return await _stream(loop.apply_decision(req.action_id, approve=False))


# ---- profiles ------------------------------------------------------------


class ProfileModel(BaseModel):
    """Profile as returned to the webui — credentials redacted to flags."""

    id: str
    label: str
    provider: str
    model: str
    base_url: str
    api_key_env: str
    verify_tls: bool
    has_api_key: bool
    has_imported_cert: bool
    max_tokens: int
    max_iterations: int


class ProfileWrite(BaseModel):
    label: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    # Empty string clears the stored key/cert; omitted (None) leaves it.
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    verify_tls: Optional[bool] = None
    ca_cert_pem: Optional[str] = None
    max_tokens: Optional[int] = None
    max_iterations: Optional[int] = None


def _to_model(p) -> ProfileModel:
    return ProfileModel(
        id=p.id,
        label=p.label,
        provider=p.provider,
        model=p.model,
        base_url=p.base_url,
        api_key_env=p.api_key_env,
        verify_tls=p.verify_tls,
        has_api_key=bool(p.api_key),
        has_imported_cert=bool(p.ca_cert_pem),
        max_tokens=p.max_tokens,
        max_iterations=p.max_iterations,
    )


@router.get("/agent/profiles")
def list_profiles():
    return {
        "active_id": profiles_store.get_active_id(),
        "profiles": [_to_model(p) for p in profiles_store.list_profiles()],
    }


@router.post("/agent/profiles", response_model=ProfileModel)
def create_profile(req: ProfileWrite):
    kwargs = {k: v for k, v in req.dict().items() if v is not None}
    if not kwargs.get("label"):
        raise HTTPException(status_code=400, detail="label is required")
    try:
        p = profiles_store.add_profile(**kwargs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _to_model(p)


@router.put("/agent/profiles/{profile_id}", response_model=ProfileModel)
def update_profile(profile_id: str, req: ProfileWrite):
    kwargs = {k: v for k, v in req.dict().items() if v is not None}
    try:
        p = profiles_store.update_profile(profile_id, **kwargs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if p is None:
        raise HTTPException(status_code=404, detail=f"no such profile: {profile_id}")
    return _to_model(p)


@router.delete("/agent/profiles/{profile_id}")
def delete_profile(profile_id: str):
    removed = profiles_store.remove_profile(profile_id)
    if removed is None:
        raise HTTPException(status_code=404, detail=f"no such profile: {profile_id}")
    return {"removed": removed.id, "active_id": profiles_store.get_active_id()}


@router.post("/agent/profiles/{profile_id}/activate")
def activate_profile(profile_id: str):
    if not profiles_store.set_active(profile_id):
        raise HTTPException(status_code=404, detail=f"no such profile: {profile_id}")
    return {"active_id": profile_id}


# ---- model listing + cert import -----------------------------------------


class ModelsRequest(BaseModel):
    """Connection to query for available models.

    For an unsaved profile the editor sends the fields directly. For a saved
    profile it can send ``profile_id`` and omit the credentials — the stored
    key / TLS settings fill in.
    """

    profile_id: Optional[str] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    verify_tls: Optional[bool] = None
    ca_cert_pem: Optional[str] = None


@router.post("/agent/models")
def list_agent_models(req: ModelsRequest):
    saved = profiles_store.get_profile(req.profile_id) if req.profile_id else None

    def pick(field: str, default: Any):
        v = getattr(req, field)
        if v is not None:
            return v
        if saved is not None:
            return getattr(saved, field)
        return default

    provider = pick("provider", "anthropic")
    base_url = pick("base_url", "")

    # Resolve the key: explicit > saved profile's key/env > in-editor env.
    # runtime.resolve_credential applies the high-value-key guard (never
    # auto-send ANTHROPIC_API_KEY to a custom base_url).
    api_key = req.api_key
    if not api_key and saved is not None:
        api_key = runtime.resolve_credential(saved.api_key, saved.api_key_env, base_url)
    if not api_key:
        api_key = runtime.resolve_credential(None, req.api_key_env, base_url)

    try:
        # Model discovery is a low-stakes probe (it returns only model ids),
        # so always skip TLS verification here — it removes a class of
        # friction (self-signed cert not yet imported) without exposing
        # anything sensitive. The actual chat connection still honors the
        # profile's real TLS posture.
        models = agent_tls.list_models(
            provider=provider,
            base_url=base_url or "",
            api_key=api_key or "",
            verify_tls=False,
            ca_cert_pem="",
        )
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code in (401, 403):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"the model server returned {code} — it requires a bearer "
                    "token. Enter the server's API key / token in the API key "
                    "field (vLLM's is in ~/.config/vllm/api-key)."
                ),
            )
        raise HTTPException(status_code=502, detail=f"upstream {code}: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return {"models": models}


class FetchCertRequest(BaseModel):
    base_url: str


@router.post("/agent/fetch-cert")
def fetch_cert(req: FetchCertRequest):
    """Retrieve the server's TLS certificate (PEM + fingerprint) for review.

    Trust-on-first-use import flow: the webui shows the fingerprint, and on
    confirmation saves the returned ``pem`` into the profile's ca_cert_pem.
    """
    try:
        return agent_tls.fetch_server_cert(req.base_url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")


@router.get("/agent/sessions/{session_id}")
def agent_session_history(session_id: str):
    conv = agent_session.get_conversation(session_id)
    if conv is None:
        raise HTTPException(status_code=404, detail=f"no such session: {session_id}")
    return {
        "session_id": conv.session_id,
        "messages": conv.messages,
        "awaiting_approval": conv.pending_turn is not None,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
    }
