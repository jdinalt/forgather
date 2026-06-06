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
from typing import AsyncIterator, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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
