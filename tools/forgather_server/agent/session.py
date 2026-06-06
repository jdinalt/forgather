"""In-memory agent session state.

Modeled on ``scheduler._state`` (a module-level singleton dataclass +
``Lock``, exposed via ``get_state()``): single-user localhost, ephemeral
by design. Conversation history and pending approvals are NOT persisted —
losing them on restart is acceptable, and a stale ``action_id`` minted
before a restart must not survive it (the previewed change could no longer
match disk). ``approve``/``reject`` of a missing id therefore 404s.

If conversation persistence is ever wanted, follow the per-key-JSON
pattern in ``overrides_store`` under a new ``paths`` helper — but that is
explicitly out of scope for the first build.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Set

from .registry import Proposal


@dataclass
class PendingApproval:
    """One previewed change awaiting the user's decision."""

    action_id: str
    session_id: str
    tool_use_id: str  # the assistant tool_use block this resolves
    tool_name: str
    risk: str
    proposal: Proposal
    created_at: float = field(default_factory=time.time)


@dataclass
class PendingTurn:
    """An assistant turn paused mid-flight waiting on approval(s).

    Every tool_use block in ``assistant_message`` must get a tool_result
    before the provider can be called again, so we hold the already-computed
    results (read tools) here and fill in the rest as approvals resolve.
    The turn resumes only when ``outstanding`` is empty.
    """

    assistant_message: Dict[str, Any]
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    outstanding: Set[str] = field(default_factory=set)


@dataclass
class Conversation:
    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    pending_turn: Optional[PendingTurn] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()


@dataclass
class AgentState:
    sessions: Dict[str, Conversation] = field(default_factory=dict)
    pending: Dict[str, PendingApproval] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)


_state = AgentState()


def get_state() -> AgentState:
    return _state


def new_session_id() -> str:
    return "sess_" + secrets.token_hex(8)


def new_action_id() -> str:
    return "act_" + secrets.token_hex(8)


def get_or_create(session_id: Optional[str]) -> Conversation:
    """Return the conversation for ``session_id``, creating one if needed.

    A ``None`` or unknown id mints a fresh conversation so the client can
    start without a round-trip to allocate an id first.
    """
    with _state._lock:
        if session_id and session_id in _state.sessions:
            return _state.sessions[session_id]
        sid = session_id or new_session_id()
        conv = Conversation(session_id=sid)
        _state.sessions[sid] = conv
        return conv


def get_conversation(session_id: str) -> Optional[Conversation]:
    with _state._lock:
        return _state.sessions.get(session_id)


def register_pending(approval: PendingApproval) -> None:
    with _state._lock:
        _state.pending[approval.action_id] = approval


def pop_pending(action_id: str) -> Optional[PendingApproval]:
    with _state._lock:
        return _state.pending.pop(action_id, None)
