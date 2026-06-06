"""Persistent registry of AI-agent connection profiles.

A profile is one named way to reach a model: provider, model, base_url,
credentials, and TLS posture. The webui manages several and switches the
*active* one at runtime — no server restart (see
:mod:`forgather_server.agent.runtime`, which rebuilds its loop whenever the
store's revision changes).

Mirrors :mod:`inference_server_registry` (JSON list, ``_lock``, atomic
0600 writes, stable 8-hex ids) — the file holds API keys / bearer tokens,
so it is mode 0600 like the other credential-bearing registries. Layout:

    {
      "active_id": "ab12cd34" | null,
      "revision": <int>,          # bumped on every write; runtime cache key
      "profiles": [ {<AgentProfile fields>}, ... ]
    }

The server-config ``agent:`` block is only a *bootstrap*: on first run, if
the store is empty, :func:`seed_if_empty` creates one profile from it. The
store is the source of truth thereafter.
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import asdict, dataclass, field, fields
from threading import Lock
from typing import Any, Dict, List, Optional

from ._atomic import atomic_write_text
from .paths import agent_profiles_file

log = logging.getLogger("forgather_server.agent_profiles")

_lock = Lock()

DEFAULT_PROVIDER = "anthropic"
DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_ITERATIONS = 12


@dataclass
class AgentProfile:
    id: str
    label: str
    provider: str = DEFAULT_PROVIDER
    # Empty model => "weakly bound": resolve from the server's model list at
    # activation time (pick the first available). vLLM serves one model, so
    # this auto-tracks a model swap on the box.
    model: str = ""
    # Empty base_url => Claude (api.anthropic.com via the SDK default).
    base_url: str = ""
    # Explicit key/bearer; empty => read api_key_env from the environment.
    api_key: str = ""
    api_key_env: str = DEFAULT_API_KEY_ENV
    # TLS posture for an https base_url:
    #   verify_tls True  + ca_cert_pem ""  -> system trust (public CA)
    #   verify_tls True  + ca_cert_pem set -> trust that imported cert
    #                                         (LAN self-signed; hostname check off)
    #   verify_tls False                    -> skip verification (accept any cert)
    verify_tls: bool = True
    ca_cert_pem: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_iterations: int = DEFAULT_MAX_ITERATIONS


_PROFILE_FIELDS = {f.name for f in fields(AgentProfile)}


@dataclass
class _StoreState:
    active_id: Optional[str] = None
    revision: int = 0
    profiles: List[AgentProfile] = field(default_factory=list)


def _read_state() -> _StoreState:
    path = agent_profiles_file()
    if not path.exists():
        return _StoreState()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        log.warning("agent_profiles.json unreadable; starting empty")
        return _StoreState()
    if not isinstance(data, dict):
        return _StoreState()
    profiles: List[AgentProfile] = []
    for item in data.get("profiles") or []:
        if not isinstance(item, dict) or "id" not in item:
            continue
        kept = {k: v for k, v in item.items() if k in _PROFILE_FIELDS}
        kept.setdefault("label", str(item.get("id")))
        profiles.append(AgentProfile(**kept))
    return _StoreState(
        active_id=data.get("active_id"),
        revision=int(data.get("revision", 0)),
        profiles=profiles,
    )


def _write_state(state: _StoreState) -> None:
    # Drop a dangling active_id (e.g. the active profile was removed).
    ids = {p.id for p in state.profiles}
    if state.active_id not in ids:
        state.active_id = state.profiles[0].id if state.profiles else None
    state.revision += 1
    payload = json.dumps(
        {
            "active_id": state.active_id,
            "revision": state.revision,
            "profiles": [asdict(p) for p in state.profiles],
        },
        indent=2,
    )
    atomic_write_text(agent_profiles_file(), payload, mode=0o600)


# ---- read API --------------------------------------------------------------


def list_profiles() -> List[AgentProfile]:
    with _lock:
        return _read_state().profiles


def get_profile(profile_id: str) -> Optional[AgentProfile]:
    with _lock:
        for p in _read_state().profiles:
            if p.id == profile_id:
                return p
    return None


def get_active() -> Optional[AgentProfile]:
    with _lock:
        state = _read_state()
        if not state.active_id:
            return state.profiles[0] if state.profiles else None
        for p in state.profiles:
            if p.id == state.active_id:
                return p
        return state.profiles[0] if state.profiles else None


def get_active_id() -> Optional[str]:
    p = get_active()
    return p.id if p else None


def revision() -> int:
    """Monotonic counter bumped on every mutation — the runtime cache key."""
    with _lock:
        return _read_state().revision


# ---- write API -------------------------------------------------------------


def _validate_token(value: str, what: str) -> str:
    value = value or ""
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in value):
        raise ValueError(f"{what} contains control characters; strip them first")
    return value


def add_profile(**kwargs: Any) -> AgentProfile:
    label = (kwargs.get("label") or "").strip()
    if not label:
        raise ValueError("label is required")
    _validate_token(kwargs.get("api_key", ""), "api_key")
    with _lock:
        state = _read_state()
        profile = AgentProfile(
            id=secrets.token_hex(4),
            **{k: v for k, v in kwargs.items() if k in _PROFILE_FIELDS and k != "id"},
        )
        profile.base_url = (profile.base_url or "").rstrip("/")
        state.profiles.append(profile)
        # First profile becomes active automatically.
        if state.active_id is None:
            state.active_id = profile.id
        _write_state(state)
        return profile


def update_profile(profile_id: str, **kwargs: Any) -> Optional[AgentProfile]:
    if "api_key" in kwargs:
        _validate_token(kwargs.get("api_key", ""), "api_key")
    with _lock:
        state = _read_state()
        target: Optional[AgentProfile] = None
        for p in state.profiles:
            if p.id == profile_id:
                target = p
                break
        if target is None:
            return None
        for k, v in kwargs.items():
            if k in _PROFILE_FIELDS and k != "id":
                setattr(target, k, v)
        target.base_url = (target.base_url or "").rstrip("/")
        _write_state(state)
        return target


def remove_profile(profile_id: str) -> Optional[AgentProfile]:
    with _lock:
        state = _read_state()
        removed: Optional[AgentProfile] = None
        keep: List[AgentProfile] = []
        for p in state.profiles:
            if p.id == profile_id and removed is None:
                removed = p
            else:
                keep.append(p)
        if removed is None:
            return None
        state.profiles = keep
        _write_state(state)
        return removed


def set_active(profile_id: str) -> bool:
    with _lock:
        state = _read_state()
        if not any(p.id == profile_id for p in state.profiles):
            return False
        state.active_id = profile_id
        _write_state(state)
        return True


def seed_if_empty(profile_kwargs: Dict[str, Any]) -> Optional[AgentProfile]:
    """Create one profile from a kwargs dict iff the store is empty.

    Used to bootstrap from the server-config ``agent:`` block on first run.
    Returns the created profile, or None if the store already has profiles
    or the kwargs don't define a usable profile (no model).
    """
    if not (profile_kwargs or {}).get("model"):
        return None
    with _lock:
        state = _read_state()
        if state.profiles:
            return None
    kwargs = dict(profile_kwargs)
    kwargs.setdefault("label", "Default (from server config)")
    return add_profile(**kwargs)
