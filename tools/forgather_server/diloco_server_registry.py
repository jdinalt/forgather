"""Persistent registry of user-added DiLoCo-server endpoints.

The webui's DiLoCo view lists *running* servers the forgather_server
itself spawned (and, in cluster mode, servers visible to peers).
Operators also have *external* servers they hit regularly — a long-lived
parameter server on another LAN segment, a teammate's server, an
SSH-tunneled remote — that no cluster knows about.

This module mirrors :mod:`inference_server_registry`: a small
JSON-backed list of (label, base_url, auth_token, verify_tls) entries
the user can add / remove via the webui, persisted across server
restarts.

TLS / auth is out of scope for the initial cut; ``auth_token`` and
``verify_tls`` are reserved fields on the entry so the wire format
doesn't need to change when TLS lands.

Intentionally a near-clone of the inference flavor rather than a shared
abstraction: surfaces evolve independently (cluster integration, status
proxy shape, port semantics differ), and a parallel module keeps the
code obvious. Refactor only if a fourth caller appears.
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import asdict, dataclass
from threading import Lock
from typing import List, Optional

from ._atomic import atomic_write_text
from .paths import diloco_server_registry_file

log = logging.getLogger("forgather_server.diloco_server_registry")

_lock = Lock()


@dataclass
class RegistryEntry:
    id: str
    label: str
    base_url: str
    # Reserved for the TLS / auth follow-up. Empty for the unauthenticated
    # plaintext servers we currently spawn. Kept in the file so the schema
    # doesn't need to change when auth lands.
    auth_token: str = ""
    # When False, outbound calls skip TLS chain + hostname validation —
    # used for SSH-tunneled remotes whose certificate won't validate
    # against the local CA. Defaults to True (chain validation on).
    verify_tls: bool = True


def _read_raw() -> List[dict]:
    path = diloco_server_registry_file()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    out: List[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "id" not in item or "base_url" not in item:
            continue
        out.append(item)
    return out


def _write_raw(entries: List[RegistryEntry]) -> None:
    payload = json.dumps([asdict(e) for e in entries], indent=2)
    atomic_write_text(diloco_server_registry_file(), payload, mode=0o600)


def _entries_from_raw(raw: List[dict]) -> List[RegistryEntry]:
    return [
        RegistryEntry(
            id=str(item["id"]),
            label=str(item.get("label") or item["base_url"]),
            base_url=str(item["base_url"]).rstrip("/"),
            auth_token=str(item.get("auth_token") or ""),
            verify_tls=bool(item.get("verify_tls", True)),
        )
        for item in raw
    ]


def list_entries() -> List[RegistryEntry]:
    with _lock:
        return _entries_from_raw(_read_raw())


def add_entry(
    *,
    label: str,
    base_url: str,
    auth_token: str = "",
    verify_tls: bool = True,
) -> RegistryEntry:
    """Append a new entry. URL is normalized to drop any trailing slash.

    Tokens are rejected if they contain control characters — same
    treatment as the inference registry, since they'd be rejected by
    the outbound HTTP layer downstream anyway.

    Pure database operation: does NOT probe the target. The operator
    validates the URL via the DiLoCo panel after registering.
    """
    base_url = (base_url or "").rstrip("/")
    if not base_url:
        raise ValueError("base_url is required")
    auth_token = auth_token or ""
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in auth_token):
        raise ValueError(
            "auth_token contains control characters (CR, LF, or other "
            "non-printable). Strip them before registering."
        )
    label = (label or "").strip() or base_url
    with _lock:
        existing = _entries_from_raw(_read_raw())
        new = RegistryEntry(
            id=secrets.token_hex(4),
            label=label,
            base_url=base_url,
            auth_token=auth_token,
            verify_tls=verify_tls,
        )
        existing.append(new)
        _write_raw(existing)
        return new


def remove_entry(entry_id: str) -> Optional[RegistryEntry]:
    """Drop the entry with matching id. Returns the removed entry or None."""
    with _lock:
        existing = _entries_from_raw(_read_raw())
        keep: List[RegistryEntry] = []
        removed: Optional[RegistryEntry] = None
        for e in existing:
            if e.id == entry_id and removed is None:
                removed = e
            else:
                keep.append(e)
        if removed is not None:
            _write_raw(keep)
        return removed


def find_token(base_url: str) -> Optional[str]:
    """Look up the bearer token for an exact base_url match.

    Returns ``None`` when the URL isn't registered or the stored token
    is empty. The plaintext-only path won't call this, but the wire
    contract is in place for the TLS follow-up.
    """
    base_url = (base_url or "").rstrip("/")
    if not base_url:
        return None
    for e in list_entries():
        if e.base_url == base_url and e.auth_token:
            return e.auth_token
    return None


def find_verify_tls(base_url: str) -> bool:
    """Look up the per-entry ``verify_tls`` setting for ``base_url``.

    Returns True (default secure posture) when the URL isn't in the
    registry. False only when an entry exists AND the operator
    explicitly opted out.
    """
    base_url = (base_url or "").rstrip("/")
    if not base_url:
        return True
    for e in list_entries():
        if e.base_url == base_url:
            return e.verify_tls
    return True
