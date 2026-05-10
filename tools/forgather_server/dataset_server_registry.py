"""Persistent registry of user-added dataset_server endpoints.

The webui's Datasets view lets a user register additional dataset_servers
beyond the local ones the forgather_server itself spawns (e.g. a peer node
in a multi-node training cluster, or a teammate's shared dataset host).
Each entry is a (label, base_url, auth_token) triple stored on disk so the
list survives server restarts.

Layout: JSON list at ``<config>/server/dataset_server_registry.json``,
mode 0600 (the file contains bearer tokens). Each entry has a stable ``id``
(8 hex chars) so the webui can edit/remove by identifier instead of by
URL — useful when the same URL is registered twice or the URL changes.
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import asdict, dataclass
from threading import Lock
from typing import List, Optional

from ._atomic import atomic_write_text
from .paths import dataset_server_registry_file

log = logging.getLogger("forgather_server.dataset_server_registry")

_lock = Lock()


@dataclass
class RegistryEntry:
    id: str
    label: str
    base_url: str
    auth_token: str  # may be empty; servers running --no-auth don't need one


def _read_raw() -> List[dict]:
    path = dataset_server_registry_file()
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
    atomic_write_text(dataset_server_registry_file(), payload, mode=0o600)


def _entries_from_raw(raw: List[dict]) -> List[RegistryEntry]:
    return [
        RegistryEntry(
            id=str(item["id"]),
            label=str(item.get("label") or item["base_url"]),
            base_url=str(item["base_url"]).rstrip("/"),
            auth_token=str(item.get("auth_token") or ""),
        )
        for item in raw
    ]


def list_entries() -> List[RegistryEntry]:
    with _lock:
        return _entries_from_raw(_read_raw())


def add_entry(*, label: str, base_url: str, auth_token: str = "") -> RegistryEntry:
    """Append a new entry. URL is normalized to drop any trailing slash.

    The label falls back to the URL when blank so the UI always has
    something to display, even if the user didn't bother naming the
    endpoint. Caller is responsible for trimming whitespace.
    """
    base_url = (base_url or "").rstrip("/")
    if not base_url:
        raise ValueError("base_url is required")
    label = (label or "").strip() or base_url
    with _lock:
        existing = _entries_from_raw(_read_raw())
        new = RegistryEntry(
            id=secrets.token_hex(4),
            label=label,
            base_url=base_url,
            auth_token=auth_token or "",
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

    Used by the proxy's token-lookup chain when a request comes in for
    a URL the user registered with an explicit token. Returns ``None``
    when the URL isn't registered or the stored token is empty.
    """
    base_url = (base_url or "").rstrip("/")
    if not base_url:
        return None
    for e in list_entries():
        if e.base_url == base_url and e.auth_token:
            return e.auth_token
    return None
