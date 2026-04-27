"""Persistent list of project-discovery roots.

The search-roots file is a tiny JSON array of absolute directory paths. The
server walks each root recursively looking for ``meta.yaml`` files to build
the project tree. Missing roots are silently tolerated (the user's drive may
not be mounted); the caller gets a flag per root indicating existence.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import List

from ._atomic import atomic_write_text
from .paths import search_roots_file

_lock = Lock()


@dataclass
class SearchRoot:
    path: str
    exists: bool


def forgather_repo_root() -> str:
    """Absolute path to the Forgather repo root.

    Derived from this file's location: ``<repo>/tools/forgather_server/search_roots.py``.
    """
    return str(Path(__file__).resolve().parent.parent.parent)


def default_roots() -> List[str]:
    """Sensible first-run search roots.

    Points at the Forgather repo root so a fresh first boot picks up
    everything under it — ``examples/`` plus any sibling project trees
    a user has parked next to the checkout. Earlier this was scoped
    down to ``<repo>/examples`` because ``docs/configuration/`` carried
    a stale ``meta.yaml`` that surfaced as a parse-error project; that
    project has since been cleaned up.
    """
    return [forgather_repo_root()]


def _read_raw() -> List[str]:
    path = search_roots_file()
    if not path.exists():
        # First boot: seed defaults and persist so the user can then remove
        # them without them coming back.
        seeded = default_roots()
        _write_raw(seeded)
        return seeded
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [str(p) for p in data if isinstance(p, str)]


def _write_raw(roots: List[str]) -> None:
    atomic_write_text(search_roots_file(), json.dumps(roots, indent=2))


def _normalize(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def list_roots() -> List[SearchRoot]:
    with _lock:
        return [SearchRoot(path=p, exists=Path(p).is_dir()) for p in _read_raw()]


def add_root(path: str) -> SearchRoot:
    normalized = _normalize(path)
    with _lock:
        roots = _read_raw()
        if normalized not in roots:
            roots.append(normalized)
            _write_raw(roots)
    return SearchRoot(path=normalized, exists=Path(normalized).is_dir())


def remove_root(path: str) -> bool:
    normalized = _normalize(path)
    with _lock:
        roots = _read_raw()
        if normalized not in roots:
            return False
        roots.remove(normalized)
        _write_raw(roots)
    return True
