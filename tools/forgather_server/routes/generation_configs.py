"""Saved generation-parameter presets.

Presets come from two layers:

1. Bundled examples under ``<forgather_repo>/generation_config/`` —
   read-only; the UI may load them but not overwrite or delete them.
2. User-saved presets under
   ``<forgather_config_dir>/generation_config/`` — read-write. Take
   precedence over bundled names of the same stem, so a user can
   effectively override an example by creating a file with the same
   name.

Each preset is a single ``<name>.json`` whose body is passed through
verbatim — the server doesn't interpret the fields, the inference
server's ``GenerationConfig`` does.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from forgather.preprocess import forgather_config_dir

from .._atomic import atomic_write_text
from ..search_roots import forgather_repo_root

# Cap on JSON body size for preset writes — these are tiny config blobs;
# anything past a few KB is almost certainly noise (or abuse).
_MAX_PRESET_BYTES = 64 * 1024

log = logging.getLogger("forgather_server.generation_configs")
router = APIRouter(tags=["generation-configs"])


def _user_dir() -> Path:
    p = Path(forgather_config_dir()) / "generation_config"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _builtin_dir() -> Path:
    return Path(forgather_repo_root()) / "generation_config"


# Filesystem-safe subset — keeps the name usable as a bare filename and
# blocks traversal (no slashes, no leading dots). Generous enough for
# human labels: letters, digits, space, dash, underscore, dot, parens.
_NAME_RE = re.compile(r"^[A-Za-z0-9 _\-.\(\)]+$")


def _validate_name(name: str) -> None:
    if not name or name.startswith(".") or ".." in name or not _NAME_RE.match(name):
        raise HTTPException(status_code=400, detail=f"invalid preset name: {name!r}")


def _user_path(name: str) -> Path:
    _validate_name(name)
    path = _user_dir() / f"{name}.json"
    # Belt-and-suspenders: reject anything that escaped the store via a
    # crafted name that slipped the regex.
    if path.parent.resolve() != _user_dir().resolve():
        raise HTTPException(status_code=400, detail="invalid preset path")
    return path


def _builtin_path(name: str) -> Path:
    _validate_name(name)
    path = _builtin_dir() / f"{name}.json"
    if path.parent.resolve() != _builtin_dir().resolve():
        raise HTTPException(status_code=400, detail="invalid preset path")
    return path


def _resolve_for_read(name: str) -> Path:
    """User copy wins over bundled copy."""
    up = _user_path(name)
    if up.exists():
        return up
    bp = _builtin_path(name)
    if bp.exists():
        return bp
    raise HTTPException(status_code=404, detail=f"preset not found: {name}")


def _is_builtin(name: str) -> bool:
    try:
        return _builtin_path(name).exists()
    except HTTPException:
        return False


class PresetInfo(BaseModel):
    name: str
    builtin: bool


class GenerationConfigEntry(BaseModel):
    name: str
    builtin: bool
    # Keep params opaque; the frontend round-trips the shape of its own
    # ``GenerationParams`` interface without the server having to mirror it.
    params: Dict[str, Any]


class GenerationConfigListResponse(BaseModel):
    presets: List[PresetInfo]


@router.get("/generation-configs", response_model=GenerationConfigListResponse)
def list_presets() -> GenerationConfigListResponse:
    user_names = {p.stem for p in _user_dir().glob("*.json") if p.is_file()}
    builtin_dir = _builtin_dir()
    builtin_names = (
        {p.stem for p in builtin_dir.glob("*.json") if p.is_file()}
        if builtin_dir.is_dir()
        else set()
    )

    # Both sources merged. A name present in both is one entry — the user
    # copy wins but the builtin flag stays false so the UI allows delete
    # (removing the override reveals the bundled version again on next
    # list).
    infos: Dict[str, PresetInfo] = {}
    for n in sorted(builtin_names):
        infos[n] = PresetInfo(name=n, builtin=True)
    for n in sorted(user_names):
        infos[n] = PresetInfo(name=n, builtin=False)
    return GenerationConfigListResponse(
        presets=sorted(infos.values(), key=lambda p: p.name)
    )


@router.get("/generation-configs/{name}", response_model=GenerationConfigEntry)
def get_preset(name: str) -> GenerationConfigEntry:
    path = _resolve_for_read(name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"malformed preset: {e}")
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="preset is not a JSON object")
    # ``builtin`` here means "the served copy came from the bundle" — if
    # the user has shadowed it, the served copy is theirs and we report
    # false so the UI can offer Delete.
    is_builtin_source = path.parent.resolve() == _builtin_dir().resolve()
    return GenerationConfigEntry(name=name, builtin=is_builtin_source, params=data)


@router.put("/generation-configs/{name}", response_model=GenerationConfigEntry)
def put_preset(
    name: str,
    params: Dict[str, Any] = Body(...),
) -> GenerationConfigEntry:
    # Writes always land in the user dir. If ``name`` also exists as a
    # builtin, this file effectively shadows it — load now returns the
    # user copy; deleting it restores the bundled version.
    path = _user_path(name)
    serialized = json.dumps(params, indent=2) + "\n"
    if len(serialized) > _MAX_PRESET_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"preset payload is {len(serialized)} bytes; "
                f"max is {_MAX_PRESET_BYTES}"
            ),
        )
    # Use the project's crash-atomic helper (write + fsync + os.replace)
    # so a crash mid-write never leaves the file truncated.
    atomic_write_text(path, serialized)
    return GenerationConfigEntry(name=name, builtin=False, params=params)


@router.delete("/generation-configs/{name}")
def delete_preset(name: str) -> Dict[str, bool]:
    up = _user_path(name)
    if up.exists():
        # Tolerate concurrent unlink — the user effectively got what they
        # asked for either way.
        up.unlink(missing_ok=True)
        return {"ok": True}
    # No user copy — either the preset doesn't exist, or it's a
    # read-only bundled example. 403 differentiates the two so the UI
    # can show a useful message.
    if _is_builtin(name):
        raise HTTPException(
            status_code=403,
            detail=(
                f"{name!r} is a built-in preset and cannot be deleted. "
                "Save a user preset with the same name to shadow it."
            ),
        )
    raise HTTPException(status_code=404, detail=f"preset not found: {name}")
