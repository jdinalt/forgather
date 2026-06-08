"""Agent filesystem tools: stat / delete / move / copy.

General file management — primarily for cleanup (removing stale runs /
output dirs / scratch files) and reorganizing. Read/inspect is already
covered by read_file / list_directory / find_files; these add the
mutating half plus a structured stat.

The mutations reuse the existing ``routes/fs.py`` handlers in their commit
closures (the same cross-module reuse ``tools_jobs`` does with
``routes.jobs``), so the authoritative guards live in exactly one place:
fs-root allowlist, symlink-chain rejection, the depth floor (>=4 path
components), the ``_FORBIDDEN_PATHS`` denylist, and the ``confirmed`` ack.
Previews are read-only — they validate fs-root + existence for fast
feedback and gather scope (size / entry count), but never touch disk;
the route handler re-runs every guard at commit. ``HTTPException`` from a
guard is translated to ``ValueError`` so the agent gets a clean message.

Core-tier (file management is a common operation, so these stay in the tool
array even in deferred mode). Everything but ``stat_path`` is CONFIRM-gated.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

from .. import paths as fs_paths
from .registry import CONFIRM, READ, Proposal, ToolRegistry, ToolSpec

log = logging.getLogger("forgather_server.agent.tools_fs")


def _resolve(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


def _require_in_fs_root(target: Path) -> None:
    if not fs_paths.is_path_in_fs_root(target):
        raise ValueError(f"path is outside the configured filesystem roots: {target}")


def _dir_size(target: Path) -> tuple[int, int]:
    """(total_bytes, file_count) for a directory tree; best-effort."""
    total = 0
    count = 0
    for root, _, files in os.walk(target, followlinks=False):
        for name in files:
            count += 1
            try:
                total += os.stat(os.path.join(root, name), follow_symlinks=False).st_size
            except OSError:
                pass
    return total, count


def _as_value_error(fn, *args):
    """Run a routes.fs handler, translating HTTPException -> ValueError."""
    from fastapi import HTTPException

    try:
        return fn(*args)
    except HTTPException as e:  # guard failure: surface the detail to the model
        raise ValueError(str(e.detail))


# ---- stat (READ) -----------------------------------------------------------


def _stat_path(args: Dict[str, Any]) -> Any:
    raw = args["path"]
    target = _resolve(raw)
    _require_in_fs_root(target)
    is_symlink = os.path.islink(os.path.abspath(os.path.expanduser(raw)))
    if not target.exists():
        return {"path": str(target), "exists": False, "is_symlink": is_symlink}
    st = target.stat()
    out: Dict[str, Any] = {
        "path": str(target),
        "exists": True,
        "is_dir": target.is_dir(),
        "is_file": target.is_file(),
        "is_symlink": is_symlink,
        "size_bytes": st.st_size,
        "mtime": st.st_mtime,
        "mode": oct(st.st_mode & 0o777),
    }
    if target.is_dir():
        try:
            out["entry_count"] = sum(1 for _ in os.scandir(target))
        except OSError:
            out["entry_count"] = None
    return out


# ---- delete (CONFIRM) ------------------------------------------------------


def _delete_path(args: Dict[str, Any]) -> Proposal:
    raw = args["path"]
    target = _resolve(raw)
    _require_in_fs_root(target)
    if not target.exists():
        raise ValueError(f"path does not exist: {target}")
    is_dir = target.is_dir()
    if is_dir:
        size, count = _dir_size(target)
        extra = {"path": str(target), "kind": "directory", "recursive": True,
                 "size_bytes": size, "entry_count": count}
    else:
        try:
            size = target.stat().st_size
        except OSError:
            size = 0
        extra = {"path": str(target), "kind": "file", "size_bytes": size}

    def commit() -> str:
        from ..routes import fs as fs_routes

        if is_dir:
            resp = _as_value_error(
                fs_routes.delete_dir,
                fs_routes.DeleteDirRequest(path=raw, confirmed=True),
            )
            return f"deleted directory {resp.deleted} ({resp.removed_bytes} bytes)"
        resp = _as_value_error(
            fs_routes.delete_file,
            fs_routes.DeleteFileRequest(path=raw, confirmed=True),
        )
        return f"deleted file {resp.deleted} ({resp.removed_bytes} bytes)"

    return Proposal(
        title=f"Delete {extra['kind']}: {target.name}",
        summary=(
            "Permanently delete this path"
            + (" and everything under it (recursive)" if is_dir else "")
            + ". This cannot be undone."
        ),
        extra=extra,
        commit=commit,
    )


# ---- move / copy (CONFIRM) -------------------------------------------------


def _move_path(args: Dict[str, Any]) -> Proposal:
    src_raw = args["src"]
    dest_raw = args["dest_dir"]
    src = _resolve(src_raw)
    dest_dir = _resolve(dest_raw)
    _require_in_fs_root(src)
    _require_in_fs_root(dest_dir)
    if not src.exists():
        raise ValueError(f"src does not exist: {src}")
    if not dest_dir.is_dir():
        raise ValueError(f"dest_dir is not a directory: {dest_dir}")
    target = dest_dir / src.name

    def commit() -> str:
        from ..routes import fs as fs_routes

        resp = _as_value_error(
            fs_routes.move,
            fs_routes.CopyOrMoveRequest(src=src_raw, dest_dir=dest_raw),
        )
        return f"moved to {resp.path}"

    return Proposal(
        title=f"Move: {src.name} -> {dest_dir.name}/",
        summary="Move a file or directory into another directory.",
        extra={"src": str(src), "dest": str(target)},
        commit=commit,
    )


def _copy_path(args: Dict[str, Any]) -> Proposal:
    src_raw = args["src"]
    dest_raw = args["dest_dir"]
    auto_rename = bool(args.get("auto_rename", False))
    target_name = args.get("target_name") or None
    src = _resolve(src_raw)
    dest_dir = _resolve(dest_raw)
    _require_in_fs_root(src)
    _require_in_fs_root(dest_dir)
    if not src.exists():
        raise ValueError(f"src does not exist: {src}")
    if not dest_dir.is_dir():
        raise ValueError(f"dest_dir is not a directory: {dest_dir}")
    target = dest_dir / (target_name or src.name)

    def commit() -> str:
        from ..routes import fs as fs_routes

        resp = _as_value_error(
            fs_routes.copy_path,
            fs_routes.CopyOrMoveRequest(
                src=src_raw, dest_dir=dest_raw, auto_rename=auto_rename,
                target_name=target_name,
            ),
        )
        return f"copied to {resp.path}"

    return Proposal(
        title=f"Copy: {src.name} -> {dest_dir.name}/",
        summary="Copy a file or directory into another directory.",
        extra={"src": str(src), "dest": str(target), "auto_rename": auto_rename},
        commit=commit,
    )


def register_all(reg: ToolRegistry) -> None:
    reg.register(
        ToolSpec(
            name="stat_path",
            description=(
                "Stat a path: existence, file/dir/symlink, size, mtime, mode, and "
                "(for a directory) its immediate entry count. Use to check what's "
                "there before delete/move/copy, or to inspect a non-project file "
                "(read_file/list_directory cover content)."
            ),
            json_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            handler=_stat_path,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="delete_path",
            description=(
                "Delete a file, or a directory and everything under it "
                "(recursive) — mainly for cleanup of stale runs / output dirs / "
                "scratch files. Approval required; the preview shows the kind, "
                "size, and entry count. Irreversible. Guarded: must be inside the "
                "filesystem roots, not a system path, and at least 4 path "
                "components deep."
            ),
            json_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            handler=_delete_path,
            risk=CONFIRM,
        )
    )
    reg.register(
        ToolSpec(
            name="move_path",
            description=(
                "Move a file or directory into dest_dir (lands at "
                "dest_dir/basename(src)). Approval required. Refuses to overwrite "
                "an existing destination; same fs-root / depth guards as delete."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "dest_dir": {"type": "string", "description": "Destination parent directory (must exist)."},
                },
                "required": ["src", "dest_dir"],
            },
            handler=_move_path,
            risk=CONFIRM,
        )
    )
    reg.register(
        ToolSpec(
            name="copy_path",
            description=(
                "Copy a file or directory into dest_dir (lands at "
                "dest_dir/basename(src), or target_name if given). Approval "
                "required. Refuses overwrite unless auto_rename is set (then it "
                "appends ' (copy)'). Same fs-root / depth guards as delete."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "dest_dir": {"type": "string", "description": "Destination parent directory (must exist)."},
                    "auto_rename": {"type": "boolean", "description": "On collision, append ' (copy)' instead of failing."},
                    "target_name": {"type": "string", "description": "Override the destination basename."},
                },
                "required": ["src", "dest_dir"],
            },
            handler=_copy_path,
            risk=CONFIRM,
        )
    )
