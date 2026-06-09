"""Agent filesystem tools: stat / create / edit / delete / move / copy.

General file management — creating and editing plain text files, cleanup
(removing stale runs / output dirs / scratch files), and reorganizing.
Read/inspect is already covered by read_file / list_directory / find_files;
these add the mutating half plus a structured stat.

``edit_file`` (write content to a file, creating it if missing, with a
before/after diff) and ``create_file`` (touch an explicitly empty file) are
the plain-file counterparts to the config/template authoring tools in
``tools_authoring`` — use those for Forgather configs (they scaffold and
parse-check); use these for markdown, notes, and other non-config text. They
reuse the same crash-atomic write primitives
(``config_ops.write_template_file`` / ``write_existing_file``), so the
fs-root allowlist, no-clobber-on-create, and optimistic mtime guards live in
one place.

Two guard paths, by operation risk — they are deliberately NOT the same:

- ``delete_path`` / ``move_path`` / ``copy_path`` reuse the existing
  ``routes/fs.py`` handlers in their commit closures (the same cross-module
  reuse ``tools_jobs`` does with ``routes.jobs``), so their authoritative
  guards live in one place: fs-root allowlist, symlink-chain rejection, the
  depth floor (>=4 path components), the ``_FORBIDDEN_PATHS`` denylist, and
  the ``confirmed`` ack. The route handler re-runs every guard at commit;
  ``HTTPException`` from a guard is translated to ``ValueError``.
- ``create_file`` / ``edit_file`` go through ``config_ops`` directly
  (``write_template_file`` / ``write_existing_file``), the same primitives
  the ``tools_authoring`` config writers use, so their guards are fs-root
  allowlist, no-clobber-on-create, and the optimistic mtime check. They do
  NOT apply the delete-style depth floor or denylist on purpose: the depth
  floor is calibrated to stop a catastrophic recursive *delete*, and would
  wrongly reject a legitimate edit near a shallow fs-root; a single
  approval-gated file write with a visible diff has a different risk profile.
  ``_resolve`` canonicalizes (follows symlinks) before the fs-root check, so
  a symlink escaping the roots is still rejected.

Previews are read-only — they validate fs-root + existence/scope for fast
feedback, but never touch disk; the write is re-checked at commit.

Core-tier (file management is a common operation, so these stay in the tool
array even in deferred mode). ``stat_path`` is READ; ``edit_file`` is
PROPOSE (it carries a diff preview); the rest are CONFIRM-gated.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .. import config_ops
from .. import paths as fs_paths
from .registry import CONFIRM, PROPOSE, READ, Proposal, ToolRegistry, ToolSpec

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


# ---- create / edit plain files (CONFIRM / PROPOSE) -------------------------


def _create_file(args: Dict[str, Any]) -> Proposal:
    raw = args["path"]
    target = _resolve(raw)
    _require_in_fs_root(target)
    if target.exists():
        raise ValueError(f"path already exists: {target}")
    # ``touch`` semantics: create the file, not its parents. Refuse a missing
    # parent rather than silently materializing a directory tree (use the
    # webui / a shell for that). This is enforced at preview only —
    # config_ops.write_template_file would makedirs at commit — which is fine:
    # the user can't approve a proposal that preview refused to build.
    if not target.parent.is_dir():
        raise ValueError(f"parent directory does not exist: {target.parent}")

    def commit() -> str:
        written = config_ops.write_template_file(str(target), "")
        return f"created empty file {written}"

    return Proposal(
        title=f"Create file: {target.name}",
        summary=f"Create a new empty file at {target}.",
        extra={"path": str(target), "kind": "file", "bytes": 0},
        commit=commit,
    )


def _edit_file(args: Dict[str, Any]) -> Proposal:
    raw = args["path"]
    new_content = args["new_content"]
    target = _resolve(raw)
    _require_in_fs_root(target)
    exists = target.exists()
    if exists and not target.is_file():
        raise ValueError(f"not a regular file: {target}")
    # Create-if-missing: writing content to a not-yet-existing path is the
    # natural "write this file" gesture, so don't force a separate create_file
    # round-trip. We still refuse to materialize a missing directory tree.
    if not exists and not target.parent.is_dir():
        raise ValueError(f"parent directory does not exist: {target.parent}")
    # Read the current content for the diff preview (``None`` for a new file,
    # which the webui renders as an all-added diff). fs-root is already
    # enforced above; read_raw also validates the file is readable text.
    before: Optional[str] = config_ops.read_raw(str(target)) if exists else None
    # Optimistic-concurrency baseline captured server-side at propose time
    # (mirrors propose_edit_config): the commit refuses only if the file
    # actually changed on disk between this read and approval.
    try:
        expected_mtime: Optional[float] = os.path.getmtime(target) if exists else None
    except OSError:
        expected_mtime = None

    def commit() -> str:
        # Re-resolve existence at commit, not preview, so a file that appeared
        # in the approval gap is handled correctly: an existing file goes
        # through the mtime guard; a still-missing one is created (and
        # write_template_file's no-clobber guard rejects a concurrent create
        # rather than silently overwriting).
        if target.exists():
            info = config_ops.write_existing_file(
                str(target), new_content, expected_mtime=expected_mtime
            )
            return (
                f"wrote {info['bytes_written']} bytes to {info['path']} "
                f"(mtime={info['mtime']})."
            )
        written = config_ops.write_template_file(str(target), new_content)
        return f"created {written} ({len(new_content.encode('utf-8'))} bytes)."

    return Proposal(
        title=(f"Edit file: {target.name}" if exists else f"Create file: {target.name}"),
        summary=(f"Overwrite {target}" if exists else f"Create {target} with content"),
        path=str(target),
        before=before,
        after=new_content,
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
            name="create_file",
            description=(
                "Create a new, empty file (like ``touch``) at an absolute path "
                "inside the filesystem roots. Approval required. Refuses if the "
                "path already exists or its parent directory is missing. Use this "
                "only when you specifically want an EMPTY file; to create a file "
                "WITH content in one step, call edit_file (it creates if missing). "
                "For a Forgather config or template use propose_new_config instead "
                "(it scaffolds from a meta-template)."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path of the file to create."},
                },
                "required": ["path"],
            },
            handler=_create_file,
            risk=CONFIRM,
        )
    )
    reg.register(
        ToolSpec(
            name="edit_file",
            description=(
                "Write content to a plain text file, shown as a before/after diff "
                "for approval. Creates the file if it does not exist (the parent "
                "directory must already exist), otherwise overwrites it. This is "
                "the one-step way to create-with-content; create_file is only for "
                "an explicitly empty file. For arbitrary files (markdown, notes, "
                "scripts) — NOT Forgather configs/templates, which have "
                "propose_edit_config (it additionally runs a post-write parse "
                "check). Guards: absolute path inside the filesystem roots, and an "
                "optimistic mtime check that refuses the write if an existing file "
                "changed on disk since it was read."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path of the existing file to overwrite."},
                    "new_content": {"type": "string", "description": "The full new file content."},
                },
                "required": ["path", "new_content"],
            },
            handler=_edit_file,
            risk=PROPOSE,
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
