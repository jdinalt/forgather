"""Filesystem-browsing endpoints used by the directory-picker UI.

Localhost-prototype caveat: these endpoints reveal arbitrary directory
contents to any caller that can reach the bound port. The server binds to
127.0.0.1 by default, and no auth is configured — matching the security
posture of the rest of the prototype. Do not expose the port on an untrusted
network.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import search_roots

log = logging.getLogger("forgather_server.fs")
router = APIRouter(tags=["fs"])


class FsEntry(BaseModel):
    name: str
    path: str
    is_dir: bool


class FsListing(BaseModel):
    path: str
    parent: Optional[str]
    entries: List[FsEntry]


class QuickPath(BaseModel):
    label: str
    path: str


@router.get("/fs/quick-paths", response_model=List[QuickPath])
def quick_paths():
    """Named shortcuts surfaced in the directory picker."""
    repo = search_roots.forgather_repo_root()
    return [
        QuickPath(label="Examples", path=os.path.join(repo, "examples")),
        QuickPath(label="Forgather repo", path=repo),
        QuickPath(label="Home", path=str(Path.home())),
    ]


class PathExistsResponse(BaseModel):
    """Cheap stat-style response. ``exists=False`` is also returned for
    paths inside directories the server can't read (PermissionError);
    callers don't get to distinguish "missing" from "unreadable" here,
    which is what we want for sanity-checking persisted UI defaults."""

    exists: bool
    is_file: bool = False
    is_dir: bool = False


@router.get("/fs/path-exists", response_model=PathExistsResponse)
def path_exists(path: str):
    """Cheap existence + type check for a single path.

    Used by modals that persist on-disk paths in localStorage (e.g.
    the MkDocs tool's ``mkdocs.yml`` default) so they can replace
    stale values that point at directories from a previous install
    location instead of failing the user on submit.
    """
    try:
        resolved = Path(os.path.expanduser(path))
        if not resolved.exists():
            return PathExistsResponse(exists=False)
        return PathExistsResponse(
            exists=True,
            is_file=resolved.is_file(),
            is_dir=resolved.is_dir(),
        )
    except OSError:
        return PathExistsResponse(exists=False)


@router.get("/fs/browse", response_model=FsListing)
def browse(path: str = "", show_hidden: bool = False, files_too: bool = False):
    """List subdirectories of ``path`` (defaults to the user's home).

    ``files_too=true`` also returns regular files so the listing can
    service a file-selection picker; the default is directories only,
    matching the original "pick a search root" use case. When files are
    included they're sorted after directories alphabetically.
    """
    if not path:
        path = str(Path.home())
    resolved = Path(os.path.expanduser(path)).resolve()

    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {resolved}")
    if not resolved.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {resolved}")

    dirs: List[FsEntry] = []
    files: List[FsEntry] = []
    try:
        for child in sorted(resolved.iterdir(), key=lambda p: p.name.lower()):
            if not show_hidden and child.name.startswith("."):
                continue
            try:
                is_dir = child.is_dir()
            except OSError:
                continue
            entry = FsEntry(name=child.name, path=str(child.resolve()), is_dir=is_dir)
            if is_dir:
                dirs.append(entry)
            elif files_too:
                files.append(entry)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    parent = str(resolved.parent) if resolved != resolved.parent else None
    return FsListing(path=str(resolved), parent=parent, entries=dirs + files)


class DeleteDirRequest(BaseModel):
    path: str
    # Required acknowledgement. A POST body that forgets to set this gets
    # a 400 instead of a deletion — acts as a belt-and-suspenders guard on
    # top of the "Are you sure?" confirm() in the UI.
    confirmed: bool = False


class DeleteDirResponse(BaseModel):
    deleted: str
    removed_bytes: int = 0


_FORBIDDEN_PATHS: set[str] = {
    "/",
    "/home",
    "/root",
    "/usr",
    "/etc",
    "/var",
    "/opt",
    "/tmp",
    "/mnt",
    "/media",
    str(Path.home()),
}


def _reject_unsafe(target: Path) -> None:
    """Reject paths that are obvious catastrophic-delete candidates.

    Defensive cheap checks: must be absolute, must exist, must be a
    directory (not a symlink to one), must be at least 4 path components
    deep (so ``/foo/bar/baz`` passes and ``/etc`` doesn't), and must not
    match a denylist of common system roots.
    """
    if not target.is_absolute():
        raise HTTPException(status_code=400, detail="path must be absolute")
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"path does not exist: {target}")
    if target.is_symlink():
        raise HTTPException(
            status_code=400, detail="refusing to follow or delete symlink"
        )
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {target}")
    resolved = str(target)
    if resolved in _FORBIDDEN_PATHS:
        raise HTTPException(
            status_code=403, detail=f"refusing to delete system path: {resolved}"
        )
    # Count "real" path components (drop the leading empty from leading /).
    depth = len([p for p in target.parts if p and p != "/"])
    if depth < 4:
        raise HTTPException(
            status_code=403,
            detail=(
                f"refusing to delete a path this shallow: {resolved} "
                f"(need at least 4 components)"
            ),
        )


class MkdirRequest(BaseModel):
    """Body for ``POST /api/fs/mkdir``.

    ``parent`` is an existing directory; ``name`` is the new directory's
    bare basename (no slashes). The resulting path is ``parent/name``.
    A nested name is intentionally rejected so the file picker's
    "+ New folder" affordance creates exactly one new directory at a
    time — users wanting deeper hierarchies can repeat the action.
    """

    parent: str
    name: str


class MkdirResponse(BaseModel):
    path: str


@router.post("/fs/mkdir", response_model=MkdirResponse)
def mkdir(req: MkdirRequest):
    """Create a new directory under an existing parent.

    Used by the directory-picker's "+ New folder" button so users can
    build out a hierarchy without leaving the modal. Refuses overwrite,
    rejects names with path separators or traversal segments, and
    requires the parent to exist as a directory.
    """
    parent = Path(os.path.expanduser(req.parent)).resolve()
    if not parent.is_absolute():
        raise HTTPException(status_code=400, detail="parent must be absolute")
    if not parent.exists() or not parent.is_dir():
        raise HTTPException(
            status_code=400, detail=f"parent is not a directory: {parent}"
        )
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if "/" in name or "\\" in name or name in (".", ".."):
        raise HTTPException(
            status_code=400, detail="name must be a bare directory name"
        )
    target = parent / name
    if target.exists():
        raise HTTPException(status_code=409, detail=f"already exists: {target}")
    try:
        target.mkdir()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"mkdir failed: {e}")
    return MkdirResponse(path=str(target))


def _check_path_safe(target: Path, *, must_exist: bool = True) -> None:
    """Cheap sanity checks shared by rename / copy / move.

    - Absolute path required.
    - Refuses symlinks (we never want to silently chase them).
    - When ``must_exist`` is true: refuses missing paths.
    - Depth floor: ≥ 4 path components, so a typo can't accidentally
      target ``/etc`` or similar.
    """
    if not target.is_absolute():
        raise HTTPException(status_code=400, detail="path must be absolute")
    if target.is_symlink():
        raise HTTPException(status_code=400, detail="refusing to operate on symlink")
    if must_exist and not target.exists():
        raise HTTPException(status_code=404, detail=f"path does not exist: {target}")
    depth = len([p for p in target.parts if p and p != "/"])
    if depth < 4:
        raise HTTPException(
            status_code=403,
            detail=(
                f"refusing to operate on a path this shallow: {target} "
                f"(need at least 4 components)"
            ),
        )


class RenameRequest(BaseModel):
    """Body for ``POST /api/fs/rename``.

    ``path`` is the existing absolute file or directory; ``new_name`` is
    the new bare basename (no separators, no `.`/`..`). The result lives
    next to the original at ``parent / new_name``.
    """

    path: str
    new_name: str


class FsPathResponse(BaseModel):
    path: str


class NewFileRequest(BaseModel):
    """Body for ``POST /api/fs/new-file``.

    Creates a single empty file at ``parent / name``. ``name`` is a bare
    basename — for nested paths, mkdir the intermediates first via
    ``/fs/mkdir``. Used by the sidebar Files tree's right-click → New
    File… affordance, the user's first stop after Browse… for putting
    a fresh file under an existing directory.
    """

    parent: str
    name: str


@router.post("/fs/new-file", response_model=FsPathResponse)
def new_file(req: NewFileRequest):
    """Create an empty file under an existing parent directory."""
    parent = Path(os.path.expanduser(req.parent)).resolve()
    if not parent.is_absolute():
        raise HTTPException(status_code=400, detail="parent must be absolute")
    if not parent.exists() or not parent.is_dir():
        raise HTTPException(
            status_code=400, detail=f"parent is not a directory: {parent}"
        )
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if "/" in name or "\\" in name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="name must be a bare file name")
    target = parent / name
    if target.exists():
        raise HTTPException(status_code=409, detail=f"already exists: {target}")
    try:
        target.touch()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"create failed: {e}")
    return FsPathResponse(path=str(target))


@router.post("/fs/rename", response_model=FsPathResponse)
def rename(req: RenameRequest):
    """Rename a file or directory in place.

    Used by the sidebar Files tree's right-click → Rename. Bare-name
    only — moving across directories goes through ``/fs/move``.
    Recoverable via reverse rename, so no ``confirmed`` flag.
    """
    src = Path(os.path.expanduser(req.path)).resolve()
    _check_path_safe(src)
    new_name = req.new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="new_name is required")
    if "/" in new_name or "\\" in new_name or new_name in (".", ".."):
        raise HTTPException(status_code=400, detail="new_name must be a bare basename")
    dest = src.parent / new_name
    if dest.exists():
        raise HTTPException(status_code=409, detail=f"already exists: {dest}")
    try:
        os.rename(src, dest)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"rename failed: {e}")
    return FsPathResponse(path=str(dest))


class CopyOrMoveRequest(BaseModel):
    """Body for ``POST /api/fs/copy`` and ``/api/fs/move``.

    ``src`` is an absolute existing file or directory; ``dest_dir`` is
    the destination *parent* (must exist). The new entry lands at
    ``dest_dir / basename(src)``. Refuses overwrite — collisions need a
    rename first.
    """

    src: str
    dest_dir: str


def _resolve_copy_target(src_path: Path, dest_dir_path: Path) -> Path:
    """Compute the resolved destination ``dest_dir / basename(src)`` and
    enforce safety: parent must be a real directory, both ends pass
    ``_check_path_safe``, and the target itself must not yet exist."""
    _check_path_safe(src_path)
    _check_path_safe(dest_dir_path)
    if not dest_dir_path.is_dir():
        raise HTTPException(
            status_code=400, detail=f"dest_dir is not a directory: {dest_dir_path}"
        )
    target = dest_dir_path / src_path.name
    if target.exists():
        raise HTTPException(status_code=409, detail=f"already exists: {target}")
    # Ensure the resulting path also clears the depth floor (it should
    # always — dest_dir already passed — but defense in depth).
    _check_path_safe(target, must_exist=False)
    return target


@router.post("/fs/copy", response_model=FsPathResponse)
def copy_path(req: CopyOrMoveRequest):
    """Copy a file or directory under ``dest_dir``.

    Files: ``shutil.copy2`` (preserves metadata). Directories:
    ``shutil.copytree``. The resulting path is ``dest_dir / basename(src)``.
    Refuses overwrite (409).
    """
    src = Path(os.path.expanduser(req.src)).resolve()
    dest_dir = Path(os.path.expanduser(req.dest_dir)).resolve()
    target = _resolve_copy_target(src, dest_dir)
    try:
        if src.is_dir():
            shutil.copytree(src, target, symlinks=False)
        else:
            shutil.copy2(src, target)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"copy failed: {e}")
    return FsPathResponse(path=str(target))


@router.post("/fs/move", response_model=FsPathResponse)
def move(req: CopyOrMoveRequest):
    """Move a file or directory under ``dest_dir``.

    Uses ``shutil.move`` so cross-device moves degrade gracefully into
    copy+unlink. Refuses overwrite — collisions need a rename first.
    """
    src = Path(os.path.expanduser(req.src)).resolve()
    dest_dir = Path(os.path.expanduser(req.dest_dir)).resolve()
    target = _resolve_copy_target(src, dest_dir)
    try:
        shutil.move(str(src), str(target))
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"move failed: {e}")
    return FsPathResponse(path=str(target))


class DeleteFileRequest(BaseModel):
    path: str
    confirmed: bool = False


class DeleteFileResponse(BaseModel):
    deleted: str
    removed_bytes: int = 0


@router.post("/fs/delete-file", response_model=DeleteFileResponse)
def delete_file(req: DeleteFileRequest):
    """Delete a single regular file.

    Companion to ``/fs/delete-dir`` for file-grained operations (e.g.
    deleting a config or a template). Requires ``confirmed=true``,
    rejects symlinks, refuses anything that isn't a regular file, and
    enforces the same path-depth floor (>=4 components) so a careless
    call can't nuke a top-level system file. The dir-shaped denylist
    doesn't apply here because the depth floor already covers it.
    """
    if not req.confirmed:
        raise HTTPException(status_code=400, detail="delete requires confirmed=true")
    target = Path(os.path.expanduser(req.path)).resolve()
    if not target.is_absolute():
        raise HTTPException(status_code=400, detail="path must be absolute")
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"path does not exist: {target}")
    if target.is_symlink():
        raise HTTPException(
            status_code=400, detail="refusing to follow or delete symlink"
        )
    if not target.is_file():
        raise HTTPException(status_code=400, detail=f"not a regular file: {target}")
    depth = len([p for p in target.parts if p and p != "/"])
    if depth < 4:
        raise HTTPException(
            status_code=403,
            detail=(
                f"refusing to delete a path this shallow: {target} "
                f"(need at least 4 components)"
            ),
        )

    try:
        size = target.stat().st_size
    except OSError:
        size = 0
    log.warning("deleting file: %s (%d bytes)", target, size)
    try:
        os.remove(target)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"unlink failed: {e}")
    return DeleteFileResponse(deleted=str(target), removed_bytes=size)


@router.post("/fs/delete-dir", response_model=DeleteDirResponse)
def delete_dir(req: DeleteDirRequest):
    """Recursively delete ``req.path``.

    Intended for clearing a config's output directory so training can
    start fresh. Guarded by :func:`_reject_unsafe` and by an explicit
    ``confirmed`` flag in the body. The client should also surface a
    user-facing confirmation before calling.
    """
    if not req.confirmed:
        raise HTTPException(status_code=400, detail="delete requires confirmed=true")
    target = Path(os.path.expanduser(req.path)).resolve()
    _reject_unsafe(target)

    # Count size before we nuke it so the UI has something to report.
    removed_bytes = 0
    try:
        for root, _, files in os.walk(target, followlinks=False):
            for name in files:
                try:
                    removed_bytes += os.stat(
                        os.path.join(root, name), follow_symlinks=False
                    ).st_size
                except OSError:
                    pass
    except Exception:
        pass

    log.warning("deleting directory: %s (%d bytes)", target, removed_bytes)
    try:
        shutil.rmtree(target)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rmtree failed: {e}")

    return DeleteDirResponse(deleted=str(target), removed_bytes=removed_bytes)
