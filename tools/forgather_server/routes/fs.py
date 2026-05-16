"""Filesystem-browsing endpoints used by the directory-picker UI.

Localhost-prototype caveat: these endpoints reveal arbitrary directory
contents to any caller that can reach the bound port. The server binds to
127.0.0.1 by default, and no auth is configured — matching the security
posture of the rest of the prototype. Do not expose the port on an untrusted
network.
"""

import logging
import os
import re
import shutil
import time
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


def _reject_symlink_in_chain(raw: str) -> None:
    """Refuse if the raw user-supplied path itself OR any ancestor is a
    symlink.

    Callers normally ``.resolve()`` the path before passing into safety
    helpers, which silently follows symlinks — so an ``is_symlink()``
    check on the resolved Path is dead code (resolve() collapses links).
    Walk the *unresolved* chain and refuse if any component is a link;
    that's the only place a symlink can hide before resolve hides it.
    """
    p = os.path.abspath(os.path.expanduser(raw))
    walk = p
    while True:
        if os.path.islink(walk):
            raise HTTPException(
                status_code=400,
                detail=f"refusing to operate on path containing symlink: {walk}",
            )
        parent = os.path.dirname(walk)
        if parent == walk:
            break
        walk = parent


def _reject_unsafe(target: Path, *, raw: Optional[str] = None) -> None:
    """Reject paths that are obvious catastrophic-delete candidates.

    Defensive cheap checks: must be absolute, must exist, must be a
    directory, must be at least 4 path components deep (so
    ``/foo/bar/baz`` passes and ``/etc`` doesn't), and must not match a
    denylist of common system roots.

    When ``raw`` is provided, also walks the unresolved path chain and
    refuses any component that is a symlink. ``target`` is expected to
    already be ``.resolve()``-d by the caller; on a resolved Path,
    ``is_symlink()`` is always False, so the symlink guard *must* run on
    the raw input.
    """
    if raw is not None:
        _reject_symlink_in_chain(raw)
    if not target.is_absolute():
        raise HTTPException(status_code=400, detail="path must be absolute")
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"path does not exist: {target}")
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


def _check_path_safe(
    target: Path,
    *,
    must_exist: bool = True,
    raw: Optional[str] = None,
) -> None:
    """Cheap sanity checks shared by rename / copy / move.

    - Absolute path required.
    - Refuses symlinks anywhere in the unresolved path chain (only when
      ``raw`` is supplied — resolved Paths can't reveal links because
      ``Path.resolve()`` already chased them).
    - When ``must_exist`` is true: refuses missing paths.
    - Depth floor: ≥ 4 path components, so a typo can't accidentally
      target ``/etc`` or similar.
    """
    if raw is not None:
        _reject_symlink_in_chain(raw)
    if not target.is_absolute():
        raise HTTPException(status_code=400, detail="path must be absolute")
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
    _check_path_safe(src, raw=req.path)
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
    ``dest_dir / basename(src)``.

    ``auto_rename`` only affects ``/fs/copy``: when set and the
    destination already exists, the server picks a non-colliding name
    by appending ``" (copy)"`` (then ``" (copy 2)"``, etc.) to the
    stem. This is how the webui implements paste-into-same-dir
    duplication and the right-click "Duplicate" actions. Move ignores
    the flag — moving a file is by definition a rename, and silently
    auto-renaming the destination would surprise the operator.
    """

    src: str
    dest_dir: str
    auto_rename: bool = False
    # Override the destination basename. Defaults to ``basename(src)``.
    # Used by the "Duplicate Config…" flow in the webui, where the
    # operator types the new config's name in a prompt before the
    # copy goes out. Must be a single filename component — no path
    # separators.
    target_name: Optional[str] = None


# Match a trailing " (copy)" or " (copy N)" on a path stem so we can
# strip it before computing the next candidate — keeps repeated
# duplicates as "foo (copy 2)" instead of accumulating
# "foo (copy) (copy)".
_COPY_SUFFIX_RE = re.compile(r" \(copy(?: \d+)?\)$")


def _next_available_copy_name(target: Path) -> Path:
    """Return ``target`` if free, else a sibling whose basename has a
    ``" (copy)"`` / ``" (copy N)"`` suffix appended to the stem until
    a free name is found.

    Preserves the original extension (``foo.yaml`` →
    ``foo (copy).yaml``). On a clean stem the first candidate is
    ``" (copy)"``; on a stem that already carries the suffix the
    increment continues from there (``foo (copy)`` → ``foo (copy 2)``).
    """
    parent = target.parent
    stem = target.stem
    suffix = target.suffix
    base = _COPY_SUFFIX_RE.sub("", stem)
    # Cap at a sane upper bound to prevent a degenerate loop if the
    # filesystem layer were lying about existence; in practice the
    # first one or two iterations always win.
    for n in range(1, 1000):
        marker = " (copy)" if n == 1 else f" (copy {n})"
        candidate = parent / f"{base}{marker}{suffix}"
        if not candidate.exists():
            return candidate
    raise HTTPException(
        status_code=500,
        detail=f"could not find a free copy name under {parent}",
    )


def _resolve_copy_target(
    src_path: Path,
    dest_dir_path: Path,
    *,
    src_raw: Optional[str] = None,
    dest_raw: Optional[str] = None,
    auto_rename: bool = False,
    target_name: Optional[str] = None,
) -> Path:
    """Compute the resolved destination ``dest_dir / target_name`` (or
    ``basename(src)`` when ``target_name`` is None) and enforce safety:
    parent must be a real directory, both ends pass
    ``_check_path_safe``, and the target itself must not yet exist
    (unless ``auto_rename`` is set, in which case a non-colliding
    sibling name is generated)."""
    _check_path_safe(src_path, raw=src_raw)
    _check_path_safe(dest_dir_path, raw=dest_raw)
    if not dest_dir_path.is_dir():
        raise HTTPException(
            status_code=400, detail=f"dest_dir is not a directory: {dest_dir_path}"
        )
    if target_name is not None:
        # Reject anything that looks like a path — basename only.
        cleaned = target_name.strip()
        if not cleaned or "/" in cleaned or "\\" in cleaned or cleaned in (".", ".."):
            raise HTTPException(
                status_code=400,
                detail=f"target_name must be a single filename: {target_name!r}",
            )
        target = dest_dir_path / cleaned
    else:
        target = dest_dir_path / src_path.name
    if target.exists():
        if not auto_rename:
            raise HTTPException(
                status_code=409, detail=f"already exists: {target}"
            )
        target = _next_available_copy_name(target)
    # Ensure the resulting path also clears the depth floor (it should
    # always — dest_dir already passed — but defense in depth).
    _check_path_safe(target, must_exist=False)
    return target


@router.post("/fs/copy", response_model=FsPathResponse)
def copy_path(req: CopyOrMoveRequest):
    """Copy a file or directory under ``dest_dir``.

    Files: ``shutil.copy2`` (preserves metadata). Directories:
    ``shutil.copytree``. The resulting path is normally
    ``dest_dir / basename(src)``; when ``auto_rename`` is set and that
    would collide, the server appends ``" (copy)"`` / ``" (copy N)"``
    to the stem until it finds a free name.
    """
    src = Path(os.path.expanduser(req.src)).resolve()
    dest_dir = Path(os.path.expanduser(req.dest_dir)).resolve()
    target = _resolve_copy_target(
        src,
        dest_dir,
        src_raw=req.src,
        dest_raw=req.dest_dir,
        auto_rename=req.auto_rename,
        target_name=req.target_name,
    )
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
    target = _resolve_copy_target(src, dest_dir, src_raw=req.src, dest_raw=req.dest_dir)
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
    # Check the unresolved chain for a symlink before resolving — once
    # ``.resolve()`` follows it, ``is_symlink()`` returns False on the
    # resolved path and the guard does nothing.
    _reject_symlink_in_chain(req.path)
    target = Path(os.path.expanduser(req.path)).resolve()
    if not target.is_absolute():
        raise HTTPException(status_code=400, detail="path must be absolute")
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"path does not exist: {target}")
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
    _reject_unsafe(target, raw=req.path)

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

    # On NFS-shared trees the operator typically follows Clean Output with a
    # restart of training on another node. That node's NFS client caches the
    # parent directory's attributes / negative lookups for up to ``acdirmax``
    # seconds (Linux default ~60s), so the just-deleted contents can briefly
    # appear to "still be there" — or, after a re-create, the re-created
    # contents can briefly look "missing". There is no NFS-level way for the
    # server to push cache invalidations to clients; the best we can do is:
    #
    #   1. ``os.sync()`` — flushes writeback so a server crash mid-cleanup
    #      doesn't leave half-deleted state on disk. Doesn't touch client
    #      caches but it's cheap insurance on the storage side.
    #   2. ``os.utime`` on the parent dir — bumps mtime so the next client
    #      lookup that *does* go over the wire (e.g. after acdirmin expires)
    #      sees a fresher value and is more likely to invalidate its cached
    #      child list. Already implicit in the rmtree of a leaf; re-issuing
    #      it explicitly is belt-and-suspenders.
    #
    # The real fix for cross-node "I just deleted that, why is it back?"
    # remains an NFS mount-side knob (``actimeo=1`` / ``noac`` / equivalent)
    # — see docs/operations/nfs-caching.md if/when that page exists.
    try:
        os.sync()
    except Exception as e:
        log.debug("post-rmtree os.sync() failed (non-fatal): %s", e)
    try:
        parent = target.parent
        if parent.is_dir():
            now = time.time()
            os.utime(parent, (now, now))
    except Exception as e:
        log.debug("post-rmtree parent utime failed (non-fatal): %s", e)

    return DeleteDirResponse(deleted=str(target), removed_bytes=removed_bytes)
