"""Crash-atomic file write helpers.

Every persistent-state write in the Forgather server should go through these
helpers to guarantee:

1. **Tmp + os.replace** — the target file is never in a partially-written
   state visible to readers.  A bare ``open(path, "w")`` truncates first,
   leaving a zero-byte window; tmp+rename closes that window.
2. **fsync before rename** — ``os.replace`` can return before the kernel
   flushes dirty pages.  Without fsync a crash can leave the *renamed* file
   intact but with stale (or zero) content.
3. **Tmp in the same directory** — so the rename stays on one filesystem and
   is truly atomic (POSIX).  Cross-device renames fall back to copy+unlink.

The optional ``mode`` parameter chmods the tmp file *before* writing, so
sensitive content (auth tokens, password hashes, anything in
``~/.config/forgather/server/``) is never readable on disk during the write
window. Without a mode argument the file inherits the process umask, which
is what user-content writes (template editor saves) want.
"""

import os
from pathlib import Path
from typing import Optional


def atomic_write_text(path: Path, content: str, *, mode: Optional[int] = None) -> None:
    """Write *content* to *path* atomically.

    Creates the parent directory if it does not exist, writes to a sibling
    ``.tmp`` file, fsyncs the fd, then renames into place. When ``mode`` is
    provided the tmp file is chmod'd to it after creation but before the
    write, closing the window where a sensitive file is briefly readable
    by other users.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        if mode is not None:
            try:
                os.chmod(tmp, mode)
            except OSError:
                pass
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_bytes(
    path: Path, content: bytes, *, mode: Optional[int] = None
) -> None:
    """Binary equivalent of :func:`atomic_write_text`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        if mode is not None:
            try:
                os.chmod(tmp, mode)
            except OSError:
                pass
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
