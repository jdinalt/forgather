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
"""

import os
from pathlib import Path


def atomic_write_text(path: Path, content: str) -> None:
    """Write *content* to *path* atomically.

    Creates the parent directory if it does not exist, writes to a sibling
    ``.tmp`` file, fsyncs the fd, then renames into place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Binary equivalent of :func:`atomic_write_text`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
