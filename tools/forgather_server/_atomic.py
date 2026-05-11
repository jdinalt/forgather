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
4. **Mode applied before any data is written** — when ``mode`` is given the
   tmp fd is opened with that mode directly via ``os.open`` and the mode is
   re-asserted via ``os.fchmod`` (defeats the process umask, which would
   otherwise mask bits off the mode passed to ``os.open``). The previous
   implementation called ``open(tmp, "w")`` then ``os.chmod`` after-the-
   fact, leaving a brief window where another local user could open the
   newly-created file at the umask-default mode (typically 0o644) and
   read sensitive content as it was being written.

Without a ``mode`` argument the file inherits the process umask, which is
what user-content writes (template editor saves) want.
"""

import os
from pathlib import Path
from typing import Optional


def _open_tmp_with_mode(tmp: Path, mode: Optional[int]) -> int:
    """Return a freshly-created tmp fd. Applies ``mode`` atomically at
    creation (subject to umask, then re-asserted via fchmod) when given.
    ``O_EXCL`` is intentionally NOT used — same-port restarts and other
    retries depend on overwriting a stale tmp from a previous run.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if mode is None:
        return os.open(str(tmp), flags)
    fd = os.open(str(tmp), flags, mode)
    try:
        os.fchmod(fd, mode)
    except OSError:
        pass
    return fd


def atomic_write_text(path: Path, content: str, *, mode: Optional[int] = None) -> None:
    """Write *content* to *path* atomically.

    Creates the parent directory if it does not exist, writes to a sibling
    ``.tmp`` file, fsyncs the fd, then renames into place. When ``mode`` is
    provided the tmp file is created with that mode AND has it re-asserted
    via fchmod — sensitive content is never readable at the umask-default
    mode, even momentarily.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = _open_tmp_with_mode(tmp, mode)
    with os.fdopen(fd, "w") as f:
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
    fd = _open_tmp_with_mode(tmp, mode)
    with os.fdopen(fd, "wb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
