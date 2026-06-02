"""Garbage collection helpers for the server's on-disk state.

The server captures the stdout/stderr of every job it dispatches to a
``q_<queue_id>.tty`` file under :func:`paths.jobs_tty_dir`. Two subsystems
in this module manage those files:

1. :func:`relocate_tty_to_logs` — runs at terminal-status time. For
   training jobs (which expose a ``logs_dir`` after PID correlation), the
   captured TTY file is moved into the run's ``logs/tty.log`` so the
   ``output_models/<run>/`` directory becomes self-contained the moment
   the job finishes. The scheduler had already symlinked
   ``logs/tty.log`` to the central file during the running phase
   (see ``scheduler._try_link_tty``); this function atomically replaces
   that symlink with the real file.

2. :func:`sweep_orphan_ttys` — runs on server startup and on a periodic
   tick. Walks :func:`paths.jobs_tty_dir` for ``q_*.tty`` files whose
   ``queue_id`` is no longer referenced by any JobRecord or queued
   QueueItem and whose mtime is older than a TTL. These are leftovers
   from non-training jobs that were removed without going through the
   relocation path, or relics of crashed servers.

Both helpers are best-effort: per-file errors are logged and swallowed,
never propagated to the scheduler tick that calls them.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from . import job_records, queue_store
from .job_records import JobRecord
from .paths import jobs_tty_dir

log = logging.getLogger("forgather_server.gc")

# Default age threshold for the orphan sweep. A small safety margin guards
# against deleting a TTY that's about to be claimed by a JobRecord write.
ORPHAN_TTY_TTL_SECONDS = int(os.environ.get("FORGATHER_ORPHAN_TTY_TTL_SECONDS", "3600"))

# Age threshold for the endpoint-dir sweep. Reuses the env var the removed
# ``forgather control cleanup`` honored, for operator continuity. Guarded so a
# bad value can't break the module import (and thus the scheduler).
try:
    ORPHAN_ENDPOINT_TTL_SECONDS = int(
        os.environ.get("FORGATHER_ORPHAN_JOB_DIR_TTL_SECONDS", "3600")
    )
except ValueError:
    ORPHAN_ENDPOINT_TTL_SECONDS = 3600


def relocate_tty_to_logs(record: JobRecord) -> Optional[Path]:
    """Move the central TTY file into the run's ``logs/`` directory.

    Returns the new path on success, ``None`` if relocation was skipped
    (no logs_dir, or TTY already lives outside the central directory) or
    failed. Updates ``record.tty_log_path`` in the persistent store on
    success.

    The destination is ``<logs_dir>/tty.log``, which during the running
    phase is a symlink back to the source. ``shutil.move`` operates on
    the symlink itself (not the link target), so the symlink is replaced
    atomically when source and destination share a filesystem.
    Cross-filesystem moves fall through to copy + unlink.
    """
    if not record.logs_dir or not record.tty_log_path:
        return None

    central_dir = jobs_tty_dir().resolve()
    try:
        src = Path(record.tty_log_path).resolve(strict=False)
    except OSError:
        return None

    # Only relocate files we own — i.e. those still under the central
    # jobs_tty_dir. Jobs whose tty_log_path was already moved (or was
    # never in the central dir to begin with) are left alone.
    try:
        src.relative_to(central_dir)
    except ValueError:
        return None

    if not src.exists():
        log.debug("relocate skipped: %s does not exist", src)
        return None

    dst_dir = Path(record.logs_dir)
    dst = dst_dir / "tty.log"

    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning("could not create logs dir %s: %s", dst_dir, e)
        return None

    # Pre-remove the destination — it's typically a symlink pointing
    # back at src that the scheduler created during PID correlation.
    # shutil.move would otherwise either replace it (rename path) or
    # raise (copy path), so make the behavior uniform.
    if dst.is_symlink() or dst.exists():
        try:
            dst.unlink()
        except OSError as e:
            log.warning("could not remove existing %s: %s", dst, e)
            return None

    try:
        shutil.move(str(src), str(dst))
    except OSError as e:
        log.warning(
            "could not relocate tty %s -> %s: %s; leaving in central dir",
            src,
            dst,
            e,
        )
        return None

    # Update the record so the WS tail endpoint and any other reader
    # follow the new location. update_record does an atomic JSON write.
    job_records.update_record(record.queue_id, tty_log_path=str(dst))
    log.info("relocated tty %s -> %s", src, dst)
    return dst


def sweep_orphan_ttys(ttl_seconds: int = ORPHAN_TTY_TTL_SECONDS) -> int:
    """Delete ``q_*.tty`` files no record references and that are older than TTL.

    Returns the number of files removed. The TTL guards against a race
    where the scheduler is mid-launch — the tty file is created in
    :func:`launcher.spawn_*` slightly before the matching ``JobRecord``
    has been persisted. A few minutes of slack is enough.

    Live ids are gathered from both ``job_records.list_records`` (covers
    running and terminal-but-not-yet-removed records) and
    ``queue_store.list_items`` (covers items dispatched on this very
    tick that may not have a record yet). Both lists are taken
    independently with their own locks; a brief race is harmless because
    fresh files are protected by the TTL.
    """
    tty_dir = jobs_tty_dir()
    try:
        entries = list(tty_dir.iterdir())
    except FileNotFoundError:
        return 0

    live_ids = {r.queue_id for r in job_records.list_records()}
    live_ids.update(it.queue_id for it in queue_store.list_items())

    now = time.time()
    removed = 0
    for entry in entries:
        if not entry.is_file() or entry.suffix != ".tty":
            continue
        queue_id = entry.stem
        if queue_id in live_ids:
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if now - mtime < ttl_seconds:
            continue
        try:
            entry.unlink()
            removed += 1
            log.info("swept orphan tty %s", entry.name)
        except OSError as e:
            log.warning("could not unlink %s: %s", entry, e)
    return removed


def _pid_definitely_gone(pid: int) -> bool:
    """True only if ``pid`` provably refers to no live process.

    Deliberately conservative: returns ``False`` when liveness is
    *indeterminate* (e.g. the process exists but is owned by another user and
    psutil can't inspect it). The endpoint-dir sweep deletes a directory only
    when its trainer is **proven** gone, never when we merely can't tell —
    deleting a live trainer's control dir (auth token + pending commands) is far
    worse than leaking a stale dir.
    """
    try:
        import psutil

        try:
            psutil.Process(pid)
            return False  # exists (alive / zombie / not-inspectable)
        except psutil.NoSuchProcess:
            return True
        except psutil.AccessDenied:
            return False  # exists, owned by another user
    except ImportError:
        try:
            os.kill(pid, 0)
            return False
        except ProcessLookupError:
            return True
        except PermissionError:
            return False  # exists, different user


def sweep_dead_endpoint_dirs(ttl_seconds: int = ORPHAN_ENDPOINT_TTL_SECONDS) -> int:
    """Remove stale TrainerControlCallback endpoint dirs under ``<config>/jobs/``.

    A trainer writes ``~/.config/forgather/jobs/<job_id>/`` (endpoint.json,
    auth_token, control_dir) and removes it on a clean exit; a crashed or
    SIGKILLed trainer leaves it behind. This sweep restores the reaper the
    removed ``forgather control cleanup`` provided.

    A dir is removed only when it is older than the TTL AND one of:
    its ``endpoint.json`` parses and its PID is **provably gone**, or it has no
    ``endpoint.json`` at all (a true orphan). A dir is **kept** whenever the
    endpoint is live, the PID can't be proven dead (indeterminate / another
    user), or ``endpoint.json`` is present but unparseable (could be a live
    trainer whose schema we don't recognize). The TTL alone is NOT treated as
    proof of death — a trainer's dir mtime is frozen at startup, so a
    long-running trainer is older than the TTL while very much alive; its live
    PID is what protects it.

    Returns the number of dirs removed. Best-effort: per-dir errors are logged
    and swallowed.
    """
    import json as _json

    from forgather.preprocess import forgather_config_dir

    jobs_dir = Path(forgather_config_dir()) / "jobs"
    try:
        entries = list(jobs_dir.iterdir())
    except FileNotFoundError:
        return 0

    now = time.time()
    removed = 0
    for entry in entries:
        # Don't follow symlinks (and rmtree refuses them anyway).
        if entry.is_symlink() or not entry.is_dir():
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if now - mtime < ttl_seconds:
            continue  # fresh; protect a just-started trainer mid-write

        ep_file = entry / "endpoint.json"
        if ep_file.exists():
            try:
                pid = _json.loads(ep_file.read_text()).get("pid")
            except Exception:
                # Present but unparseable — could be a live trainer with a
                # schema we don't recognize. Leave it alone.
                continue
            if pid is None or not _pid_definitely_gone(int(pid)):
                continue  # alive or indeterminate — keep
        # else: no endpoint.json and older than the TTL → a genuine orphan
        # (a live trainer writes endpoint.json at startup, well within the TTL).

        try:
            shutil.rmtree(entry)
            removed += 1
            log.info("swept dead endpoint dir %s", entry.name)
        except OSError as e:
            log.warning("could not remove %s: %s", entry, e)
    return removed


def delete_central_tty_for(record: JobRecord) -> bool:
    """Unlink the TTY file iff it still lives in the central jobs_tty_dir.

    Used by the record-removal API: when the user deletes a JobRecord
    whose TTY was never relocated (typically a non-training job that
    has no logs_dir), we should free the central file rather than
    leaving it for the orphan sweep.

    Returns True if a file was removed, False otherwise.
    """
    if not record.tty_log_path:
        return False

    central_dir = jobs_tty_dir().resolve()
    try:
        path = Path(record.tty_log_path).resolve(strict=False)
    except OSError:
        return False

    try:
        path.relative_to(central_dir)
    except ValueError:
        # tty_log_path points elsewhere (already relocated into a run dir).
        # Leave it alone — it's part of the run's artifacts now.
        return False

    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as e:
        log.warning("could not unlink central tty %s: %s", path, e)
        return False
    return True
