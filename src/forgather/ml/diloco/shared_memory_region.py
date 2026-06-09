"""
Shared-memory region mechanics for single-host DiLoCo (issue #154).

The low-level byte layout, cross-process ``flock``, control header, manifest,
and fp32 master/accumulator views — factored out of
:mod:`shared_memory_backend` so the **worker** side
(:class:`~forgather.ml.diloco.shared_memory_backend.SharedMemoryBackend`,
follower) and the **server** side
(:class:`~forgather.ml.diloco.shared_memory_aggregator.SharedMemoryAggregator`,
aggregator) share one definition of the region. Two independent
implementations of the layout would be a correctness landmine (a drifting
header index silently corrupts every sync), so both import this.

Region (a memory-mapped file under a per-group dir, the path as the rendezvous):

* ``manifest.json`` — canonical param names / shapes / fp32 offsets, written
  once by the region's creator (the aggregator); attachers poll for it.
* ``region.bin`` — mmap: a small int64 control header (magic, generation,
  arrival count, group size, live attach count) + the master params (fp32) +
  an accumulator (fp32, same size). The attach count drives self-cleanup.
* ``region.lock`` — the rendezvous mutex (a persistent ``flock``).
* ``owner.lock`` — the aggregator's process-lifetime ownership lease.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import mmap
import os
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    StateDict = Dict[str, torch.Tensor]

_MAGIC = 0x44494C4F434F0001  # "DILOCO" + version
_HEADER_WORDS = 8  # int64 control words
_HEADER_BYTES = _HEADER_WORDS * 8
# Control-word indices.
_W_MAGIC = 0
_W_GENERATION = 1
_W_ARRIVALS = 2
_W_GROUP_SIZE = 3
_W_ATTACH = 4  # live attach count; last worker out (==0) unlinks the region

_SHM_SUBDIR = "diloco_shm"
_MANIFEST = "manifest.json"
_REGION = "region.bin"
_OWNER_LOCK = "owner.lock"  # aggregator's process-lifetime ownership lease
_REGION_LOCK = "region.lock"  # rendezvous mutex
_POLL_INTERVAL = 0.01


class ShmRegion:
    """The shared region's on-disk mechanics, with no role policy.

    Construct with the rendezvous ``group_dir``; the creator additionally
    supplies the master ``state_dict`` + ``group_size`` to :meth:`create`,
    attachers call :meth:`attach`. The control header and the master /
    accumulator fp32 views are exposed for the aggregator's step; all header
    mutation must happen under :meth:`locked`.
    """

    def __init__(self, group_dir: str):
        # realpath (not abspath) so co-located peers handed a symlinked or
        # relative group dir still resolve to the same rendezvous.
        self.group_dir = os.path.realpath(group_dir)
        self._shm_dir = os.path.join(self.group_dir, _SHM_SUBDIR)
        self._region_path = os.path.join(self._shm_dir, _REGION)
        self._manifest_path = os.path.join(self._shm_dir, _MANIFEST)
        self._lock_path = os.path.join(self._shm_dir, _REGION_LOCK)
        self._owner_lock_path = os.path.join(self._shm_dir, _OWNER_LOCK)
        self._lock_fd: Optional[int] = None
        self._owner_lock_fd: Optional[int] = None

        self._names: List[str] = []
        self._layout: Dict[str, tuple] = {}  # name -> (float_offset, numel, shape)
        self._total_floats = 0
        self._fd: Optional[int] = None
        self._mm: Optional[mmap.mmap] = None
        self._ctrl: Optional[np.ndarray] = None
        self._master_t: Optional[torch.Tensor] = None  # 1-D fp32 view (shared)
        self._accum_t: Optional[torch.Tensor] = None  # 1-D fp32 view (shared)

    # ----- properties -------------------------------------------------------

    @property
    def names(self) -> List[str]:
        return self._names

    @property
    def total_floats(self) -> int:
        return self._total_floats

    @property
    def ctrl(self) -> Optional[np.ndarray]:
        return self._ctrl

    @property
    def shm_dir(self) -> str:
        return self._shm_dir

    @property
    def manifest_path(self) -> str:
        return self._manifest_path

    @property
    def region_path(self) -> str:
        return self._region_path

    def generation(self) -> int:
        return int(self._ctrl[_W_GENERATION])

    def arrivals(self) -> int:
        return int(self._ctrl[_W_ARRIVALS])

    def group_size(self) -> int:
        return int(self._ctrl[_W_GROUP_SIZE])

    def attach_count(self) -> int:
        return int(self._ctrl[_W_ATTACH])

    def magic(self) -> int:
        return int(self._ctrl[_W_MAGIC])

    # ----- locking + ownership ----------------------------------------------

    @contextlib.contextmanager
    def locked(self):
        """A real cross-process mutex: a persistent ``flock`` on a long-lived
        lock fd. (Deliberately NOT ``construct.file_lock_build`` — that unlinks
        the lock file on release, which breaks ``flock`` mutual exclusion under
        the high-frequency re-acquisition this region does every round.) The fd
        is opened once and never unlinked; the OS releases the lock if the
        holder dies, so a crashed peer cannot deadlock the survivors.
        """
        if self._lock_fd is None:
            os.makedirs(self._shm_dir, exist_ok=True)
            self._lock_fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def try_acquire_ownership(self) -> bool:
        """Try to take the aggregator ownership lease.

        A non-blocking exclusive ``flock`` on ``owner.lock``, held for this
        process's lifetime when acquired. Returns True if this process is the
        aggregator. Call under :meth:`locked` so the accept/attach decision is
        serialized across co-located peers — exactly one wins. The OS frees the
        lease when the holder dies, so a crashed aggregator's lease is
        reclaimable by the next launch.
        """
        if self._owner_lock_fd is None:
            self._owner_lock_fd = os.open(
                self._owner_lock_path, os.O_CREAT | os.O_RDWR, 0o644
            )
        try:
            fcntl.flock(self._owner_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            # EWOULDBLOCK / EAGAIN: a live aggregator holds the lease.
            return False

    def release_ownership(self) -> None:
        """Close the owner-lock fd, dropping the lease (no-op for an attacher,
        which opened the fd but never acquired the flock). Idempotent."""
        try:
            if self._owner_lock_fd is not None:
                os.close(self._owner_lock_fd)
                self._owner_lock_fd = None
        except OSError:
            pass

    def discard_stale(self) -> None:
        """Unlink a region orphaned by a crashed prior group.

        Safe only while holding the ownership lease — no live aggregator can be
        using these files. Only the data files (manifest + region) are removed;
        ``region.lock`` (the rendezvous mutex) and ``owner.lock`` (the lease)
        are deliberately left in place — unlinking the mutex path here would
        split it for the next joiner.
        """
        for path in (self._manifest_path, self._region_path):
            try:
                os.unlink(path)
            except OSError:
                pass

    # ----- layout + mapping -------------------------------------------------

    def _build_layout(self, state_dict: "StateDict") -> None:
        names = list(state_dict.keys())
        layout = {}
        offset = 0
        for name in names:
            t = state_dict[name]
            numel = t.numel()
            layout[name] = (offset, numel, tuple(t.shape))
            offset += numel
        self._names = names
        self._layout = layout
        self._total_floats = offset

    def _map(self) -> None:
        params_bytes = self._total_floats * 4
        region_size = _HEADER_BYTES + 2 * params_bytes
        self._fd = os.open(self._region_path, os.O_RDWR)
        self._mm = mmap.mmap(self._fd, region_size)
        self._ctrl = np.frombuffer(self._mm, dtype=np.int64, count=_HEADER_WORDS)
        master_np = np.frombuffer(
            self._mm, dtype=np.float32, count=self._total_floats, offset=_HEADER_BYTES
        )
        accum_np = np.frombuffer(
            self._mm,
            dtype=np.float32,
            count=self._total_floats,
            offset=_HEADER_BYTES + params_bytes,
        )
        self._master_t = torch.from_numpy(master_np)
        self._accum_t = torch.from_numpy(accum_np)

    def create(self, master: "StateDict", group_size: int) -> None:
        """Create + map the region, seed the master buffer, write the manifest.

        Caller must hold the ownership lease and the rendezvous lock. The
        manifest is published last — its presence is the "region ready" signal
        attachers poll for.
        """
        master = {k: v.detach().float().cpu().contiguous() for k, v in master.items()}
        self._build_layout(master)

        params_bytes = self._total_floats * 4
        region_size = _HEADER_BYTES + 2 * params_bytes
        with open(self._region_path, "wb") as fh:
            fh.truncate(region_size)
        self._map()

        # Initialize control + master buffer; zero accumulator (truncate gives
        # zeros, but be explicit).
        self._accum_t.zero_()
        for name in self._names:
            self.slice(self._master_t, name).copy_(master[name].reshape(-1))
        self._ctrl[_W_GENERATION] = 0
        self._ctrl[_W_ARRIVALS] = 0
        self._ctrl[_W_GROUP_SIZE] = int(group_size)
        self._ctrl[_W_MAGIC] = _MAGIC

        manifest = {
            "names": self._names,
            "shapes": {n: list(self._layout[n][2]) for n in self._names},
            "offsets": {n: self._layout[n][0] for n in self._names},
            "numels": {n: self._layout[n][1] for n in self._names},
            "total_floats": self._total_floats,
            "group_size": int(group_size),
        }
        tmp = self._manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        os.replace(tmp, self._manifest_path)

    def attach(self) -> dict:
        """Read the manifest, map the region, validate magic. Returns the
        manifest dict (so the caller can cross-check group_size). Raises on a
        magic mismatch (closes the mapping first)."""
        with open(self._manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        self._names = list(manifest["names"])
        self._layout = {
            name: (
                manifest["offsets"][name],
                manifest["numels"][name],
                tuple(manifest["shapes"][name]),
            )
            for name in self._names
        }
        self._total_floats = int(manifest["total_floats"])
        self._map()
        if self.magic() != _MAGIC:
            self.close()
            raise RuntimeError("ShmRegion: region magic mismatch")
        return manifest

    # ----- views ------------------------------------------------------------

    def slice(self, t: torch.Tensor, name: str) -> torch.Tensor:
        off, numel, _shape = self._layout[name]
        return t[off : off + numel]

    def master_slice(self, name: str) -> torch.Tensor:
        return self.slice(self._master_t, name)

    def accum_slice(self, name: str) -> torch.Tensor:
        return self.slice(self._accum_t, name)

    def read_master_snapshot(self) -> "StateDict":
        out = {}
        for name in self._names:
            _off, _numel, shape = self._layout[name]
            out[name] = self.slice(self._master_t, name).clone().reshape(shape)
        return out

    # ----- attach accounting (caller holds :meth:`locked`) ------------------

    def incr_attach(self) -> int:
        """Count one more live attacher; returns the new count. The attach
        count drives self-cleanup: the last one out (==0) unlinks the region."""
        n = int(self._ctrl[_W_ATTACH]) + 1
        self._ctrl[_W_ATTACH] = n
        return n

    def decr_attach(self) -> int:
        """Drop one attacher; returns the remaining count (never below 0)."""
        n = max(0, int(self._ctrl[_W_ATTACH]) - 1)
        self._ctrl[_W_ATTACH] = n
        return n

    def mark_dead(self) -> None:
        """Clear the magic word so any out-of-lifecycle attacher that still
        maps the region fails loud on the magic check."""
        self._ctrl[_W_MAGIC] = 0

    # ----- round bookkeeping (caller holds :meth:`locked`) ------------------

    def shape(self, name: str) -> tuple:
        return self._layout[name][2]

    def zero_accum(self) -> None:
        self._accum_t.zero_()

    def set_arrivals(self, n: int) -> None:
        self._ctrl[_W_ARRIVALS] = int(n)

    def bump_generation(self) -> int:
        """Advance the generation by one; returns the new value."""
        n = int(self._ctrl[_W_GENERATION]) + 1
        self._ctrl[_W_GENERATION] = n
        return n

    # ----- teardown ---------------------------------------------------------

    def close(self) -> None:
        """Close this process's mmap + region fd (not the lock fds)."""
        try:
            if self._mm is not None:
                self._ctrl = None
                self._master_t = None
                self._accum_t = None
                self._mm.close()
                self._mm = None
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
        except OSError:
            pass

    def cleanup_files(self) -> None:
        """Unlink the shared region + lock files and remove the dirs if empty.

        Best-effort — a group that *crashes* (no clean last leave) can still
        leave files behind, but that's no longer a hazard: the ownership lease
        means the next launch reclaims and rebuilds an orphaned region rather
        than attaching to it.
        """
        for path in (
            self._region_path,
            self._manifest_path,
            self._lock_path,
            self._owner_lock_path,
        ):
            try:
                os.unlink(path)
            except OSError:
                pass
        for d in (self._shm_dir, self.group_dir):
            try:
                os.rmdir(d)
            except OSError:
                pass

    def close_lock(self) -> None:
        """Close the rendezvous-lock fd (dropping any held flock)."""
        try:
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
        except OSError:
            pass
