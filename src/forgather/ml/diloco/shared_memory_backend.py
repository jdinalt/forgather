"""
Single-host shared-memory DiLoCo backend (issue #154).

The first non-HTTP :class:`~forgather.ml.diloco.sync_backend.OuterSyncBackend`.
Instead of round-tripping a central HTTP parameter server, co-located worker
processes on one machine share a CPU master-weights region: one shared master
region per host (the aggregator additionally holds a working param copy +
optimizer momentum, like the server), zero serialization on the wire. This is
the single-host regime from #154 — DiLoCo as a *DDP alternative*, syncing every
H steps at a fraction of DDP's per-step all-reduce.

Because the worker hands the backend the **raw** pseudo-gradient (the upload cast
moved into the backend in #157), this backend operates entirely in fp32 with no
wire cast.

Roles (same class; decided at join by the ``owner.lock`` ownership lease — the
worker that takes the lease is the aggregator, the rest are followers):

* **Aggregator** — creates the region, loads the initial weights, owns the master
  ``ParameterList`` + the outer optimizer (its momentum lives in this process,
  never shared, so it stays consistent across rounds exactly like the server).
  Each round it averages the contributed pseudo-gradients, steps the optimizer,
  and *publishes* the new master into the shared region.
* **Follower** — attaches to the region and reads the published master.

Every worker (aggregator included) trains locally and contributes its own
pseudo-gradient into a shared accumulator.

Shared region (a memory-mapped file under a per-group dir, guarded by a
persistent ``flock``, with the path as the rendezvous):

* ``manifest.json`` — canonical param names / shapes / fp32 offsets, written once
  by the aggregator; followers poll for it.
* ``region.bin`` — mmap: a small int64 control header (generation, arrival count,
  group size, live attach count) + the master params (fp32) + an accumulator
  (fp32, same size). The attach count drives self-cleanup: ``join`` increments it,
  ``leave`` decrements it, and the last worker out unlinks the region.
* ``owner.lock`` — the aggregator role is an **ownership lease**: the aggregator
  holds an exclusive ``flock`` on this file for its whole lifetime. A joiner that
  can take the lease is the aggregator (and discards any region orphaned by a
  crashed prior group before recreating); one that can't attaches as a follower.
  Because the OS frees the lease when the holder dies, a re-launched group never
  attaches to an ownerless region left by a crash — which would otherwise
  deadlock (all followers, no aggregator publishing).

Scope (increment 1): the backend + protocol + correctness, constructed directly
(rendezvous params passed in). Trainer/CLI integration, the co-located-group
rendezvous, and streaming-fragment support are follow-ups.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import mmap
import os
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np
import torch

from .sync_backend import OuterSyncBackend, SyncResult

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
_POLL_INTERVAL = 0.01


def _default_outer_optimizer_factory(params):
    """SGD with Nesterov momentum — the DiLoCo paper defaults, matching the
    server's ``_default_outer_optimizer_factory``."""
    return torch.optim.SGD(params, lr=0.7, momentum=0.9, nesterov=True)


class SharedMemoryBackend(OuterSyncBackend):
    """Co-located, single-host shared-memory backend.

    Constructed with the group rendezvous (a shared ``group_dir`` and the
    ``group_size`` of co-located workers), an ``init_checkpoint`` directory the
    aggregator loads the initial weights from, and the ``outer_opt_factory`` the
    aggregator builds its outer optimizer with.
    """

    runs_outer_optimizer = "shared-region"
    supports_async = False
    fault_tolerant = False  # single host: a dead process kills the group
    # join attaches to the region, not the HTTP server — the worker registers
    # separately for coordinator membership / diagnostics.
    registers_with_coordinator = False

    def __init__(
        self,
        *,
        group_dir: str,
        group_size: int,
        init_checkpoint: Optional[str] = None,
        outer_opt_factory: Optional[Callable] = None,
        lock_timeout: float = 300.0,
        follower_only: bool = False,
    ):
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        # realpath (not abspath) so co-located workers handed a symlinked or
        # relative group dir still resolve to the same rendezvous.
        self.group_dir = os.path.realpath(group_dir)
        self.group_size = int(group_size)
        self.init_checkpoint = init_checkpoint
        self.outer_opt_factory = outer_opt_factory or _default_outer_optimizer_factory
        self.lock_timeout = lock_timeout
        # Flavor 2 (issue #154): when the co-located server is the aggregator,
        # every worker is a pure follower — it must NEVER take the ownership
        # lease or create a region (no self-elected aggregator, no silent
        # fallback to a workerless region), only attach to the region the server
        # created. ``init_checkpoint`` is unused on this path (the server seeds
        # the master). The legacy lease-based role election (worker-as-aggregator)
        # remains for the serverless/test regime when this is False.
        self.follower_only = bool(follower_only)

        self._shm_dir = os.path.join(self.group_dir, _SHM_SUBDIR)
        self._region_path = os.path.join(self._shm_dir, _REGION)
        self._manifest_path = os.path.join(self._shm_dir, _MANIFEST)
        self._lock_path = os.path.join(self._shm_dir, "region.lock")
        self._lock_fd: Optional[int] = None
        self._owner_lock_path = os.path.join(self._shm_dir, _OWNER_LOCK)
        # Aggregator ownership lease: held (LOCK_EX) for this process's lifetime
        # when this worker is the aggregator; the OS frees it on death. None
        # until join() decides the role.
        self._owner_lock_fd: Optional[int] = None

        # Set at join().
        self._is_aggregator = False
        self._names: List[str] = []
        self._layout: Dict[str, tuple] = {}  # name -> (float_offset, numel, shape)
        self._total_floats = 0
        self._fd: Optional[int] = None
        self._mm: Optional[mmap.mmap] = None
        self._ctrl: Optional[np.ndarray] = None
        self._master_t: Optional[torch.Tensor] = None  # 1-D fp32 view (shared)
        self._accum_t: Optional[torch.Tensor] = None  # 1-D fp32 view (shared)
        # Aggregator-only working state.
        self._master_params: Optional[torch.nn.ParameterList] = None
        self._outer_optimizer: Optional[torch.optim.Optimizer] = None

    # ----- region helpers ---------------------------------------------------

    @contextlib.contextmanager
    def _locked(self):
        """A real cross-process mutex: a persistent ``flock`` on a long-lived
        lock fd. (Deliberately NOT ``construct.file_lock_build`` — that unlinks
        the lock file on release, which breaks ``flock`` mutual exclusion under
        the high-frequency re-acquisition this backend does every round.) The
        fd is opened once and never unlinked; the OS releases the lock if the
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

    def _map_region(self) -> None:
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

    def _slice(self, t: torch.Tensor, name: str) -> torch.Tensor:
        off, numel, _shape = self._layout[name]
        return t[off : off + numel]

    def _read_master_snapshot(self) -> "StateDict":
        out = {}
        for name in self._names:
            _off, _numel, shape = self._layout[name]
            out[name] = self._slice(self._master_t, name).clone().reshape(shape)
        return out

    # ----- OuterSyncBackend -------------------------------------------------

    def join(
        self,
        *,
        worker_id: str,
        worker_info: Optional[dict] = None,
        outer_opt_factory: Optional[Callable] = None,
    ) -> "StateDict":
        """Create or attach to the shared region; return the master snapshot.

        The aggregator role is an ownership lease, not "first to write the
        manifest": the worker that can take the ``owner.lock`` flock is the
        aggregator — it loads the initial weights from ``init_checkpoint``, lays
        out + creates the region, copies the weights into the master buffer, and
        builds the outer optimizer over its own fp32 master ``ParameterList``.
        Workers that can't take the lease (a live aggregator holds it) attach as
        followers. The lease is what makes re-launch after a crash safe: a region
        orphaned by a dead group has no live lease holder, so the next launch
        reclaims ownership and rebuilds it instead of every worker attaching to
        an ownerless region (which would deadlock — no aggregator to publish).
        """
        from forgather.ml.sharded_checkpoint import load_checkpoint

        factory = outer_opt_factory or self.outer_opt_factory
        os.makedirs(self._shm_dir, exist_ok=True)

        # Flavor 2: pure follower — wait for the server's region, attach, never
        # create or self-elect.
        if self.follower_only:
            self._join_follower_only()
            return self._read_master_snapshot()

        with self._locked():
            # Release the lease/fd if the role decision raises (a bad
            # init_checkpoint, OOM building the master, a group_size mismatch on
            # the follower path). The OS would free it on process exit anyway,
            # but an in-process retry against the same group_dir must not find
            # the lease held by this dead attempt — that would reintroduce the
            # exact "no aggregator" deadlock the lease prevents.
            try:
                if self._try_acquire_ownership():
                    # No live aggregator holds the lease: a fresh group, or a
                    # region orphaned by a crashed one (leave() never ran, so its
                    # stale manifest/region were never unlinked). Discard any
                    # leftovers and (re)create — otherwise every worker would
                    # attach as a follower to an ownerless region and deadlock.
                    self._discard_stale_region()
                    self._create_as_aggregator(load_checkpoint, factory)
                else:
                    # A live aggregator holds the lease; its manifest (written
                    # before it released the rendezvous mutex) is the
                    # region-ready signal.
                    self._attach_as_follower()
            except BaseException:
                self._release_ownership()
                raise
            # Count this worker as attached (under the same lock as create/attach
            # so the aggregator's fresh region — zeroed by truncate — goes 0->1
            # atomically). leave() decrements; the last one out unlinks.
            self._ctrl[_W_ATTACH] = int(self._ctrl[_W_ATTACH]) + 1

        return self._read_master_snapshot()

    def _join_follower_only(self) -> None:
        """Attach to a region created by the server-side aggregator (Flavor 2).

        Polls for the server's manifest (its presence is the region-ready
        signal), then attaches under the rendezvous lock and counts itself in
        the attach total. Never takes the ownership lease and never creates a
        region: a shared-memory worker must not self-elect — it waits for the
        server. Fails loud if the server never publishes a region.
        """
        deadline = time.time() + self.lock_timeout
        while not os.path.exists(self._manifest_path):
            if time.time() > deadline:
                raise TimeoutError(
                    "SharedMemoryBackend: timed out waiting for the DiLoCo "
                    "server to create the shared-memory region under "
                    f"{self.group_dir}. Is the server running with "
                    "--backend shared_memory?"
                )
            time.sleep(_POLL_INTERVAL)
        with self._locked():
            self._attach_as_follower()
            self._ctrl[_W_ATTACH] = int(self._ctrl[_W_ATTACH]) + 1

    def _try_acquire_ownership(self) -> bool:
        """Try to take the aggregator ownership lease.

        A non-blocking exclusive ``flock`` on ``owner.lock``, held for this
        process's lifetime when acquired. Returns True if this worker is the
        aggregator. Called under the rendezvous mutex (``_locked``), so the
        accept/attach decision is serialized across co-located workers — exactly
        one wins. The OS frees the lease when the holder dies, so a crashed
        aggregator's lease is reclaimable by the next launch.
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

    def _release_ownership(self) -> None:
        """Close the owner-lock fd, dropping the lease (no-op for a follower,
        which opened the fd but never acquired the flock). Idempotent."""
        try:
            if self._owner_lock_fd is not None:
                os.close(self._owner_lock_fd)
                self._owner_lock_fd = None
        except OSError:
            pass

    def _discard_stale_region(self) -> None:
        """Unlink a region orphaned by a crashed prior group.

        Safe because we hold the ownership lease — no live aggregator can be
        using these files. Only the data files (manifest + region) are removed;
        ``region.lock`` (the rendezvous mutex this call holds) and ``owner.lock``
        (our lease) are deliberately left in place — unlinking the mutex path
        here would split it for the next joiner.
        """
        for path in (self._manifest_path, self._region_path):
            try:
                os.unlink(path)
            except OSError:
                pass

    def _attach_as_follower(self) -> None:
        with open(self._manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        if manifest.get("group_size") != self.group_size:
            raise ValueError(
                f"SharedMemoryBackend: group_size mismatch — region has "
                f"{manifest.get('group_size')}, this worker has {self.group_size}"
            )
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
        self._is_aggregator = False
        self._map_region()
        if int(self._ctrl[_W_MAGIC]) != _MAGIC:
            self._close_region()
            raise RuntimeError("SharedMemoryBackend: region magic mismatch")

    def _create_as_aggregator(self, load_checkpoint, factory) -> None:
        init = load_checkpoint(self.init_checkpoint, module=None, device="cpu")
        # Canonical fp32 CPU master, mirroring the server's master ParameterList.
        master = {k: v.detach().float().cpu().contiguous() for k, v in init.items()}
        self._build_layout(master)

        params_bytes = self._total_floats * 4
        region_size = _HEADER_BYTES + 2 * params_bytes
        # Allocate the backing file, then map.
        with open(self._region_path, "wb") as fh:
            fh.truncate(region_size)
        self._is_aggregator = True
        self._map_region()

        # Initialize control + master buffer; zero accumulator (truncate gives
        # zeros, but be explicit).
        self._accum_t.zero_()
        for name in self._names:
            self._slice(self._master_t, name).copy_(master[name].reshape(-1))
        self._ctrl[_W_GENERATION] = 0
        self._ctrl[_W_ARRIVALS] = 0
        self._ctrl[_W_GROUP_SIZE] = self.group_size
        self._ctrl[_W_MAGIC] = _MAGIC

        # Aggregator working copy + outer optimizer (momentum stays local).
        self._master_params = torch.nn.ParameterList(
            [
                torch.nn.Parameter(master[name].clone(), requires_grad=False)
                for name in self._names
            ]
        )
        self._outer_optimizer = factory(self._master_params.parameters())

        # Publish the manifest last (its presence is the "region ready" signal).
        manifest = {
            "names": self._names,
            "shapes": {n: list(self._layout[n][2]) for n in self._names},
            "offsets": {n: self._layout[n][0] for n in self._names},
            "numels": {n: self._layout[n][1] for n in self._names},
            "total_floats": self._total_floats,
            "group_size": self.group_size,
        }
        tmp = self._manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        os.replace(tmp, self._manifest_path)

    def synchronize(self, *, worker_id: str, pseudograds: "StateDict") -> SyncResult:
        """Contribute this worker's raw pseudo-gradient, then read back the
        averaged-and-outer-optimized master once all co-located workers have."""
        # Phase 1: add my pseudo-grad (upcast to fp32, matching the server) into
        # the shared accumulator and record the round I contributed to.
        with self._locked():
            my_gen = int(self._ctrl[_W_GENERATION])
            for name in self._names:
                pg = pseudograds.get(name)
                if pg is None:
                    # Fail loud rather than under-weight the average: this
                    # backend divides by group_size, which is only correct when
                    # every co-located worker contributes the full param set
                    # (the single-host, non-pipeline regime it targets).
                    raise ValueError(
                        f"SharedMemoryBackend: pseudograds missing '{name}'; "
                        "every co-located worker must contribute the full "
                        "parameter set."
                    )
                self._slice(self._accum_t, name).add_(pg.detach().float().reshape(-1))
            self._ctrl[_W_ARRIVALS] = int(self._ctrl[_W_ARRIVALS]) + 1

        # Phase 2: the aggregator waits for everyone, then steps + publishes.
        if self._is_aggregator:
            self._aggregate_when_ready(my_gen)

        # Phase 3: wait for the round to commit, then read the new master.
        params = self._await_generation(my_gen + 1)
        return SyncResult(params=params, committed=True, sent_bytes=0, recv_bytes=0)

    def _aggregate_when_ready(self, my_gen: int) -> None:
        deadline = time.time() + self.lock_timeout
        # Lock-free poll: the control words are only written under _locked(), and
        # an aligned int64 read is atomic on the platforms we target — so an
        # unsynchronized read here can at worst cost an extra poll, never a wrong
        # value. (Keep these reads single-word.)
        while True:
            if int(self._ctrl[_W_ARRIVALS]) >= self.group_size:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    "SharedMemoryBackend: timed out waiting for all workers to "
                    f"contribute (got {int(self._ctrl[_W_ARRIVALS])}/"
                    f"{self.group_size})"
                )
            time.sleep(_POLL_INTERVAL)

        with self._locked():
            # Average over the contributors and outer-step, reproducing
            # DiLoCoServer._apply_outer_optimizer. Phase 1 fails loud if any
            # worker omits a name, so the contributor count is exactly
            # group_size for every name here.
            for i, name in enumerate(self._names):
                avg = self._slice(self._accum_t, name).clone() / self.group_size
                self._master_params[i].grad = avg.reshape(self._layout[name][2])
            self._outer_optimizer.step()
            self._outer_optimizer.zero_grad()
            # Publish the stepped master into the shared region for followers.
            for i, name in enumerate(self._names):
                self._slice(self._master_t, name).copy_(
                    self._master_params[i].data.reshape(-1)
                )
            self._accum_t.zero_()
            self._ctrl[_W_ARRIVALS] = 0
            self._ctrl[_W_GENERATION] = my_gen + 1

    def _await_generation(self, target_gen: int) -> "StateDict":
        deadline = time.time() + self.lock_timeout
        while int(self._ctrl[_W_GENERATION]) < target_gen:
            if time.time() > deadline:
                raise TimeoutError(
                    "SharedMemoryBackend: timed out waiting for sync round "
                    f"{target_gen}"
                )
            time.sleep(_POLL_INTERVAL)
        with self._locked():
            return self._read_master_snapshot()

    def synchronize_fragment(
        self, *, worker_id: str, fragment_id: int, pseudograds: "StateDict"
    ) -> SyncResult:
        raise NotImplementedError(
            "SharedMemoryBackend does not support streaming-fragment sync "
            "(num_fragments > 1) yet; use full-model sync."
        )

    def current_global_params(self) -> "StateDict":
        with self._locked():
            return self._read_master_snapshot()

    def _close_region(self) -> None:
        """Close this process's mmap + region fd (not the lock fd)."""
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

    def _cleanup_region_files(self) -> None:
        """Last worker out: unlink the shared region so a completed group leaves
        nothing behind. Best-effort — a group that *crashes* (no last leave) can
        still leave files behind, but that's no longer a hazard: the ownership
        lease means the next launch reclaims and rebuilds an orphaned region
        rather than attaching to it (see ``join`` / ``_try_acquire_ownership``).
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
        # Remove the region subdir, then the per-submit group dir if now empty.
        for d in (self._shm_dir, self.group_dir):
            try:
                os.rmdir(d)
            except OSError:
                pass

    def leave(self, *, worker_id: str) -> None:
        # Decrement the attach count and, if last out, unlink the region — all
        # under the held flock, in one critical section. Doing the unlink under
        # the lock (rather than after releasing it) closes the window where a
        # peer racing the last leaver could see a half-removed region. We also
        # clear the magic word first so that any out-of-lifecycle joiner that
        # still maps the region fails loud on the magic check (the intended
        # lifecycle is: every worker joins during rendezvous, then all leave at
        # teardown — no join arrives after the first leave).
        if self._ctrl is not None:
            try:
                with self._locked():
                    remaining = max(0, int(self._ctrl[_W_ATTACH]) - 1)
                    self._ctrl[_W_ATTACH] = remaining
                    if remaining == 0:
                        self._ctrl[_W_MAGIC] = 0
                        # Unlinking the lock file while still holding its flock
                        # is fine: the lock lives on the open fd, released when
                        # we close it below.
                        self._cleanup_region_files()
            except OSError:
                pass

        # Best-effort teardown of this process's handles.
        self._close_region()

        try:
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
        except OSError:
            pass

        # Release the aggregator ownership lease (no-op for a follower, which
        # never holds it). Closing the fd drops the flock; the OS would do the
        # same on process exit, but a clean leave frees it promptly so an
        # out-of-lifecycle relaunch can reclaim ownership immediately.
        self._release_ownership()
