"""
Single-host shared-memory DiLoCo backend (issue #154).

The first non-HTTP :class:`~forgather.ml.diloco.sync_backend.OuterSyncBackend`.
Instead of round-tripping a central HTTP parameter server, co-located worker
processes on one machine share a CPU master-weights region: one master copy per
host, zero serialization, the outer optimizer applied in place. This is the
single-host regime from #154 — DiLoCo as a *DDP alternative*, syncing every H
steps at a fraction of DDP's per-step all-reduce.

Because the worker hands the backend the **raw** pseudo-gradient (the upload cast
moved into the backend in #157), this backend operates entirely in fp32 with no
wire cast.

Roles (same class; decided at join — first arriver wins):

* **Aggregator** — creates the region, loads the initial weights, owns the master
  ``ParameterList`` + the outer optimizer (its momentum lives in this process,
  never shared, so it stays consistent across rounds exactly like the server).
  Each round it averages the contributed pseudo-gradients, steps the optimizer,
  and *publishes* the new master into the shared region.
* **Follower** — attaches to the region and reads the published master.

Every worker (aggregator included) trains locally and contributes its own
pseudo-gradient into a shared accumulator.

Shared region (a memory-mapped file under a per-group dir — reuses
``file_lock_build`` for the critical section, with the path as the rendezvous):

* ``manifest.json`` — canonical param names / shapes / fp32 offsets, written once
  by the aggregator; followers poll for it.
* ``region.bin`` — mmap: a small int64 control header (generation, arrival count,
  group size) + the master params (fp32) + an accumulator (fp32, same size).

Scope (increment 1): the backend + protocol + correctness, constructed directly
(rendezvous params passed in). Trainer/CLI integration, the co-located-group
rendezvous, and streaming-fragment support are follow-ups.
"""

from __future__ import annotations

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

_SHM_SUBDIR = "diloco_shm"
_MANIFEST = "manifest.json"
_REGION = "region.bin"
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

    def __init__(
        self,
        *,
        group_dir: str,
        group_size: int,
        init_checkpoint: str,
        outer_opt_factory: Optional[Callable] = None,
        lock_timeout: float = 300.0,
    ):
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self.group_dir = os.path.abspath(group_dir)
        self.group_size = int(group_size)
        self.init_checkpoint = init_checkpoint
        self.outer_opt_factory = outer_opt_factory or _default_outer_optimizer_factory
        self.lock_timeout = lock_timeout

        self._shm_dir = os.path.join(self.group_dir, _SHM_SUBDIR)
        self._region_path = os.path.join(self._shm_dir, _REGION)
        self._manifest_path = os.path.join(self._shm_dir, _MANIFEST)
        self._lock_target = os.path.join(self._shm_dir, "region")  # -> region.lock

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

    def _lock(self):
        from forgather.ml.construct import file_lock_build

        return file_lock_build(
            self._lock_target, timeout=self.lock_timeout, force_lock=True
        )

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

        First arriver (no manifest yet) is the aggregator: it loads the initial
        weights from ``init_checkpoint``, lays out + creates the region, copies
        the weights into the master buffer, and builds the outer optimizer over
        its own fp32 master ``ParameterList``. Later arrivers attach.
        """
        from forgather.ml.sharded_checkpoint import load_checkpoint

        factory = outer_opt_factory or self.outer_opt_factory
        os.makedirs(self._shm_dir, exist_ok=True)

        with self._lock():
            if os.path.exists(self._manifest_path):
                self._attach_as_follower()
            else:
                self._create_as_aggregator(load_checkpoint, factory)

        return self._read_master_snapshot()

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
        with self._lock():
            my_gen = int(self._ctrl[_W_GENERATION])
            for name in self._names:
                pg = pseudograds.get(name)
                if pg is None:
                    continue
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

        with self._lock():
            # Average over contributors (== group_size here) and outer-step,
            # reproducing DiLoCoServer._apply_outer_optimizer.
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
        with self._lock():
            return self._read_master_snapshot()

    def synchronize_fragment(
        self, *, worker_id: str, fragment_id: int, pseudograds: "StateDict"
    ) -> SyncResult:
        raise NotImplementedError(
            "SharedMemoryBackend does not support streaming-fragment sync "
            "(num_fragments > 1) yet; use full-model sync."
        )

    def current_global_params(self) -> "StateDict":
        with self._lock():
            return self._read_master_snapshot()

    def leave(self, *, worker_id: str) -> None:
        # Best-effort teardown of this process's handles. Region-file lifecycle
        # (unlink) is the integration layer's concern (increment 2).
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
