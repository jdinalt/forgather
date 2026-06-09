"""
Single-host shared-memory DiLoCo follower (issue #154).

The worker side of the shared-memory backend. shared-memory DiLoCo is
single-host: co-located worker processes share a CPU master-weights region
(memory-mapped, zero serialization on the wire) instead of round-tripping a
central HTTP parameter server. Because shared-memory is single-host and the
DiLoCo param server is co-located, the **server** owns that region and runs the
outer optimizer
(:class:`~forgather.ml.diloco.shared_memory_aggregator.SharedMemoryAggregator`).
Every worker is therefore a pure **follower**:

* :meth:`join` waits for the server's region (polls for the manifest the server
  publishes) and attaches — it never creates a region or self-elects as an
  aggregator.
* :meth:`synchronize` contributes the worker's raw fp32 pseudo-gradient into the
  shared accumulator and reads back the master the server publishes once the
  round commits.
* :meth:`leave` drops the worker's attach; the server (counted as an attacher)
  owns the region's lifecycle and cleans it up on its own teardown.

All region mechanics (byte layout, the cross-process ``flock``, the control
header, the manifest, the fp32 master/accumulator views) live in
:class:`~forgather.ml.diloco.shared_memory_region.ShmRegion`, shared with the
server-side aggregator so the two never disagree on the region format.

Because the worker hands the backend the **raw** pseudo-gradient (the upload cast
moved into the backend in #157), this backend operates entirely in fp32 with no
wire cast.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Dict, Optional

import torch

from .shared_memory_region import ShmRegion
from .sync_backend import OuterSyncBackend, SyncResult

if TYPE_CHECKING:
    StateDict = Dict[str, torch.Tensor]

_POLL_INTERVAL = 0.01


class SharedMemoryBackend(OuterSyncBackend):
    """Co-located follower for server-aggregated shared-memory DiLoCo.

    Constructed with the group rendezvous (the shared ``group_dir`` the server
    created the region under, and the ``group_size`` of co-located workers).
    Both are advertised by the server in ``/info`` (``shm_group_dir`` /
    ``shm_group_size``); the worker validates ``group_size`` against the region's
    manifest on attach.
    """

    # The server (shared-region aggregator) runs the outer optimizer; the worker
    # only contributes pseudo-gradients and reads back the published master.
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
        lock_timeout: float = 300.0,
    ):
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self._region = ShmRegion(group_dir)
        self.group_size = int(group_size)
        self.lock_timeout = lock_timeout

    @property
    def group_dir(self) -> str:
        return self._region.group_dir

    # ----- OuterSyncBackend -------------------------------------------------

    def join(
        self,
        *,
        worker_id: str,
        worker_info: Optional[dict] = None,
        outer_opt_factory: Optional[object] = None,
    ) -> "StateDict":
        """Attach to the region the server created; return the master snapshot.

        Polls for the server's manifest (its presence is the region-ready
        signal), then attaches under the rendezvous lock and counts itself in
        the attach total. The worker never creates a region or takes the
        ownership lease — it waits for the server. Fails loud if the server
        never publishes a region (e.g. it isn't running with
        ``--backend shared_memory``).
        """
        os.makedirs(self._region.shm_dir, exist_ok=True)
        deadline = time.time() + self.lock_timeout
        while not os.path.exists(self._region.manifest_path):
            if time.time() > deadline:
                raise TimeoutError(
                    "SharedMemoryBackend: timed out waiting for the DiLoCo "
                    "server to create the shared-memory region under "
                    f"{self._region.group_dir}. Is the server running with "
                    "--backend shared_memory?"
                )
            time.sleep(_POLL_INTERVAL)
        with self._region.locked():
            manifest = self._region.attach()
            if manifest.get("group_size") != self.group_size:
                self._region.close()
                raise ValueError(
                    "SharedMemoryBackend: group_size mismatch — region has "
                    f"{manifest.get('group_size')}, this worker has "
                    f"{self.group_size}"
                )
            self._region.incr_attach()
        return self._region.read_master_snapshot()

    def synchronize(self, *, worker_id: str, pseudograds: "StateDict") -> SyncResult:
        """Contribute this worker's raw pseudo-gradient, then read back the
        averaged-and-outer-optimized master the server publishes for the round."""
        # Add my pseudo-grad (upcast to fp32) into the shared accumulator and
        # record the round I contributed to.
        with self._region.locked():
            my_gen = self._region.generation()
            for name in self._region.names:
                pg = pseudograds.get(name)
                if pg is None:
                    # Fail loud rather than under-weight the average: the server
                    # divides by the contributor count, which is only correct
                    # when every co-located worker contributes the full param
                    # set (the single-host, non-pipeline regime this targets).
                    raise ValueError(
                        f"SharedMemoryBackend: pseudograds missing '{name}'; "
                        "every co-located worker must contribute the full "
                        "parameter set."
                    )
                self._region.accum_slice(name).add_(pg.detach().float().reshape(-1))
            self._region.set_arrivals(self._region.arrivals() + 1)

        # Wait for the server to commit the round (bump the generation), then
        # read the new master.
        params = self._await_generation(my_gen + 1)
        return SyncResult(params=params, committed=True, sent_bytes=0, recv_bytes=0)

    def _await_generation(self, target_gen: int) -> "StateDict":
        deadline = time.time() + self.lock_timeout
        while self._region.generation() < target_gen:
            if not self._region.is_alive():
                # The server cleared the magic — it aborted the group (its
                # aggregation loop died) or tore the region down. Fail loud now
                # rather than block out the timeout on a generation that will
                # never advance.
                raise RuntimeError(
                    "SharedMemoryBackend: the server ended the shared-memory "
                    "group (region marked dead) while awaiting sync round "
                    f"{target_gen}; the aggregator is gone."
                )
            if time.time() > deadline:
                raise TimeoutError(
                    "SharedMemoryBackend: timed out waiting for sync round "
                    f"{target_gen}"
                )
            time.sleep(_POLL_INTERVAL)
        with self._region.locked():
            return self._region.read_master_snapshot()

    def synchronize_fragment(
        self, *, worker_id: str, fragment_id: int, pseudograds: "StateDict"
    ) -> SyncResult:
        raise NotImplementedError(
            "SharedMemoryBackend does not support streaming-fragment sync "
            "(num_fragments > 1) yet; use full-model sync."
        )

    def current_global_params(self) -> "StateDict":
        with self._region.locked():
            return self._region.read_master_snapshot()

    def leave(self, *, worker_id: str) -> None:
        # Drop this worker's attach. The server is itself counted as an attacher
        # and owns the region's lifecycle, so a follower leaving never unlinks
        # it (remaining stays >= 1 while the server is alive). The
        # ``remaining == 0`` cleanup is a defensive fallback for a serverless
        # last-out; the magic is cleared first so any out-of-lifecycle attacher
        # fails loud on its magic check.
        if self._region.ctrl is not None:
            try:
                with self._region.locked():
                    remaining = self._region.decr_attach()
                    if remaining == 0:
                        self._region.mark_dead()
                        self._region.cleanup_files()
            except OSError:
                pass
        self._region.close()
        self._region.close_lock()
