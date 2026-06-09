"""
Server-side shared-memory aggregator for single-host DiLoCo (issue #154).

Flavor 2 of the shared-memory backend: because shared-memory DiLoCo is
single-host and the param server is co-located, the **server** maps the same
:class:`~forgather.ml.diloco.shared_memory_region.ShmRegion` and runs the outer
optimizer itself — instead of electing one of the workers as the aggregator.
shared-memory becomes a *transport swap* of the HTTP star, not a second
aggregation owner:

* The server holds the region's ownership lease, creates the region from its own
  master weights, and is the sole writer of the master / generation.
* Every worker is a **follower**: it contributes its raw pseudo-gradient into the
  shared accumulator and reads back the published master each round.
* The server reuses its existing master ``ParameterList`` + outer optimizer +
  ``save_state`` / ``load_state``, so the outer-optimizer momentum is persisted
  and the round counter advances exactly like the HTTP path — which is what makes
  a shared-memory run checkpoint and resume coherently (issues #197, #198).

This class owns only the region side of that: lease + create, the per-round
barrier (wait for every follower to contribute), reading the averaged
pseudo-gradient, and publishing the server's stepped master back into the region.
The optimizer step itself stays in the server (passed in as ``step_fn``) so the
HTTP and shared-memory paths share one outer-step implementation.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable, Dict

from .shared_memory_region import ShmRegion

if TYPE_CHECKING:
    import torch

    StateDict = Dict[str, "torch.Tensor"]

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.01


class SharedMemoryAggregator:
    """Drives the shared-memory region from the server side.

    Lifecycle:

    * :meth:`start` — acquire the ownership lease (fail loud if another live
      aggregator holds it), discard any crash-orphaned region, create + seed the
      region from the server's master, and count the server itself as an
      attacher (so a follower leaving never unlinks the region while the server
      is alive — the server cleans up on :meth:`stop`).
    * :meth:`wait_for_round` — block until every currently-attached follower has
      contributed (a dynamic barrier: a follower that ``leave()``s shrinks the
      expectation, so the drain on shutdown can't deadlock), or ``timeout``.
    * :meth:`aggregate` — under the region lock, average the accumulator, run the
      server's outer step (``step_fn``), publish the new master, and bump the
      generation so the followers release.
    * :meth:`stop` — drop the server's attach, unlink the region if last out,
      release the lease.
    """

    def __init__(self, group_dir: str, lock_timeout: float = 300.0):
        self._region = ShmRegion(group_dir)
        self.lock_timeout = lock_timeout
        self._group_size = 0
        self._started = False
        # The barrier only shrinks dynamically AFTER the full configured group
        # has formed (every follower has attached at least once). Until then it
        # waits for all ``group_size`` contributions, so a slow-to-arrive worker
        # can't let the server aggregate a partial group at startup. ``_formed``
        # latches True once the live-follower high-water mark reaches group_size.
        self._formed = False
        self._max_live = 0

    @property
    def group_dir(self) -> str:
        return self._region.group_dir

    def start(self, master: "StateDict", group_size: int) -> None:
        """Create + seed the region as its owner.

        Fails loud if another process already holds the ownership lease for this
        ``group_dir`` — the server is the sole shared-memory aggregator, so a
        held lease means a stale server (or a misconfigured second one) is still
        running. The OS frees a crashed owner's lease, so this only blocks a
        genuinely live conflict.
        """
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        with self._region.locked():
            if not self._region.try_acquire_ownership():
                raise RuntimeError(
                    "SharedMemoryAggregator: the shared-memory region at "
                    f"{self._region.group_dir} is already owned by a live "
                    "aggregator. A stale DiLoCo server is still holding it — "
                    "stop it before starting a new one."
                )
            # A region orphaned by a crashed prior group has no live lease
            # holder; discard its stale files before recreating.
            self._region.discard_stale()
            self._region.create(master, group_size)
            # Count the server as an attacher so the last follower out doesn't
            # unlink the region from under us; the server unlinks on stop().
            self._region.incr_attach()
        self._group_size = int(group_size)
        self._started = True
        logger.info(
            "SharedMemoryAggregator: region created at %s (group_size=%d)",
            self._region.group_dir,
            self._group_size,
        )

    def wait_for_round(self, timeout: float | None = None) -> bool:
        """Block until every currently-attached follower has contributed.

        The barrier is **dynamic**, mirroring the HTTP path's
        ``_round_expected_workers``: it releases once arrivals reach the number
        of *live* followers (the region's attach count minus the server's own
        +1), not a fixed ``group_size``. This is essential for the drain on
        shutdown — a follower that finishes and ``leave()``s shrinks the
        expectation, so the remaining followers (parked having already
        contributed) are released instead of deadlocking on a 4th contribution
        that will never come. A follower that *crashes* without ``leave()``
        keeps its slot (shared_memory is ``fault_tolerant=False``).

        Returns True when the round is ready, False on timeout. Lock-free poll:
        the control words are only written under the region lock and an aligned
        int64 read is atomic on the platforms we target, so an unsynchronized
        read here at worst costs an extra poll.
        """
        if not self._started:
            raise RuntimeError("SharedMemoryAggregator.start() not called")
        deadline = time.time() + (self.lock_timeout if timeout is None else timeout)
        while True:
            arrivals = self._region.arrivals()
            live_followers = self._region.attach_count() - 1  # exclude the server
            # Latch "formed" once the full configured group has attached; only
            # then may the barrier shrink (the drain/leave case). Before that,
            # require all group_size contributions so a slow arriver at startup
            # can't trigger a partial-group outer step.
            self._max_live = max(self._max_live, live_followers)
            if self._max_live >= self._group_size:
                self._formed = True
            threshold = max(1, live_followers) if self._formed else self._group_size
            if arrivals >= 1 and arrivals >= threshold:
                return True
            if time.time() > deadline:
                return False
            time.sleep(_POLL_INTERVAL)

    def aggregate(self, step_fn: Callable[["StateDict"], "StateDict"]) -> int:
        """Apply one outer step and publish, under the region lock.

        ``step_fn`` receives the averaged pseudo-gradient (the accumulator
        divided by the number of contributors, keyed by param name, reshaped)
        and returns the new master state dict (the server sets these as grads on
        its ``_param_list``, steps the outer optimizer, and returns
        ``get_global_params()``). Holding the lock across the step is cheap (a
        CPU SGD step); followers contributing the next round are parked on the
        generation bump until we publish, so the accumulator is stable here.

        The divisor is the *actual* arrival count read under the lock, not the
        nominal ``group_size`` — so a dynamic-barrier round with fewer
        contributors (a follower left during the drain) is still a correct mean,
        matching the HTTP path's per-contributor averaging. ``arrivals`` and the
        accumulator are mutated together under the lock in the follower's
        contribute step, so they are always consistent here.

        Returns the new generation.
        """
        with self._region.locked():
            n = max(1, self._region.arrivals())
            avg: "StateDict" = {}
            for name in self._region.names:
                avg[name] = (self._region.accum_slice(name).clone() / n).reshape(
                    self._region.shape(name)
                )

            new_master = step_fn(avg)

            for name in self._region.names:
                self._region.master_slice(name).copy_(new_master[name].reshape(-1))
            # Zero the accumulator and reset arrivals, then bump the generation
            # last — its advance is the followers' "round committed" signal.
            self._region.zero_accum()
            self._region.set_arrivals(0)
            new_gen = self._region.bump_generation()
        return new_gen

    def abort(self) -> None:
        """Mark the region dead so parked followers fail loud immediately.

        Called when the server's aggregation loop dies (e.g. the outer step
        raised): clearing the magic makes every follower's ``_await_generation``
        poll raise at once, instead of each blocking out its ``lock_timeout`` on
        a generation that will never advance. Best-effort; the region is cleaned
        up by :meth:`stop` on the server's teardown.
        """
        try:
            with self._region.locked():
                if self._region.ctrl is not None:
                    self._region.mark_dead()
        except OSError:
            pass

    def stop(self) -> None:
        """Drop the server's attach (unlinking the region if last out) and
        release the lease. Idempotent."""
        if not self._started:
            self._region.release_ownership()
            return
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
        self._region.release_ownership()
        self._started = False
