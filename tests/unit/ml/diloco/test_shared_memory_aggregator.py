"""Server-side shared-memory aggregator (Flavor 2, issues #197/#198).

The server owns the region and runs the outer step; workers are followers. These
tests drive a :class:`SharedMemoryAggregator` against real
:class:`SharedMemoryBackend` followers, which also proves the region byte layout
is shared correctly between the two implementations (the followers still use the
original backend mapping code).
"""

import multiprocessing as mp
import os
import signal
import threading
import time

import pytest
import torch

from forgather.ml.diloco.shared_memory_aggregator import SharedMemoryAggregator
from forgather.ml.diloco.shared_memory_backend import SharedMemoryBackend
from forgather.ml.diloco.shared_memory_region import _W_ATTACH, ShmRegion


def _aggregator_then_hang(group_dir, ready_path):
    """Child process: become the shared-memory aggregator, signal readiness,
    then hang holding the ownership lease. The parent SIGKILLs it to simulate a
    crash (no stop()), so the region persists and the OS frees the lease."""
    agg = SharedMemoryAggregator(group_dir)
    agg.start(_master(), group_size=1)
    with open(ready_path, "w", encoding="utf-8") as fh:
        fh.write("ready")
    time.sleep(3600)


def _master():
    torch.manual_seed(0)
    return {"w": torch.randn(4, 3), "b": torch.randn(5)}


def _follower(group_dir, group_size, wid, pseudograd, out):
    be = SharedMemoryBackend(group_dir=group_dir, group_size=group_size)
    be.join(worker_id=wid)
    res = be.synchronize(worker_id=wid, pseudograds=pseudograd)
    out[wid] = {k: v.clone() for k, v in res.params.items()}
    be.leave(worker_id=wid)


def test_server_aggregates_followers_outer_step(tmp_path):
    group_dir = str(tmp_path / "grp")
    master = _master()
    lr = 0.5
    group_size = 2

    agg = SharedMemoryAggregator(group_dir)
    agg.start(master, group_size)

    # The server's master + a plain SGD-style outer step (no momentum) so the
    # expected result is closed-form: master <- master - lr * mean(pseudograds).
    server_master = {k: v.clone() for k, v in master.items()}

    def step_fn(avg):
        for k in server_master:
            server_master[k] = server_master[k] - lr * avg[k]
        return {k: v.clone() for k, v in server_master.items()}

    def run_agg():
        assert agg.wait_for_round(timeout=10.0)
        agg.aggregate(step_fn)

    agg_thread = threading.Thread(target=run_agg)
    agg_thread.start()

    g0 = {"w": torch.ones(4, 3), "b": torch.full((5,), 2.0)}
    g1 = {"w": torch.full((4, 3), 3.0), "b": torch.zeros(5)}
    out = {}
    t0 = threading.Thread(target=_follower, args=(group_dir, group_size, "w0", g0, out))
    t1 = threading.Thread(target=_follower, args=(group_dir, group_size, "w1", g1, out))
    t0.start()
    t1.start()
    t0.join(timeout=15)
    t1.join(timeout=15)
    agg_thread.join(timeout=15)

    # Expected published master: master - lr * (g0 + g1) / 2.
    expected = {}
    for k in master:
        avg = (g0[k] + g1[k]) / 2.0
        expected[k] = master[k] - lr * avg

    for wid in ("w0", "w1"):
        assert wid in out, f"{wid} did not return a synced master"
        for k in expected:
            assert torch.allclose(
                out[wid][k], expected[k], atol=1e-5
            ), f"{wid}/{k} mismatch"

    agg.stop()


def test_server_owns_cleanup_until_stop(tmp_path):
    """A follower leaving must NOT unlink the region while the server is alive;
    the server (counted as an attacher) cleans up on stop()."""
    group_dir = str(tmp_path / "grp")
    master = _master()
    agg = SharedMemoryAggregator(group_dir)
    agg.start(master, group_size=1)

    region = ShmRegion(group_dir)
    region.attach()
    # server (+1) only so far.
    assert int(region.ctrl[_W_ATTACH]) == 1
    region.close()

    g = {"w": torch.zeros(4, 3), "b": torch.zeros(5)}
    out = {}

    def run_agg():
        assert agg.wait_for_round(timeout=10.0)
        agg.aggregate(lambda avg: {k: v.clone() for k, v in master.items()})

    th = threading.Thread(target=run_agg)
    th.start()
    _follower(group_dir, 1, "w0", g, out)
    th.join(timeout=15)

    # Follower has left; region files must still exist (server owns them).
    import os

    assert os.path.exists(region.region_path), "region unlinked while server alive"
    assert os.path.exists(region.manifest_path)

    agg.stop()
    assert not os.path.exists(region.region_path), "server stop() did not clean up"


def test_dynamic_barrier_releases_when_a_follower_leaves(tmp_path):
    """The barrier is dynamic: a follower that leaves (clean shutdown/drain)
    shrinks the live count so the remaining followers — already parked having
    contributed — are released, instead of deadlocking on a contribution that
    will never come. This is the shutdown deadlock a fixed group_size barrier
    caused.
    """
    group_dir = str(tmp_path / "grp")
    master = _master()
    agg = SharedMemoryAggregator(group_dir)
    agg.start(master, group_size=3)  # nominal 3, but one leaves without syncing

    out = {}
    g = {"w": torch.ones(4, 3), "b": torch.zeros(5)}

    def contributor(wid):
        _follower(group_dir, 3, wid, g, out)

    def leaver():
        # Joins (attach++) then leaves WITHOUT contributing — the drain case
        # where a worker got save_and_stop between rounds.
        be = SharedMemoryBackend(group_dir=group_dir, group_size=3)
        be.join(worker_id="C")
        be.leave(worker_id="C")

    def run_agg():
        assert agg.wait_for_round(timeout=10.0)
        agg.aggregate(lambda avg: {k: v.clone() for k, v in master.items()})

    threads = [
        threading.Thread(target=contributor, args=("A",)),
        threading.Thread(target=contributor, args=("B",)),
        threading.Thread(target=leaver),
        threading.Thread(target=run_agg),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    # A and B both released despite only 2 of the nominal 3 contributing.
    assert "A" in out and "B" in out, "dynamic barrier did not release the rest"
    agg.stop()


def test_start_fails_loud_if_region_already_owned(tmp_path):
    group_dir = str(tmp_path / "grp")
    master = _master()
    a = SharedMemoryAggregator(group_dir)
    a.start(master, group_size=1)

    b = SharedMemoryAggregator(group_dir)
    try:
        with pytest.raises(RuntimeError, match="already owned"):
            b.start(master, group_size=1)
    finally:
        a.stop()


def test_start_reclaims_stale_region(tmp_path):
    """A region orphaned by a crashed prior aggregator (files on disk, lease
    freed) must not strand the next launch: start() reclaims ownership (the OS
    freed the lease on death) and rebuilds a fresh region rather than attaching
    to the ownerless one."""
    group_dir = str(tmp_path / "grp")
    master = _master()
    a = SharedMemoryAggregator(group_dir)
    a.start(master, group_size=1)
    assert os.path.exists(a._region.manifest_path)
    # Simulate a crash: free the lease + drop the mapping WITHOUT cleanup, so the
    # region files persist with no live lease holder (what a crash leaves).
    a._region.release_ownership()
    a._region.close()
    a._region.close_lock()
    assert os.path.exists(a._region.manifest_path)  # still there

    b = SharedMemoryAggregator(group_dir)
    b.start(master, group_size=1)
    # Fresh region: generation 0, attach == 1 (server only), not a re-attach of
    # the stale one (which would read its old generation / attach count).
    assert b._region.generation() == 0
    assert b._region.attach_count() == 1
    b.stop()


def test_crash_reclaim_real_process_death(tmp_path):
    """The marquee guarantee against a real process death: an aggregator process
    is SIGKILLed (no stop()), leaving the region on disk with the lease freed by
    the OS. A fresh aggregator must reclaim + rebuild, and a follower must be
    able to complete a round against it (a stranded region would hang)."""
    group_dir = str(tmp_path / "grp")
    ready = str(tmp_path / "ready")
    ctx = mp.get_context("fork")
    p = ctx.Process(target=_aggregator_then_hang, args=(group_dir, ready))
    p.start()
    try:
        deadline = time.time() + 30
        while not os.path.exists(ready):
            if time.time() > deadline or not p.is_alive():
                pytest.fail("aggregator child never became ready")
            time.sleep(0.05)
        os.kill(p.pid, signal.SIGKILL)
        p.join(timeout=30)
    finally:
        if p.is_alive():
            p.kill()
            p.join(timeout=10)

    shm_dir = os.path.join(os.path.realpath(group_dir), "diloco_shm")
    assert os.path.exists(os.path.join(shm_dir, "manifest.json"))  # leftover

    agg = SharedMemoryAggregator(group_dir)
    agg.start(_master(), group_size=1)
    assert agg._region.attach_count() == 1  # fresh region, not a re-attach
    out = {}

    def run_agg():
        assert agg.wait_for_round(timeout=10.0)
        agg.aggregate(lambda avg: {k: v.clone() for k, v in _master().items()})

    th = threading.Thread(target=run_agg)
    th.start()
    _follower(group_dir, 1, "w0", {"w": torch.zeros(4, 3), "b": torch.zeros(5)}, out)
    th.join(timeout=15)
    assert "w0" in out
    agg.stop()
