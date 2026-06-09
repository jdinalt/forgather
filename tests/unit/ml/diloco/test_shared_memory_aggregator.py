"""Server-side shared-memory aggregator (Flavor 2, issues #197/#198).

The server owns the region and runs the outer step; workers are followers. These
tests drive a :class:`SharedMemoryAggregator` against real
:class:`SharedMemoryBackend` followers, which also proves the region byte layout
is shared correctly between the two implementations (the followers still use the
original backend mapping code).
"""

import threading

import torch

from forgather.ml.diloco.shared_memory_aggregator import SharedMemoryAggregator
from forgather.ml.diloco.shared_memory_backend import SharedMemoryBackend
from forgather.ml.diloco.shared_memory_region import _W_ATTACH, ShmRegion


def _master():
    torch.manual_seed(0)
    return {"w": torch.randn(4, 3), "b": torch.randn(5)}


def _follower(group_dir, group_size, wid, pseudograd, out):
    be = SharedMemoryBackend(
        group_dir=group_dir,
        group_size=group_size,
        init_checkpoint="unused-by-follower",
    )
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


def test_start_fails_loud_if_region_already_owned(tmp_path):
    group_dir = str(tmp_path / "grp")
    master = _master()
    a = SharedMemoryAggregator(group_dir)
    a.start(master, group_size=1)

    b = SharedMemoryAggregator(group_dir)
    try:
        import pytest

        with pytest.raises(RuntimeError, match="already owned"):
            b.start(master, group_size=1)
    finally:
        a.stop()
