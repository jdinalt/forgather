"""Tests for the async DiLoCo grace period (Liu et al. 2024, Section 3).

A grace window is a soft barrier with a wall-clock timeout on the async submit
path: near-simultaneous submissions are parked and aggregated into ONE outer
step, so workers that finish close together resync against the same model.

Direct tests exercise the flush/aggregation/DN-layering logic deterministically;
the threaded tests drive the real HTTP submit path for the window semantics,
worker-death release, and the periodic-save re-entrancy (deadlock) regression.
"""

import threading
import time

import torch

from forgather.ml.diloco.client import DiLoCoClient
from forgather.ml.diloco.server import DiLoCoServer

from .conftest import make_initial_checkpoint


def _sgd(lr, momentum=0.0):
    def factory(params):
        return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=momentum > 0)

    return factory


def _server(tmp_path, sd, **kw):
    ckpt = make_initial_checkpoint(sd, tmp_path / "init")
    kw.setdefault("num_workers", 2)
    kw.setdefault("save_every_n_rounds", 0)
    return DiLoCoServer(
        output_dir=str(tmp_path),
        from_checkpoint=ckpt,
        port=0,
        async_mode=True,
        **kw,
    )


# ---------------------------------------------------------------------------
# Direct (deterministic) flush / aggregation / DN-layering tests
# ---------------------------------------------------------------------------


class TestGraceFlushDirect:
    def test_aggregate_pseudograds_mean(self, tmp_path):
        server = _server(tmp_path, {"w": torch.zeros(4)}, grace_period=1.0)
        out = server._aggregate_pseudograds(
            [{"w": torch.ones(4)}, {"w": torch.full((4,), 3.0)}]
        )
        assert torch.allclose(out["w"], torch.full((4,), 2.0))
        # A batch of one is returned as-is (single-worker window == immediate).
        single = {"w": torch.ones(4)}
        assert server._aggregate_pseudograds([single]) is single

    def test_flush_one_outer_step_and_stats(self, tmp_path):
        sd = {"w": torch.zeros(4)}
        server = _server(
            tmp_path,
            sd,
            grace_period=1.0,
            dn_buffer_size=0,
            outer_optimizer_factory=_sgd(1.0),
        )
        # Two parked workers: grads 1.0 and 3.0 -> mean 2.0; sgd lr=1 -> theta=-2.
        server._grace_pending = {
            "w0": {"w": torch.ones(4)},
            "w1": {"w": torch.full((4,), 3.0)},
        }
        server._grace_tau_sync = server._now()
        with server._grace_cond:  # flush notifies, so it needs the lock held
            server._flush_grace_window([])

        assert server._sync_round == 1  # ONE outer step for the batch
        assert server._total_submissions == 2  # but two real submissions counted
        assert server._grace_epoch == 1
        assert 0 in server._grace_results
        assert server._grace_batches == 1
        assert server._grace_batch_hist == {2: 1}
        assert torch.allclose(server.get_global_params()["w"], torch.full((4,), -2.0))

    def test_dn_one_tick_per_batch(self, tmp_path):
        sd = {"w": torch.zeros(4)}
        server = _server(
            tmp_path,
            sd,
            grace_period=1.0,
            dn_buffer_size=2,
            outer_optimizer_factory=_sgd(0.7, 0.9),
        )
        # Two grace batches (each of two workers). DN buffer N=2 must tick once
        # per BATCH, so momentum refreshes on the 2nd batch, not the 2nd worker.
        for _ in range(2):
            server._grace_pending = {
                "w0": {"w": torch.ones(4)},
                "w1": {"w": torch.ones(4)},
            }
            server._grace_tau_sync = server._now()
            with server._grace_cond:
                server._flush_grace_window([])
        assert server._dn_count == 0  # N=2 -> two batches -> exactly one full cycle
        assert server._sync_round == 2  # two outer steps (one per batch)
        assert server._grace_batched_submissions == 4


# ---------------------------------------------------------------------------
# Threaded HTTP submit-path tests (window semantics, death, deadlock)
# ---------------------------------------------------------------------------


def _client(server):
    return DiLoCoClient(f"localhost:{server.port}", timeout=20)


def _pg(sd, val=0.1):
    return {k: torch.full_like(v, val) for k, v in sd.items()}


class TestGraceWindowThreaded:
    def test_k_within_window_one_step(self, tmp_path):
        """Two workers submitting together short-circuit into ONE outer step."""
        sd = {"w": torch.zeros(4)}
        server = _server(
            tmp_path,
            sd,
            num_workers=2,
            grace_period=5.0,
            outer_optimizer_factory=_sgd(1.0),
        )
        server.start()
        try:
            c0, c1 = _client(server), _client(server)
            c0.register("w0")
            c1.register("w1")
            results = {}

            def submit(c, wid):
                results[wid] = c.submit_pseudogradients(wid, _pg(sd))

            t0 = threading.Thread(target=submit, args=(c0, "w0"))
            t1 = threading.Thread(target=submit, args=(c1, "w1"))
            t0.start()
            t1.start()
            t0.join(timeout=15)
            t1.join(timeout=15)

            assert set(results) == {"w0", "w1"}
            assert server._sync_round == 1  # one aggregated step, not two
            assert server._total_submissions == 2
            assert server._grace_batches == 1
            # Both workers got the SAME post-step params (resynced together).
            assert torch.allclose(results["w0"]["w"], results["w1"]["w"])
        finally:
            server.stop()

    def test_window_expiry_separate_steps(self, tmp_path):
        """A lone submitter flushes at the deadline; a later one is a new batch."""
        sd = {"w": torch.zeros(4)}
        server = _server(
            tmp_path,
            sd,
            num_workers=2,
            grace_period=0.15,
            outer_optimizer_factory=_sgd(1.0),
        )
        server.start()
        try:
            c0, c1 = _client(server), _client(server)
            c0.register("w0")
            c1.register("w1")  # registered (so live=2) but never submits
            t0 = time.time()
            # submit only w0: the window can't short-circuit (1 < 2 live)
            c0.submit_pseudogradients("w0", _pg(sd))
            assert time.time() - t0 >= 0.12  # waited out the window
            assert server._sync_round == 1
            assert server._grace_batches == 1
            c0.submit_pseudogradients("w0", _pg(sd))
            assert server._sync_round == 2  # a fresh window/epoch
        finally:
            server.stop()

    def test_worker_death_releases_batch(self, tmp_path):
        """A death lowers the live count so the parked batch flushes (no hang)."""
        sd = {"w": torch.zeros(4)}
        server = _server(
            tmp_path,
            sd,
            num_workers=2,
            grace_period=10.0,
            outer_optimizer_factory=_sgd(1.0),
        )
        server.start()
        try:
            c0, c1 = _client(server), _client(server)
            c0.register("w0")
            c1.register("w1")
            results = []

            def submit():
                results.append(c0.submit_pseudogradients("w0", _pg(sd)))

            t = threading.Thread(target=submit)
            t.start()
            time.sleep(0.3)  # let w0 park as the driver (1 < 2 live)
            server._handle_worker_death("w1")  # w1 never submitted
            t.join(timeout=15)

            assert len(results) == 1  # released rather than hanging on the dead peer
            assert server._sync_round == 1
        finally:
            server.stop()

    def test_grace_flush_periodic_save_no_deadlock(self, tmp_path):
        """grace flush -> periodic save_state re-enters _async_lock; must not hang."""
        sd = {"w": torch.zeros(4)}
        server = _server(
            tmp_path,
            sd,
            num_workers=1,
            grace_period=0.1,
            dn_buffer_size=1,
            save_every_n_rounds=1,
            outer_optimizer_factory=_sgd(0.7, 0.9),
        )
        server.start()
        try:
            c0 = _client(server)
            c0.register("w0")
            done = threading.Event()

            def go():
                # Single live worker -> short-circuit flush -> _apply_async ->
                # save_state re-acquires _async_lock (RLock) from inside the
                # grace flush. A non-reentrant lock would deadlock here.
                c0.submit_pseudogradients("w0", _pg(sd))
                done.set()

            t = threading.Thread(target=go)
            t.start()
            t.join(timeout=20)
            assert done.is_set(), "grace flush + periodic save deadlocked"
            assert server._sync_round == 1
        finally:
            server.stop()
