"""Tests for SharedMemoryBackend (issue #154, increment 1).

The core check is multi-process: N co-located processes share a region, run M
synchronization rounds with deterministic per-worker pseudo-gradients, and the
resulting master must equal an in-process reference that runs the *server's*
outer-step math (per-name mean over contributors, fp32, SGD-Nesterov) on the same
inputs — i.e. the shared-memory path is bit-equivalent to the HTTP path's outer
step.
"""

import multiprocessing as mp

import pytest
import torch

from forgather.ml.diloco.shared_memory_backend import (
    _W_ATTACH,
    _W_GENERATION,
    SharedMemoryBackend,
    _default_outer_optimizer_factory,
)


def _pg_value(worker_idx: int, round_idx: int, param_idx: int) -> float:
    """Deterministic per-(worker, round, param) pseudo-gradient fill value."""
    return (worker_idx + 1) * 0.1 + round_idx * 0.01 + param_idx * 1.0


def _reference_master(init_sd, group_size, num_rounds):
    """Run the server-equivalent outer step over the same averaged pseudo-grads.

    ``init_sd`` must be in the *canonical* (checkpoint-load) order, the same
    order workers see from ``join()`` — the per-param index ``j`` feeds the
    deterministic pseudo-grad values, so both sides must agree on it.
    """
    names = list(init_sd.keys())
    params = torch.nn.ParameterList(
        [
            torch.nn.Parameter(init_sd[n].clone().float(), requires_grad=False)
            for n in names
        ]
    )
    opt = _default_outer_optimizer_factory(params.parameters())
    for r in range(num_rounds):
        for j, name in enumerate(names):
            avg = sum(_pg_value(w, r, j) for w in range(group_size)) / group_size
            params[j].grad = torch.full_like(params[j].data, avg)
        opt.step()
        opt.zero_grad()
    return {name: params[j].data.clone() for j, name in enumerate(names)}


def _shm_worker(worker_idx, group_dir, group_size, ckpt, num_rounds, result_path):
    """Child process: join the region and run num_rounds of synchronize."""
    backend = SharedMemoryBackend(
        group_dir=group_dir, group_size=group_size, init_checkpoint=ckpt
    )
    init = backend.join(worker_id=f"w{worker_idx}")
    names = list(init.keys())
    last = None
    for r in range(num_rounds):
        pg = {
            name: torch.full_like(init[name], _pg_value(worker_idx, r, j))
            for j, name in enumerate(names)
        }
        last = backend.synchronize(worker_id=f"w{worker_idx}", pseudograds=pg).params
    torch.save({"worker": worker_idx, "master": last}, result_path)
    backend.leave(worker_id=f"w{worker_idx}")


def _make_checkpoint(tmp_path):
    from .conftest import make_initial_checkpoint

    torch.manual_seed(1)
    sd = {
        "layer0.weight": torch.randn(3, 4),
        "layer1.weight": torch.randn(4, 3),
        "layer1.bias": torch.randn(4),
    }
    return sd, make_initial_checkpoint(sd, str(tmp_path))


class TestSingleProcess:
    def test_join_loads_init_and_step_matches_server(self, tmp_path):
        sd, ckpt = _make_checkpoint(tmp_path)
        backend = SharedMemoryBackend(
            group_dir=str(tmp_path / "g"), group_size=1, init_checkpoint=ckpt
        )
        init = backend.join(worker_id="w0")
        # join returns the fp32 master, equal to the checkpoint weights.
        for k in sd:
            assert torch.allclose(init[k], sd[k].float())

        names = list(sd.keys())
        pg = {
            name: torch.full_like(init[name], _pg_value(0, 0, j))
            for j, name in enumerate(names)
        }
        result = backend.synchronize(worker_id="w0", pseudograds=pg)
        assert result.committed is True
        assert result.sent_bytes == 0  # shared memory: zero wire bytes
        assert result.recv_bytes == 0

        ref = _reference_master(sd, group_size=1, num_rounds=1)
        for k in sd:
            assert torch.allclose(result.params[k], ref[k], atol=1e-6), k
        backend.leave(worker_id="w0")

    def test_multi_round_single_process_matches_server(self, tmp_path):
        """Several rounds in one process — catches a momentum-across-rounds bug
        in the aggregator's optimizer reuse without needing multiprocessing."""
        _sd, ckpt = _make_checkpoint(tmp_path)
        backend = SharedMemoryBackend(
            group_dir=str(tmp_path / "g"), group_size=1, init_checkpoint=ckpt
        )
        init = backend.join(worker_id="w0")  # canonical order
        names = list(init.keys())
        num_rounds = 3
        last = None
        for r in range(num_rounds):
            pg = {
                name: torch.full_like(init[name], _pg_value(0, r, j))
                for j, name in enumerate(names)
            }
            last = backend.synchronize(worker_id="w0", pseudograds=pg).params
        ref = _reference_master(init, group_size=1, num_rounds=num_rounds)
        for name in names:
            assert torch.allclose(last[name], ref[name], atol=1e-6), name
        backend.leave(worker_id="w0")

    def test_current_global_params_returns_master(self, tmp_path):
        _sd, ckpt = _make_checkpoint(tmp_path)
        backend = SharedMemoryBackend(
            group_dir=str(tmp_path / "g"), group_size=1, init_checkpoint=ckpt
        )
        init = backend.join(worker_id="w0")
        names = list(init.keys())
        pg = {
            name: torch.full_like(init[name], _pg_value(0, 0, j))
            for j, name in enumerate(names)
        }
        result = backend.synchronize(worker_id="w0", pseudograds=pg)
        current = backend.current_global_params()
        for name in names:
            assert torch.allclose(current[name], result.params[name])
        backend.leave(worker_id="w0")

    def test_missing_name_fails_loud(self, tmp_path):
        """A worker omitting a name must raise, not silently under-weight the
        average (the backend divides by group_size)."""
        _sd, ckpt = _make_checkpoint(tmp_path)
        backend = SharedMemoryBackend(
            group_dir=str(tmp_path / "g"), group_size=1, init_checkpoint=ckpt
        )
        init = backend.join(worker_id="w0")
        names = list(init.keys())
        partial = {names[0]: torch.zeros_like(init[names[0]])}  # missing the rest
        with pytest.raises(ValueError):
            backend.synchronize(worker_id="w0", pseudograds=partial)
        backend.leave(worker_id="w0")

    def test_capability_flags(self, tmp_path):
        _sd, ckpt = _make_checkpoint(tmp_path)
        backend = SharedMemoryBackend(
            group_dir=str(tmp_path / "g"), group_size=1, init_checkpoint=ckpt
        )
        assert backend.runs_outer_optimizer == "shared-region"
        assert backend.supports_async is False
        assert backend.fault_tolerant is False

    def test_fragment_sync_unsupported(self, tmp_path):
        _sd, ckpt = _make_checkpoint(tmp_path)
        backend = SharedMemoryBackend(
            group_dir=str(tmp_path / "g"), group_size=1, init_checkpoint=ckpt
        )
        backend.join(worker_id="w0")
        with pytest.raises(NotImplementedError):
            backend.synchronize_fragment(worker_id="w0", fragment_id=0, pseudograds={})
        backend.leave(worker_id="w0")

    def test_last_out_cleans_up_region(self, tmp_path):
        """A single-worker group: join+leave is also last-out, so the region
        files and the per-group dir are unlinked (a submit leaves nothing
        behind)."""
        import os

        _sd, ckpt = _make_checkpoint(tmp_path)
        gdir = str(tmp_path / "g")
        backend = SharedMemoryBackend(
            group_dir=gdir, group_size=1, init_checkpoint=ckpt
        )
        backend.join(worker_id="w0")
        assert os.path.exists(backend._region_path)
        assert os.path.exists(backend._manifest_path)
        backend.leave(worker_id="w0")
        assert not os.path.exists(backend._region_path)
        assert not os.path.exists(backend._manifest_path)
        assert not os.path.exists(backend._lock_path)
        assert not os.path.isdir(backend._shm_dir)

    def test_early_leaver_does_not_clean_up(self, tmp_path):
        """With two attached, the first to leave is not last-out and must not
        unlink the region out from under the survivor."""
        import os

        _sd, ckpt = _make_checkpoint(tmp_path)
        gdir = str(tmp_path / "g")
        agg = SharedMemoryBackend(group_dir=gdir, group_size=2, init_checkpoint=ckpt)
        agg.join(worker_id="w0")
        follower = SharedMemoryBackend(
            group_dir=gdir, group_size=2, init_checkpoint=ckpt
        )
        follower.join(worker_id="w1")
        # Follower leaves first: region must survive for the aggregator.
        follower.leave(worker_id="w1")
        assert os.path.exists(agg._region_path)
        assert os.path.exists(agg._manifest_path)
        # Aggregator leaves last: now everything is gone.
        agg.leave(worker_id="w0")
        assert not os.path.exists(agg._region_path)
        assert not os.path.isdir(agg._shm_dir)

    def test_group_size_mismatch_rejected(self, tmp_path):
        _sd, ckpt = _make_checkpoint(tmp_path)
        gdir = str(tmp_path / "g")
        # Aggregator creates the region with group_size=2.
        agg = SharedMemoryBackend(group_dir=gdir, group_size=2, init_checkpoint=ckpt)
        agg.join(worker_id="w0")
        # A follower advertising a different group_size must be rejected.
        bad = SharedMemoryBackend(group_dir=gdir, group_size=3, init_checkpoint=ckpt)
        with pytest.raises(ValueError):
            bad.join(worker_id="w1")
        agg.leave(worker_id="w0")

    def test_live_aggregator_makes_next_joiner_a_follower(self, tmp_path):
        """The ownership lease is what assigns the role: while the aggregator
        holds it, the next joiner attaches as a follower (not a second
        aggregator)."""
        _sd, ckpt = _make_checkpoint(tmp_path)
        gdir = str(tmp_path / "g")
        agg = SharedMemoryBackend(group_dir=gdir, group_size=2, init_checkpoint=ckpt)
        agg.join(worker_id="w0")
        assert agg._is_aggregator is True
        follower = SharedMemoryBackend(
            group_dir=gdir, group_size=2, init_checkpoint=ckpt
        )
        follower.join(worker_id="w1")
        assert follower._is_aggregator is False
        follower.leave(worker_id="w1")
        agg.leave(worker_id="w0")

    def test_stale_region_reclaimed_after_crash(self, tmp_path):
        """A group that crashed (region left on disk, no live lease holder) must
        not strand the next launch: with the role decided by an OS ownership
        lease (freed when the holder dies), the fresh worker reclaims ownership
        and rebuilds the region instead of attaching to the ownerless one as a
        follower — which would deadlock (no aggregator publishing)."""
        import os

        _sd, ckpt = _make_checkpoint(tmp_path)
        gdir = str(tmp_path / "g")

        crashed = SharedMemoryBackend(
            group_dir=gdir, group_size=1, init_checkpoint=ckpt
        )
        crashed.join(worker_id="w0")
        assert os.path.exists(crashed._manifest_path)
        assert os.path.exists(crashed._region_path)
        # Simulate the process dying mid-run: drop the OS locks (the ownership
        # lease + the rendezvous mutex) and the region mapping WITHOUT the
        # cleanup that leave() would do, so the manifest/region persist on disk
        # with no live lease holder — exactly what a crash leaves behind.
        os.close(crashed._owner_lock_fd)
        crashed._owner_lock_fd = None
        if crashed._lock_fd is not None:
            os.close(crashed._lock_fd)
            crashed._lock_fd = None
        crashed._close_region()
        assert os.path.exists(crashed._manifest_path)  # still there (no cleanup)

        # A fresh launch against the same (now stable) per-server dir.
        fresh = SharedMemoryBackend(group_dir=gdir, group_size=1, init_checkpoint=ckpt)
        init = fresh.join(worker_id="w0")
        # It reclaimed ownership and rebuilt: aggregator role, a fresh region
        # (generation 0, attach count 1 — not 2, which is what attaching to the
        # stale region would have produced).
        assert fresh._is_aggregator is True
        assert int(fresh._ctrl[_W_GENERATION]) == 0
        assert int(fresh._ctrl[_W_ATTACH]) == 1
        # And it can actually complete a round (a stranded follower would hang).
        names = list(init.keys())
        pg = {
            name: torch.full_like(init[name], _pg_value(0, 0, j))
            for j, name in enumerate(names)
        }
        result = fresh.synchronize(worker_id="w0", pseudograds=pg)
        assert result.committed is True
        fresh.leave(worker_id="w0")


class TestMultiProcess:
    @pytest.mark.parametrize("group_size", [2, 3])
    def test_matches_server_math(self, tmp_path, group_size):
        sd, ckpt = _make_checkpoint(tmp_path)
        num_rounds = 4
        gdir = str(tmp_path / "group")
        ctx = mp.get_context("fork")  # CPU-only; fork avoids spawn re-import

        procs, result_paths = [], []
        for w in range(group_size):
            rp = str(tmp_path / f"result_{w}.pt")
            result_paths.append(rp)
            p = ctx.Process(
                target=_shm_worker, args=(w, gdir, group_size, ckpt, num_rounds, rp)
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
                pytest.fail("a shared-memory worker process hung")
            assert p.exitcode == 0, f"worker exited {p.exitcode}"

        # Reference uses the canonical checkpoint-load order — the same order the
        # workers see from join(), so the per-param pseudo-grad index agrees.
        from forgather.ml.sharded_checkpoint import load_checkpoint

        canonical = load_checkpoint(ckpt, module=None, device="cpu")
        ref = _reference_master(canonical, group_size, num_rounds)
        for rp in result_paths:
            got = torch.load(rp, weights_only=False)["master"]
            assert got is not None
            for name in canonical:
                assert torch.allclose(got[name], ref[name], atol=1e-5), (rp, name)

        # After every worker has left, the last one out unlinked the region —
        # a completed group leaves nothing behind under the group dir.
        import os

        shm_dir = os.path.join(os.path.realpath(gdir), "diloco_shm")
        assert not os.path.exists(os.path.join(shm_dir, "region.bin"))
        assert not os.path.exists(os.path.join(shm_dir, "manifest.json"))
        assert not os.path.isdir(shm_dir)
