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
