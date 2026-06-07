"""Tests for CollectiveBackend (issue #154).

The core check is multi-process: N gloo ranks on CPU run M synchronization
rounds with deterministic per-rank pseudo-gradients, and the resulting master
must equal an in-process reference that runs the *server's* outer-step math
(per-name mean over contributors, fp32, SGD-Nesterov) on the same inputs — i.e.
the collective path is bit-equivalent to the HTTP path's outer step. The test
*also* asserts the masters are bit-identical across ranks (``torch.equal``),
which is the determinism guarantee that lets ``synchronize`` skip broadcasting
its result.
"""

import multiprocessing as mp
import socket

import pytest
import torch

from forgather.ml.diloco.collective_backend import (
    CollectiveBackend,
    _default_outer_optimizer_factory,
)


def _pg_value(worker_idx: int, round_idx: int, param_idx: int) -> float:
    """Deterministic per-(worker, round, param) pseudo-gradient fill value."""
    return (worker_idx + 1) * 0.1 + round_idx * 0.01 + param_idx * 1.0


def _reference_master(init_sd, group_size, num_rounds):
    """Run the server-equivalent outer step over the same averaged pseudo-grads.

    ``init_sd`` must be in the *canonical* (checkpoint-load) order — the same
    order ranks see from ``join()`` (rank 0 broadcasts that order) — so the
    per-param index ``j`` feeds the deterministic pseudo-grad values consistently
    on both sides.
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


def _make_checkpoint(tmp_path):
    from .conftest import make_initial_checkpoint

    torch.manual_seed(1)
    sd = {
        "layer0.weight": torch.randn(3, 4),
        "layer1.weight": torch.randn(4, 3),
        "layer1.bias": torch.randn(4),
    }
    return sd, make_initial_checkpoint(sd, str(tmp_path))


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _collective_worker(rank, world_size, master_port, ckpt, num_rounds, result_path):
    """Child process: init a gloo group, run num_rounds of synchronize."""
    import os

    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    dist.init_process_group(backend="gloo")
    try:
        backend = CollectiveBackend(init_checkpoint=ckpt)  # default group, init bcast
        init = backend.join(worker_id=f"w{rank}")
        names = list(init.keys())
        last = None
        for r in range(num_rounds):
            pg = {
                name: torch.full_like(init[name], _pg_value(rank, r, j))
                for j, name in enumerate(names)
            }
            last = backend.synchronize(worker_id=f"w{rank}", pseudograds=pg).params
        torch.save({"rank": rank, "master": last}, result_path)
        backend.leave(worker_id=f"w{rank}")
    finally:
        dist.destroy_process_group()


class TestSingleProcess:
    """No torch.distributed: group_size=1 short-circuits the all_reduce, so the
    replicated outer-step math is provable in isolation (no GPU, no group)."""

    def test_join_loads_init_and_step_matches_server(self, tmp_path):
        sd, ckpt = _make_checkpoint(tmp_path)
        backend = CollectiveBackend(init_checkpoint=ckpt, group_size=1, rank=0)
        init = backend.join(worker_id="w0")
        for k in sd:
            assert torch.allclose(init[k], sd[k].float())

        names = list(sd.keys())
        pg = {
            name: torch.full_like(init[name], _pg_value(0, 0, j))
            for j, name in enumerate(names)
        }
        result = backend.synchronize(worker_id="w0", pseudograds=pg)
        assert result.committed is True
        # fp32 on the wire, no cast: the logical all_reduce volume each way.
        expected_bytes = sum(t.numel() * 4 for t in sd.values())
        assert result.sent_bytes == expected_bytes
        assert result.recv_bytes == expected_bytes

        ref = _reference_master(sd, group_size=1, num_rounds=1)
        for k in sd:
            assert torch.allclose(result.params[k], ref[k], atol=1e-6), k
        backend.leave(worker_id="w0")

    def test_multi_round_matches_server(self, tmp_path):
        """Several rounds in one process — catches a momentum-across-rounds bug
        in the replicated optimizer's reuse without needing multiprocessing."""
        _sd, ckpt = _make_checkpoint(tmp_path)
        backend = CollectiveBackend(init_checkpoint=ckpt, group_size=1, rank=0)
        init = backend.join(worker_id="w0")
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
        backend = CollectiveBackend(init_checkpoint=ckpt, group_size=1, rank=0)
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
        """A replica omitting a name must raise, not silently under-weight the
        average (the backend divides by group_size)."""
        _sd, ckpt = _make_checkpoint(tmp_path)
        backend = CollectiveBackend(init_checkpoint=ckpt, group_size=1, rank=0)
        init = backend.join(worker_id="w0")
        names = list(init.keys())
        partial = {names[0]: torch.zeros_like(init[names[0]])}  # missing the rest
        with pytest.raises(ValueError):
            backend.synchronize(worker_id="w0", pseudograds=partial)
        backend.leave(worker_id="w0")

    def test_capability_flags(self, tmp_path):
        _sd, ckpt = _make_checkpoint(tmp_path)
        backend = CollectiveBackend(init_checkpoint=ckpt, group_size=1, rank=0)
        assert backend.runs_outer_optimizer == "replicated"
        assert backend.supports_async is False
        assert backend.fault_tolerant is False
        assert backend.registers_with_coordinator is False

    def test_fragment_sync_unsupported(self, tmp_path):
        _sd, ckpt = _make_checkpoint(tmp_path)
        backend = CollectiveBackend(init_checkpoint=ckpt, group_size=1, rank=0)
        backend.join(worker_id="w0")
        with pytest.raises(NotImplementedError):
            backend.synchronize_fragment(worker_id="w0", fragment_id=0, pseudograds={})
        backend.leave(worker_id="w0")

    def test_invalid_group_size_rejected(self, tmp_path):
        _sd, ckpt = _make_checkpoint(tmp_path)
        with pytest.raises(ValueError):
            CollectiveBackend(init_checkpoint=ckpt, group_size=0, rank=0)


class TestMultiProcess:
    @pytest.mark.parametrize("world_size", [2, 3])
    def test_matches_server_math_and_ranks_agree(self, tmp_path, world_size):
        sd, ckpt = _make_checkpoint(tmp_path)
        num_rounds = 4
        port = _free_port()
        ctx = mp.get_context("fork")  # CPU-only; fork avoids spawn re-import

        procs, result_paths = [], []
        for r in range(world_size):
            rp = str(tmp_path / f"result_{r}.pt")
            result_paths.append(rp)
            p = ctx.Process(
                target=_collective_worker,
                args=(r, world_size, port, ckpt, num_rounds, rp),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
                pytest.fail("a collective worker process hung")
            assert p.exitcode == 0, f"worker exited {p.exitcode}"

        # Reference uses the canonical checkpoint-load order — the order rank 0
        # broadcasts to the group, so the per-param pseudo-grad index agrees.
        from forgather.ml.sharded_checkpoint import load_checkpoint

        canonical = load_checkpoint(ckpt, module=None, device="cpu")
        ref = _reference_master(canonical, world_size, num_rounds)
        masters = [torch.load(rp, weights_only=False)["master"] for rp in result_paths]

        # (a) every rank's master matches the server outer-step math.
        for m in masters:
            assert m is not None
            for name in canonical:
                assert torch.allclose(m[name], ref[name], atol=1e-5), name

        # (b) the masters are bit-identical across ranks — the determinism
        # guarantee that lets synchronize() skip broadcasting its result.
        for name in canonical:
            for m in masters[1:]:
                assert torch.equal(m[name], masters[0][name]), name
