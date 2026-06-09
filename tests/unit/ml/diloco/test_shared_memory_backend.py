"""Tests for SharedMemoryBackend — the co-located *follower* (issue #154).

shared-memory DiLoCo is server-aggregated: the co-located DiLoCoServer owns the
region and runs the outer step (SharedMemoryAggregator), and every worker is a
pure follower. These tests drive real followers against a SharedMemoryAggregator
running the DiLoCo default outer optimizer (SGD-Nesterov), and check that a
follower receives the correctly aggregated + outer-stepped master (including
momentum across rounds), plus the follower's fail-loud guards. The
lease/stale-region/crash-reclaim guarantees moved to the aggregator and are
covered in test_shared_memory_aggregator.py.
"""

import threading

import pytest
import torch

from forgather.ml.diloco.shared_memory_aggregator import SharedMemoryAggregator
from forgather.ml.diloco.shared_memory_backend import SharedMemoryBackend


def _master():
    torch.manual_seed(1)
    return {
        "layer0.weight": torch.randn(3, 4),
        "layer1.weight": torch.randn(4, 3),
        "layer1.bias": torch.randn(4),
    }


def _outer_opt(params):
    # The DiLoCo default outer optimizer (matches the server's default).
    return torch.optim.SGD(params, lr=0.7, momentum=0.9, nesterov=True)


def _make_step_fn(master):
    """A server-equivalent outer step over a private master ParameterList."""
    names = list(master.keys())
    plist = torch.nn.ParameterList(
        [
            torch.nn.Parameter(master[n].clone().float(), requires_grad=False)
            for n in names
        ]
    )
    opt = _outer_opt(plist.parameters())

    def step_fn(avg):
        for j, n in enumerate(names):
            plist[j].grad = avg[n].reshape(plist[j].shape)
        opt.step()
        opt.zero_grad()
        return {n: plist[j].data.clone() for j, n in enumerate(names)}

    return step_fn


def _reference_master(master, group_size, num_rounds, pg_value):
    """Independent replay of the same outer steps over the mean pseudo-grads."""
    names = list(master.keys())
    plist = torch.nn.ParameterList(
        [
            torch.nn.Parameter(master[n].clone().float(), requires_grad=False)
            for n in names
        ]
    )
    opt = _outer_opt(plist.parameters())
    for r in range(num_rounds):
        for j, n in enumerate(names):
            avg = sum(pg_value(w, r, j) for w in range(group_size)) / group_size
            plist[j].grad = torch.full_like(plist[j].data, avg)
        opt.step()
        opt.zero_grad()
    return {n: plist[j].data.clone() for j, n in enumerate(names)}


def _pg_value(worker_idx, round_idx, param_idx):
    return (worker_idx + 1) * 0.1 + round_idx * 0.01 + param_idx * 1.0


def _run_aggregator(agg, step_fn, num_rounds, stop):
    """Aggregate ``num_rounds`` rounds (or until ``stop`` is set)."""
    done = 0
    while done < num_rounds and not stop.is_set():
        if agg.wait_for_round(timeout=0.2):
            agg.aggregate(step_fn)
            done += 1


def _follower_rounds(group_dir, group_size, wid, master, num_rounds, out):
    be = SharedMemoryBackend(group_dir=group_dir, group_size=group_size)
    init = be.join(worker_id=wid)
    names = list(init.keys())
    widx = int(wid[1:])
    last = None
    try:
        for r in range(num_rounds):
            pg = {
                n: torch.full_like(init[n], _pg_value(widx, r, j))
                for j, n in enumerate(names)
            }
            last = be.synchronize(worker_id=wid, pseudograds=pg).params
    finally:
        be.leave(worker_id=wid)
    out[wid] = {k: v.clone() for k, v in last.items()} if last else None


def _drive(tmp_path, group_size, num_rounds):
    """Run group_size follower threads against an aggregator; return their
    last-received masters keyed by worker id, plus the reference."""
    master = _master()
    group_dir = str(tmp_path / "g")
    agg = SharedMemoryAggregator(group_dir)
    agg.start(master, group_size)
    stop = threading.Event()
    out = {}
    threads = [
        threading.Thread(
            target=_run_aggregator, args=(agg, _make_step_fn(master), num_rounds, stop)
        )
    ]
    for w in range(group_size):
        threads.append(
            threading.Thread(
                target=_follower_rounds,
                args=(group_dir, group_size, f"w{w}", master, num_rounds, out),
            )
        )
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    stop.set()
    agg.stop()
    ref = _reference_master(master, group_size, num_rounds, _pg_value)
    return out, ref


class TestFollowerOuterStep:
    def test_single_round_matches_server(self, tmp_path):
        out, ref = _drive(tmp_path, group_size=1, num_rounds=1)
        for k in ref:
            assert torch.allclose(out["w0"][k], ref[k], atol=1e-6), k

    def test_multi_round_momentum_matches_server(self, tmp_path):
        # Several rounds — catches a momentum-across-rounds bug in the server's
        # optimizer reuse.
        out, ref = _drive(tmp_path, group_size=1, num_rounds=3)
        for k in ref:
            assert torch.allclose(out["w0"][k], ref[k], atol=1e-6), k

    def test_multi_follower_mean_matches_server(self, tmp_path):
        out, ref = _drive(tmp_path, group_size=3, num_rounds=4)
        assert set(out) == {"w0", "w1", "w2"}
        for wid in out:
            for k in ref:
                assert torch.allclose(out[wid][k], ref[k], atol=1e-5), (wid, k)


class TestFollowerGuards:
    def _aggregator(self, tmp_path, group_size=1):
        agg = SharedMemoryAggregator(str(tmp_path / "g"))
        agg.start(_master(), group_size)
        return agg

    def test_capability_flags(self):
        assert SharedMemoryBackend.runs_outer_optimizer == "shared-region"
        assert SharedMemoryBackend.supports_async is False
        assert SharedMemoryBackend.fault_tolerant is False
        assert SharedMemoryBackend.registers_with_coordinator is False

    def test_missing_name_fails_loud(self, tmp_path):
        agg = self._aggregator(tmp_path)
        be = SharedMemoryBackend(group_dir=agg.group_dir, group_size=1)
        init = be.join(worker_id="w0")
        names = list(init.keys())
        partial = {names[0]: torch.zeros_like(init[names[0]])}  # missing the rest
        with pytest.raises(ValueError):
            be.synchronize(worker_id="w0", pseudograds=partial)
        be.leave(worker_id="w0")
        agg.stop()

    def test_group_size_mismatch_rejected(self, tmp_path):
        agg = self._aggregator(tmp_path, group_size=2)
        bad = SharedMemoryBackend(group_dir=agg.group_dir, group_size=3)
        with pytest.raises(ValueError, match="group_size mismatch"):
            bad.join(worker_id="w1")
        agg.stop()

    def test_fragment_sync_unsupported(self, tmp_path):
        agg = self._aggregator(tmp_path)
        be = SharedMemoryBackend(group_dir=agg.group_dir, group_size=1)
        be.join(worker_id="w0")
        with pytest.raises(NotImplementedError):
            be.synchronize_fragment(worker_id="w0", fragment_id=0, pseudograds={})
        be.leave(worker_id="w0")
        agg.stop()

    def test_current_global_params_returns_master(self, tmp_path):
        agg = self._aggregator(tmp_path)
        be = SharedMemoryBackend(group_dir=agg.group_dir, group_size=1)
        init = be.join(worker_id="w0")
        cur = be.current_global_params()
        for k in init:
            assert torch.allclose(cur[k], init[k])
        be.leave(worker_id="w0")
        agg.stop()

    def test_join_times_out_without_a_server(self, tmp_path):
        # No aggregator created a region: the follower must fail loud rather than
        # self-elect (no silent fallback).
        be = SharedMemoryBackend(
            group_dir=str(tmp_path / "g"), group_size=1, lock_timeout=0.3
        )
        with pytest.raises(TimeoutError, match="server to create"):
            be.join(worker_id="w0")
