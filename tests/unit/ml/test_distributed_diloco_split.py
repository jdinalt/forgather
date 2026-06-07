"""Tests for the DiLoCo replicate split in DistributedEnvironment (issue #154).

When DILOCO_REPLICATE > 1, DistributedEnvironment reports the INNER view
(world_size/rank within one replica) to the trainer and exposes the diloco
sub-group for the collective backend. The global process group stays over all
ranks. Verified on CPU/gloo via forked workers.
"""

import multiprocessing as mp
import socket

import pytest


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _split_worker(rank, world_size, port, degree, result_path, inner_axis):
    import os

    import torch
    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world_size),
        DILOCO_BACKEND="collective",  # the split's required consumer
    )
    os.environ.pop("DILOCO_REPLICATE", None)
    from forgather.ml.distributed import DistributedEnvironment

    env = DistributedEnvironment(
        no_accelerator=True, diloco_replicate=degree, diloco_inner_axis=inner_axis
    )
    out = {
        "rank": rank,
        "trainer_world_size": env.world_size,
        "trainer_rank": env.rank,
        "diloco_degree": env.diloco_degree,
        "diloco_size": env.diloco_size,
        "diloco_rank": env.diloco_rank,
        "diloco_ranks": (
            sorted(dist.get_process_group_ranks(env.diloco_group))
            if env.diloco_group is not None
            else None
        ),
        "inner_ranks": (
            sorted(dist.get_process_group_ranks(env.inner_group))
            if env.inner_group is not None
            else None
        ),
    }
    torch.save(out, result_path)
    dist.destroy_process_group()


def _run(tmp_path, world_size, degree, inner_axis="data_parallel"):
    import torch

    port = _free_port()
    ctx = mp.get_context("fork")
    procs, paths = [], []
    for r in range(world_size):
        rp = str(tmp_path / f"split_{r}.pt")
        paths.append(rp)
        p = ctx.Process(
            target=_split_worker, args=(r, world_size, port, degree, rp, inner_axis)
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
            pytest.fail("a split worker hung")
        assert p.exitcode == 0, f"worker exited {p.exitcode}"
    return [torch.load(rp, weights_only=False) for rp in paths]


def test_inner_one_reports_world_size_one(tmp_path):
    # degree=2 over world=2 -> inner=1: each rank is its own single-device
    # replica (the trainer sees world_size=1), the diloco group spans both.
    results = _run(tmp_path, world_size=2, degree=2)
    for r in results:
        assert r["trainer_world_size"] == 1
        assert r["trainer_rank"] == 0
        assert r["diloco_degree"] == 2 and r["diloco_size"] == 2
        assert r["diloco_rank"] == r["rank"]
        assert r["diloco_ranks"] == [0, 1]


def test_inner_gt_one_data_parallel_fails_loud(tmp_path):
    # degree=2 over world=4 -> inner=2 with a *data_parallel* inner axis is the
    # diloco x DDP/FSDP composition, which is not yet supported; the split must
    # fail loud, not silently mis-parallelize.
    port = _free_port()
    ctx = mp.get_context("fork")
    procs = []
    for r in range(4):
        p = ctx.Process(
            target=_split_worker,
            args=(r, 4, port, 2, str(tmp_path / f"s_{r}.pt"), "data_parallel"),
        )
        p.start()
        procs.append(p)
    exitcodes = []
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
            pytest.fail("a split worker hung")
        exitcodes.append(p.exitcode)
    # Every rank raises (data_parallel inner=2 rejected) -> all non-zero.
    assert all(code != 0 for code in exitcodes), exitcodes


def test_inner_gt_one_pipeline_succeeds(tmp_path):
    # degree=2 over world=4 -> inner=2 with a *pipeline_parallel* inner axis is
    # the priority composition (diloco x pipeline). Each replica is a 2-rank
    # pipeline; the trainer sees world_size=2; the diloco group strides across
    # replicas at the same pp position; the inner group is the pipeline.
    results = _run(tmp_path, world_size=4, degree=2, inner_axis="pipeline_parallel")
    by_rank = {r["rank"]: r for r in results}
    for r in results:
        assert r["trainer_world_size"] == 2
        assert r["diloco_degree"] == 2 and r["diloco_size"] == 2
    # Trainer (inner) rank = pp position; diloco rank = replica index.
    assert by_rank[0]["trainer_rank"] == 0 and by_rank[0]["diloco_rank"] == 0
    assert by_rank[1]["trainer_rank"] == 1 and by_rank[1]["diloco_rank"] == 0
    assert by_rank[2]["trainer_rank"] == 0 and by_rank[2]["diloco_rank"] == 1
    assert by_rank[3]["trainer_rank"] == 1 and by_rank[3]["diloco_rank"] == 1
    # diloco groups stride across replicas at the same pp position.
    assert by_rank[0]["diloco_ranks"] == [0, 2]
    assert by_rank[1]["diloco_ranks"] == [1, 3]
    # inner groups are the contiguous per-replica pipelines.
    assert by_rank[0]["inner_ranks"] == [0, 1]
    assert by_rank[2]["inner_ranks"] == [2, 3]


def test_degree_one_is_noop(tmp_path):
    # No split: trainer sees the global world; no diloco group.
    results = _run(tmp_path, world_size=2, degree=1)
    for r in results:
        assert r["trainer_world_size"] == 2
        assert r["diloco_degree"] == 1 and r["diloco_size"] == 1
        assert r["diloco_ranks"] is None
