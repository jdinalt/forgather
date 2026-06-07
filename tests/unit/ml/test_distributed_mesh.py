"""Tests for the (diloco, inner) device-mesh builder (issue #154)."""

import multiprocessing as mp
import socket

import pytest

from forgather.ml.distributed_mesh import ForgatherParallelDims


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestDegreeValidation:
    """The degree math is validated at construction, no torch.distributed needed."""

    def test_product_must_equal_world(self):
        with pytest.raises(ValueError, match="!= "):
            ForgatherParallelDims(
                diloco=2, inner=2, inner_axis="data_parallel", world_size=5
            )

    def test_ok_product(self):
        pd = ForgatherParallelDims(
            diloco=2, inner=2, inner_axis="data_parallel", world_size=4
        )
        assert pd.diloco_size() == 2 and pd.inner_size() == 2

    def test_inner_one_is_whole_world_diloco(self):
        pd = ForgatherParallelDims(
            diloco=4, inner=1, inner_axis="data_parallel", world_size=4
        )
        assert pd.diloco_size() == 4 and pd.inner_size() == 1

    def test_bad_inner_axis(self):
        with pytest.raises(ValueError, match="inner_axis"):
            ForgatherParallelDims(diloco=2, inner=1, inner_axis="bogus", world_size=2)

    def test_degrees_must_be_positive(self):
        with pytest.raises(ValueError, match=">= 1"):
            ForgatherParallelDims(
                diloco=0, inner=1, inner_axis="data_parallel", world_size=0
            )


def _mesh_worker(rank, world_size, port, diloco, inner, result_path, inner_axis):
    """Child: init gloo, build the mesh, report this rank's coordinates + the
    global ranks of its diloco and inner sub-groups."""
    import os

    import torch
    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    dist.init_process_group(backend="gloo")
    try:
        pd = ForgatherParallelDims(
            diloco=diloco,
            inner=inner,
            inner_axis=inner_axis,
            world_size=world_size,
            device_type="cpu",
        )
        out = {
            "rank": rank,
            "diloco_rank": pd.diloco_rank(),
            "inner_rank": pd.inner_rank(),
            "diloco_ranks": sorted(dist.get_process_group_ranks(pd.diloco_group())),
            "inner_ranks": sorted(dist.get_process_group_ranks(pd.inner_group())),
        }
        torch.save(out, result_path)
    finally:
        dist.destroy_process_group()


class TestMultiProcessMesh:
    def _run(self, tmp_path, diloco, inner, inner_axis="data_parallel"):
        import torch

        world_size = diloco * inner
        port = _free_port()
        ctx = mp.get_context("fork")
        procs, paths = [], []
        for r in range(world_size):
            rp = str(tmp_path / f"mesh_{r}.pt")
            paths.append(rp)
            p = ctx.Process(
                target=_mesh_worker,
                args=(r, world_size, port, diloco, inner, rp, inner_axis),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
                pytest.fail("a mesh worker hung")
            assert p.exitcode == 0, f"worker exited {p.exitcode}"
        return [torch.load(rp, weights_only=False) for rp in paths]

    def test_diloco_x_inner_grouping(self, tmp_path):
        # (diloco=2, inner=2), world=4. rank = diloco_idx*inner + inner_idx.
        # diloco group at inner-position k = ranks sharing inner_rank k.
        results = self._run(tmp_path, diloco=2, inner=2)
        by_rank = {r["rank"]: r for r in results}
        # Coordinates.
        assert by_rank[0]["diloco_rank"] == 0 and by_rank[0]["inner_rank"] == 0
        assert by_rank[1]["diloco_rank"] == 0 and by_rank[1]["inner_rank"] == 1
        assert by_rank[2]["diloco_rank"] == 1 and by_rank[2]["inner_rank"] == 0
        assert by_rank[3]["diloco_rank"] == 1 and by_rank[3]["inner_rank"] == 1
        # diloco groups stride across replicas at the same inner position.
        assert by_rank[0]["diloco_ranks"] == [0, 2]
        assert by_rank[1]["diloco_ranks"] == [1, 3]
        # inner groups are contiguous within a replica.
        assert by_rank[0]["inner_ranks"] == [0, 1]
        assert by_rank[2]["inner_ranks"] == [2, 3]

    def test_inner_one_diloco_is_whole_world(self, tmp_path):
        # (diloco=3, inner=1): the diloco group spans the whole world.
        results = self._run(tmp_path, diloco=3, inner=1)
        for r in results:
            assert r["inner_rank"] == 0
            assert r["diloco_rank"] == r["rank"]
            assert r["diloco_ranks"] == [0, 1, 2]

    def test_diloco_x_pipeline_grouping(self, tmp_path):
        # (diloco=2, pipeline_parallel=2), world=4 — the priority composition.
        # The inner axis name differs but the rank arithmetic is identical to
        # the data_parallel case: pp group is contiguous within a replica, the
        # diloco group strides across replicas at the same pp position.
        results = self._run(tmp_path, diloco=2, inner=2, inner_axis="pipeline_parallel")
        by_rank = {r["rank"]: r for r in results}
        assert by_rank[0]["diloco_rank"] == 0 and by_rank[0]["inner_rank"] == 0
        assert by_rank[3]["diloco_rank"] == 1 and by_rank[3]["inner_rank"] == 1
        # Each pp rank all-reduces its slice across its replicas at that pp pos.
        assert by_rank[0]["diloco_ranks"] == [0, 2]
        assert by_rank[1]["diloco_ranks"] == [1, 3]
        # The pipeline runs on the contiguous inner sub-group.
        assert by_rank[0]["inner_ranks"] == [0, 1]
        assert by_rank[2]["inner_ranks"] == [2, 3]
