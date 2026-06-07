"""Tests for the per-replica DILOCO_WORKER_ID rewrite (issue #154).

``diloco_apply_collective_worker_id`` runs at the torchrun entrypoint, before
config preprocessing, and makes DILOCO_WORKER_ID per-replica-distinct so each
collective replica gets its own output dir / run logs / data shard.
"""

import pytest

from forgather.ml.diloco import diloco_apply_collective_worker_id

_KEYS = ("DILOCO_REPLICATE", "WORLD_SIZE", "RANK", "DILOCO_WORKER_ID")


@pytest.fixture
def clean_env(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_degree_one_is_noop(clean_env):
    clean_env.setenv("DILOCO_WORKER_ID", "base")
    clean_env.setenv("WORLD_SIZE", "1")
    clean_env.setenv("RANK", "0")
    diloco_apply_collective_worker_id()  # DILOCO_REPLICATE unset -> 1 -> no-op
    import os

    assert os.environ["DILOCO_WORKER_ID"] == "base"


def test_per_replica_suffix(clean_env):
    import os

    # inner = WORLD_SIZE // REPLICATE = 1, so diloco_rank == RANK.
    clean_env.setenv("DILOCO_REPLICATE", "3")
    clean_env.setenv("WORLD_SIZE", "3")
    clean_env.setenv("DILOCO_WORKER_ID", "base")
    for rank, expect in ((0, "base_r0"), (1, "base_r1"), (2, "base_r2")):
        clean_env.setenv("RANK", str(rank))
        os.environ["DILOCO_WORKER_ID"] = "base"
        diloco_apply_collective_worker_id()
        assert os.environ["DILOCO_WORKER_ID"] == expect


def test_diloco_rank_uses_inner_division(clean_env):
    # WORLD=4, REPLICATE=2 -> inner=2 -> diloco_rank = rank // 2.
    import os

    clean_env.setenv("DILOCO_REPLICATE", "2")
    clean_env.setenv("WORLD_SIZE", "4")
    for rank, expect in ((0, "b_r0"), (1, "b_r0"), (2, "b_r1"), (3, "b_r1")):
        clean_env.setenv("RANK", str(rank))
        os.environ["DILOCO_WORKER_ID"] = "b"
        diloco_apply_collective_worker_id()
        assert os.environ["DILOCO_WORKER_ID"] == expect


def test_idempotent(clean_env):
    import os

    clean_env.setenv("DILOCO_REPLICATE", "2")
    clean_env.setenv("WORLD_SIZE", "2")
    clean_env.setenv("RANK", "1")
    clean_env.setenv("DILOCO_WORKER_ID", "base")
    diloco_apply_collective_worker_id()
    assert os.environ["DILOCO_WORKER_ID"] == "base_r1"
    diloco_apply_collective_worker_id()  # already suffixed -> no double-append
    assert os.environ["DILOCO_WORKER_ID"] == "base_r1"


def test_unset_base_is_noop(clean_env):
    import os

    clean_env.setenv("DILOCO_REPLICATE", "2")
    clean_env.setenv("WORLD_SIZE", "2")
    clean_env.setenv("RANK", "0")
    # No DILOCO_WORKER_ID -> leave it for the downstream 'unset' guidance.
    diloco_apply_collective_worker_id()
    assert os.environ.get("DILOCO_WORKER_ID", "") == ""
