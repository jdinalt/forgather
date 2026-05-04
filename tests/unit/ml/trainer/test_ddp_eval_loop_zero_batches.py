"""
Tests for DDPTrainer._eval_loop_all_shards zero-batch and uneven-shard
handling.

The loop has two correctness obligations under DDP:

1. Pre-flight: raise on every rank if any rank's eval shard is empty
   (otherwise asymmetric self.model(...) calls deadlock the process
   group). This is the failure mode that surfaced in the tiny_llama
   tutorial at world_size 5 or 6: 16-example shards with packed batches
   meant some ranks dropped to zero batches under
   dataloader_drop_last=True.

2. Symmetric main loop: even when all shards are non-empty, ranks may
   hold *different* numbers of batches (e.g. shards [85, 167, 239, 247]
   under sequence packing). The loop must keep ``self.model(...)`` calls
   symmetric across ranks until every shard is exhausted, while only
   counting real batches in the eval loss.

The tests below exercise the pre-flight via ``_check_eval_shards_nonempty``
directly (it is the single point where the per-rank presence is decided)
and exercise the diagnostic message in all three header shapes.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from forgather.ml.loss import RescaleLoss
from forgather.ml.trainer.ddp.ddp_trainer import (
    DDPTrainer,
    DDPTrainingArguments,
)
from forgather.ml.trainer.synchronized_dataloader import SynchronizedDataLoader


def _make_trainer_skeleton(world_size, rank, dispatch_eval_batches=False):
    """Build a minimal stub that satisfies _eval_loop_all_shards's needs.

    We bypass DDPTrainer.__init__ (constructing the full trainer pulls
    in the model, optimizer, distributed env, etc.) and instead bind the
    handful of attributes the pre-flight branch reads.
    """
    trainer = DDPTrainer.__new__(DDPTrainer)

    # Args: only the fields the diagnostic and pre-flight read.
    trainer.args = SimpleNamespace(
        device=torch.device("cpu"),
        dispatch_batches=False,
        dispatch_eval_batches=dispatch_eval_batches,
        dataloader_drop_last=True,
        per_device_eval_batch_size=16,
        max_eval_steps=-1,
    )

    trainer.dist = SimpleNamespace(world_size=world_size, rank=rank)

    # The model and loss_fn aren't reached on the pre-flight failure path,
    # but the asserts and isinstance checks earlier in the method are. We
    # need spec= so isinstance(self.loss_fn, RescaleLoss) holds.
    trainer.model = MagicMock()
    trainer.loss_fn = MagicMock(spec=RescaleLoss)
    trainer.amp_context = SimpleNamespace(autocast=MagicMock())
    trainer.use_fused_loss = False

    return trainer


def _patch_dist_min_max(monkeypatch, per_rank_has_first):
    """Make dist.all_reduce(SUM) return the actual per-rank vector.

    The pre-flight zero-fills a tensor of length world_size, sets index
    `rank` to local has_first, then SUM-reduces across ranks. We
    monkey-patch the reduction to deposit the supplied per-rank vector
    directly, which is equivalent to having every rank participate.
    """
    from torch import distributed as dist

    expected = torch.tensor(per_rank_has_first, dtype=torch.int32)

    def fake_all_reduce(tensor, op=None, group=None):
        # The pre-flight's tensor is shape (world_size,). For SUM, copy in
        # the expected per-rank vector. For MAX/MIN with a 1-element
        # tensor (used elsewhere in the loop), no-op so unrelated paths
        # don't blow up.
        if tensor.shape == expected.shape:
            tensor.copy_(expected)

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: len(per_rank_has_first))


def test_partial_empty_ranks_raises_with_listing(monkeypatch):
    """When only some ranks have zero batches, every rank raises and
    the diagnostic names the empty ranks rather than claiming all are
    empty.
    """
    world_size = 6
    # Ranks 0..2 have a batch; ranks 3..5 do not.
    per_rank_has_first = [1, 1, 1, 0, 0, 0]

    _patch_dist_min_max(monkeypatch, per_rank_has_first)

    # Run the pre-flight from the perspective of rank 0 (which has a batch
    # locally but should still raise because peers are empty).
    trainer = _make_trainer_skeleton(world_size=world_size, rank=0)

    # An iterator that yields one batch (this rank is non-empty locally).
    iterator = iter([{"input_ids": torch.zeros(1, 4, dtype=torch.long)}])

    with pytest.raises(RuntimeError) as excinfo:
        trainer._check_eval_shards_nonempty(iterator)

    msg = str(excinfo.value)
    assert "3 of 6 ranks" in msg
    assert "empty ranks: 3, 4, 5" in msg
    assert "would deadlock" in msg


def test_all_empty_ranks_raises_all_n_message(monkeypatch):
    """When every rank's shard is empty the diagnostic uses the
    'across all N ranks' header rather than the partial form.
    """
    world_size = 4
    per_rank_has_first = [0, 0, 0, 0]

    _patch_dist_min_max(monkeypatch, per_rank_has_first)

    trainer = _make_trainer_skeleton(world_size=world_size, rank=2)

    # iter(empty list) → StopIteration on first next() → local_has_first=0
    with pytest.raises(RuntimeError) as excinfo:
        trainer._check_eval_shards_nonempty(iter([]))

    msg = str(excinfo.value)
    assert "across all 4 ranks" in msg
    # Negative: the partial-rank phrasing must NOT appear here.
    assert " of 4 ranks (empty rank" not in msg


def test_all_ranks_have_first_returns_first_batch(monkeypatch):
    """When every rank has a first batch, the helper returns it so the
    caller can prepend it back into the eval-loop iterator (the
    invariant the symmetric main loop relies on).
    """
    world_size = 3
    per_rank_has_first = [1, 1, 1]
    _patch_dist_min_max(monkeypatch, per_rank_has_first)

    trainer = _make_trainer_skeleton(world_size=world_size, rank=1)

    sentinel_batch = {"input_ids": torch.tensor([[7, 7, 7]])}
    iterator = iter([sentinel_batch, {"input_ids": torch.tensor([[8, 8, 8]])}])

    returned = trainer._check_eval_shards_nonempty(iterator)
    assert returned is sentinel_batch  # caller will chain it back in


def test_zero_eval_batches_message_dispatcher_branch():
    """When dispatch_eval_batches is effectively True the message
    explains the dispatcher's "needs one full batch per rank" failure
    mode rather than the sharded-shard-empty case.
    """
    trainer = _make_trainer_skeleton(world_size=4, rank=0, dispatch_eval_batches=True)

    msg = trainer._zero_eval_batches_message()

    assert "across all 4 ranks" in msg
    assert "rank 0 loads" in msg
    assert "dispatcher needs to assemble at least one batch" in msg


def test_zero_eval_batches_message_settings_block_lists_world_size():
    """Diagnostic should always surface world_size so the user sees
    the topology that produced the failure.
    """
    trainer = _make_trainer_skeleton(world_size=6, rank=0)

    msg = trainer._zero_eval_batches_message(empty_ranks=[3, 4, 5])

    assert "world_size" in msg
    assert "= 6" in msg
