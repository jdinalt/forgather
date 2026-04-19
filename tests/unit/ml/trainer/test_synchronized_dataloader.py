"""
Tests for SynchronizedDataLoader.

Covers:
  - Pass-through when disabled (non-distributed).
  - Clean termination when the local iterator exhausts earlier than
    len() suggested (regression for the original PEP 479 bug and for
    the later hang when a rank ran past its real length in DDP).
  - Exact-length iteration when all ranks agree.
  - Per-step MIN synchronization: when a peer rank signals exhaustion,
    this rank stops on the same iteration and drops any prefetched
    batch to stay in lockstep with DDP gradient all-reduces.
  - state_dict / load_state_dict forwarding for checkpoint round-trip.
"""

from unittest.mock import MagicMock

import pytest

from forgather.ml.trainer.synchronized_dataloader import SynchronizedDataLoader


class _FiniteLoader:
    """A fake dataloader with a fixed length and an optional early exhaust.

    `real_len` is what iter() actually yields; `reported_len` is what
    len() returns. When real_len < reported_len, the iterator raises
    StopIteration before reaching reported_len — this models the case
    where an iterable dataset's dynamic length estimate overshoots.
    """

    def __init__(self, real_len, reported_len=None):
        self.real_len = real_len
        self.reported_len = real_len if reported_len is None else reported_len
        self.state = {}

    def __iter__(self):
        return iter(range(self.real_len))

    def __len__(self):
        return self.reported_len

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict):
        self.state = state_dict


def test_passthrough_when_disabled():
    """When distributed is not initialized, iter delegates without sync."""
    loader = _FiniteLoader(real_len=5)
    wrapped = SynchronizedDataLoader(loader, device=MagicMock(), enabled=False)

    items = list(wrapped)

    assert items == [0, 1, 2, 3, 4]


def test_early_exhaustion_returns_cleanly(monkeypatch):
    """When the local iterator runs out, iteration terminates cleanly.

    With per-step MIN synchronization this also regresses against two
    historical bugs:
      (1) PEP 479 `RuntimeError: generator raised StopIteration` when
          StopIteration from `next()` leaked out of the generator body.
      (2) A DDP hang when one rank ran past its real length: the old
          one-shot len()-based sync let the shorter rank StopIteration
          early and move on to end-of-training collectives while peers
          were still doing gradient all-reduces.
    """
    import torch
    from torch import distributed as dist

    # Reported length (10) overshoots real length (3); iteration must
    # end cleanly at 3.
    loader = _FiniteLoader(real_len=3, reported_len=10)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    # No-op all_reduce: simulates a peer rank that never signals
    # exhaustion, so stopping is driven entirely by local StopIteration.
    monkeypatch.setattr(dist, "all_reduce", lambda t, op=None, group=None: None)

    wrapped = SynchronizedDataLoader(loader, device=torch.device("cpu"))

    items = list(wrapped)
    assert items == [0, 1, 2]


def test_peer_rank_exhaustion_stops_this_rank(monkeypatch):
    """When a peer rank signals StopIteration (has_batch=0 after MIN),
    this rank stops on the same iteration even if its own iterator still
    has batches. Any prefetched local batch is dropped so all ranks
    remain in lockstep for DDP gradient all-reduces.
    """
    import torch
    from torch import distributed as dist

    # This rank has 10 batches locally.
    loader = _FiniteLoader(real_len=10)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    # Simulate a peer rank that announces exhaustion after 3 batches:
    # the first 3 all_reduce calls leave has_batch at 1, the 4th clamps
    # it to 0 as if the peer had finished its shard.
    call_count = {"n": 0}

    def simulated_all_reduce(tensor, op=None, group=None):
        call_count["n"] += 1
        # After 3 successful all_reduces, the peer signals done.
        if call_count["n"] >= 4:
            tensor.zero_()
        return None

    monkeypatch.setattr(dist, "all_reduce", simulated_all_reduce)

    wrapped = SynchronizedDataLoader(loader, device=torch.device("cpu"))

    items = list(wrapped)

    # We expect exactly 3 items to be yielded before this rank stops to
    # stay aligned with the peer that exhausted first. The remaining
    # 7 local batches are dropped (warning logged).
    assert items == [0, 1, 2]


def test_exact_length_iterates_fully(monkeypatch):
    """When real and reported lengths match, iteration yields all items."""
    import torch
    from torch import distributed as dist

    loader = _FiniteLoader(real_len=5)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_reduce", lambda t, op=None, group=None: None)

    wrapped = SynchronizedDataLoader(loader, device=torch.device("cpu"))
    items = list(wrapped)
    assert items == [0, 1, 2, 3, 4]


def test_state_dict_forwarding():
    """state_dict / load_state_dict forward to the wrapped loader.

    SynchronizedDataLoader is a transparent wrapper; checkpoint state
    has to round-trip through it or resume-from-checkpoint breaks.
    """
    loader = _FiniteLoader(real_len=3)
    wrapped = SynchronizedDataLoader(loader, device=MagicMock(), enabled=False)

    loader.state = {"position": 42, "seed": 1337}
    assert wrapped.state_dict() == {"position": 42, "seed": 1337}

    wrapped.load_state_dict({"position": 99})
    assert loader.state == {"position": 99}


def test_state_dict_no_support():
    """A loader without state_dict / load_state_dict methods shouldn't crash.

    The wrapper returns an empty dict on save and logs a warning on
    load. This matches what the trainer expects for plain DataLoader.
    """

    class _Bare:
        def __iter__(self):
            return iter([])

        def __len__(self):
            return 0

    wrapped = SynchronizedDataLoader(_Bare(), device=MagicMock(), enabled=False)
    assert wrapped.state_dict() == {}
    # load_state_dict on a loader without support should warn, not raise
    wrapped.load_state_dict({"anything": 1})
