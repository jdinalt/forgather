"""
Tests for SynchronizedDataLoader.

Focused regression tests for the StopIteration → PEP 479 conversion bug
that caused an open-orca run to crash without a final checkpoint save:
`yield next(iterator)` inside the __iter__ generator leaked StopIteration
out of the generator body, which Python 3.7+ converts to `RuntimeError:
generator raised StopIteration`. The fix catches StopIteration explicitly
and returns from the generator so callers see a normal iteration end.
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
    """StopIteration raised before min_len must not propagate as RuntimeError.

    Regression for: `yield next(iterator)` inside __iter__ leaked
    StopIteration out of the generator body and triggered PEP 479
    (`RuntimeError: generator raised StopIteration`), bypassing the
    trainer's clean end-of-training save path.

    We simulate a dataloader whose reported length overshoots the real
    length by stubbing the distributed all_reduce so min_len matches the
    reported length; then iteration should terminate cleanly once the
    underlying loader runs out.
    """
    import torch
    from torch import distributed as dist

    # Reported length is 10, real length is 3: iteration must end at 3
    # without raising RuntimeError.
    loader = _FiniteLoader(real_len=3, reported_len=10)

    # Force the wrapper to believe distributed is enabled so the
    # synchronization path runs. We then stub dist.is_initialized /
    # get_world_size / all_reduce so the wrapper's min_len becomes
    # reported_len (10).
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def fake_all_reduce(tensor, op=None, group=None):
        # Leave the tensor alone; its initial value is reported_len(10),
        # which will be used as min_len. This simulates "all ranks agree
        # on len=10" while our real iterator only has 3 items.
        return None

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    wrapped = SynchronizedDataLoader(loader, device=torch.device("cpu"))

    # Must not raise RuntimeError from PEP 479. The loop should just
    # terminate early when the underlying iterator is exhausted.
    items = list(wrapped)
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
