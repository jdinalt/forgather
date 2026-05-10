"""
In-memory implementation of `IterableDatasetBackend`.

Used as a small reference backend for tests of the composing wrapper
and as a sanity-check that the abstract interface is sufficient. Not
intended for production use — large datasets should use
`ArrowIterableDataset` (memory-mapped Arrow files).
"""

from __future__ import annotations

import random
from typing import Iterator, List, Optional

from .iterable_backend import IterableDatasetBackend


class InMemoryBackend(IterableDatasetBackend):
    """
    Backend that holds a list of dict examples in memory.

    Immutable across `shuffle` and `seek` (they return new instances
    that share the underlying list reference). Iteration mutates the
    instance's `_position` cursor so `position()` reflects progress.
    """

    def __init__(
        self,
        examples: List[dict],
        order: Optional[List[int]] = None,
        position: int = 0,
    ):
        self._examples = examples
        # `order` is a permutation of indices into `_examples`. None
        # means natural order.
        self._order = order
        self._position = position

    def __iter__(self) -> Iterator[dict]:
        n = len(self._examples)
        order = self._order
        # Iterate from current position to end. Mutate `_position` as
        # we go so callers can capture it via `position()`.
        while self._position < n:
            idx = order[self._position] if order is not None else self._position
            self._position += 1
            yield self._examples[idx]

    def __len__(self) -> int:
        return len(self._examples)

    def shuffle(self, seed: Optional[int] = None) -> "InMemoryBackend":
        rng = random.Random(seed)
        order = list(range(len(self._examples)))
        rng.shuffle(order)
        return InMemoryBackend(self._examples, order=order, position=0)

    def seek(self, position: int) -> "InMemoryBackend":
        if position < 0:
            raise ValueError(f"position must be non-negative, got {position}")
        return InMemoryBackend(
            self._examples,
            order=self._order,
            position=min(position, len(self._examples)),
        )

    def position(self) -> int:
        return self._position

    @property
    def column_names(self) -> List[str]:
        if not self._examples:
            return []
        return sorted(self._examples[0].keys())

    @property
    def n_shards(self) -> int:
        return 1
