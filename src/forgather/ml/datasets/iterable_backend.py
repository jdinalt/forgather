"""
Abstract iterable-dataset backend interface.

Defines the minimal contract every storage backend must satisfy so that
higher-level operations (`map`, `filter`, `select`, `slice`, `shard`,
the example-level shuffle buffer, `state_dict`/`load_state_dict`,
`set_epoch`) can be implemented once in a shared wrapper that composes
over any backend.

Concrete backends:

- `ArrowBackend` — local Arrow-file backend (today).
- `RemoteBackend` — network proxy talking to a dataset server
  (planned).

Backend ops are immutable: `shuffle` and `seek` return a new backend
instance rather than mutating the receiver. This matches the existing
functional-chaining style and lets the proxy implement these ops as
pure client-side updates to a `(handle, seed, position)` tuple
without RPCs.

Things explicitly NOT on the backend interface (handled by the
composing wrapper, never by the backend):

- map / filter / select / slice / shard
- example-level shuffle buffer (reservoir sampling)
- state_dict / load_state_dict
- set_epoch
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional


class IterableDatasetBackend(ABC):
    """
    Abstract storage backend for an iterable dataset.

    The contract: `__iter__` yields `dict` examples in some order,
    `__len__` returns the total example count, `position()` reports
    the flat example index where the next iteration would start, and
    `shuffle`/`seek` return a new backend instance with the requested
    state change.

    Implementations must:

    - Be deterministic given the same shuffle seed and seek position.
    - Update `position()` as iteration progresses, so a wrapper can
      capture it for `state_dict` at any point.
    - After `shuffle(seed)`, position resets to 0 (the new instance is
      fresh). After `seek(n)`, position is `n`.

    Implementations may optionally expose:

    - `column_names: list[str]` — schema column names.
    - `features` — schema feature dict (HuggingFace-style).
    - `n_shards: int` — number of natural physical shards (e.g. files).

    The wrapper forwards these via attribute access and tolerates
    `AttributeError` for backends that don't provide them.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[dict]:
        """Yield `dict` examples starting at `position()`."""

    @abstractmethod
    def __len__(self) -> int:
        """Total number of examples in the underlying dataset."""

    @abstractmethod
    def shuffle(self, seed: Optional[int] = None) -> "IterableDatasetBackend":
        """
        Return a new backend with the underlying example order
        re-permuted.

        No buffer parameter — the example-level reservoir buffer lives
        in the composing wrapper, not in the backend. The seed
        determines the new order deterministically; if ``None`` an
        implementation-chosen seed is generated and surfaced via the
        new instance's state so it can be reproduced from a checkpoint.

        The returned instance has `position()` reset to 0.
        """

    @abstractmethod
    def seek(self, position: int) -> "IterableDatasetBackend":
        """
        Return a new backend whose next `__iter__` begins at the given
        flat example index.

        Not expected to be O(1) — implementations may need to walk
        index metadata to translate the flat position into their
        internal representation. The returned instance has `position()`
        equal to the requested value.
        """

    @abstractmethod
    def position(self) -> int:
        """
        Current flat example index where the next `__iter__` would
        start.

        Must update during iteration so a wrapper can capture it for
        `state_dict()` after any number of yielded examples.
        """
