"""
ArrowBackend — pure storage backend over HuggingFace Arrow files.

Implements `IterableDatasetBackend` and nothing else. Higher-level
operations (map/filter/select/slice/shard, the example-level shuffle
buffer, multi-worker support, length estimation, set_epoch,
state_dict/load_state_dict at the wrapper level) are provided by
`ComposableIterableDataset`, which wraps this backend.

Constructed by `FastDatasetLoaderSimple`/`fast_load_iterable_dataset`
once an Arrow file index is available; rarely instantiated by user
code directly.

The backend keeps a cursor over a (possibly shuffled) ordering of
the underlying Arrow files. ``shuffle(seed)`` and ``seek(position)``
return new instances; iteration mutates the cursor on the active
instance so callers can capture progress via ``position()``.

A backend-level ``state_dict``/``load_state_dict`` is provided so the
wrapper can verify dataset identity (file-list SHA-256) and
reconstruct the shuffled file order from the saved seed. Other
backends in this package don't expose state_dict and the wrapper
falls back to the bare ``position()`` on those.
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

from datasets import Dataset

from .iterable_backend import IterableDatasetBackend

logger = logging.getLogger(__name__)


class ArrowBackend(IterableDatasetBackend):
    """
    Storage backend over a list of memory-mapped Arrow files.

    Parameters
    ----------
    arrow_files : list of str
        Ordered list of Arrow file paths that make up the dataset.
        Each file is treated as one natural shard.
    file_lengths : list of int, optional
        Per-file example counts, parallel to ``arrow_files``. When
        provided, ``__len__`` and ``seek`` are O(num_files) without
        any file I/O. When ``None``, file lengths are read on
        construction by opening each file (slow path; the loader
        normally avoids this by passing cached lengths from the
        on-disk index).

    Notes
    -----
    `__iter__` mutates the cursor; multiple concurrent iterators on
    the same instance would interfere. In multi-worker DataLoader
    setups each worker receives its own copy (via fork or pickle), so
    concurrent cursors aren't an issue in practice.
    """

    def __init__(
        self,
        arrow_files: List[str],
        file_lengths: Optional[List[int]] = None,
    ):
        self.arrow_files: List[str] = list(arrow_files)
        if file_lengths is None:
            # Slow fallback — open each file to read its length.
            self.file_lengths = [len(Dataset.from_file(f)) for f in self.arrow_files]
        else:
            if len(file_lengths) != len(self.arrow_files):
                raise ValueError(
                    f"file_lengths length ({len(file_lengths)}) does not match "
                    f"arrow_files length ({len(self.arrow_files)})"
                )
            self.file_lengths = list(file_lengths)

        # Current iteration order (after any shuffle).
        self._seed: Optional[int] = None
        self._order_files: List[str] = self.arrow_files
        self._order_lengths: List[int] = self.file_lengths

        # Flat cursor — index of the NEXT example to yield.
        self._position: int = 0

        # Lazy schema cache (read from the first Arrow file).
        self._column_names: Optional[List[str]] = None
        self._features = None

    # ----- Backend interface -----

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Walk Arrow files from the current cursor to end, yielding
        examples and updating ``_position`` as each one is emitted.

        ``_position`` is incremented BEFORE yield so callers that read
        it (e.g. the wrapper's `_iter_window` in check-then-consume
        mode) see "index of the next example" semantics consistently
        with InMemoryBackend / RemoteBackend.
        """
        cumul = 0
        for path, file_len in zip(self._order_files, self._order_lengths):
            file_end = cumul + file_len
            if file_end <= self._position:
                cumul = file_end
                continue
            local_start = max(0, self._position - cumul)
            ds = Dataset.from_file(path)
            if local_start > 0:
                ds = ds.select(range(local_start, file_len))
            for example in ds:
                self._position += 1
                yield example
            cumul = file_end

    def __len__(self) -> int:
        return sum(self.file_lengths)

    def shuffle(self, seed: Optional[int] = None) -> "ArrowBackend":
        """
        Return a new backend with files re-permuted under ``seed``.
        Cursor is reset to 0. No example-level buffer — that lives in
        the wrapper.
        """
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        new = ArrowBackend.__new__(ArrowBackend)
        new.arrow_files = self.arrow_files
        new.file_lengths = self.file_lengths
        new._seed = seed
        new._order_files, new._order_lengths = self._shuffled_order(
            self.arrow_files, self.file_lengths, seed
        )
        new._position = 0
        new._column_names = self._column_names
        new._features = self._features
        return new

    def seek(self, position: int) -> "ArrowBackend":
        """
        Return a new backend with the cursor at ``position``. Past-the-end
        positions are clamped to the end (next iteration yields nothing).
        """
        if position < 0:
            raise ValueError(f"position must be non-negative, got {position}")
        new = ArrowBackend.__new__(ArrowBackend)
        new.arrow_files = self.arrow_files
        new.file_lengths = self.file_lengths
        new._seed = self._seed
        new._order_files = self._order_files
        new._order_lengths = self._order_lengths
        new._position = min(position, len(self))
        new._column_names = self._column_names
        new._features = self._features
        return new

    def position(self) -> int:
        return self._position

    # ----- Optional metadata -----

    @property
    def column_names(self) -> List[str]:
        if self._column_names is None:
            if not self.arrow_files:
                return []
            self._column_names = Dataset.from_file(self.arrow_files[0]).column_names
        return self._column_names

    @property
    def features(self):
        if self._features is None:
            if not self.arrow_files:
                return None
            self._features = Dataset.from_file(self.arrow_files[0]).features
        return self._features

    @property
    def n_shards(self) -> int:
        return len(self.arrow_files)

    # ----- Optional checkpoint protocol (used by the wrapper) -----

    def state_dict(self) -> Dict[str, Any]:
        """
        Capture cursor + order seed + dataset-identity fingerprint.

        The wrapper picks this up via the optional-backend-state_dict
        path so a checkpoint round-trip can detect "different files
        behind the same handle" early.
        """
        return {
            "version": 1,
            "fingerprint": self._fingerprint(),
            "num_files": len(self.arrow_files),
            "seed": self._seed,
            "position": self._position,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("version") != 1:
            raise ValueError(
                f"Unknown ArrowBackend state version: {state.get('version')!r}"
            )
        saved_fp = state.get("fingerprint")
        if saved_fp is not None and saved_fp != self._fingerprint():
            raise ValueError(
                "Dataset fingerprint mismatch — checkpoint refers to a "
                "different set of Arrow files."
            )
        saved_n = state.get("num_files")
        if saved_n is not None and saved_n != len(self.arrow_files):
            raise ValueError(
                f"Number of files mismatch: checkpoint has {saved_n}, "
                f"backend has {len(self.arrow_files)}."
            )
        self._seed = state.get("seed")
        if self._seed is not None:
            self._order_files, self._order_lengths = self._shuffled_order(
                self.arrow_files, self.file_lengths, self._seed
            )
        else:
            self._order_files = self.arrow_files
            self._order_lengths = self.file_lengths
        self._position = int(state.get("position", 0))

    # ----- Helpers -----

    def _fingerprint(self) -> str:
        return hashlib.sha256("\n".join(self.arrow_files).encode("utf-8")).hexdigest()

    @staticmethod
    def _shuffled_order(
        files: List[str],
        lengths: List[int],
        seed: int,
    ) -> Tuple[List[str], List[int]]:
        rng = random.Random(seed)
        paired = list(zip(files, lengths))
        rng.shuffle(paired)
        files_shuffled, lengths_shuffled = zip(*paired)
        return list(files_shuffled), list(lengths_shuffled)

    def __repr__(self) -> str:
        return (
            f"ArrowBackend(files={len(self.arrow_files)}, "
            f"examples={len(self)}, position={self._position}, "
            f"seed={self._seed})"
        )
