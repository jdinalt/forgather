"""
Composable iterable dataset.

Backend-agnostic implementations of every higher-level dataset operation
on top of the small `IterableDatasetBackend` contract — `map`, `filter`,
`select`, `slice`, `shard`, `shuffle` (with example-level reservoir
buffer), `set_epoch`, `state_dict` / `load_state_dict`, multi-worker
DataLoader sub-window partitioning, and progressive length estimation
for cardinality-changing maps.

This is the canonical wrapper that all backends (Arrow files, in-memory,
network proxy) flow through. Backends only need to implement the
storage primitives (iter / len / shuffle / seek / position); the
wrapper supplies everything else uniformly so a future remote backend
gets `.map()` / `.shard()` / `state_dict` "for free."
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch.utils.data
from torch.utils.data import IterableDataset as TorchIterableDataset

from .iterable_backend import IterableDatasetBackend

logger = logging.getLogger(__name__)


_LENGTH_MODES = ("static", "dynamic", "exact")


def _identity(x: Dict) -> Dict:
    return x


class ComposableIterableDataset(TorchIterableDataset):
    """
    Backend-agnostic iterable dataset wrapper.

    Wraps any `IterableDatasetBackend`. Composable transformations
    (`map`, `slice`, `shard`, `shuffle`, …) return new wrapper
    instances; `set_epoch` mutates in place (callers re-use the same
    wrapper instance across epochs). Backend-mutating ops (`shuffle`,
    `seek`) return new backend instances and the wrapper holds a
    reference to the latest one.

    The shard `mode` parameter that the legacy Arrow class supported
    is intentionally absent: at this layer sharding is purely logical
    (compute a contiguous example range; restrict iteration to it).
    Backends that want to do physical optimizations (e.g. file-level
    affinity) can do so privately on their own; the wrapper does not
    surface that distinction.

    Multi-worker DataLoader support is built in: when iterated under
    `torch.utils.data.DataLoader(num_workers > 1)` each worker takes a
    contiguous sub-window of the visible range. Per-worker checkpoint
    state is captured by `state_dict` and restored by `load_state_dict`.

    Length estimation has three modes (`length_estimate_mode`):

    - ``"static"`` — `__len__` always returns the view length (after
      slice/shard), ignoring map-induced cardinality changes.
    - ``"dynamic"`` (default) — progressive ratio-based estimate during
      the first complete pass, then locked to the exact count via
      ``_cached_exact_length`` once iteration runs to completion.
    - ``"exact"`` — alias for ``"dynamic"``.

    Parameters
    ----------
    backend : IterableDatasetBackend
        Underlying storage backend.
    length_estimate : {"dynamic", "static", "exact"}, optional
        Initial length-estimation mode. Default ``"dynamic"``.
    reset_length_on_iter : bool, optional
        If ``True``, reset input/output counters at the start of every
        new iteration. Default ``False`` (counters accumulate across
        passes).
    """

    def __init__(
        self,
        backend: IterableDatasetBackend,
        length_estimate: str = "dynamic",
        reset_length_on_iter: bool = False,
    ):
        if length_estimate not in _LENGTH_MODES:
            raise ValueError(
                f"Invalid length_estimate mode: {length_estimate!r}. "
                f"Must be one of {_LENGTH_MODES}."
            )

        self._backend = backend

        # Slice (virtual split) — absolute indices in backend space.
        self._split_start_idx: Optional[int] = None
        self._split_end_idx: Optional[int] = None

        # Shard — implemented logically as a slice computed from
        # (num_shards, shard_index) at construction time.
        self._shard: Optional[Tuple[int, int]] = None  # (num, idx) for repr

        # Shuffle state.
        self._base_shuffle_seed: Optional[int] = None
        self._epoch: int = 0
        self._shuffle_buffer_size: Optional[int] = None

        # Map chain — list of dicts so we can compose multiple maps.
        # Each entry: {"fn", "batched", "batch_size", "drop_last_batch",
        #              "with_indices", "input_columns", "remove_columns",
        #              "fn_kwargs"}
        self._maps: List[Dict[str, Any]] = []

        # Length estimation state.
        self.length_estimate_mode: str = length_estimate
        self._reset_length_on_iter: bool = reset_length_on_iter
        self._input_count: int = 0
        self._output_count: int = 0
        self._cached_exact_length: Optional[int] = None
        self._length_invalidated: bool = False
        self._current_batch_buffer_size: int = 0
        # Set by load_state_dict so the next __iter__ honours the
        # restored backend cursor and counts instead of resetting them.
        self._restored_from_checkpoint: bool = False

    # ----- Construction helpers -----

    def _clone(self, **overrides) -> "ComposableIterableDataset":
        """Return a shallow copy with overrides applied to instance attrs."""
        new = ComposableIterableDataset.__new__(ComposableIterableDataset)
        new._backend = overrides.get("backend", self._backend)
        new._split_start_idx = overrides.get("slice_start", self._split_start_idx)
        new._split_end_idx = overrides.get("slice_end", self._split_end_idx)
        new._shard = overrides.get("shard", self._shard)
        new._base_shuffle_seed = overrides.get(
            "base_shuffle_seed", self._base_shuffle_seed
        )
        new._epoch = overrides.get("epoch", self._epoch)
        new._shuffle_buffer_size = overrides.get(
            "shuffle_buffer_size", self._shuffle_buffer_size
        )
        new._maps = overrides.get("maps", list(self._maps))
        new.length_estimate_mode = overrides.get(
            "length_estimate_mode", self.length_estimate_mode
        )
        new._reset_length_on_iter = overrides.get(
            "reset_length_on_iter", self._reset_length_on_iter
        )
        new._input_count = overrides.get("input_count", self._input_count)
        new._output_count = overrides.get("output_count", self._output_count)
        new._cached_exact_length = overrides.get(
            "cached_exact_length", self._cached_exact_length
        )
        new._length_invalidated = overrides.get(
            "length_invalidated", self._length_invalidated
        )
        new._current_batch_buffer_size = overrides.get(
            "current_batch_buffer_size", self._current_batch_buffer_size
        )
        new._restored_from_checkpoint = overrides.get(
            "restored_from_checkpoint", self._restored_from_checkpoint
        )
        return new

    # ----- Backend metadata pass-through -----

    @property
    def backend(self) -> IterableDatasetBackend:
        return self._backend

    @property
    def column_names(self) -> Optional[List[str]]:
        return getattr(self._backend, "column_names", None)

    @property
    def features(self):
        return getattr(self._backend, "features", None)

    @property
    def n_shards(self) -> int:
        n = getattr(self._backend, "n_shards", None)
        return n if n is not None else 1

    @property
    def _shuffle_seed(self) -> Optional[int]:
        """
        Effective shuffle seed currently in use (``base_seed + epoch``
        when both are set, just ``epoch`` if epoch>0 with no base seed,
        else the base seed if any). Exposed for introspection / parity
        with the legacy class — internal logic computes it on the fly
        via `_effective_buffer_seed`.
        """
        if self._base_shuffle_seed is not None:
            return self._base_shuffle_seed + self._epoch
        if self._epoch > 0:
            return self._epoch
        return None

    # ----- Effective view bounds in backend space -----

    def _view_bounds(self) -> Tuple[int, int]:
        """
        (start, end) in backend space after slice + shard. End exclusive.
        """
        backend_len = len(self._backend)
        start = self._split_start_idx if self._split_start_idx is not None else 0
        end = self._split_end_idx if self._split_end_idx is not None else backend_len
        return start, end

    @staticmethod
    def _get_worker_info() -> Tuple[int, int]:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return 0, 1
        return int(info.id), int(info.num_workers)

    def _worker_view_bounds(self) -> Tuple[int, int]:
        """
        (start, end) restricted to the current worker's slice of the
        view. Identical to `_view_bounds()` outside DataLoader workers
        or when num_workers <= 1.
        """
        start, end = self._view_bounds()
        worker_id, num_workers = self._get_worker_info()
        if num_workers <= 1:
            return start, end
        total = end - start
        if total <= 0:
            return start, end
        per_worker = int(math.ceil(total / num_workers))
        worker_start = start + worker_id * per_worker
        worker_end = min(worker_start + per_worker, end)
        if worker_start >= end:
            # This worker has no work — yield an empty range.
            return end, end
        return worker_start, worker_end

    # ----- Length -----

    def __len__(self) -> int:
        start, end = self._view_bounds()
        base = max(0, end - start)
        if self.length_estimate_mode == "static":
            return base
        if self._cached_exact_length is not None:
            return self._cached_exact_length
        if self._output_count > 0 and self._input_count > 0:
            ratio = self._output_count / self._input_count
            return int(base * ratio)
        return base

    def set_length_estimate_mode(self, mode: str) -> None:
        if mode not in _LENGTH_MODES:
            raise ValueError(f"Invalid mode: {mode!r}. Must be one of {_LENGTH_MODES}.")
        self.length_estimate_mode = mode

    def get_length_stats(self) -> Dict[str, Any]:
        start, end = self._view_bounds()
        base = max(0, end - start)
        ratio: Optional[float]
        if self._output_count > 0 and self._input_count > 0:
            ratio = self._output_count / self._input_count
        else:
            ratio = None
        return {
            "mode": self.length_estimate_mode,
            "original_length": base,
            "input_count": self._input_count,
            "output_count": self._output_count,
            "cached_exact": self._cached_exact_length,
            "ratio": ratio,
            "invalidated": self._length_invalidated,
            "batch_buffer_size": self._current_batch_buffer_size,
            "reset_on_iter": self._reset_length_on_iter,
            "current_estimate": len(self),
        }

    # ----- shuffle / set_epoch -----

    def shuffle(
        self,
        seed: Optional[int] = None,
        buffer_size: Optional[int] = 1000,
    ) -> "ComposableIterableDataset":
        """
        Re-permute the underlying example order via the backend and
        configure an example-level reservoir shuffle buffer.

        Length-estimation cache is invalidated; existing input/output
        counts are preserved as a ratio carry-over.
        """
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        if buffer_size is None or buffer_size <= 0:
            buffer_size = None
        new_backend = self._backend.shuffle(seed)
        return self._clone(
            backend=new_backend,
            base_shuffle_seed=seed,
            epoch=0,
            shuffle_buffer_size=buffer_size,
            # Invalidate but preserve ratio.
            cached_exact_length=None,
            length_invalidated=True,
            # Counts intentionally carried over.
        )

    def set_epoch(self, epoch: int) -> None:
        """
        Set the current epoch and re-shuffle the backend if any seed
        is in play. Mutates in place.
        """
        self._epoch = epoch
        if self._base_shuffle_seed is not None:
            effective = self._base_shuffle_seed + epoch
        elif epoch > 0:
            effective = epoch
        else:
            return
        self._backend = self._backend.shuffle(effective)

    # ----- slice / select / shard -----

    def slice(
        self,
        start: Optional[int | float | str] = None,
        end: Optional[int | float | str] = None,
    ) -> "ComposableIterableDataset":
        """Return a view restricted to ``[start, end)``."""

        def parse(idx, total):
            if idx is None:
                return None
            if isinstance(idx, str):
                if idx.endswith("%"):
                    idx = float(idx[:-1]) / 100.0
                else:
                    idx = float(idx)
            if isinstance(idx, float):
                if not 0 <= idx <= 1:
                    raise ValueError(f"Percentage must be in range [0, 1], got {idx}")
                return int(idx * total)
            if isinstance(idx, int):
                if idx < 0:
                    return total + idx
                return idx
            raise ValueError(f"Invalid index type: {type(idx)}")

        cur_start, cur_end = self._view_bounds()
        cur_len = cur_end - cur_start

        rel_start = parse(start, cur_len) if start is not None else 0
        rel_end = parse(end, cur_len) if end is not None else cur_len

        if not 0 <= rel_start <= cur_len:
            raise ValueError(f"Start index {rel_start} out of range [0, {cur_len}]")
        if not 0 <= rel_end <= cur_len:
            raise ValueError(f"End index {rel_end} out of range [0, {cur_len}]")
        if rel_start >= rel_end:
            raise ValueError(f"Start index {rel_start} must be < end index {rel_end}")

        return self._clone(
            slice_start=cur_start + rel_start,
            slice_end=cur_start + rel_end,
            # Different view; counts and cache no longer apply.
            input_count=0,
            output_count=0,
            cached_exact_length=None,
            length_invalidated=False,
        )

    def select(self, indices) -> "ComposableIterableDataset":
        """Contiguous-range select; non-contiguous indices not supported."""
        if hasattr(indices, "tolist"):
            indices = indices.tolist()
        elif not isinstance(indices, list):
            indices = list(indices)
        if not indices:
            raise ValueError("Cannot select from empty indices")
        start = indices[0]
        end = indices[-1] + 1
        if indices != list(range(start, end)):
            raise NotImplementedError(
                "Only contiguous, ordered index sequences are supported."
            )
        return self.slice(start, end)

    def shard(self, num_shards: int, index: int) -> "ComposableIterableDataset":
        """
        Split into ``num_shards`` disjoint slices and return the one
        at ``index``. Logical sharding only — there is no ``mode``
        parameter at this layer; the backend may do whatever physical
        optimization it wants internally.
        """
        if num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {num_shards}")
        if not 0 <= index < num_shards:
            raise ValueError(f"index ({index}) must be in [0, {num_shards})")

        cur_start, cur_end = self._view_bounds()
        total = cur_end - cur_start
        per_shard = total // num_shards
        remainder = total % num_shards
        # Distribute remainder examples to first `remainder` shards.
        if index < remainder:
            shard_offset = index * (per_shard + 1)
            shard_size = per_shard + 1
        else:
            shard_offset = index * per_shard + remainder
            shard_size = per_shard

        return self._clone(
            slice_start=cur_start + shard_offset,
            slice_end=cur_start + shard_offset + shard_size,
            shard=(num_shards, index),
            input_count=0,
            output_count=0,
            cached_exact_length=None,
            length_invalidated=False,
        )

    # ----- map / filter -----

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[str | List[str]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[str | List[str]] = None,
        fn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "ComposableIterableDataset":
        """
        Append a map step to the chain. Multiple map calls compose.

        A non-batched function returning ``None`` filters the example
        out (matches the legacy Arrow class behavior).

        Mixed batched / non-batched chains are not supported (raises).
        """
        if function is None:
            function = _identity
        if isinstance(input_columns, str):
            input_columns = [input_columns]
        if isinstance(remove_columns, str):
            remove_columns = [remove_columns]
        if fn_kwargs is None:
            fn_kwargs = {}

        if self._maps:
            existing_batched = self._maps[0]["batched"]
            if existing_batched != batched:
                raise ValueError("Cannot chain maps with different batched modes.")

        new_maps = list(self._maps)
        new_maps.append(
            {
                "fn": function,
                "batched": batched,
                "batch_size": batch_size,
                "drop_last_batch": drop_last_batch,
                "with_indices": with_indices,
                "input_columns": input_columns,
                "remove_columns": remove_columns,
                "fn_kwargs": fn_kwargs,
            }
        )
        return self._clone(
            maps=new_maps,
            input_count=0,
            output_count=0,
            cached_exact_length=None,
            length_invalidated=False,
        )

    def filter(
        self,
        function: Callable,
        with_indices: bool = False,
        input_columns: Optional[str | List[str]] = None,
        fn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "ComposableIterableDataset":
        """Keep examples where ``function(example)`` returns truthy."""
        if isinstance(input_columns, str):
            input_columns = [input_columns]
        if fn_kwargs is None:
            fn_kwargs = {}

        def _filter_map(example, *args, **kwargs):
            if input_columns is not None:
                fn_input = {c: example[c] for c in input_columns if c in example}
            else:
                fn_input = example
            keep = (
                function(fn_input, *args, **fn_kwargs)
                if (with_indices and args)
                else function(fn_input, **fn_kwargs)
            )
            return example if keep else None

        return self.map(_filter_map, with_indices=with_indices)

    # ----- iteration -----

    def __iter__(self) -> Iterator[Dict]:
        # Decide whether to reset count state at the start of this pass.
        self._maybe_reset_counts()

        start, end = self._worker_view_bounds()

        # Position the backend at our window start unless we're resuming
        # mid-window (after load_state_dict or partial iteration that
        # left the cursor inside [start, end)).
        cur = self._backend.position()
        if cur < start or cur >= end:
            self._backend = self._backend.seek(start)

        # Clear the restored flag — by the time we're iterating we've
        # honored it.
        self._restored_from_checkpoint = False

        gen = self._iter_window(self._backend, start, end)
        if self._shuffle_buffer_size:
            gen = self._reservoir_buffer(
                gen, self._shuffle_buffer_size, self._effective_buffer_seed()
            )
        if self._maps:
            if self._maps[0]["batched"]:
                gen = self._apply_batched_maps(gen, start)
            else:
                gen = self._apply_single_maps(gen, start)
        else:
            gen = self._track_passthrough(gen)

        completed = False
        try:
            for ex in gen:
                yield ex
            completed = True
        finally:
            self._on_iter_done(completed)

    def _maybe_reset_counts(self) -> None:
        # Don't touch state if we just restored from a checkpoint —
        # the saved counts must survive into the next iteration.
        if self._restored_from_checkpoint:
            return
        if self._reset_length_on_iter or self._length_invalidated:
            preserve_ratio = (
                self._length_invalidated
                and self._input_count > 0
                and self._output_count > 0
            )
            if not preserve_ratio:
                self._input_count = 0
                self._output_count = 0
            self._cached_exact_length = None
            self._length_invalidated = False
            self._current_batch_buffer_size = 0
        else:
            # Fresh iteration with no invalidation: reset counts so the
            # next pass tracks itself, but keep the cached exact length
            # (it persists across iterations once observed).
            self._input_count = 0
            self._output_count = 0
            self._current_batch_buffer_size = 0

    def _on_iter_done(self, completed: bool) -> None:
        self._current_batch_buffer_size = 0
        if not completed:
            return
        # Iteration ran to natural end — cache the exact output count
        # for cardinality-changing maps in dynamic/exact mode.
        if self.length_estimate_mode in ("dynamic", "exact"):
            if self._output_count > 0:
                self._cached_exact_length = self._output_count

    def _iter_window(
        self,
        backend: IterableDatasetBackend,
        start: int,
        end: int,
    ) -> Iterator[Dict]:
        """Yield examples from backend in [start, end). Check-then-consume
        to avoid over-fetching past the window."""
        if start >= end:
            return
        it = iter(backend)
        while True:
            if backend.position() >= end:
                return
            try:
                ex = next(it)
            except StopIteration:
                return
            yielded_idx = backend.position() - 1
            if yielded_idx < start:
                continue
            yield ex

    def _effective_buffer_seed(self) -> int:
        if self._base_shuffle_seed is not None:
            return self._base_shuffle_seed + self._epoch
        return self._epoch or 0

    @staticmethod
    def _reservoir_buffer(
        it: Iterator[Dict], buffer_size: int, seed: int
    ) -> Iterator[Dict]:
        rng = random.Random(seed)
        buf: List[Dict] = []
        for ex in it:
            buf.append(ex)
            if len(buf) >= buffer_size:
                break
        if not buf:
            return
        for ex in it:
            idx = rng.randint(0, buffer_size - 1)
            yield buf[idx]
            buf[idx] = ex
        rng.shuffle(buf)
        yield from buf

    def _track_passthrough(self, it: Iterator[Dict]) -> Iterator[Dict]:
        for ex in it:
            self._input_count += 1
            self._output_count += 1
            yield ex

    def _apply_single_maps(self, it: Iterator[Dict], start_idx: int) -> Iterator[Dict]:
        idx = start_idx
        for example in it:
            self._input_count += 1
            current = example
            keep = True
            for spec in self._maps:
                current = self._call_single(spec, current, idx)
                if current is None:
                    keep = False
                    break
            idx += 1
            if keep:
                self._output_count += 1
                yield current

    @staticmethod
    def _call_single(spec: Dict[str, Any], example: Dict, idx: int) -> Optional[Dict]:
        if spec["input_columns"] is not None:
            fn_input = {c: example[c] for c in spec["input_columns"] if c in example}
        else:
            fn_input = example
        if spec["with_indices"]:
            result = spec["fn"](fn_input, idx, **spec["fn_kwargs"])
        else:
            result = spec["fn"](fn_input, **spec["fn_kwargs"])
        if result is None:
            return None
        if not isinstance(result, dict):
            raise ValueError(
                f"Map function must return a dict or None, got {type(result)}"
            )
        merged = example.copy()
        merged.update(result)
        if spec["remove_columns"] is not None:
            for col in spec["remove_columns"]:
                merged.pop(col, None)
        return merged

    def _apply_batched_maps(self, it: Iterator[Dict], start_idx: int) -> Iterator[Dict]:
        """Batch-collect, run all maps in sequence over the batch."""
        batch_size = self._maps[0]["batch_size"] or 1000
        drop_last = self._maps[0]["drop_last_batch"]

        batch: List[Dict] = []
        batch_start = start_idx
        for example in it:
            batch.append(example)
            self._current_batch_buffer_size = len(batch)
            if len(batch) >= batch_size:
                results = self._run_batched_chain(batch, batch_start)
                self._input_count += len(batch)
                self._output_count += len(results)
                self._current_batch_buffer_size = 0
                yield from results
                batch_start += len(batch)
                batch = []
        if batch and not drop_last:
            results = self._run_batched_chain(batch, batch_start)
            self._input_count += len(batch)
            self._output_count += len(results)
            self._current_batch_buffer_size = 0
            yield from results

    def _run_batched_chain(self, batch: List[Dict], batch_start: int) -> List[Dict]:
        examples = batch
        for spec in self._maps:
            examples = self._apply_batched_step(spec, examples, batch_start)
            if not examples:
                return []
        return examples

    @staticmethod
    def _apply_batched_step(
        spec: Dict[str, Any], examples: List[Dict], batch_start: int
    ) -> List[Dict]:
        if not examples:
            return examples
        # Collect into dict-of-lists.
        if spec["input_columns"] is not None:
            cols = spec["input_columns"]
        else:
            cols = sorted({k for ex in examples for k in ex.keys()})
        batch_dict = {c: [ex.get(c) for ex in examples] for c in cols}

        if spec["with_indices"]:
            indices = list(range(batch_start, batch_start + len(examples)))
            result = spec["fn"](batch_dict, indices, **spec["fn_kwargs"])
        else:
            result = spec["fn"](batch_dict, **spec["fn_kwargs"])

        if result is None:
            return []
        if not isinstance(result, dict):
            raise ValueError(
                f"Batched map function must return a dict or None, got {type(result)}"
            )
        # Determine output count from the first list-valued column.
        n_out = 0
        for v in result.values():
            if isinstance(v, list):
                n_out = len(v)
                break

        out: List[Dict] = []
        for i in range(n_out):
            row = {}
            for c, v in result.items():
                row[c] = v[i] if isinstance(v, list) else v
            if i < len(examples):
                merged = examples[i].copy()
                merged.update(row)
            else:
                merged = row
            if spec["remove_columns"] is not None:
                for c in spec["remove_columns"]:
                    merged.pop(c, None)
            out.append(merged)
        return out

    # ----- HF compatibility -----

    def to_hf_iterable(self):
        """
        Wrap this dataset in a HuggingFace ``IterableDataset`` for APIs
        that require one. The returned object exposes ``__len__`` via
        `IterableDatasetWithLength` so it can drive ``torch.DataLoader``;
        the wrapper checkpoint protocol is *not* preserved on the
        returned value.
        """
        from datasets import IterableDataset as HFIterableDataset

        from .iterable_with_length import IterableDatasetWithLength

        def gen():
            yield from self

        return IterableDatasetWithLength(
            HFIterableDataset.from_generator(gen), len(self)
        )

    # ----- checkpoint protocol -----

    def state_dict(self) -> Dict[str, Any]:
        """
        Capture wrapper state plus the backend's flat position.

        The backend's `position()` is in underlying-example space, not
        in user-facing post-slice/shard/map space — that's deliberate
        so resume can call `backend.seek(saved_position)` and continue
        consuming examples regardless of how a map function may have
        changed cardinality.
        """
        backend_state: Dict[str, Any] = {"position": self._backend.position()}
        if hasattr(self._backend, "state_dict") and callable(self._backend.state_dict):
            try:
                backend_state["state_dict"] = self._backend.state_dict()
            except Exception as exc:  # pragma: no cover
                logger.debug(
                    "backend.state_dict() failed; falling back to position only: %s",
                    exc,
                )

        return {
            "wrapper_version": 1,
            "slice_start": self._split_start_idx,
            "slice_end": self._split_end_idx,
            "shard": self._shard,
            "base_shuffle_seed": self._base_shuffle_seed,
            "epoch": self._epoch,
            "shuffle_buffer_size": self._shuffle_buffer_size,
            "n_maps": len(self._maps),
            "maps_fingerprint": self._maps_fingerprint(),
            "length_estimate_mode": self.length_estimate_mode,
            "reset_length_on_iter": self._reset_length_on_iter,
            "input_count": self._input_count,
            "output_count": self._output_count,
            "cached_exact_length": self._cached_exact_length,
            "length_invalidated": self._length_invalidated,
            "backend": backend_state,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Restore wrapper state and seek the backend to the saved
        position. Map functions themselves are not serialised — the
        caller must reconstruct the same map chain before calling
        `load_state_dict` (a fingerprint is checked).
        """
        if state.get("wrapper_version") != 1:
            raise ValueError(f"Unknown wrapper_version: {state.get('wrapper_version')}")
        if state.get("maps_fingerprint") != self._maps_fingerprint():
            raise ValueError(
                "Map chain fingerprint mismatch — reconstruct the same "
                "map chain (with the same number/order of maps) before "
                "calling load_state_dict."
            )
        self._split_start_idx = state.get("slice_start")
        self._split_end_idx = state.get("slice_end")
        self._shard = state.get("shard")
        self._base_shuffle_seed = state.get("base_shuffle_seed")
        self._epoch = state.get("epoch", 0)
        self._shuffle_buffer_size = state.get("shuffle_buffer_size")

        self.length_estimate_mode = state.get(
            "length_estimate_mode", self.length_estimate_mode
        )
        self._reset_length_on_iter = state.get(
            "reset_length_on_iter", self._reset_length_on_iter
        )
        self._input_count = int(state.get("input_count", 0))
        self._output_count = int(state.get("output_count", 0))
        self._cached_exact_length = state.get("cached_exact_length")
        self._length_invalidated = bool(state.get("length_invalidated", False))

        backend_state = state.get("backend", {})
        if (
            "state_dict" in backend_state
            and hasattr(self._backend, "load_state_dict")
            and callable(self._backend.load_state_dict)
        ):
            self._backend.load_state_dict(backend_state["state_dict"])
        else:
            self._backend = self._backend.seek(backend_state["position"])

        # Mark restored so the next iteration honors saved counts.
        self._restored_from_checkpoint = True

    def _maps_fingerprint(self) -> str:
        """
        Cheap fingerprint of the map chain so load_state_dict can
        catch the obvious "you added/removed maps" foot-gun. Uses
        function qualnames and kwarg keys, not the function bodies.
        """
        h = hashlib.sha256()
        for spec in self._maps:
            fn = spec["fn"]
            qual = getattr(fn, "__qualname__", repr(fn))
            mod = getattr(fn, "__module__", "")
            h.update(f"{mod}.{qual}|".encode())
            h.update(f"batched={spec['batched']}|".encode())
            h.update(f"with_indices={spec['with_indices']}|".encode())
            ic = ",".join(spec["input_columns"] or [])
            rc = ",".join(spec["remove_columns"] or [])
            h.update(f"input={ic}|remove={rc}|".encode())
            kwarg_keys = ",".join(sorted((spec["fn_kwargs"] or {}).keys()))
            h.update(f"kw={kwarg_keys}|".encode())
        return h.hexdigest()

    # ----- repr -----

    def __repr__(self) -> str:
        return (
            f"ComposableIterableDataset(backend={type(self._backend).__name__}, "
            f"len={len(self)}, slice=({self._split_start_idx},{self._split_end_idx}), "
            f"shard={self._shard}, n_maps={len(self._maps)}, "
            f"buffer={self._shuffle_buffer_size}, epoch={self._epoch}, "
            f"mode={self.length_estimate_mode})"
        )
