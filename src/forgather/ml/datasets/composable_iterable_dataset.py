"""
Composable iterable dataset.

Provides backend-agnostic implementations of higher-level dataset
operations — `map`, `filter`, `select`, `slice`, `shard`,
`set_epoch`, `shuffle` (with a reservoir buffer),
`state_dict`/`load_state_dict` — on top of any
`IterableDatasetBackend`.

The intent is that backends only need to implement the small core
contract (iter, len, shuffle, seek, position) and this wrapper
supplies everything else uniformly. That way a future
`RemoteIterableDataset` proxy doesn't have to re-implement map /
filter / shard etc. — they all run client-side over the proxy's
streamed examples.

The local Arrow backend (`ArrowIterableDataset`) also provides these
operations directly for backwards compatibility with code that uses
the loader's return value as-is. This wrapper is the path forward
for new backends.
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Any, Callable, Dict, Iterator, List, Optional

from torch.utils.data import IterableDataset as TorchIterableDataset

from .iterable_backend import IterableDatasetBackend

logger = logging.getLogger(__name__)


def _identity(x: Dict) -> Dict:
    return x


class ComposableIterableDataset(TorchIterableDataset):
    """
    Backend-agnostic iterable dataset wrapper.

    Wraps an `IterableDatasetBackend` and provides higher-level
    composable operations. All transformations are immutable: each
    method returns a new wrapper. The underlying backend is treated
    as immutable as well — backend-mutating ops (`shuffle`, `seek`)
    return new backend instances and the wrapper holds a reference to
    the latest one.

    The shard `mode` parameter that the legacy Arrow class supported
    is intentionally absent: at this layer sharding is purely logical
    (compute a contiguous example range; restrict iteration to it).
    Backends that want to do physical optimizations (e.g. file-level
    affinity) can do so privately on their own; the wrapper does not
    surface that distinction.

    Multi-worker DataLoader support and progressive length estimation
    for cardinality-changing maps are NOT implemented here — the
    wrapper assumes single-worker iteration and length is computed
    naively from the backend length, slice/shard bounds, and a
    static map-cardinality factor (default 1.0). Use
    `ArrowIterableDataset` directly if you need those features today.
    """

    def __init__(self, backend: IterableDatasetBackend):
        self._backend = backend

        # Slice (virtual split) — absolute indices in backend space.
        self._slice_start: Optional[int] = None
        self._slice_end: Optional[int] = None

        # Shard — implemented logically as a slice computed from
        # (num_shards, shard_index) at construction time.
        self._shard: Optional[tuple[int, int]] = None  # (num, idx) for repr

        # Shuffle state.
        self._base_shuffle_seed: Optional[int] = None
        self._epoch: int = 0
        self._shuffle_buffer_size: Optional[int] = None

        # Map chain — list of dicts so we can compose multiple maps.
        # Each entry: {"fn", "batched", "batch_size", "drop_last_batch",
        #              "with_indices", "input_columns", "remove_columns",
        #              "fn_kwargs"}
        self._maps: List[Dict[str, Any]] = []

    # ----- Construction helpers -----

    def _clone(self, **overrides) -> "ComposableIterableDataset":
        """Return a shallow copy with overrides applied to instance attrs."""
        new = ComposableIterableDataset.__new__(ComposableIterableDataset)
        new._backend = overrides.get("backend", self._backend)
        new._slice_start = overrides.get("slice_start", self._slice_start)
        new._slice_end = overrides.get("slice_end", self._slice_end)
        new._shard = overrides.get("shard", self._shard)
        new._base_shuffle_seed = overrides.get(
            "base_shuffle_seed", self._base_shuffle_seed
        )
        new._epoch = overrides.get("epoch", self._epoch)
        new._shuffle_buffer_size = overrides.get(
            "shuffle_buffer_size", self._shuffle_buffer_size
        )
        new._maps = overrides.get("maps", list(self._maps))
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

    # ----- Effective view bounds in backend space -----

    def _view_bounds(self) -> tuple[int, int]:
        """
        Return the (start, end) example bounds in backend space that
        this wrapper exposes after slice + shard. End is exclusive.
        """
        backend_len = len(self._backend)
        start = self._slice_start if self._slice_start is not None else 0
        end = self._slice_end if self._slice_end is not None else backend_len
        return start, end

    # ----- Length -----

    def __len__(self) -> int:
        start, end = self._view_bounds()
        # Note: cardinality-changing maps will make this an upper bound
        # for filter or anything that returns None; we don't try to
        # estimate map ratios in this wrapper. Use ArrowIterableDataset
        # directly for progressive-estimate behavior.
        return max(0, end - start)

    # ----- shuffle / set_epoch -----

    def shuffle(
        self,
        seed: Optional[int] = None,
        buffer_size: Optional[int] = 1000,
    ) -> "ComposableIterableDataset":
        """
        Re-permute the underlying example order via the backend and
        configure an example-level reservoir shuffle buffer.

        Two-level shuffle: backend re-orders the underlying sequence,
        then iteration applies a buffer for within-window randomness.
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
        )

    def set_epoch(self, epoch: int) -> None:
        """
        Set epoch for per-epoch re-shuffling. Mutates in place because
        callers expect to invoke this once per epoch on a stable
        wrapper instance.

        Re-shuffles the backend with seed = base_seed + epoch (or just
        epoch if no base seed has been set).
        """
        self._epoch = epoch
        if self._base_shuffle_seed is not None:
            effective = self._base_shuffle_seed + epoch
        elif epoch > 0:
            effective = epoch
        else:
            return  # No shuffle requested.
        self._backend = self._backend.shuffle(effective)

    # ----- slice / select / shard -----

    def slice(
        self,
        start: Optional[int | float | str] = None,
        end: Optional[int | float | str] = None,
    ) -> "ComposableIterableDataset":
        """
        Return a view restricted to ``[start, end)``. start/end accept
        ``None``, ``int`` (negative supported), ``float`` in [0, 1],
        or a percentage string like ``"80%"``. The slice is applied
        relative to the current view.
        """

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
        return self._clone(maps=new_maps)

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
        start, end = self._view_bounds()
        # Resume from wherever the backend currently sits if it's already
        # within our window (load_state_dict / partial iteration); otherwise
        # seek to the window start. Mutate self._backend so position()
        # reflects live iteration progress for state_dict capture.
        cur = self._backend.position()
        if cur < start or cur >= end:
            self._backend = self._backend.seek(start)
        gen = self._iter_window(self._backend, start, end)
        if self._shuffle_buffer_size:
            seed = self._effective_buffer_seed()
            gen = self._reservoir_buffer(gen, self._shuffle_buffer_size, seed)
        if self._maps:
            gen = self._apply_maps(gen, start)
        yield from gen

    def _iter_window(
        self,
        backend: IterableDatasetBackend,
        start: int,
        end: int,
    ) -> Iterator[Dict]:
        """Yield examples from backend in the half-open [start, end) window.

        Check-then-consume: we read `position()` before pulling the next
        example so we never over-fetch (which would silently advance past
        the window and skip the first example of the next shard/slice
        view that shares the same backend reference).
        """
        it = iter(backend)
        while True:
            if backend.position() >= end:
                return
            try:
                ex = next(it)
            except StopIteration:
                return
            # `position()` is now one past the just-yielded example.
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

    def _apply_maps(self, it: Iterator[Dict], start_idx: int) -> Iterator[Dict]:
        """Apply the chained map operations. All maps share batched mode."""
        if not self._maps:
            yield from it
            return

        if self._maps[0]["batched"]:
            yield from self._apply_batched_maps(it, start_idx)
        else:
            yield from self._apply_single_maps(it, start_idx)

    def _apply_single_maps(self, it: Iterator[Dict], start_idx: int) -> Iterator[Dict]:
        idx = start_idx
        for example in it:
            current = example
            keep = True
            for spec in self._maps:
                current = self._call_single(spec, current, idx)
                if current is None:
                    keep = False
                    break
            idx += 1
            if keep:
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
        # All maps share batched mode (validated in `map`); use the
        # first map's batch_size and drop_last_batch policy.
        batch_size = self._maps[0]["batch_size"] or 1000
        drop_last = self._maps[0]["drop_last_batch"]

        batch: List[Dict] = []
        batch_start = start_idx
        for example in it:
            batch.append(example)
            if len(batch) >= batch_size:
                yield from self._run_batched_chain(batch, batch_start)
                batch_start += len(batch)
                batch = []
        if batch and not drop_last:
            yield from self._run_batched_chain(batch, batch_start)

    def _run_batched_chain(self, batch: List[Dict], batch_start: int) -> Iterator[Dict]:
        examples = batch
        for spec in self._maps:
            examples = self._apply_batched_step(spec, examples, batch_start)
            if not examples:
                return
        yield from examples

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
        # If the backend exposes its own state_dict (e.g. Arrow's), nest
        # it for richer restore. Otherwise position alone suffices.
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
            "slice_start": self._slice_start,
            "slice_end": self._slice_end,
            "shard": self._shard,
            "base_shuffle_seed": self._base_shuffle_seed,
            "epoch": self._epoch,
            "shuffle_buffer_size": self._shuffle_buffer_size,
            "n_maps": len(self._maps),
            "maps_fingerprint": self._maps_fingerprint(),
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
        self._slice_start = state.get("slice_start")
        self._slice_end = state.get("slice_end")
        self._shard = state.get("shard")
        self._base_shuffle_seed = state.get("base_shuffle_seed")
        self._epoch = state.get("epoch", 0)
        self._shuffle_buffer_size = state.get("shuffle_buffer_size")

        backend_state = state.get("backend", {})
        if (
            "state_dict" in backend_state
            and hasattr(self._backend, "load_state_dict")
            and callable(self._backend.load_state_dict)
        ):
            self._backend.load_state_dict(backend_state["state_dict"])
        else:
            self._backend = self._backend.seek(backend_state["position"])

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
            f"len={len(self)}, slice=({self._slice_start},{self._slice_end}), "
            f"shard={self._shard}, n_maps={len(self._maps)}, "
            f"buffer={self._shuffle_buffer_size}, epoch={self._epoch})"
        )
