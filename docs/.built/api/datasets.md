# Datasets

Dataset utilities for loading, tokenizing, and packing sequences for causal language model training.

**Related documentation:**

- [Fast HF Loader](../datasets/fast-hf-loader.md) — indexed Arrow loading with seconds-to-open on large datasets
- [Fast HF Loader Checkpoints](../datasets/fast-hf-loader-checkpoints.md) — stateful resume from any dataset position
- [Sequence Packing](../datasets/sequence-packing.md) — packing multiple documents per batch with document-boundary tracking
- [Sequence Packing Quick Reference](../datasets/sequence-packing-quick-reference.md)
- [Document Boundaries](../datasets/document-boundaries.md) — enforcing no cross-document attention with Flex Attention
- [Dataset Projects](../datasets/dataset-projects.md) — organising datasets as standalone Forgather projects
- [Dataset CLI](../datasets/dataset-cli.md) — `forgather dataset` commands for inspecting and sampling datasets

## Fast HuggingFace Loader

### `FastDatasetLoaderSimple` {#forgather-ml-datasets-fast_hf_loader-fastdatasetloadersimple}

`forgather.ml.datasets.fast_hf_loader.FastDatasetLoaderSimple`

```python
class FastDatasetLoaderSimple(index_dir: Optional[str] = None)
```

Fast HuggingFace dataset loader backed by an Arrow file index.

On the first call for a given dataset/split combination the loader
downloads (or locates) the dataset via the HuggingFace ``datasets``
library, records the paths and per-file example counts of the
underlying Arrow cache files in a compact JSON index, and returns
a `ComposableIterableDataset` wrapping an `ArrowBackend`. All
subsequent calls for the same configuration load in milliseconds
by reading the index directly.

Both HuggingFace Hub datasets and locally saved datasets (produced
by ``Dataset.save_to_disk()``) are supported.

**Parameters**

- `index_dir` (str) — Directory in which the JSON index files are stored. Defaults to
``~/.cache/fast_hf_indexes_simple``.

**Examples**

```python
>>> loader = FastDatasetLoaderSimple()
>>> ds = loader.load_iterable("allenai/c4", name="en", split="train")
>>> ds = ds.shuffle(seed=42).shard(num_shards=4, index=0)
>>> for example in ds:
...     pass
```

**Attributes**

- `index_dir`

**Methods**

#### `load_iterable` {#forgather-ml-datasets-fast_hf_loader-fastdatasetloadersimple-load_iterable}

```python
def load_iterable(path: str, name: Optional[str] = None, split: Optional[str] = None, data_files: Optional[Union[str, list]] = None, revision: Optional[str] = None, force_reindex: bool = False, num_proc: Optional[int] = None, length_estimate: str = 'dynamic', reset_length_on_iter: bool = False, **load_dataset_kwargs)
```

Load a dataset as a `ComposableIterableDataset` over an
`ArrowBackend`.

**Parameters**

- `path` (str) — HuggingFace Hub identifier or a local saved-dataset path.
- `name` (str) — Dataset configuration name.
- `split` (str) — Split, with optional slice notation (e.g. ``"train[10000:]"``).
- `data_files` (optional) — Forwarded to ``datasets.load_dataset`` on the slow path.
- `revision` (optional) — Forwarded to ``datasets.load_dataset`` on the slow path.
- `num_proc` (optional) — Forwarded to ``datasets.load_dataset`` on the slow path.
- `force_reindex` (bool) — Rebuild the Arrow file index even when a valid cached index
already exists.
- `length_estimate` ((dynamic, static, exact)) — Length-estimation mode for the wrapper. Default ``"dynamic"``.
- `reset_length_on_iter` (bool) — Reset wrapper length-estimation counters at the start of each
new iteration. Default ``False``.
- `**load_dataset_kwargs` — Forwarded to ``datasets.load_dataset`` on the slow path.

**Returns**

- `ComposableIterableDataset` — Wrapper around an `ArrowBackend` ready for shuffling,
sharding, mapping, and checkpointing.

---

### `fast_load_iterable_dataset` {#forgather-ml-datasets-fast_hf_loader-fast_load_iterable_dataset}

```python
def fast_load_iterable_dataset(path: str, name: Optional[str] = None, split: Optional[str] = None, data_files: Optional[Union[str, list]] = None, revision: Optional[str] = None, force_reindex: bool = False, num_proc: Optional[int] = None, index_dir: Optional[str] = None, length_estimate: str = 'dynamic', reset_length_on_iter: bool = False, **load_dataset_kwargs)
```

Load a HuggingFace dataset as a fast iterable with sharding and
checkpoint support.

> **Routing**
>
> - If the ``FORGATHER_DATASET_SERVER`` environment variable is set
>   to a URL (e.g. ``http://host:8765``), the load is routed
>   transparently through the dataset server and a
>   `RemoteBackend`-wrapped dataset is returned. The server must
>   have been started with ``--allow-load``. Server-only options
>   (``force_reindex``, ``num_proc``, ``index_dir``,
>   ``**load_dataset_kwargs``) are not forwarded over the wire and
>   take effect only on the local path.
> - Otherwise, loads locally via `FastDatasetLoaderSimple`. The
>   first call for a given dataset is slow (it builds an Arrow
>   file index); all subsequent calls are instant.

**Parameters**

- `path` (str) — HuggingFace Hub identifier (e.g. ``"allenai/c4"``) **or** a local
path to a dataset saved with ``Dataset.save_to_disk()``.
- `name` (str) — Dataset configuration name (e.g. ``"en"`` for C4 English).
- `split` (str) — Split to load. Supports HuggingFace slice notation such as
``"train[10000:]"`` or ``"validation[:500]"``.
- `data_files` (str or list of str) — Specific data files to load (forwarded to ``load_dataset``).
- `revision` (str) — Dataset revision or commit hash (forwarded to ``load_dataset``).
- `force_reindex` (bool) — Rebuild the Arrow file index from scratch (local path only).
- `num_proc` (int) — Number of processes for the initial dataset download/indexing
step (local path only).
- `index_dir` (str) — Directory where JSON index files are stored (local path only).
- `length_estimate` ((dynamic, static, exact)) — Length-estimation mode for the wrapper.
- `reset_length_on_iter` (bool) — Whether to reset length-estimation counters at the start of each
new iteration pass.
- `**load_dataset_kwargs` — Extra keyword arguments forwarded to ``datasets.load_dataset``
on the initial (slow-path) local load. Not forwarded to the
remote server.

**Returns**

- `ComposableIterableDataset` — Iterable dataset (wrapper over `ArrowBackend` locally or
`RemoteBackend` when routed through the server) supporting:

- `.shuffle(seed)` for backend-level + buffer-level shuffling
- `.shard(num_shards, index)` for DDP data partitioning
- `.map(fn)` for lazy transformations
- `.slice()` / `.select()` for virtual splits
- `state_dict` / `load_state_dict` for stateful checkpointing

**Examples**

```python
>>> ds = fast_load_iterable_dataset("allenai/c4", name="en", split="train")
>>> ds = ds.shuffle(seed=42)
>>> ds = ds.shard(num_shards=world_size, index=rank)
>>> ds = ds.map(tokenize)
>>> for example in ds:
...     pass
```

## Backend abstraction

The loader returns a `ComposableIterableDataset` wrapped around an
`ArrowBackend`. The same wrapper can sit on top of an
`InMemoryBackend` or a `RemoteBackend` (network proxy to a
[Dataset Server](../tools/dataset_server/README.md)) without
client code changes.

### `ComposableIterableDataset` {#forgather-ml-datasets-composable_iterable_dataset-composableiterabledataset}

`forgather.ml.datasets.composable_iterable_dataset.ComposableIterableDataset`

```python
class ComposableIterableDataset(backend: IterableDatasetBackend, length_estimate: str = 'dynamic', reset_length_on_iter: bool = False)
```

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

**Parameters**

- `backend` (IterableDatasetBackend) — Underlying storage backend.
- `length_estimate` (('dynamic', 'static', 'exact')) — Initial length-estimation mode. Default ``"dynamic"``.
- `reset_length_on_iter` (bool) — If ``True``, reset input/output counters at the start of every
new iteration. Default ``False`` (counters accumulate across
passes).

**Attributes**

- `length_estimate_mode` (str)
- `backend` (IterableDatasetBackend)
- `column_names` (Optional[List[str]])
- `features`
- `n_shards` (int)

**Methods**

#### `set_length_estimate_mode` {#forgather-ml-datasets-composable_iterable_dataset-composableiterabledataset-set_length_estimate_mode}

```python
def set_length_estimate_mode(mode: str)
```

_No documentation._

#### `get_length_stats` {#forgather-ml-datasets-composable_iterable_dataset-composableiterabledataset-get_length_stats}

```python
def get_length_stats()
```

_No documentation._

#### `shuffle` {#forgather-ml-datasets-composable_iterable_dataset-composableiterabledataset-shuffle}

```python
def shuffle(seed: Optional[int] = None, buffer_size: Optional[int] = 1000)
```

Re-permute the underlying example order via the backend and
configure an example-level reservoir shuffle buffer.

Length-estimation cache is invalidated; existing input/output
counts are preserved as a ratio carry-over.

#### `set_epoch` {#forgather-ml-datasets-composable_iterable_dataset-composableiterabledataset-set_epoch}

```python
def set_epoch(epoch: int)
```

Set the current epoch and re-shuffle the backend if any seed
is in play. Mutates in place.

``set_epoch(0)`` always restores the wrapper's natural
backend state (the post-construction or post-``shuffle()``
baseline) — even if a previous ``set_epoch(N>0)`` left the
backend in an N-shuffled state. Without this, going back to
epoch 0 would silently reuse the stale epoch-N order.

#### `slice` {#forgather-ml-datasets-composable_iterable_dataset-composableiterabledataset-slice}

```python
def slice(start: Optional[int | float | str] = None, end: Optional[int | float | str] = None)
```

Return a view restricted to ``[start, end)``.

#### `select` {#forgather-ml-datasets-composable_iterable_dataset-composableiterabledataset-select}

```python
def select(indices)
```

Contiguous-range select; non-contiguous indices not supported.

#### `shard` {#forgather-ml-datasets-composable_iterable_dataset-composableiterabledataset-shard}

```python
def shard(num_shards: int, index: int)
```

Split into ``num_shards`` disjoint slices and return the one
at ``index``. Logical sharding only — there is no ``mode``
parameter at this layer; the backend may do whatever physical
optimization it wants internally.

#### `map` {#forgather-ml-datasets-composable_iterable_dataset-composableiterabledataset-map}

```python
def map(function: Optional[Callable] = None, with_indices: bool = False, input_columns: Optional[str | List[str]] = None, batched: bool = False, batch_size: Optional[int] = 1000, drop_last_batch: bool = False, remove_columns: Optional[str | List[str]] = None, fn_kwargs: Optional[Dict[str, Any]] = None)
```

Append a map step to the chain. Multiple map calls compose.

A non-batched function returning ``None`` filters the example
out (matches the legacy Arrow class behavior).

Mixed batched / non-batched chains are not supported (raises).

#### `filter` {#forgather-ml-datasets-composable_iterable_dataset-composableiterabledataset-filter}

```python
def filter(function: Callable, with_indices: bool = False, input_columns: Optional[str | List[str]] = None, fn_kwargs: Optional[Dict[str, Any]] = None)
```

Keep examples where ``function(example)`` returns truthy.

#### `to_hf_iterable` {#forgather-ml-datasets-composable_iterable_dataset-composableiterabledataset-to_hf_iterable}

```python
def to_hf_iterable()
```

Wrap this dataset in a HuggingFace ``IterableDataset`` for APIs
that require one. The returned object exposes ``__len__`` via
`IterableDatasetWithLength` so it can drive ``torch.DataLoader``;
the wrapper checkpoint protocol is *not* preserved on the
returned value.

#### `state_dict` {#forgather-ml-datasets-composable_iterable_dataset-composableiterabledataset-state_dict}

```python
def state_dict()
```

Capture wrapper state plus the backend's flat position.

The backend's `position()` is in underlying-example space, not
in user-facing post-slice/shard/map space — that's deliberate
so resume can call `backend.seek(saved_position)` and continue
consuming examples regardless of how a map function may have
changed cardinality.

#### `load_state_dict` {#forgather-ml-datasets-composable_iterable_dataset-composableiterabledataset-load_state_dict}

```python
def load_state_dict(state: Dict[str, Any])
```

Restore wrapper state and seek the backend to the saved
position. Map functions themselves are not serialised — the
caller must reconstruct the same map chain before calling
`load_state_dict` (a fingerprint is checked).

---

### `IterableDatasetBackend` {#forgather-ml-datasets-iterable_backend-iterabledatasetbackend}

`forgather.ml.datasets.iterable_backend.IterableDatasetBackend`

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

**Methods**

#### `shuffle` {#forgather-ml-datasets-iterable_backend-iterabledatasetbackend-shuffle}

```python
def shuffle(seed: Optional[int] = None)
```

Return a new backend with the underlying example order
re-permuted.

No buffer parameter — the example-level reservoir buffer lives
in the composing wrapper, not in the backend. The seed
determines the new order deterministically; if ``None`` an
implementation-chosen seed is generated and surfaced via the
new instance's state so it can be reproduced from a checkpoint.

The returned instance has `position()` reset to 0.

#### `seek` {#forgather-ml-datasets-iterable_backend-iterabledatasetbackend-seek}

```python
def seek(position: int)
```

Return a new backend whose next `__iter__` begins at the given
flat example index.

Not expected to be O(1) — implementations may need to walk
index metadata to translate the flat position into their
internal representation. The returned instance has `position()`
equal to the requested value.

#### `position` {#forgather-ml-datasets-iterable_backend-iterabledatasetbackend-position}

```python
def position()
```

Current flat example index where the next `__iter__` would
start.

Must update during iteration so a wrapper can capture it for
`state_dict()` after any number of yielded examples.

---

### `ArrowBackend` {#forgather-ml-datasets-arrow_backend-arrowbackend}

`forgather.ml.datasets.arrow_backend.ArrowBackend`

```python
class ArrowBackend(arrow_files: List[str], file_lengths: Optional[List[int]] = None)
```

Storage backend over a list of memory-mapped Arrow files.

**Parameters**

- `arrow_files` (list of str) — Ordered list of Arrow file paths that make up the dataset.
Each file is treated as one natural shard.
- `file_lengths` (list of int) — Per-file example counts, parallel to ``arrow_files``. When
provided, ``__len__`` and ``seek`` are O(num_files) without
any file I/O. When ``None``, file lengths are read on
construction by opening each file (slow path; the loader
normally avoids this by passing cached lengths from the
on-disk index).

> **Note**
>
> `__iter__` mutates the cursor; multiple concurrent iterators on
> the same instance would interfere. In multi-worker DataLoader
> setups each worker receives its own copy (via fork or pickle), so
> concurrent cursors aren't an issue in practice.

**Attributes**

- `arrow_files` (List[str])
- `file_lengths`
- `column_names` (List[str])
- `features`
- `n_shards` (int)

**Methods**

#### `shuffle` {#forgather-ml-datasets-arrow_backend-arrowbackend-shuffle}

```python
def shuffle(seed: Optional[int] = None)
```

Return a new backend with files re-permuted under ``seed``.
Cursor is reset to 0. No example-level buffer — that lives in
the wrapper.

#### `seek` {#forgather-ml-datasets-arrow_backend-arrowbackend-seek}

```python
def seek(position: int)
```

Return a new backend with the cursor at ``position``. Past-the-end
positions are clamped to the end (next iteration yields nothing).

#### `position` {#forgather-ml-datasets-arrow_backend-arrowbackend-position}

```python
def position()
```

_No documentation._

#### `state_dict` {#forgather-ml-datasets-arrow_backend-arrowbackend-state_dict}

```python
def state_dict()
```

Capture cursor + order seed + dataset-identity fingerprint.

The wrapper picks this up via the optional-backend-state_dict
path so a checkpoint round-trip can detect "different files
behind the same handle" early.

#### `load_state_dict` {#forgather-ml-datasets-arrow_backend-arrowbackend-load_state_dict}

```python
def load_state_dict(state: Dict[str, Any])
```

_No documentation._

---

### `RemoteBackend` {#forgather-ml-datasets-remote_backend-remotebackend}

`forgather.ml.datasets.remote_backend.RemoteBackend`

```python
class RemoteBackend(url: str, handle: str, seed: Optional[int] = None, position: int = 0, timeout: float = 60.0, token: Optional[str] = None, column_names: Optional[list[str]] = None)
```

Network-proxy backend.

**Parameters**

- `url` (str) — Base URL of the dataset server, e.g. ``"http://host:8766"``.
- `handle` (str) — Server-side identifier for the registered backend to consume.
- `seed` (int) — Shuffle seed; ``None`` means no shuffle requested.
- `position` (int) — Initial flat example index. Default ``0``.
- `timeout` (float) — Per-request HTTP timeout (seconds). Default ``60``.
- `token` (str) — Explicit bearer token. If omitted, the constructor consults
``$FORGATHER_DATASET_SERVER_TOKEN`` and (for localhost URLs)
``<forgather_config_dir>/dataset_server/<port>.token``.

**Attributes**

- `n_shards` (int)
- `column_names` (Optional[list[str]]) — Column names of the underlying dataset.

**Methods**

#### `shuffle` {#forgather-ml-datasets-remote_backend-remotebackend-shuffle}

```python
def shuffle(seed: Optional[int] = None)
```

Return a new client with the new seed; position resets to 0.

No RPC is issued — the seed travels with the next ``/iter``
request. The cached length is preserved (shuffling doesn't
change the underlying example count).

#### `seek` {#forgather-ml-datasets-remote_backend-remotebackend-seek}

```python
def seek(position: int)
```

Return a new client positioned at the given flat example index.

No RPC is issued — the position travels with the next
``/iter`` request.

#### `position` {#forgather-ml-datasets-remote_backend-remotebackend-position}

```python
def position()
```

_No documentation._

## Interleaved Datasets

### `InterleavedDataset` {#forgather-ml-datasets-interleaved-interleaveddataset}

`forgather.ml.datasets.interleaved.InterleavedDataset`

```python
class InterleavedDataset(datasets: List, probabilities: Optional[Union[List[float], Callable]] = None, seed: Optional[int] = None, stopping_strategy: str = 'first_exhausted')
```

An iterable dataset that interleaves examples from multiple child datasets.

Works with any iterable dataset that supports the stateful checkpoint
protocol (``state_dict`` / ``load_state_dict``), including
`ComposableIterableDataset`. Designed for multi-dataset pre-training where
examples from several corpora need to be mixed in a single training loop.

**Parameters**

- `datasets` (list) — Child datasets to interleave. Must be non-empty. Each element can be
any iterable; checkpointing is available for elements that implement
``state_dict`` / ``load_state_dict``.
- `probabilities` (list of float or callable) — Controls which child dataset is sampled at each step:

- ``None`` (default) — round-robin: datasets are visited in order,
  cycling back to the first after the last.
- ``list of float`` — static per-dataset weights. Values are
  normalised automatically; all must be non-negative and their sum
  must be positive.
- ``callable`` — dynamic weight function called at each step with
  signature ``(step, datasets, examples_per_dataset, exhausted)
  -> list of float``. See `balance_remaining_examples` for an
  example implementation.
- `seed` (int) — Random seed for reproducible probabilistic sampling. Ignored when
``probabilities`` is ``None`` (round-robin).
- `stopping_strategy` ((first_exhausted, all_exhausted)) — When to stop iteration:

- ``"first_exhausted"`` (default) — stop as soon as any child dataset
  is exhausted.
- ``"all_exhausted"`` — continue until every child dataset is
  exhausted, oversampling shorter datasets.

**Raises**

- `ValueError` — If ``datasets`` is empty, probabilities fail validation, or an
unsupported ``stopping_strategy`` is given.

**Examples**

```python
>>> ds1 = fast_load_iterable_dataset("corpus_a", split="train")
>>> ds2 = fast_load_iterable_dataset("corpus_b", split="train")
>>> combined = InterleavedDataset([ds1, ds2], probabilities=[0.7, 0.3], seed=42)
>>> for example in combined:
...     pass
```

**Attributes**

- `datasets`
- `seed`
- `stopping_strategy`
- `probabilities`
- `column_names` (List[str]) — Get column names from first dataset.
- `features` — Get features from first dataset.
- `n_shards` (int) — Total number of shards across all datasets.

**Methods**

#### `state_dict` {#forgather-ml-datasets-interleaved-interleaveddataset-state_dict}

```python
def state_dict()
```

Serialize the interleaving position and all child dataset states.

**Returns**

- `dict` — Dictionary with the following keys:

``current_dataset_index``
    Index of the child dataset that was most recently sampled.
``current_example_count``
    Total examples yielded so far across all children.
``datasets_exhausted``
    Boolean list indicating which children are exhausted.
``probabilities``
    Normalised static probabilities (``None`` if round-robin or
    dynamic).
``seed``
    Random seed.
``stopping_strategy``
    Configured stopping strategy string.
``child_states``
    List of per-child state dicts (``None`` for children that do
    not implement ``state_dict``).
``examples_per_dataset``
    Per-child example counts at the time of the last yield
    (present only when available; required for dynamic probability
    functions).

#### `load_state_dict` {#forgather-ml-datasets-interleaved-interleaveddataset-load_state_dict}

```python
def load_state_dict(state_dict: Dict[str, Any])
```

Restore the interleaving position and all child dataset states.

After calling this method, the next iteration resumes from the saved
position. Child datasets that implement ``load_state_dict`` are
restored individually; others are left at their natural start position.

**Parameters**

- `state_dict` (dict) — Dictionary previously returned by `state_dict`.

## Utilities

### `IterableDatasetWithLength` {#forgather-ml-datasets-iterable_with_length-iterabledatasetwithlength}

`forgather.ml.datasets.iterable_with_length.IterableDatasetWithLength`

```python
class IterableDatasetWithLength(iterable_dataset, length: int)
```

A thin wrapper that adds a known length to an iterable dataset.

PyTorch's ``IterableDataset`` does not require ``__len__``, but trainers
and data-loader utilities often query it to calculate epoch step counts.
When a map-style ``Dataset`` is converted to an iterable form with
``to_iterable_dataset()``, the length information is lost. This wrapper
re-attaches it.

All attribute and method accesses that are not handled by this class are
forwarded transparently to the wrapped dataset via ``__getattr__``,
including ``state_dict`` / ``load_state_dict`` for checkpointing, and
HuggingFace Dataset attributes such as ``column_names`` and ``features``.

**Parameters**

- `iterable_dataset` (IterableDataset) — The dataset to wrap. Any iterable dataset is accepted.
- `length` (int) — The length to report from ``__len__``. This value is not validated
against the actual number of examples; the caller is responsible for
supplying a consistent value.

> **Note**
>
> The `map` and `shuffle` methods are overridden to return a new
> ``IterableDatasetWithLength`` with the same reported length, so that
> the length is preserved through chained transformations.

> `filter` is *not* overridden: the filtered dataset is returned as-is
> because the new length cannot be determined without iterating.

**Examples**

```python
>>> from torch.utils.data import IterableDataset
>>> ds = some_map_style_dataset.to_iterable_dataset()
>>> ds_with_len = IterableDatasetWithLength(ds, length=len(some_map_style_dataset))
>>> len(ds_with_len)
1000
```

**Methods**

#### `map` {#forgather-ml-datasets-iterable_with_length-iterabledatasetwithlength-map}

```python
def map(*args, **kwargs)
```

Apply a map transformation while preserving the reported length.

Delegates to the wrapped dataset's ``map`` method and re-wraps the
result in a new ``IterableDatasetWithLength`` with the same length.

**Parameters**

- `*args` — Positional arguments forwarded to the wrapped dataset's ``map``.
- `**kwargs` — Keyword arguments forwarded to the wrapped dataset's ``map``.

**Returns**

- `IterableDatasetWithLength` — Mapped dataset with the same reported length as this instance.

#### `shuffle` {#forgather-ml-datasets-iterable_with_length-iterabledatasetwithlength-shuffle}

```python
def shuffle(*args, **kwargs)
```

Shuffle the dataset while preserving the reported length.

Delegates to the wrapped dataset's ``shuffle`` method and re-wraps the
result in a new ``IterableDatasetWithLength`` with the same length.

**Parameters**

- `*args` — Positional arguments forwarded to the wrapped dataset's ``shuffle``.
- `**kwargs` — Keyword arguments forwarded to the wrapped dataset's ``shuffle``.

**Returns**

- `IterableDatasetWithLength` — Shuffled dataset with the same reported length as this instance.

#### `filter` {#forgather-ml-datasets-iterable_with_length-iterabledatasetwithlength-filter}

```python
def filter(*args, **kwargs)
```

Filter the dataset.

Delegates to the wrapped dataset's ``filter`` method. The length
information is *not* preserved because the post-filter count cannot
be determined without iterating.

**Parameters**

- `*args` — Positional arguments forwarded to the wrapped dataset's ``filter``.
- `**kwargs` — Keyword arguments forwarded to the wrapped dataset's ``filter``.

**Returns**

- `IterableDataset` — Filtered dataset without a ``__len__`` method.
