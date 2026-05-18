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

::: forgather.ml.datasets.fast_hf_loader.FastDatasetLoaderSimple

---

::: forgather.ml.datasets.fast_hf_loader.fast_load_iterable_dataset

## Backend abstraction

The loader returns a `ComposableIterableDataset` wrapped around an
`ArrowBackend`. The same wrapper can sit on top of an
`InMemoryBackend` or a `RemoteBackend` (network proxy to a
[Dataset Server](../tools/dataset_server/README.md)) without
client code changes.

::: forgather.ml.datasets.composable_iterable_dataset.ComposableIterableDataset

---

::: forgather.ml.datasets.iterable_backend.IterableDatasetBackend

---

::: forgather.ml.datasets.arrow_backend.ArrowBackend

---

::: forgather.ml.datasets.remote_backend.RemoteBackend

## Interleaved Datasets

::: forgather.ml.datasets.interleaved.InterleavedDataset

## Utilities

::: forgather.ml.datasets.iterable_with_length.IterableDatasetWithLength
