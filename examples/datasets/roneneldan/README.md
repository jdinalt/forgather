# TinyStories

https://huggingface.co/datasets/roneneldan/TinyStories

> A dataset of 2.1M synthetically generated short stories created by GPT-3.5 and GPT-4 that use only a small vocabulary, designed for training small language models that can speak coherent English. Described in the paper "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" (https://arxiv.org/abs/2305.07759).

## Configurations

- [tinystories.yaml](./templatelib/configs/tinystories.yaml) Tiny Stories, full dataset
- [tinystories-abridged.yaml](./templatelib/configs/tinystories-abridged.yaml) First 15% of `train` only -- quick iteration
- [tinystories-iter.yaml](./templatelib/configs/tinystories-iter.yaml) Tiny Stories as an `IterableDataset`
- [tinystories-packed.yaml](./templatelib/configs/tinystories-packed.yaml) Tiny Stories with sequence packing
- [fast-iter.yaml](./templatelib/configs/fast-iter.yaml) Tiny Stories via `fast_load_iterable_dataset` (instant loads, position-based checkpoint resumption)
- [fast-iter-packed.yaml](./templatelib/configs/fast-iter-packed.yaml) Fast iterable variant with packing
