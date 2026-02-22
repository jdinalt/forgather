# Dataset Projects

This guide covers how to create, configure, and use dataset projects in Forgather. Dataset projects are standalone Forgather projects that encapsulate dataset loading, splitting, preprocessing, and tokenization. They can be used directly from Python or referenced from training configurations via `from_project`.

## Overview

A dataset project is a standard Forgather project whose configuration class is `type.dataset`. It typically exposes these materialization targets:

| Target | Description |
|--------|-------------|
| `train_dataset_split` | Raw training split (before tokenization) |
| `validation_dataset_split` | Raw validation split |
| `test_dataset_split` | Raw test split |
| `train_dataset` | Preprocessed/tokenized training dataset |
| `eval_dataset` | Preprocessed/tokenized evaluation dataset |
| `test_dataset` | Preprocessed/tokenized test dataset |
| `meta` | Configuration metadata (config name, class, main feature) |

The `train_dataset_split` targets return untokenized data, while `train_dataset` and friends return data that has been passed through `preprocess_dataset`, which handles tokenization, sharding, shuffling, and range selection.

## Project Structure

A dataset project follows the standard Forgather project layout:

```
examples/datasets/roneneldan/
    meta.yaml                              # Project metadata
    templatelib/
        configs/
            tinystories.yaml               # Full TinyStories dataset
            tinystories-abridged.yaml      # 10% subset
            tinystories-packed.yaml        # Packed sequences variant
```

The `meta.yaml` declares the project name and default configuration:

```yaml
-- extends "meta_defaults.yaml"

-- block configs
name: "roneneldan"
description: "Datasets published by Ronen Eldan"
config_prefix: "configs"
default_config: "tinystories-abridged.yaml"
<< endblock configs
```

## Base Templates

Forgather provides base templates in `templatelib/base/datasets/` that dataset configurations extend:

### `datasets/load_dataset.yaml`

The most commonly used base template. It provides a complete pipeline for loading a HuggingFace dataset and preprocessing it. Key blocks to override:

- `[load_dataset_args]` -- the `path`, `name`, `revision`, and split definitions
- `[train_dataset_split]`, `[validation_dataset_split]`, `[test_dataset_split]` -- split selection
- `[map_function]` -- custom tokenization/mapping function
- `[map_kwargs]` -- additional arguments for the map call

### `datasets/tokenized_dataset.yaml`

A more flexible base template with explicit blocks for every stage of the pipeline. Use this when you need full control over dataset loading (e.g., loading from disk instead of HuggingFace Hub).

### `datasets/dataset_type.yaml`

The root base template for all dataset projects. Defines the `type.dataset` config class and the `main` output target.

## Using Dataset Projects from Python

### Basic Usage

```python
from forgather import Project

# Load a dataset project
dataset_project = Project(
    "tinystories-abridged.yaml",
    "examples/datasets/roneneldan",
)

# Materialize the raw training split (no tokenization)
raw_train = dataset_project("train_dataset_split")

# Inspect examples
for example in raw_train:
    print(example["text"][:200])
    break
```

### With Tokenization

To get tokenized datasets, pass a tokenizer and preprocessing args:

```python
from transformers import AutoTokenizer
from forgather import Project

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")

# Load dataset project
dataset_project = Project(
    "tinystories-abridged.yaml",
    "examples/datasets/roneneldan",
)

# Materialize tokenized train and eval datasets
train_dataset, eval_dataset = dataset_project(
    "train_dataset",
    "eval_dataset",
    tokenizer=tokenizer,
    preprocess_args=dict(truncation=True, max_length=512),
)

# Each example now has 'input_ids'
for example in train_dataset:
    print(f"Token count: {len(example['input_ids'])}")
    break
```

### With Dataset Sharding (Distributed Training)

For distributed training, pass `shard_dataset` to split data across ranks:

```python
train_dataset, eval_dataset = dataset_project(
    "train_dataset",
    "eval_dataset",
    tokenizer=tokenizer,
    preprocess_args=dict(truncation=True, max_length=4096),
    shard_dataset=True,  # Auto-shards by WORLD_SIZE and RANK
)
```

See the section on `shard_dataset` below for the full details.

## Using Dataset Projects in Training Configurations

Training configurations reference dataset projects using `from_project`, which loads a sub-project and materializes its targets:

```yaml
[datasets_definition]
.define: &dataset_dict !call:forgather:from_project
    project_dir: "{{ ns.dataset_proj }}"
    config_template: "{{ ns.dataset_config }}"
    targets: [ "train_dataset", "eval_dataset" ]
    preprocess_args: *tokenizer_args
    tokenizer: *tokenizer

train_dataset: &train_dataset !call:getitem [ *dataset_dict, 'train_dataset' ]
eval_dataset: &eval_dataset !call:getitem [ *dataset_dict, 'eval_dataset' ]
```

### Passing Preprocessor Arguments via `pp_kwargs`

The `pp_kwargs` parameter passes arguments to the sub-project's Jinja2 preprocessor (template variables), while other keyword arguments are passed at materialization time (runtime variables accessed via `!var`):

```yaml
.define: &dataset_dict !call:forgather:from_project
    project_dir: "{{ ns.dataset_proj }}"
    config_template: "{{ ns.dataset_config }}"
    targets: [ "train_dataset", "eval_dataset" ]
    # pp_kwargs are Jinja2 template variables (resolved at config preprocessing time)
    pp_kwargs: *dataset_project_pp_args
    # These are runtime variables (resolved when the config graph is materialized)
    preprocess_args: *tokenizer_args
    tokenizer: *tokenizer
    shard_dataset: True
```

### Dataset Sharding for Distributed Training

When using multi-GPU training without batch dispatching (i.e., `dispatch_batches: False`), each rank needs its own shard of the dataset. Pass `shard_dataset: True` to enable automatic sharding:

```yaml
.define: &dataset_dict !call:forgather:from_project
    project_dir: "{{ ns.dataset_proj }}"
    config_template: "{{ ns.dataset_config }}"
    targets: [ "train_dataset", "eval_dataset" ]
    preprocess_args: *tokenizer_args
    tokenizer: *tokenizer
    shard_dataset: {{ ns.dispatch_batches == False }}
```

## The `shard_dataset` Parameter

The `shard_dataset` parameter on `preprocess_dataset` controls how datasets are split for distributed training. It accepts two forms.

### Boolean Form

```python
train_dataset = dataset_project(
    "train_dataset",
    tokenizer=tokenizer,
    shard_dataset=True,
)
```

When `True`, the dataset is automatically sharded using the current distributed environment:
- `num_shards` defaults to `WORLD_SIZE` (total number of processes)
- `index` defaults to `RANK` (current process rank)

When `False` or `None`, no sharding is performed.

### Dictionary Form

```python
train_dataset = dataset_project(
    "train_dataset",
    tokenizer=tokenizer,
    shard_dataset=dict(
        num_shards=4,
        index=1,  # This process gets shard 1 of 4
    ),
)
```

This gives explicit control over the number of shards and which shard the current process receives. This is useful for:
- Testing a specific shard locally
- Custom parallelism strategies where the shard count differs from world size
- DiLoCo distributed training where workers are independent processes (not torch distributed ranks)

### How Sharding Works Internally

The implementation in `preprocess_dataset` (`src/forgather/ml/datasets/preprocess.py`):

1. If `shard_dataset` is `True`, it resolves to `{"num_shards": WORLD_SIZE, "index": RANK}`.
2. If `shard_dataset` is `False`, it becomes `None` (no sharding).
3. For HuggingFace `Dataset` or `IterableDataset`, it uses `split_dataset_by_node` from the `datasets` library.
4. For other dataset types (e.g., `SimpleArrowIterableDataset`), it calls the `.shard()` method.
5. When sharding is enabled, `main_process_first()` is **not** used, because each rank processes its own independent shard. When sharding is disabled, `main_process_first()` ensures rank 0 preprocesses and caches the dataset before other ranks load the cache.

### In YAML Configurations

The `load_dataset.yaml` base template exposes `shard_dataset` as a runtime variable with a default of `False`:

```yaml
[dataset_args]
.define: &dataset_args
    shard_dataset: !var [ "shard_dataset", False ]
```

This means the calling configuration (or Python code) controls whether sharding is active.

### CLI Usage

The `forgather dataset` command also supports sharding for testing:

```bash
# Test shard 0 of 4
forgather -p examples/datasets/roneneldan dataset --num-shards 4 --shard-index 0 -n 5

# Test shard 3 of 4
forgather -p examples/datasets/roneneldan dataset --num-shards 4 --shard-index 3 -n 5
```

## The `select_range` Parameter

The `preprocess_dataset` function accepts a `select_range` parameter that allows subsetting the dataset before processing. It supports several formats:

| Format | Example | Description |
|--------|---------|-------------|
| `int` | `500` | First 500 records |
| `float` | `0.25` | First 25% of records |
| `str` (slice) | `"100:500"` | Records 100 through 499 |
| `str` (percent) | `"10%:80%"` | Records from 10% to 80% |
| `str` (open) | `"100:"` | Records from 100 to end |
| `Sequence` | `[100, 900]` | Records 100 through 899 |
| `range` | `range(10, 100)` | Direct range object |

## CLI: Testing and Inspecting Datasets

The `forgather dataset` command provides tools for inspecting dataset projects without writing Python code.

### Viewing Raw Examples

```bash
# Show 5 raw (untokenized) examples from the default target (train_dataset_split)
forgather -p examples/datasets/roneneldan dataset -n 5

# Show examples from a specific config
forgather -p examples/datasets/roneneldan -t tinystories.yaml dataset -n 3
```

### Viewing Tokenized Examples

```bash
# Tokenize and show examples (requires a tokenizer)
forgather -p examples/datasets/roneneldan dataset \
    -T path/to/tokenizer \
    --target train_dataset \
    -s -n 5
```

### Generating Token Length Histograms

```bash
forgather -p examples/datasets/roneneldan dataset \
    -T path/to/tokenizer \
    -H --histogram-samples 2000
```

## Example Dataset Projects

Forgather ships with several example dataset projects under `examples/datasets/`:

| Project | Path | Description |
|---------|------|-------------|
| TinyStories | `examples/datasets/roneneldan/` | Small synthetic stories dataset, good for quick experiments |
| SmolLM Corpus | `examples/datasets/HuggingFaceTB/` | FineWeb-Edu, Cosmopedia-v2, and interleaved variants |
| Wikitext | `examples/datasets/EleutherAI/` | Document-level Wikipedia text |
| Local Dataset | `examples/datasets/local_dataset/` | Template for loading datasets from disk |

## Key Relationships

- **Dataset projects** are regular Forgather projects with `config_class: type.dataset`.
- **Training projects** reference dataset projects using `!call:forgather:from_project`.
- **`from_project`** creates a `Project` instance for the dataset sub-project and calls it with the given targets and keyword arguments. The result is a dictionary keyed by target name.
- **Runtime variables** in dataset configs (e.g., `!var "tokenizer"`, `!var "shard_dataset"`) are resolved from the kwargs passed by the calling project or Python code.
- **`preprocess_dataset`** is the central preprocessing function that handles tokenization, sharding, shuffling, and range selection.
