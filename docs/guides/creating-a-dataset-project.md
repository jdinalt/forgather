# Creating a Dataset Project

This guide walks through creating dataset configurations for use with Forgather
training projects. A dataset project defines how to load, tokenize, and optionally
pack or interleave datasets from HuggingFace Hub.

For existing dataset projects to use as reference, see
[`examples/datasets/`](../examples/datasets/README.md). The
[HuggingFaceTB](../examples/datasets/HuggingFaceTB/README.md) project is particularly
comprehensive, demonstrating packed and interleaved configurations.

For detailed reference on individual features, see:
- [Dataset CLI Reference](../datasets/dataset-cli.md) -- dataset CLI commands, target inspection, histogram generation, and testing workflows
- [Fast HF Loader](../datasets/fast-hf-loader.md) -- the optimized dataset loader
- [Sequence Packing](../datasets/sequence-packing.md) -- packing strategies and parameters
- [Dataset Projects](../datasets/dataset-projects.md) -- API reference

## Creating a project with the CLI

If you don't already have a workspace, create one:

```bash
forgather ws create \
    --name "My Datasets" \
    --description "Dataset configurations" \
    --forgather-dir /path/to/forgather \
    -l base -l examples \
    my_datasets
```

Then create the dataset project inside the workspace:

```bash
cd my_datasets
forgather project create \
    --name "My Dataset Collection" \
    --description "HuggingFace dataset configurations"
```

This scaffolds the project directory with `meta.yaml` and a default config.

## Project structure

A dataset project lives inside a Forgather workspace and has this layout:

```
my_datasets/
├── meta.yaml
└── templatelib/
    ├── packed.yaml              # Optional: shared packing macros
    └── configs/
        ├── dataset_a.yaml       # Individual dataset configs
        ├── dataset_a-packed.yaml
        ├── dataset_b.yaml
        ├── dataset_b-packed.yaml
        └── interleaved.yaml     # Combines multiple datasets
```

Note that the `forgather project create` command creates a `templates/` directory
by default. Dataset projects in `examples/datasets/` use `templatelib/` instead
-- either convention works as long as `meta.yaml` is consistent with the
directory name (via `config_prefix`).

## Step 1: Create a simple dataset config

The simplest dataset config extends `datasets/load_dataset.yaml` and specifies
a HuggingFace Hub dataset path and splits.

Create `templatelib/configs/my-dataset.yaml`:

```yaml
-- extends "datasets/load_dataset.yaml"

[config_metadata]
    == super()
    -- set ns.config_name = "My Dataset"
    -- set ns.config_description = "Description of my dataset"
    -- set ns.source = "https://huggingface.co/datasets/author/dataset-name"
    -- set ns.load_method = "forgather.ml.datasets:fast_load_iterable_dataset"

[map_kwargs]
    == super()
    batch_size: 32

[load_dataset_args]
    == super()
    path: "author/dataset-name"

[train_dataset_split]
    == super()
    split: "train[10000:]"

[validation_dataset_split]
    == super()
    split: "train[0:1000]"

[test_dataset_split]
    == super()
    split: "train[1000:10000]"
```

Key points:

- `ns.load_method` controls which loader function is used. Setting it to
  `forgather.ml.datasets:fast_load_iterable_dataset` uses Forgather's optimized
  loader, which memory-maps Arrow files from the HuggingFace cache for fast
  subsequent loads. Omitting it (or clearing the value) falls back to the
  standard `datasets.load_dataset`. This makes it easy to switch between the
  two without changing anything else in the config.
- **`ns.main_feature`** identifies which field in the dataset contains the text
  to tokenize. Most datasets use `"text"` (the default), but some use other
  names like `"code"` or `"page"`. Check the dataset's HuggingFace page or
  inspect it directly to find the feature name. If this is wrong, tokenization
  will silently produce empty results.
- **Virtual splits** (e.g., `"train[10000:]"`) let you carve train/eval/test
  splits from a single dataset split. The numbers are row indices. Only create
  virtual splits when the dataset has a single split -- if it already has
  `train`, `validation`, and `test` splits, use those directly.
- `batch_size` in `[map_kwargs]` controls the tokenization batch size (how many
  examples are tokenized at once). This is separate from the training batch size.

## Step 2: Verify

```bash
cd my_datasets/
forgather ls                              # Check all configs parse
forgather -t my-dataset.yaml pp           # Preview expanded config
forgather -t my-dataset.yaml targets      # List materializable targets
```

After verifying the config parses, test that it can actually load data. A
dataset config exposes two tiers of split targets:

- **Raw splits** — `train_dataset_split`, `validation_dataset_split`,
  `test_dataset_split` — the source data with no tokenization. These need no
  tokenizer.
- **Tokenized splits** — `train_dataset`, `eval_dataset`, `test_dataset` —
  the preprocessed/tokenized data. These require a tokenizer (`-T`).

All of these should materialize cleanly; verify each.

**Build the raw data first.** Materializing `train_dataset_split` is what
triggers the actual download and build of the underlying dataset:

```bash
# Triggers the download + build of the dataset.
forgather -t my-dataset.yaml dataset --target train_dataset_split
```

> The first build downloads from the HuggingFace Hub and writes the Arrow
> cache. For a large dataset this can take a **long time** — let it finish
> before doing anything that reads the dataset's metadata. Subsequent loads use
> the cached files and are fast.

Once it's built, load a few examples from each raw split, truncating long text
so the output stays readable:

```bash
# Print the first 3 examples, truncating each feature to 64 characters.
forgather -t my-dataset.yaml dataset --target train_dataset_split -n 3 --truncate 64
forgather -t my-dataset.yaml dataset --target validation_dataset_split -n 3 --truncate 64
forgather -t my-dataset.yaml dataset --target test_dataset_split -n 3 --truncate 64
```

Then test the tokenized splits with a tokenizer:

```bash
forgather -t my-dataset.yaml dataset --target train_dataset \
    -T ../../../tokenizers/wikitext_32k --truncate 64 -n 3
# Repeat for eval_dataset and test_dataset.
```

**Finding the splits, example counts, and feature names.** When you don't
already know a dataset's splits or which feature holds the text (needed to
write the split blocks and `main_feature`), build the raw data as above, then
inspect it through a running **dataset server** — which reports the splits,
the number of examples per split, and the feature/column names. (Some source
datasets only have a single `train` split; in that case slice it into the
others in the config, e.g. `validation = "train[0:1000]"`, as shown in Step 1.)

> **In the webui agent.** The agent doesn't run the CLI; it uses the
> equivalent tools: `run_dataset` (submits the build/inspect as a scheduler
> job — it returns immediately, so watch it with `list_jobs` /
> `read_job_output`), and `dataset_info` (splits / #examples / features from a
> dataset server; see `list_dataset_servers`). If no dataset server is
> reachable, the agent can start one with `start_service(type="dataset")`.
> `forgather ls` → `list_projects` / `list_configs`; `forgather pp` →
> `render_config_pp`; `forgather graph` (validate) → `check_config`.

For interleaved and packed configs, perform the same tests to verify the
composition works end-to-end.

## Step 3: Create a packed variant

Packed datasets concatenate short sequences into fixed-length blocks, reducing
padding waste and improving training throughput. A packed config extends the base
dataset and overrides the tokenization function with `block_tokenize_fn`.

**`templatelib/configs/my-dataset-packed.yaml`:**

```yaml
-- extends "configs/my-dataset.yaml"

[config_metadata]
    == super()
    -- set ns.config_name = "My Dataset Packed"
    -- set ns.config_description = "My Dataset with packed sequences"

[map_function]
.define: &map_function !partial:forgather.ml.datasets:block_tokenize_fn
    max_length: {{ max_length | default(512) }}
    overflow: True
    packed: True
    shuffle_output: True
    stride: {{ stride | default(0) }}
    packing_strategy: "best_fit"
    min_len: 32
    add_bos: True
    add_eos: True

[map_kwargs]
    == super()
    batch_size: 250

[dynamic_args]
    == super()
    max_length:
        names: "--max-length"
        type: "int"
        help: "Maximum length of output sequences."
    stride:
        names: "--stride"
        type: "int"
        help: "Number of tokens to repeat from previous overflowing example."
```

The packed config inherits the dataset path and splits from the base, then
overrides `[map_function]` with `block_tokenize_fn` which concatenates and
packs sequences. The `[dynamic_args]` block exposes `--max-length` and
`--stride` as CLI arguments for runtime override.

When you have multiple packed configs that share the same packing settings,
Jinja2 macros can be used to avoid duplicating the `[map_function]`,
`[map_kwargs]`, and `[dynamic_args]` blocks. See
`examples/datasets/HuggingFaceTB/templatelib/packed.yaml` for an example of
this pattern.

**Key packing parameters:**

| Parameter | Description |
|-----------|-------------|
| `max_length` | Target sequence length for packed blocks |
| `overflow` | When `True`, documents longer than `max_length` are split into multiple blocks. When `False`, they are truncated to `max_length`. Use `True` for long-form text (books, articles) where you want to train on the full content. |
| `stride` | Number of tokens of overlap between consecutive blocks from a split document. Only meaningful when `overflow: True`. For example, with `max_length: 4096` and `stride: 512`, each block shares 512 tokens of context with the previous one. This provides continuity for long texts that span many blocks. |
| `packing_strategy` | Packing algorithm: `"best_fit"` (most efficient, slowest), `"first_fit"` (good balance), or `"greedy"` (fastest, least efficient) |
| `shuffle_output: True` | Randomize block order (important with `best_fit`) |
| `min_len` | Discard sequences shorter than this |
| `add_bos` / `add_eos` | Add special tokens at sequence boundaries |
| `batch_size: 250` | Process 250 examples at a time for packing efficiency |

For a full explanation of packing strategies, see
[Sequence Packing](../datasets/sequence-packing.md).

## Step 4: Create an interleaved dataset

Interleaved datasets combine multiple datasets into a single stream, sampling
from each according to configurable probabilities. This is useful for training
on a mix of data sources (e.g., multiple programming languages, or code + text).

**`templatelib/configs/interleaved.yaml`:**

```yaml
-- extends "datasets/dataset_type.yaml"

[config_metadata]
    == super()
    -- set ns.config_name = "Interleaved"
    -- set ns.config_description = "Interleave all datasets"
    -- set ns.main_feature = "text"

[main_body]
.define: &tokenizer !var "tokenizer"

    [pp_kwargs]
.define: &pp_kwargs !dict

    [dataset_a]
.define: &dataset_a !call:forgather:from_project
    <<: &dataset_args
        preprocess_args: !var "preprocess_args"
        tokenizer: *tokenizer
        pp_kwargs: *pp_kwargs
        shard_dataset: !var [ "shard_dataset", False ]
        select_range: !var [ "select_range", null ]
        shuffle: !var [ "shuffle", False ]
        seed: !var [ "seed", 42 ]
    project_dir: "{{ project_dir }}"
    config_template: "dataset-a.yaml"
    targets: [ "train_dataset", "eval_dataset", "test_dataset" ]

    [dataset_b]
.define: &dataset_b !call:forgather:from_project
    <<: *dataset_args
    project_dir: "{{ project_dir }}"
    config_template: "dataset-b.yaml"
    targets: [ "train_dataset", "eval_dataset", "test_dataset" ]

    [train_dataset]
train_dataset: &train_dataset !singleton:forgather.ml.datasets:interleave_datasets
    <<: &interleave_args
        probabilities: !partial:forgather.ml.datasets:balance_remaining_examples
        seed: 42
        stopping_strategy: "all_exhausted"
    datasets:
        - !call:getitem [ *dataset_a, 'train_dataset' ]
        - !call:getitem [ *dataset_b, 'train_dataset' ]

    [eval_dataset]
eval_dataset: !singleton:forgather.ml.datasets:interleave_datasets
    <<: *interleave_args
    datasets:
        - !call:getitem [ *dataset_a, 'eval_dataset' ]
        - !call:getitem [ *dataset_b, 'eval_dataset' ]

    [test_dataset]
test_dataset: !singleton:forgather.ml.datasets:interleave_datasets
    <<: *interleave_args
    datasets:
        - !call:getitem [ *dataset_a, 'test_dataset' ]
        - !call:getitem [ *dataset_b, 'test_dataset' ]
```

Key patterns in the interleaved config:

- Extends `datasets/dataset_type.yaml` directly (not `load_dataset.yaml`,
  since we're composing existing datasets rather than loading one).
- Each sub-dataset is loaded via `!call:forgather:from_project`, which
  materializes another dataset config within the same project.
- The `<<: &dataset_args` / `<<: *dataset_args` pattern defines common
  arguments once and merges them into each sub-dataset.
- `!call:getitem` extracts individual splits from the sub-dataset results.
- `balance_remaining_examples` dynamically adjusts sampling probabilities
  proportional to how many examples remain in each dataset.
- `stopping_strategy: "all_exhausted"` continues sampling until every dataset
  is fully consumed (oversampling shorter datasets as needed).

## Step 5: Create an interleaved-packed variant

Combine interleaving with packing by extending the interleaved config and
pointing each sub-dataset to its packed variant:

```yaml
-- extends "configs/interleaved.yaml"
-- from "packed.yaml" import dynamic_args

[config_metadata]
    == super()
    -- set ns.config_name = "Interleaved Packed"
    -- set ns.config_description = "Interleave and pack all datasets"

[pp_kwargs]
    == super()
    stride: {{ stride | default(0) }}
    max_length: {{ max_length | default(512) }}

[dataset_a]
    == super()
    config_template: "dataset-a-packed.yaml"

[dataset_b]
    == super()
    config_template: "dataset-b-packed.yaml"

[dynamic_args]
    == super()
{{ dynamic_args() }}
```

The `[pp_kwargs]` block passes `max_length` and `stride` down to the packed
sub-configs via Jinja2 preprocessing. Each sub-dataset block overrides just the
`config_template` to point to the packed variant while keeping all other
arguments unchanged via `== super()`.

## Using datasets from a training project

Training projects load datasets from dataset projects using
`!call:forgather:from_project`:

```yaml
[datasets_definition]
.define: &dataset_dict !call:forgather:from_project
    project_dir: "/path/to/my_datasets"
    config_template: "my-dataset-packed.yaml"
    targets: [ "train_dataset", "eval_dataset" ]
    preprocess_args: *tokenizer_args
    tokenizer: *tokenizer

train_dataset: &train_dataset !call:getitem [ *dataset_dict, 'train_dataset' ]
eval_dataset: &eval_dataset !call:getitem [ *dataset_dict, 'eval_dataset' ]
```

The `tokenizer` and `preprocess_args` are passed as runtime variables (via
`!var`) that the dataset config receives when materialized.

## Custom preprocessing

Some datasets require custom Python code to transform examples before
tokenization -- for example, applying chat templates, filtering by quality,
or extracting text from structured data. In these cases, you write a Python
module in a `src/` directory and reference it from the config.

For examples of datasets with custom preprocessing, see:
- [`examples/datasets/QuixiAI/`](../examples/datasets/QuixiAI/) -- custom
  preprocessing for synthetic data
- [`examples/datasets/Open-Orca/`](../examples/datasets/Open-Orca/) -- chat
  template application and filtering
- [`examples/datasets/OpenAssistant/`](../examples/datasets/OpenAssistant/README.md) --
  quality-weighted sampling from conversation trees

## Worked example

The `examples/datasets/ajibawa-2023/` project demonstrates these patterns with
real datasets: four code datasets (Python, C++, C, JavaScript), each with a
packed variant, plus interleaved and interleaved-packed configs that combine
all four.
