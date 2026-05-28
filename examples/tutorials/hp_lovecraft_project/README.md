## H.P. Lovecraft Project Tutorial

Finetune a 7B-parameter model on the complete works of H.P. Lovecraft to
summon the Elder Gods.

## What You'll Learn

This tutorial walks through:
- Creating a Forgather workspace from scratch
- Turning a directory of raw text files into a tokenized training dataset
- Building a finetuning project that reuses another project's dataset
- Fine-tuning a 7B-parameter model on a single 24 GB consumer GPU
- Pushing training context length well past the usual limits -- this tutorial
  reaches **~42K tokens on Mistral-7B and ~53K tokens on Llama-2-7B** on a
  single 24 GB card, and documents exactly how
- Serving the resulting model and generating long-form Lovecraftian prose
- Long-context RoPE-variant experiments (plain, YaRN, Llama-3 NTK-by-parts
  scaling, bumped `rope_theta`) -- see
  [long_context_experiments.md](long_context_experiments.md) for the
  full writeup, including implications for pretraining recipes

**Time required**: ~2-3 hours, depending on context length and epoch count.\
**Hardware requirements**: One GPU with 24 GB of VRAM (RTX 3090, 4090, 5090).

## Quick Start: Just Use the Reference Project

If you want to skip straight to training, a working copy of the whole
workspace is already checked in at `lovecraft_reference/`. Extract the
corpus, download + convert the base model (next section), then:

```bash
# 4K single-GPU default; ~30 min on a 24 GB card
cd lovecraft_reference/finetune_lovecraft
forgather train -M ~/models/fg_mistral_7b -d 0

# 16K long-context variant (requires extended tokenizer max_length, see below)
forgather -t 16k.yaml train -M ~/models/fg_mistral_7b \
    --attn-implementation sdpa -d 0
```

The rest of this document explains how the reference was built, what each
piece does, and how to push the context length.

## Setup

The tutorial assumes that everything lives under the tutorial directory,
but feel free to work outside the Forgather tree; you will just need to
adjust paths accordingly.

### Extract the text corpus

```bash
# From examples/tutorials/hp_lovecraft_project/
tar -xzf hp_lovecraft.tgz

# Produces a hp_lovecraft/ directory with 63 .txt files
less hp_lovecraft/the_call_of_cthulhu.txt
```

### Download and convert a base model

The tutorial targets Mistral-7B-v0.1 because its GQA attention saves memory
at long context. Llama-2-7B and other 7B Llama variants also work with
minimal changes.

```bash
# Pick a models directory
MODELS_DIR=~/models  # or wherever you keep models
mkdir -p "${MODELS_DIR}"

# Download the base model
SRC_MODEL="${MODELS_DIR}/mistral_7b"
hf download mistralai/Mistral-7B-v0.1 --local-dir "${SRC_MODEL}" \
    --exclude "*.safetensors" "model.safetensors.index.json"

# Convert to Forgather format
FG_MODEL="${MODELS_DIR}/fg_mistral_7b"
forgather convert --dtype bfloat16 "${SRC_MODEL}" "${FG_MODEL}"
```

Forgather's conversion produces a self-contained model directory with
generated PyTorch code, the original tokenizer, and the weights in
`pytorch_model-*.bin` shards. This format unlocks the fused
linear+cross-entropy loss kernel and CPU activation offloading, which
together are what make long-context training on 24 GB actually fit.

To convert a trained Forgather checkpoint back to HF format (e.g. to serve
with vLLM or load with plain `AutoModelForCausalLM`):

```bash
forgather convert "${FG_MODEL}" OUTPUT_MODEL_PATH
```

#### Optional: extend the Mistral context limit

Mistral's tokenizer ships with `model_max_length: 32768`, which caps how
long a sequence the data collator will accept regardless of anything the
model can physically handle. To train at longer contexts:

```bash
# Re-convert with an extended max_length in the model config
forgather convert --dtype bfloat16 --max-length 65536 \
    "${SRC_MODEL}" "${FG_MODEL}"

# Bump the tokenizer limit too; the converter does not rewrite this file
sed -i 's/"model_max_length": 32768/"model_max_length": 65536/' \
    "${FG_MODEL}/tokenizer_config.json"
```

Llama-2-7B does not have this cap -- the converter ships it with an
effectively-unbounded `max_position_embeddings`, and rotary embeddings
scale naturally with sequence length.

### Syntax highlighting and the interactive CLI

The tutorial edits many Forgather config files. If you use Vim or VS Code,
the syntax-highlighting plugins in `syntax_highlighting/` will make them
much more readable. Otherwise YAML mode is the closest stock option.

For VS Code users: if you launch `forgather` from a terminal that isn't
attached to VS Code, export `VSCODE_IPC_HOOK_CLI` from a VS Code
terminal into your working shell and Forgather's `edit` command will
open files directly in the editor.

```bash
# From a VS Code terminal
env | grep VSCODE_IPC_HOOK_CLI

# Paste the value into your external terminal
export VSCODE_IPC_HOOK_CLI=/tmp/vscode-ipc-XXXXXX.sock
```

## Create a Forgather Workspace

A workspace groups related projects and centralises search paths. The CLI
scaffolds one in seconds:

```bash
# From examples/tutorials/hp_lovecraft_project/
forgather ws create --name "H.P. Lovecraft Workspace" \
    --description "H.P. Lovecraft tutorial workspace" \
    --forgather-dir ../../../ -l base -l finetune

cd hp_lovecraft_workspace/
cat forgather_workspace/base_directories.yaml
cat forgather_workspace/meta_defaults.yaml
```

`meta_defaults.yaml` defines the default template search paths that every
project in the workspace inherits. `base_directories.yaml` holds path
definitions shared by both the meta-config and all projects -- the CLI
auto-generates a pointer to the Forgather installation; you can add more
(e.g. `ns.models_dir`, `ns.datasets_dir`) here if you like.

### Base directories reference

Standard base-directory names have defaults in
[templatelib/base/config_type.yaml](../../templatelib/base/config_type.yaml).
Override them at the workspace level by editing `base_directories.yaml`.
Always anchor paths to a symbolic location rather than using raw relatives,
so configs work regardless of the current working directory.

**Required**
- `ns.forgather_dir` -- the installed Forgather directory

**Overridable**
- `ns.models_dir` -- where models are stored
- `ns.datasets_dir` -- where datasets are stored
- `ns.tokenizers_dir`, `ns.model_src_dir`, `ns.project_model_src_dir`

**Set by the preprocessor**
- `project_dir`, `workspace_root`
- `user_home_dir()`, `forgather_config_dir()`, `getcwd()`
- `user_data_dir()`, `user_cache_dir()`, `user_config_dir()`,
  `site_data_dir()`, `site_config_dir()`
  (see [platformdirs](https://pypi.org/project/platformdirs/))

`forgather pp` prints the runtime values in the preprocessed header for
diagnostics.

## Create a Dataset Project

The dataset project tokenises raw text into training blocks and exposes
splits. We start from the `local_dataset` example's sliding-window config
because our examples (complete stories) are too long for a single block
and benefit from overlapping windows.

```bash
# From hp_lovecraft_workspace/
forgather project create --name "Lovecraft Dataset" \
    --description "The complete works of H.P. Lovecraft" \
    --default-config lovecraft.yaml \
    ../../../datasets/local_dataset/templatelib/configs/sliding_window.yaml

cd lovecraft_dataset/

# Recommended: use the interactive shell for the rest of this section
forgather -i
```

When running interactively, drop the `forgather` prefix from the command
examples (so `pp` instead of `forgather pp`, `ls` instead of `forgather ls`,
etc.).

### Customize the dataset configuration

Open `templates/configs/lovecraft.yaml` (in interactive mode: `edit`, then
pick `lovecraft.yaml` from the menu).

The stock `sliding_window.yaml` uses `load_from_disk`; we have loose `.txt`
files so we swap in `load_dataset`, point at the corpus directory, and
switch the `block_tokenize_fn` over to the newer `preprocess_args`-based
API so the *training* project can inject `max_length = seq_len` at runtime:

```yaml
-- extends 'datasets/tokenized_dataset.yaml'

[config_metadata]
    == super()
    -- set ns.config_name = "Lovecraft"
    -- set ns.config_description = "The complete works of H.P. Lovecraft"
    -- set ns.dataset_path = joinpath(project_dir, "../../hp_lovecraft")

[map_function]
.define: &map_function !partial:forgather.ml.datasets:block_tokenize_fn
    max_length: {{ max_length | default(4096) }}
    stride: {{ stride | default(0) }}
    min_len: {{ min_length | default(1) }}

[dataset_dict]
dataset_dict: &dataset_dict !singleton:datasets:load_dataset
    arg0: "text"
    data_dir: {{ ns.dataset_path }}
    sample_by: "document"
    data_files:
        train: "*.txt"                           # train on all files
        validation: "the_call_of_cthulhu.txt"    # validate on this one
        test: "at_the_mountains_of_madness.txt"

# Use a YAML merge to share the common dataset arguments across splits;
# `fn_kwargs: !var "preprocess_args"` is the critical bit -- the training
# project injects its `seq_len` through this variable at runtime.
#
# `partition_purpose: "train"` / `"eval"` distinguishes how each split
# is partitioned across DDP / DiLoCo workers. Train is the
# parameter-update split; eval / test are replicated across DiLoCo
# hosts (every host runs the full eval; metrics averaged). See
# `docs/trainers/diloco.md` for the full validity matrix.

[train_dataset]
train_dataset: &train_dataset !singleton:forgather.ml.datasets:preprocess_dataset@train_dataset
    <<: &common_dataset_args
        tokenizer: *tokenizer
        fn_kwargs: !var { name: "preprocess_args", default: null }
        map_fn: *map_function
    dataset: *train_dataset_split
    desc: "Tokenizing train"
    partition_purpose: "train"

[eval_dataset]
eval_dataset: &eval_dataset !singleton:forgather.ml.datasets:preprocess_dataset@eval_dataset
    <<: *common_dataset_args
    dataset: *validation_dataset_split
    desc: "Tokenizing validation"
    partition_purpose: "eval"

[test_dataset]
test_dataset: &test_dataset !singleton:forgather.ml.datasets:preprocess_dataset@test_dataset
    <<: *common_dataset_args
    dataset: *validation_dataset_split
    desc: "Tokenizing test"
    partition_purpose: "eval"

[dynamic_args]
    == super()
    dataset_path:
        names: "--dataset-path"
        type: path
        help: "Local path to dataset"
    max_length:
        names: "--max-length"
        type: "int"
        help: "Maximum tokens per output block"
    stride:
        names: "--stride"
        type: "int"
        help: "Number of tokens to overlap between blocks"
    min_length:
        names: "--min-length"
        type: "int"
        help: "Minimum example length (tokens)"
```

For simplicity, we validate on a file that is also in the training split.
In a real run you would exclude it, but this keeps the tutorial small.

The `max_length` template variable has two roles:

- When the dataset is exercised standalone (e.g. `forgather -t lovecraft.yaml
  dataset --max-length 4096 ...`), it takes its default `4096` unless the
  `--max-length` CLI flag is supplied.
- When the dataset is consumed by a finetune project, the parent's
  `[datasets_preprocessor_args]` block passes `max_length` via the
  `preprocess_args` runtime variable.  The `!var { name: "preprocess_args",
  default: null }` spelling causes `fn_kwargs` to *override* the template
  default at runtime, so the training `seq_len` wins when both are present.

This two-way plumbing is what lets a single `lovecraft.yaml` power both
standalone inspection and training at any `seq_len` the parent asks for.

### Test the configuration

```bash
# Show the preprocessed configuration
pp

# Debug Jinja2 template errors by dumping every preprocessed template
pp --debug

# Same idea for ls when configs fail to parse
ls --debug

# Construct and inspect the raw dataset splits
construct --target dataset_dict

# Dump the first training example (a complete story).  In interactive mode
# this pipes through less automatically.
dataset --target train_dataset_split -n 1
```

Next, try the block tokenizer. It needs a tokenizer path (`-T`), and
optionally `--max-length` and `--stride`. Small values make it easy to
see what is happening:

```bash
dataset --target train_dataset \
    -T ~/models/fg_mistral_7b \
    --max-length 64 --stride 8 -s -n 3
```

### Add a packed block-size config

For long-context training we want packed blocks (multiple short stories
concatenated into each block, with document-start markers for the
collator) rather than one-document-per-block. Create a derived config:

```bash
project new_config lovecraft-packed.yaml templates/configs/lovecraft.yaml
edit                              # pick lovecraft-packed.yaml
```

Replace everything except the metadata with:

```yaml
-- extends "configs/lovecraft.yaml"

[config_metadata]
    == super()
    -- set ns.config_name = "Lovecraft Packed"
    -- set ns.config_description = "Densely-packed Lovecraft blocks"

[map_function]
.define: &map_function !partial:forgather.ml.datasets:block_tokenize_fn
    max_length: {{ max_length | default(4096) }}
    stride: {{ stride | default(0) }}
    packed: True
    packing_strategy: "best_fit"
    shuffle_output: True
    min_len: {{ min_length | default(64) }}

[train_dataset]
    == super()
    map_kwargs:
        batch_size: 1000    # Larger batches let the packer find better fits.

[eval_dataset]
    == super()
    map_kwargs:
        batch_size: 1000

[test_dataset]
    == super()
    map_kwargs:
        batch_size: 1000
```

Verify it parses and produces the expected distribution:

```bash
forgather ls
# Lovecraft Dataset : The complete works of H.P. Lovecraft
#     lovecraft-packed.yaml          Lovecraft Packed : Densely-packed Lovecraft blocks
#     [lovecraft.yaml]               Lovecraft : The complete works of H.P. Lovecraft

# Peek at a couple of tokenized blocks
forgather -t lovecraft-packed.yaml dataset --target train_dataset \
    -T ~/models/fg_mistral_7b --max-length 4096 -s -n 2 | head

# Token-length histogram (--tokenized / -s tells it the split is already
# tokenised, so it reads `input_ids` instead of retokenising `text`).
forgather -t lovecraft-packed.yaml dataset --target train_dataset \
    -T ~/models/fg_mistral_7b --max-length 4096 -s --histogram
```

Packed blocks are tightly clustered just below the `max_length` cap -- the
`best_fit` packer bundles short stories together to minimise padding
waste. A histogram SVG is written next to the config.

## Create the Finetune Project

`cd` back out of the dataset project:

```bash
quit                  # exit the interactive shell
cd ..                 # back to the workspace root
```

The finetune project is a separate project that reads examples from the
dataset project. We build it on `projects/finetune_v2.yaml`, which extends
`projects/lm_training_project.yaml`. Between them these templates give us:

- A token-budget-driven step count (specify `total_tokens`, `warmup_tokens`,
  `annealing_tokens` in millions; step counts are derived from sequence
  length and batch size).
- Automatic LR scaling from global batch size (power-law rule, sqrt scaling
  by default).
- A WSD (warmup-stable-decay) learning-rate scheduler with auto-triggered
  annealing at `total_steps - annealing_steps`.
- A text-generation eval callback that periodically samples from the model
  during training so quality regressions are visible in TensorBoard.
- Critically for this tutorial: `[datasets_preprocessor_args].max_length`
  is wired to `ns.seq_len`, so the dataset's `block_tokenize_fn` really
  receives the training sequence length -- no silent 4K defaults.

Create an empty project and we'll fill in the template:

```bash
forgather project create --name "Finetune Lovecraft" \
    --description "Finetune a model on the complete works of H.P. Lovecraft" \
    --default-config default.yaml

cd finetune_lovecraft/
forgather -i
```

### Write the project template

Create `templates/project.yaml` (inside the project, not in `configs/`).
This is where we pin the dataset project, sensible defaults for the
tutorial's token budget, and the LR-scheduler's auto-decay trigger:

```bash
project new_config --type project project.yaml
edit                              # pick project.yaml
```

Contents:

```yaml
-- extends 'projects/finetune_v2.yaml'

[config_metadata]
    == super()
    -- set ns.config_name = "Lovecraft Finetune"
    -- set ns.config_description = "Fine-tune a causal LM on H.P. Lovecraft"

    ## Dataset: point at the sibling dataset project in this workspace.
    -- set ns.dataset_proj = joinpath(workspace_root, 'lovecraft_dataset')
    -- set ns.dataset_config = dataset_config | default("lovecraft-packed.yaml")

    ## Batching.  seq_len is threaded into the dataset's block_tokenize_fn
    ## via [datasets_preprocessor_args].max_length, so blocks are really
    ## this long.
    -- set ns.seq_len = seq_len | default(4096)
    -- set ns.per_device_train_batch_size = batch_size | default(1)

    ## Token budget (millions).  ~7M tokens is roughly 10 epochs on the
    ## 63-story corpus at 4K context.
    -- set ns.total_tokens = total_tokens | default(7)

    ## Compile: finetune_v2 defaults torch_compile=True with max-autotune,
    ## which is memory-hungry.  Off by default for the 24 GB target.
    -- set ns.torch_compile = compile | default(False)

    ## Small corpus: log more often than the library defaults by scaling
    ## ns.step_cadence (which multiplies base log/eval/save token intervals).
    -- set ns.step_cadence = step_cadence | default(0.02)

    ## LR annealing budget (millions of tokens).  WSDScheduler holds LR
    ## constant until decay_start_step kicks in below.
    -- set ns.annealing_tokens = annealing_tokens | default(10)

    ## LR calibration (from an LR sweep on this project).  lm_training_project
    ## scales ns.base_lr by (actual_tokens_per_step / ns.base_batch_size) ^
    ## ns.lr_alpha (sqrt scaling by default), so a longer seq_len is
    ## automatically given a proportionally larger LR.
    -- set ns.base_lr = lr | default(5.0e-5)
    -- set ns.base_batch_size = 4096

    ## Lovecraft-flavoured prompts for the TextgenCallback.  Resolved from
    ## project_dir so it's independent of the shell's CWD at training time.
    -- set ns.eval_prompts_file = eval_prompts_file | default(abspath(joinpath(project_dir, "../../prompts/lovecraft_seeds.yaml")))
    -- set ns.eval_max_new_tokens = eval_max_new_tokens | default(256)

[globals]
    == super()
    ## Auto-decay trigger: WSD switches from stable -> decay at this step.
    -- set ns.decay_start_step = [ns.warmup_steps, (ns.total_steps - ns.annealing_steps)] | max | int

[variable_listing]
    == super()
# ns.decay_start_step: {{ ns.decay_start_step }}

[trainer_args]
    == super()
    ## Single-GPU memory knobs.  Together these fit 7B + 16K context on
    ## a 24 GB card; on larger cards they can be disabled for speed.
    gradient_checkpointing: {{ gradient_checkpointing | default(True) }}
    fuse_optim_with_backward: {{ fuse_optim_with_backward | default(True) }}
    enable_activation_offloading: {{ activation_offloading | default(True) }}

    ## finetune_v2 defaults max_steps = -1 (one epoch).  Rebind to ns.total_steps
    ## so --total-tokens actually bounds training.
    max_steps: {{ max_steps | toyaml(ns.total_steps) }}

[lr_scheduler]
lr_scheduler: &lr_scheduler !partial:forgather.ml.optim:WSDScheduler@lr_scheduler
    warmup_steps: {{ ns.warmup_steps }}
    min_lr: {{ ns.min_lr | toyaml }}
    decay_steps: {{ ns.annealing_steps }}
    decay_start_step: {{ ns.decay_start_step }}
    start_decay: {{ start_annealing | toyaml(False) }}
```

### Write the default training config

Because `project.yaml` already holds the batch size, token budget, compile
setting, and memory knobs, the per-config file is very thin -- it just
names the run. Sibling configs (like the `16k.yaml` below) can inherit
everything from `project.yaml` without copying it.

```bash
project new_config default.yaml        # create templates/configs/default.yaml
edit                                    # pick default.yaml
```

Contents:

```yaml
-- extends 'project.yaml'

[config_metadata]
    == super()
    -- set ns.config_name = "Lovecraft Default"
    -- set ns.config_description = "Single-GPU 4K-context fine-tune on the Lovecraft corpus"
    -- set ns.log_name = log_name | default("default")
```

And the 16K sibling, which overrides only the sequence length:

```yaml
-- extends 'project.yaml'

[config_metadata]
    == super()
    -- set ns.config_name = "Lovecraft 16K"
    -- set ns.config_description = "16K-context single-GPU fine-tune"
    -- set ns.log_name = log_name | default("16k")
    -- set ns.seq_len = seq_len | default(16384)
```

Add a matching `prompts/lovecraft_seeds.yaml` at the tutorial root
(`../../prompts/lovecraft_seeds.yaml` from the finetune project, per the
`ns.eval_prompts_file` path above).  The format is a YAML list of strings;
see [`prompts/lovecraft_seeds.yaml`](prompts/lovecraft_seeds.yaml) for the
reference's invented Lovecraftian openings.

Verify:

```bash
forgather ls
# Finetune Lovecraft : Finetune a model on the complete works of H.P. Lovecraft
#     [default.yaml]                 Lovecraft Default : Single-GPU 4K-context fine-tune on the Lovecraft corpus

forgather pp                        # inspect the fully-resolved config
```

## Train

See what parameters the config accepts:

```bash
train --help
```

Key flags from `finetune_v2`:

- `-M / --model-id-or-path` -- HF ID or local Forgather model directory.
- `--total-tokens N` -- total training tokens (millions).
- `--seq-len N` -- sequence length; passed through to the dataset as
  `max_length`, so blocks are really this long.
- `--batch-size N` -- per-device training batch size.
- `--lr X` -- base learning rate (scaled by global batch size).
- `--attn-implementation {flex_attention, sdpa, eager}`.

Smoke-test first to confirm everything is wired correctly:

```bash
train --max-steps 10 --save-strategy no \
    -M ~/models/fg_mistral_7b -d 1
```

(We pin the smoke test to `-d 1` so GPU 0 stays free for other work --
adjust to whichever device you prefer.)

Then the real run (on an RTX 4090 the 4K default finishes its ~7M-token
budget in about 30 minutes):

```bash
train -M ~/models/fg_mistral_7b -d 1
```

Checkpoints land under `${FG_MODEL}/checkpoints/`; training logs under
`${FG_MODEL}/runs/`. Override either with `--output-dir PATH`.

## Push the Context Further

The current training stack -- gradient checkpointing + `fuse_optim_with_backward`
+ activation offloading + SDPA (flash / mem-efficient backend) +
Adafactor + the fused cross-entropy loss -- has made a big dent in
per-token memory since the original tutorial was written. On a single
24 GB card (batch size 1, bf16, SDPA, packed-dense training data so
every token is real), peak memory comes out as:

| Context | Llama-2-7B (MHA, 32/32 KV heads) | Mistral-7B-v0.1 (GQA, 32/8 KV heads) |
|---------|----------------------------------|---------------------------------------|
| 4K      | 13.9 GiB                         | 14.9 GiB                              |
| 8K      | 14.1 GiB                         | -                                     |
| 16K     | 15.5 GiB                         | 16.8 GiB                              |
| 24K     | 17.0 GiB                         | -                                     |
| 32K     | 18.5 GiB                         | 20.0 GiB                              |
| 40K     | 19.9 GiB                         | 21.7 GiB                              |
| 43K     | -                                | 22.1 GiB (practical ceiling)          |
| 48K     | 21.4 GiB                         | OOM                                   |
| 51K     | 21.9 GiB                         | --                                    |
| 53K     | 22.1 GiB (practical ceiling)     | --                                    |

Llama-2-7B wins the long-context race despite having full MHA instead of
GQA. The reason is not what you might expect. With a modern SDPA backend
(flash / mem-efficient) *or* flex-attention, the attention kernel itself
is already O(N) in memory -- it never materialises the full NxN score
matrix. GQA's K/V shrinkage therefore buys essentially nothing here, and
what ends up mattering is the per-token activation outside the attention
kernel. Mistral's MLP intermediate is 14336 vs Llama's 11008 (~30%
larger), so Mistral's MLP hidden state dominates at long context and
Llama gets roughly 10K more usable tokens on the same card.

To double-check: re-running the 4K and 8K rows with `--attn-implementation
eager` (which *does* allocate the NxN matrix) immediately OOMs at 8K --
eager attention's quadratic allocation alone needs 8 GiB at 8K on
Llama-7B's 32 heads, and the card can't absorb it on top of weights and
saved activations. That's why the modern backends matter: they're the
difference between "ceiling at 4K" and "ceiling at 50K+".

A `16k.yaml` config is checked into the reference project as a long-context
sibling of `default.yaml`. Because `finetune_v2` threads `ns.seq_len` all
the way into the dataset's `block_tokenize_fn`, a single `--seq-len` flag
reshapes both training and the packed dataset -- no separate dataset
config toggle is needed:

```bash
# From lovecraft_reference/finetune_lovecraft/
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
forgather -t 16k.yaml train \
    --max-steps 3 --save-strategy no \
    -M ~/models/fg_mistral_7b \
    --attn-implementation sdpa \
    -d 1
```

To sweep context length from the default 16K upwards, just override
`--seq-len`:

```bash
forgather -t 16k.yaml train \
    --max-steps 3 --save-strategy no \
    --seq-len 32768 \
    -M ~/models/fg_mistral_7b \
    --attn-implementation sdpa \
    -d 1
```

At 32K+, `flex_attention` (the `finetune_v2` default) currently hits a
tensor-stride assertion; force `sdpa` until that's investigated. At
seq_len > 32768 on Mistral, also bump the tokenizer's `model_max_length`
(see "Optional: extend the Mistral context limit" above).

### What actually pays off

- `gradient_checkpointing: True` -- the single biggest win; trades compute
  for activation memory.
- `fuse_optim_with_backward: True` -- merges the optimizer step into the
  backward pass so gradients are freed as soon as the parameter is
  updated, instead of after the whole backward.
- `enable_activation_offloading: True` -- moves saved activations to CPU
  RAM between forward and backward. Requires a Forgather-format model
  (the converter sets up the hooks correctly).
- `attn_implementation: sdpa` (or `flex_attention`) -- the decisive
  choice. SDPA's flash and mem-efficient backends, and PyTorch's
  flex-attention, all chunk the attention kernel internally and never
  allocate the NxN score matrix. Raw `eager` attention does allocate it
  and OOMs at roughly 8K on a 24 GB card. Pad-only flex-attention is
  *additionally* sparse across pad positions, which is nice for training
  on non-packed data but irrelevant to the memory ceiling.
- **Adafactor** (vs AdamW): one state tensor instead of two, and the
  factored form saves memory at the 7B param count.
- **Fused linear-cross-entropy loss** (`LinearCrossEntropyLoss`): avoids
  ever materialising a `(seq_len, vocab_size)` logits tensor. At 32K
  context and 32000 vocab that alone would be a 2 GB bf16 tensor.

### When you hit OOM

- If you see `failed to CUDA calloc` during training setup, the model +
  optimizer + first forward exceeded VRAM; drop context length, batch
  size, or both.
- If training OOMs mid-step, try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  first -- it often buys the 1-2 GiB you need without changing anything
  else.
- Beyond that, bump `gradient_checkpointing` to include more layers, drop
  `per_device_train_batch_size` to 1, or move to a multi-GPU config (see
  "Multiple GPUs" below).

## Inference

Serve the fine-tuned model with Forgather's OpenAI-compatible inference
server. For raw speed, convert back to HF format first
(`forgather convert`); Forgather's reference implementation skips the KV
cache and will be slow on long prompts.

### Start the server

```bash
# Start with the latest training checkpoint auto-selected
forgather inf server -c -m ~/models/fg_mistral_7b
```

### Continue a seeded prompt

The model was not taught a chat format, so use completion mode. Seed
with the opening of a story and let the model continue:

```bash
forgather inf client --completion \
    "Of such great powers or beings there may be conceivably a survival" \
    --max-tokens 512
```

### Long-context generation

With a fine-tuned checkpoint you can generate well past the 4K training
window (the rotary embeddings handle it; quality tails off gradually).
A favourite prompt for this tutorial is an invented Lovecraft title --
the model will confabulate a complete story to go with it:

```bash
forgather inf client --temperature 1.0 \
    --completion "The Stranger (1923)" \
    --max-tokens 8192 | tee the_stranger.txt
```

### Experiments to try

A few quick inference experiments that are fun to run after training:

- **Sampling sweep**: generate the same seed at `--temperature 0.3, 0.7,
  1.0, 1.3` and compare. Lower temperatures produce more "Lovecraft
  cliche" averages; higher temperatures produce weirder and occasionally
  broken prose.
- **Base vs fine-tuned**: start a second inference server pointed at the
  un-trained converted model (omit `-c` so it loads the base weights from
  the model directory) and feed it the same seed. The fine-tuned model's
  affect for adjective-stacking, archaic diction, and unnamed cosmic
  dread is very clearly absent from the base.
- **Long continuation**: generate the same `--completion` at
  `--max-tokens 2048` and `--max-tokens 16384`. Look for where the model
  starts losing plot coherence; this gives an informal sense of how far
  the finetune's quality extends past the 4K training window.

## Monitoring and Control

Forgather has a control interface for monitoring and safely stopping
running jobs. Prefer this over Ctrl-C, which can leave worker processes
hanging (especially for pipeline-parallel runs).

```bash
forgather control list                 # discover running jobs
forgather control status JOB_ID        # inspect a specific job
forgather control save JOB_ID          # force checkpoint save
forgather control stop JOB_ID          # graceful stop (saves a final checkpoint)
forgather control save-stop JOB_ID     # save then exit
forgather control abort JOB_ID         # kill without saving
forgather control cleanup              # prune dead job endpoint files
```

### Training dashboards

TensorBoard reads the logs produced during training:

```bash
forgather tb --output-dir ~/models/fg_mistral_7b
# or, to expose it on the LAN:
forgather tb --output-dir ~/models/fg_mistral_7b -- --bind_all
```

For quick offline inspection:

```bash
forgather logs summary ~/models/fg_mistral_7b/runs/*/trainer_logs.json
forgather logs plot --loss-curves ~/models/fg_mistral_7b/runs/*/trainer_logs.json
```

## Extra Credit

### Multiple GPUs / multiple nodes

`finetune_v2` exposes a `--trainer-type` CLI flag that selects between
`basic` (single-GPU, the default), `ddp`, `fsdp2`, and `pipeline`. The
same `default.yaml` / `16k.yaml` configs work across all of them; the
training project reads `nproc_per_node` from the trainer class when
launched. For example:

```bash
# 2-GPU DDP
forgather -t default.yaml train -M ~/models/fg_mistral_7b \
    --trainer-type ddp -d 1,2

# Pipeline parallel across 4 GPUs (roughly 32K context at PBS > 1)
forgather -t 16k.yaml train -M ~/models/fg_mistral_7b \
    --trainer-type pipeline -d 1,2,3,4 \
    --attn-implementation sdpa
```

See `docs/trainers/trainer_options.md` for the per-trainer option matrix.

### Alternative optimizers

At 7B scale your optimizer choices are thin, but a few are worth trying:

- **SGD with momentum**: minimal state, but needs a much smaller LR.
- **Adafactor variants**: the default in this project; try
  `Adafactor(lr=..., decay_rate=-0.8)` for LLaMA-style decoupling.
- **torchao 4-bit AdamW**: see `torchao.optim.AdamW4bit`. Fits AdamW-like
  adaptivity into roughly the same footprint as Adafactor. Works with
  stochastic rounding via the `stochastic_rounding=True` kwarg.

Override the optimizer by adding a `[optimizer]` block to a config that
extends `default.yaml`:

```yaml
[optimizer]
optimizer: &optimizer !partial:torchao.optim:AdamW4bit
    lr: {{ ns.global_lr | toyaml }}
    stochastic_rounding: True
```

### Tune the schedule

`project.yaml` wires in a WSDScheduler that holds LR flat during the
"stable" phase and anneals over the final `ns.annealing_tokens` tokens.
The auto-decay trigger is derived from `ns.total_steps` and
`ns.annealing_steps`; override the annealing budget via `--annealing-tokens`
(millions), or force an early decay at any step with
`forgather control save` followed by `--start-annealing` on resume.

### Push to Llama-2-7B for the highest context

Per the memory table above, Llama-2-7B fits about 11K more tokens than
Mistral on the same card. Run through the conversion once more with your
Llama-2 checkout:

```bash
forgather convert --dtype bfloat16 \
    /path/to/meta-llama--Llama-2-7b-hf ~/models/fg_llama_7b
```

Then point `--model-id-or-path` at the Llama directory. The same dataset
project works -- the tokenizer swap happens automatically because the
dataset project reads its tokenizer from the model path.

## Reference Project Layout

A fully-working copy of the workspace is at `lovecraft_reference/`.
Files worth looking at:

- [`lovecraft_dataset/templates/configs/lovecraft.yaml`](lovecraft_reference/lovecraft_dataset/templates/configs/lovecraft.yaml) -- base dataset config (non-packed; one document per block)
- [`lovecraft_dataset/templates/configs/lovecraft-packed.yaml`](lovecraft_reference/lovecraft_dataset/templates/configs/lovecraft-packed.yaml) -- packed-block variant used for training (parametrised by `--max-length`)
- [`finetune_lovecraft/templates/project.yaml`](lovecraft_reference/finetune_lovecraft/templates/project.yaml) -- extends `projects/finetune_v2.yaml`; pins the dataset project, the TextgenCallback prompts, and the WSDScheduler auto-decay trigger
- [`finetune_lovecraft/templates/configs/default.yaml`](lovecraft_reference/finetune_lovecraft/templates/configs/default.yaml) -- 4K single-GPU fine-tune (memory-saving knobs enabled by default)
- [`finetune_lovecraft/templates/configs/16k.yaml`](lovecraft_reference/finetune_lovecraft/templates/configs/16k.yaml) -- 16K-context single-GPU variant; override `--seq-len` for a broader context sweep
- [`prompts/lovecraft_seeds.yaml`](prompts/lovecraft_seeds.yaml) -- invented Lovecraftian openings used by the TextgenCallback during training

## Long-context generation quality experiments

Fitting a 16K-context training run into VRAM is only half the problem. The
other half is getting the trained model to *generate coherently beyond its
training window*.
[long_context_experiments.md](long_context_experiments.md) documents a
four-way Llama-2-7B comparison — plain RoPE, YaRN, Llama-3 NTK-by-parts
scaling, and a bumped base frequency (`rope_theta=500 000`) — all
fine-tuned identically at 8K context and evaluated at 2K-16K on held-out
text.

Headline findings:

- **Plain Llama-2 RoPE cannot extrapolate past its training window.**
  Fine-tuned at 8K, evaluated at 16K → PPL grows 3.5×.
- **A bumped `rope_theta` alone captures most of the extrapolation benefit.**
  θ=500 000 with no other scaling: 15% PPL growth from 8K → 16K.
- **Llama-3-style NTK-by-parts scaling adds a small further improvement**
  on top of the high θ.
- **YaRN with a factor that doesn't cover the eval window is catastrophic.**
  `factor=2, orig=4096` trained at 8K blows up to 24× the 8K PPL at 16K.

The experiments doc also includes a section on **implications for
pretraining**: in short, if `θ` can be swapped at fine-tune time and
recover long-context extrapolation from a 4K-pretrained model in ~4M
tokens of adaptation, the compute-efficient recipe is to pretrain short
with a deployment-sized `θ`, not to pretrain long. A proposed follow-up
study is sketched there.

For the earlier investigation into a 4K-periodic NLL spike pattern —
ultimately traced to a configuration gap in the pre-`finetune_v2`
template — see [`4k_spike_investigation.md`](4k_spike_investigation.md).
That artefact does not appear in the current properly-plumbed runs.
