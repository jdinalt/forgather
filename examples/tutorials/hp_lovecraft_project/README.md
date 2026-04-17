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
- **Making long-context generation actually work:** training at 16K isn't
  enough to get coherent 16K generation -- see
  [long_context_experiments.md](long_context_experiments.md) for YaRN, sliding-window,
  and precision experiments across Mistral-7B and Llama-2-7B

**Time required**: ~2-3 hours, depending on context length and epoch count.\
**Hardware requirements**: One GPU with 24 GB of VRAM (RTX 3090, 4090, 5090).

## Quick Start: Just Use the Reference Project

If you want to skip straight to training, a working copy of the whole
workspace is already checked in at `lovecraft_reference/`. Extract the
corpus, download + convert the base model (next section), then:

```bash
cd lovecraft_reference/finetune_lovecraft
forgather train --epochs 3 -M ~/models/fg_mistral_7b -d 0
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
[templatelib/base/config_type.yaml](../../../templatelib/base/config_type.yaml).
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

The stock config is almost right, but needs a few edits:

```yaml
[config_metadata]
    == super()
    -- set ns.config_name = "Lovecraft"
    -- set ns.config_description = "The complete works of H.P. Lovecraft"
    -- set ns.dataset_path = joinpath(project_dir, "../../hp_lovecraft")

[dataset_dict]
dataset_dict: &dataset_dict !singleton:datasets:load_dataset
    arg0: "text"
    data_dir: {{ ns.dataset_path }}
    sample_by: "document"
    data_files:
        train: "*.txt"                           # train on all files
        validation: "the_call_of_cthulhu.txt"    # validate on this one
        test: "at_the_mountains_of_madness.txt"
```

For simplicity, we validate on a file that is also in the training split.
In a real run you would exclude it, but this keeps the tutorial small.

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

Next, try the block tokenizer. It needs a tokenizer path (`-T`), a
`--window-size` (tokens per block), and a `--stride` (token overlap
between consecutive blocks). Small values make it easy to see what is
happening:

```bash
dataset --target train_dataset \
    -T ~/models/fg_mistral_7b \
    --window-size 64 --stride 8 -s -n 3
```

### Add a 4K block-size config

We will train with 4096-token blocks. Create a derived config so the
settings stay in one place:

```bash
project new_config 4k.yaml templates/configs/lovecraft.yaml
edit                              # pick 4k.yaml
```

Replace everything except the metadata with:

```yaml
-- extends 'configs/lovecraft.yaml'

[config_metadata]
    == super()
    -- set ns.config_name = "Lovecraft 4K"
    -- set ns.config_description = "Lovecraft with 4K Blocks"

[map_function]
.define: &map_function !partial:forgather.ml.datasets:block_tokenize_fn
    max_length: 4096
    stride: 512
```

Verify it parses and produces the expected distribution:

```bash
forgather ls
# Lovecraft Dataset : The complete works of H.P. Lovecraft
#     4k.yaml                        Lovecraft 4K : Lovecraft with 4K Blocks
#     [lovecraft.yaml]               Lovecraft : The complete works of H.P. Lovecraft

# Peek at a couple of tokenized blocks
forgather -t 4k.yaml dataset --target train_dataset \
    -T ~/models/fg_mistral_7b -s -n 2 | head

# Token-length histogram (--tokenized / -s tells it the split is already
# tokenised, so it reads `input_ids` instead of retokenising `text`).
forgather -t 4k.yaml dataset --target train_dataset \
    -T ~/models/fg_mistral_7b -s --histogram
```

Typical output:

```
sample size: 208
min: 519
max: 4096
mean: 3491.60
median: 4096.0
```

Most blocks hit the 4K cap; the short tail comes from end-of-story
windows shorter than `window_size`. A histogram SVG is written next to
the config.

## Create the Finetune Project

`cd` back out of the dataset project:

```bash
quit                  # exit the interactive shell
cd ..                 # back to the workspace root
```

The finetune project is a separate project that reads examples from the
dataset project. We start from Samantha's single-GPU Llama-7B config
because it already wires up activation offloading, gradient
checkpointing, and the fused loss kernel.

```bash
forgather project create --name "Finetune Lovecraft" \
    --description "Finetune a model on the complete works of H.P. Lovecraft" \
    --default-config 1gpu/default.yaml \
    ../../../finetune/samantha/templates/configs/llama2_7b/1gpu_default.yaml

cd finetune_lovecraft/
forgather -i
```

### Wire up the project template

Samantha's default config depends on a project-level `samantha.yaml`.
Copy that template in too -- `--type project` tells `new_config` that
this goes into `templates/` directly, not `templates/configs/`:

```bash
project new_config --type project project.yaml \
    ../../../../finetune/samantha/templates/samantha.yaml
```

You now have:
- `templates/project.yaml` -- project-wide defaults
- `templates/configs/1gpu/default.yaml` -- the single-GPU training config

Edit `project.yaml` so it points at our dataset project instead of
Samantha's:

```yaml
[config_metadata]
    == super()
    -- set ns.default_dataset_proj = joinpath(workspace_root, 'lovecraft_dataset')
    -- set ns.default_dataset_config = "4k.yaml"
```

Edit `configs/1gpu/default.yaml` and update its metadata + max_length so
it matches our 4K dataset config:

```yaml
-- extends 'project.yaml'

[config_metadata]
    == super()
    -- set ns.config_name = "Finetune Lovecraft Default"
    -- set ns.config_description = "Train with 4096 token context on single GPU, 24 GB"
    -- set ns.log_name = "1gpu_4096"

[trainer_args]
    == super()
    per_device_train_batch_size: 4
    per_device_eval_batch_size: 6
    gradient_checkpointing: True
    fuse_optim_with_backward: True
    enable_activation_offloading: True

[datacollator]
    == super()
    max_length: 4096

[optimizer]
    == super()
    lr: 3.5e-6           # rescaled for the smaller batch
```

Verify:

```bash
forgather ls
# Finetune Lovecraft : Finetune a model on the complete works of H.P. Lovecraft
#     [1gpu/default.yaml]            Finetune Lovecraft Default : Train with 4096 token context on single GPU, 24 GB

forgather pp               # inspect the fully-resolved config
```

## Train

See what parameters the config accepts:

```bash
train --help
```

Smoke-test first to confirm everything is wired correctly:

```bash
train --max-steps 10 --save-strategy no \
    -M ~/models/fg_mistral_7b -d 0
```

Then the real run (on an RTX 4090 the 4K config finishes 3 epochs in
about 30 minutes with ~208 training blocks):

```bash
train --epochs 3 -M ~/models/fg_mistral_7b -d 0
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

A generic `long_context.yaml` is checked into the reference project with
`--seq-len`, `--window-size`, and `--attn-implementation` as CLI args.
Paired with `long_context_packed.yaml` (a packed-dataset variant so every
token is real, not padding), you can reproduce the numbers above:

```bash
# From lovecraft_reference/finetune_lovecraft/
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
forgather -t long_context.yaml train \
    --max-steps 3 --save-strategy no \
    --dataset-config long_context_packed.yaml \
    -M ~/models/fg_mistral_7b \
    --seq-len 32768 --window-size 32768 --batch-size 1 \
    --attn-implementation sdpa \
    -d 0 -P
```

`-P` / `--log-peak-memory` prints peak CUDA memory at each log step, which
is the easiest way to find where your particular card starts rejecting
allocations.

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

The Samantha tutorial's `llama2_7b/2gpu_pp.yaml`, `llama2_7b/4gpu_pp.yaml`,
and `llama2_7b/fsdp2.yaml` configs transfer directly to this project --
copy one into `templates/configs/`, swap the `extends` line to inherit
from `project.yaml`, and you have a multi-GPU Lovecraft run. With
pipeline parallel across 4 GPUs you can train Mistral at roughly 32K
context with per-device batch size >1.

### Alternative optimizers

At 7B scale your optimizer choices are thin, but a few are worth trying:

- **SGD with momentum**: minimal state, but needs a much smaller LR.
- **Adafactor variants**: the default in this project; try
  `Adafactor(lr=..., decay_rate=-0.8)` for LLaMA-style decoupling.
- **torchao 4-bit AdamW**: see `torchao.optim.AdamW4bit`. Fits AdamW-like
  adaptivity into roughly the same footprint as Adafactor. Works with
  stochastic rounding via the `stochastic_rounding=True` kwarg. See the
  `llama3_1b/ddp_adam4bit.yaml` config in the Samantha project for a
  working example.

### Tune the schedule, not just the optimizer

- The reference uses `warmup_steps: 50` and constant LR after. Try cosine
  annealing by setting `cooldown_steps > 0` in the `lr_scheduler` block.
  With ~200 blocks x 3 epochs that's a very short run -- warmup might
  not even be worth it.
- Log more frequently (`logging_steps: 2`) if you want a higher-resolution
  loss curve to compare schedules.

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

- [`lovecraft_dataset/templates/configs/lovecraft.yaml`](lovecraft_reference/lovecraft_dataset/templates/configs/lovecraft.yaml) -- the base dataset config
- [`lovecraft_dataset/templates/configs/4k.yaml`](lovecraft_reference/lovecraft_dataset/templates/configs/4k.yaml) -- 4K block-size child config
- [`lovecraft_dataset/templates/configs/32k.yaml`](lovecraft_reference/lovecraft_dataset/templates/configs/32k.yaml) -- 32K block-size child config
- [`lovecraft_dataset/templates/configs/long_context.yaml`](lovecraft_reference/lovecraft_dataset/templates/configs/long_context.yaml) -- parametrised `--window-size` (non-packed)
- [`lovecraft_dataset/templates/configs/long_context_packed.yaml`](lovecraft_reference/lovecraft_dataset/templates/configs/long_context_packed.yaml) -- densely-packed variant used for the memory table
- [`finetune_lovecraft/templates/project.yaml`](lovecraft_reference/finetune_lovecraft/templates/project.yaml) -- project defaults
- [`finetune_lovecraft/templates/configs/1gpu/default.yaml`](lovecraft_reference/finetune_lovecraft/templates/configs/1gpu/default.yaml) -- 4K/single-GPU training
- [`finetune_lovecraft/templates/configs/long_context.yaml`](lovecraft_reference/finetune_lovecraft/templates/configs/long_context.yaml) -- parametrised context-length training config, used to reproduce the memory table above

## Long-context generation quality experiments

Fitting a 16K-context training run into VRAM is only half the problem.  The
other half is getting the trained model to *generate coherently at 16K*.
[long_context_experiments.md](long_context_experiments.md) documents five
Mistral / Llama variants (sliding-window on / off, YaRN on / off) trained at
matched step counts and evaluated on a held-out 16K story.

Headline findings:
- **YaRN on Llama-2-7B reduces 16K perplexity by 30%** (29.7 → 20.5).
  Llama-2-7B was pretrained at 4K, so any 16K use is extrapolation, and
  YaRN's per-frequency-band rescaling directly addresses that.
- **YaRN is neutral for Mistral** — it was pretrained at 32K, so a 16K
  fine-tune isn't extrapolating in the first place.
- **Mistral's sliding window is not the main limiter** for next-token
  prediction quality at the token-level.  An early version of this
  experiment suggested it was, but that was a step-mismatch artifact; with
  step-matched training every variant shows the same story-content NLL
  spikes at the same positions, regardless of attention pattern.
- **The default tutorial LR of 3.5e-6 was severely under-trained.**  A
  binary-search LR probe established 5e-5 as stable-and-effective for
  Adafactor at this batch/seq size.  At 5e-5, loss plateaus around 2.0
  instead of 5.3 in comparable step budgets.

YaRN support was added to Forgather's rotary-embedding module as part of
this work; enable it by setting `rope_type: "yarn"` in your model's
`config.json`.
