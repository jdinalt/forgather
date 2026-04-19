# Forgather

Forgather is a configuration-driven ML framework that uses template
inheritance and code generation to eliminate configuration duplication
and enable systematic experimentation. Instead of copying and modifying
entire training scripts, you inherit from base templates and specify
only what changes.

> 📚 **Documentation:** [`docs/README.md`](./docs/README.md) is the
> complete documentation index. New users should head straight to
> **[Getting Started](./docs/getting-started/README.md)**.

## Why Forgather?

Most research ML codebases accrete: one training script becomes ten
training scripts, each a near-copy of the others with subtle
differences. Every variation is expensive to try. Small bugs -- a loss
function wired wrong, a scheduler silently reset on resume, a CLI flag
that didn't actually reach the tokenizer -- hide across forks.

Forgather addresses this with three layers:

- **Template inheritance.** A project config extends a parent; both
  are plain YAML with Jinja2 preprocessing. Overrides are explicit
  (`-- set ns.seq_len = 16384`), and every knob is documented on the
  parent.
- **Portable model source.** Model construction emits standalone
  Python source into the training run's output directory, and the
  training run then loads that source. You can zip that directory
  and load the model on another machine with just `transformers`
  installed -- no Forgather dependency. (The trainer, optimiser,
  datasets, and callbacks materialise straight from the config
  graph; the Python export is specifically for the model so it can
  live outside Forgather's runtime.)
- **Live job control.** Running training jobs register a small
  control endpoint. From another shell you can save a checkpoint on
  demand, gracefully stop a run, or abort a failed experiment --
  works uniformly across DDP, FSDP2, and pipeline-parallel jobs.

## Key Benefits

- **No config duplication** -- inherit from base templates and override only what changes; every knob is an explicit overridable block.
- **Types as hyperparameters** -- swap optimizers, models, datasets, trainers, and samplers directly in YAML via custom tags (`!partial`, `!factory`, `!singleton`). No Python edits required to try a new optimizer or a different layer norm.
- **Full reproducibility** -- automatic snapshots of configs and generated code with each run.
- **Standalone generated models** -- the PyTorch source written into each run's `output_models/` directory loads with plain `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`. No Forgather dependency at inference time; zip and ship.
- **Pipeline-parallel trainer for bandwidth-limited environments** -- requires *dramatically* less cross-device communication than DDP or FSDP. Forgather has trained a 7B model across two machines linked only by 1 Gbit Ethernet with bandwidth to spare, and the same design means it scales well on consumer hardware where FSDP stalls on PCIe. Should extend to more than two nodes on the same fabric.
- **Low-memory training suite** -- gradient checkpointing, CPU activation offloading, fused optimizer step, fused linear+cross-entropy loss (Liger / Apple CCE / torch.compile), Triton Adafactor with stochastic bf16 rounding, FP8 via torchao. Full-parameter (not LoRA) finetuning of 7B models at up to ~53K context on a single 24 GB GPU.
- **Fast dataset loading** -- large HF datasets (e.g. C4) normally take 10-20 minutes to open. Forgather indexes the Arrow files on first use and is ready in a few seconds after that, with stateful resume from any position.
- **Packed sequences + Flex Attention** -- pack multiple documents into every batch with explicit document-boundary tracking; flex-attention masks enforce no cross-document attention, so packing doesn't cost attention quality. Combined with the fused-loss kernel, this is a large throughput win on small-document corpora.
- **Live job control** -- save, stop, save-stop, or abort running training jobs from another shell. Coordinates automatically across DDP / FSDP-2 / pipeline-parallel workers.
- **Extensive examples and documentation** -- tutorials covering every step from scratch, pretraining recipes from 4M to ~1B params, finetune recipes for Llama-2/3, Mistral, Qwen3, Gemma-3, and a growing set of ablation-focused experiments. Every major subsystem has its own docs page.
- **Speed-optimised** -- meaningful effort has gone into wall-clock performance (Triton kernels, packed iterators, SDPA-backend selection, `torch.compile` everywhere it helps). A formal head-to-head benchmark against alternative frameworks is still to be done.

## News

- **Apr 2026** -- New recommended base template **[`projects/lm_training_project.yaml`](templatelib/examples/projects/lm_training_project.yaml)** (pretraining and finetuning) and **[`projects/finetune_v2.yaml`](templatelib/examples/projects/finetune_v2.yaml)** (finetune-specific layer). Token-budget-driven step computation, automatic batch-size-aware LR scaling, WSD scheduler, fully-documented parameter surface. Replaces several drifting older base templates.
- **Apr 2026** -- **[Tiny Llama](./examples/tutorials/tiny_llama/README.md)** and **[H.P. Lovecraft](./examples/tutorials/hp_lovecraft_project/README.md)** tutorials rewritten around the v2 templates as README-first (no Jupyter required). Tiny Llama covers the full train → monitor → control → eval → inference → export flow.
- **Mar 2026** -- **YaRN** and **Llama-3 style RoPE scaling** in the rotary-embeddings module. Configure via `rope_parameters` with `rope_type: yarn` or `rope_type: llama3`.
- **Mar 2026** -- **`forgather eval test`** -- run any named eval config against a trained model and write markdown + JSON results to `{model}/evals/`.
- **Feb 2026** -- **Trainer job control** (`forgather control list / status / save / stop / save-stop / abort`). Distributed-safe; works across DDP and pipeline-parallel runs.
- **Feb 2026** -- **Sharded-checkpoint abstraction** with explicit state-sharing patterns (GLOBAL / PER_RANK / REPLICATED / PER_GROUP / PER_NODE) and per-checkpoint manifests. See [`docs/checkpointing/`](./docs/checkpointing/).
- **Dec 2025** -- **Fused linear-cross-entropy loss** ([paper](https://arxiv.org/abs/2411.09009)) -- Liger / Apple CCE / PyTorch-compiled implementations. Large peak-memory reduction for training with big vocabularies. Example: [`examples/finetune/samantha/templates/configs/llama3_1b/1gpu_default.yaml`](./examples/finetune/samantha/templates/configs/llama3_1b/1gpu_default.yaml).
- **Dec 2025** -- **Triton Adafactor** -- [`src/forgather/ml/optim/adafactor_triton.py`](./src/forgather/ml/optim/adafactor_triton.py) -- lower peak memory and faster training than the reference Adafactor.
- **Dec 2025** -- Inference server supports `device_map="auto"`, so models too large for one GPU can be sharded across all visible GPUs for serving.
- **Nov 2025** -- Overhauled [model-conversion tool](./tools/convert_model/README.md) with support for Llama (incl. RoPE scaling, tied embeddings), Mistral, Qwen3, Gemma-3.
- **Nov 2025** -- **[OpenAssistant dataset](./examples/datasets/OpenAssistant/README.md)** -- high-quality example of a custom dataset that generates examples on the fly (quality-weighted sampling from conversation trees, sequence packing, multi-language, deterministic). [Demo finetune project](./examples/finetune/openassistant/README.md).
- **Nov 2025** -- Support for [packed sequences](./docs/datasets/sequence-packing.md) and [Flex Attention](https://pytorch.org/blog/flexattention/); KV cache in models.
- **[Torch Titan integration](./examples/torchtitan/README.md)** -- Use Forgather to configure Torch Titan.

*vLLM integration is currently broken due to Forgather's move to Transformers v5, which vLLM does not yet support. Upstream is working on v5 compatibility; we'll re-enable the integration once that lands.*

## Table of Contents

- [Why Forgather?](#why-forgather)
- [Key Benefits](#key-benefits)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Core Concepts](#core-concepts)
- [Learning Forgather](#learning-forgather)
- [Featured Examples](#featured-examples)
- [Command Overview](#command-overview)
- [Project Structure](#project-structure)
- [Development Setup](#development-setup)
- [Contributing](#contributing)

## Quick Start

Full install walkthrough and first-training-run tutorial:
**[docs/getting-started/README.md](./docs/getting-started/README.md)**.

The short version, assuming Python 3.12+ and
`build-essential` / `python3-dev` / `graphviz` already installed.
An NVIDIA GPU is strongly recommended but not required -- CPU-only
training works (Tiny Llama runs end-to-end on a Chromebook in a day);
it's just ~two to three orders of magnitude slower than the same
workload on a 4090. Non-CUDA accelerators (Intel, AMD) may work in
theory -- we've avoided hard CUDA dependencies -- but have not been
tested outside of CUDA and CPU.

```bash
# Create and activate a virtual environment first -- most modern distros
# refuse `pip install` into the system Python (PEP 668).  venv, conda, or
# uv all work; pick one:
python3.12 -m venv ~/venvs/forgather
source ~/venvs/forgather/bin/activate

git clone https://github.com/jdinalt/forgather.git
cd forgather
pip install -e .
pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"

cd examples/tutorials/tiny_llama
forgather -t v2.yaml train        # ~10 min on a single RTX 4090
```

See [`examples/tutorials/tiny_llama/README.md`](./examples/tutorials/tiny_llama/README.md)
for the full "train → monitor → control → eval → inference → export"
walkthrough, or [`docs/getting-started/README.md`](./docs/getting-started/README.md)
for the install details.

## Key Features

### Template inheritance

Create new experiments by inheriting from existing configs and
specifying only the differences:

```yaml
-- extends 'base_experiment.yaml'

[config_metadata]
    == super()
    -- set ns.seq_len = 16384        # longer context

[optimizer]
    == super()
    lr: 1.0e-3                       # override the LR, keep everything else
```

### Dynamic type system

Use any Python class or function directly in configs. Custom YAML tags
(`!partial`, `!factory`, `!singleton`, `!var`, `!call`) describe how to
build live Python objects from the parsed graph:

```yaml
optimizer: !partial:torch.optim.AdamW
    lr: 1.0e-3
    weight_decay: 0.01

[layer_factory]
# Experiment: swap PreLayerNorm for PostLayerNorm
layer_factory: &layer_factory !partial:.post_ln_layer:PostLNLayer@layer_factory
    feedforward_factory: *feedforward_factory
    attention_factory: *attention_factory
    norm_factory: *layer_norm_factory
    dropout: !var "layer_dropout"
    residual_dropout: !var "residual_dropout"
```

See [Syntax Reference](./docs/configuration/syntax-reference.md) for
the full list of line statements and YAML tags.

### Code generation (export, not an interpreter step)

**At runtime Forgather materialises the parsed node graph directly
into Python objects -- no intermediate code-generation phase is
involved.** Python-source export is a separate function with two
uses:

1. **Custom model source.** When you construct a model for the first
   time, Forgather writes the equivalent Python source into the
   training run's output directory. The generated code has no
   Forgather dependency: any HF-compatible consumer (`transformers`,
   vLLM, etc.) can load the model without Forgather installed. This
   is what `trust_remote_code=True` resolves.

   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained(
       "output_models/v2",
       trust_remote_code=True,
   )
   ```

   If you want plain-HF weights without `trust_remote_code`, convert
   via `forgather convert --reverse --model-type llama <src> <dst>`.
   The converter supports Llama, Mistral, Qwen3, and Gemma-3.

2. **Config debugging / pedagogy.** `forgather code --target X`
   prints the Python equivalent of any node in your config graph --
   useful when you want to understand what a complex `!partial` / 
   `!factory` chain actually constructs, or to see how template
   overrides materialise.  Not used by training itself.

### Built-in training infrastructure

- **`basic` trainer** -- single-GPU, the fast path for small-model experiments.
- **`ddp` trainer** -- multi-GPU DistributedDataParallel, with optional
  **PostLocalSGD** for reduced communication frequency.
- **`fsdp2` trainer** -- FSDP-2 sharded data parallel, with configurable
  parameter/reduce/buffer dtypes and CPU offload.
- **`pipeline` trainer** -- pipeline parallelism. GPipe, 1F1B,
  Interleaved-1F1B, and zero-bubble schedules, multi-stage support,
  per-stage `torch.compile`. Designed for bandwidth-limited setups
  (multi-node over Ethernet or consumer GPUs over PCIe).
- **DiLoCo** -- distributed local SGD for very-low-bandwidth
  multi-machine training. Sync and async modes, Delayed Nesterov
  momentum, dynamic local-update adaptation. See
  [`docs/trainers/diloco.md`](./docs/trainers/diloco.md).
- **`AccelTrainer`** -- legacy Hugging Face Accelerate wrapper, kept
  for a few older examples. Maintenance is low priority; prefer
  `ddp` / `fsdp2` / `pipeline` for new work.
- A small **Transformers-Trainer compatibility shim** also exists for
  pre-Forgather-trainer scripts. Legacy; low priority.

### Optimizers and precision

- **Adafactor with bf16 stochastic rounding** *(the distinctive one)*
  -- Forgather's Triton Adafactor combines factored second-moment
  estimation with per-parameter SR applied to bf16 weight updates, in
  a single fused kernel. Stochastic rounding is critical for
  *pure-bf16* training (no fp32 master weights) -- without it,
  small updates below the bf16 precision step are systematically
  rounded to zero and the model's weight norms slowly drift.
  To our knowledge this is the only Adafactor+SR implementation
  available, and it runs faster in our tests than any other Adafactor
  we've benchmarked.  File:
  [`src/forgather/ml/optim/adafactor_triton.py`](./src/forgather/ml/optim/adafactor_triton.py).
- **AdamW with SR** -- Forgather ships a stochastic-rounding AdamW
  ([`src/forgather/ml/optim/adamw.py`](./src/forgather/ml/optim/adamw.py))
  that makes a real difference in pure-bf16 runs, but if you want
  quantized state on top of SR, prefer **`torchao.optim.AdamW4bit`**
  (4-bit optimizer state, SR-enabled via `stochastic_rounding=True`).
  Example config:
  [`examples/finetune/samantha/templates/configs/llama3_1b/ddp_adam4bit.yaml`](./examples/finetune/samantha/templates/configs/llama3_1b/ddp_adam4bit.yaml).
- **Apollo / Apollo-mini** ([arXiv:2412.05270](https://arxiv.org/abs/2412.05270))
  -- low-rank gradient projection for SGD-level memory with
  AdamW-level performance.  **Experimental** -- interesting for small
  ablations and memory-constrained single-GPU runs, not
  production-hardened.
- **Other optimizers** -- SinkGD
  ([arXiv:2502.06742](https://arxiv.org/abs/2502.06742), stateless
  matrix normalization), SGD, Muon (see the optimizer-comparison
  experiment), plus a regex-based `multiopt` helper for
  per-parameter-group optimizer assignment.
- **FP8 via torchao** -- adapters for `tensorwise` / `rowwise` /
  `rowwise_with_gw_hp` recipes, orthogonal to bf16 mixed precision.
- **Mixed precision** -- bf16 (default) and fp16 (with GradScaler);
  TF32 matmul controls; SDPA backend selection (flash / mem-efficient
  / math); FP8 via torchao (`tensorwise`, `rowwise`,
  `rowwise_with_gw_hp` recipes).
- **Learning-rate schedulers** -- Warmup-Stable-Decay, Cosine,
  Infinite-LR, all with configurable warmup / decay budgets in tokens
  or steps.

### Distributed checkpointing

**Model parameters are HF-compatible.** The weight shards are written
as a standard Hugging Face Safetensors layout (a `*.safetensors` shard
set plus a `pytorch_model.bin.index.json` / `model.safetensors.index.json`
manifest), not a bespoke Forgather format. That's the critical part:
any tool that reads an HF checkpoint -- `transformers`, vLLM,
llama.cpp conversion, remote eval harnesses -- can read the trained
model. Combined with `forgather checkpoint link` (which symlinks the
latest checkpoint's shards up into the model directory), a plain
`AutoModelForCausalLM.from_pretrained("output_models/my_run")` works
without `trust_remote_code` once the model has been converted to a
canonical HF architecture via `forgather convert --reverse`.

The *rest* of the checkpoint (optimizer state, LR-scheduler state,
dataset iterator state, per-rank RNG state, trainer step counter,
manifest) is Forgather-specific -- it has to be, since it encodes
Forgather-specific trainer internals -- and is used only by Forgather
itself for resume. So "zip and ship" to another framework is
supported for the *model*; full-state resume is Forgather-only.

The Forgather coordination layer sits *above* the on-disk format:
**explicit state-sharing patterns**. Every checkpoint component
declares whether it's `GLOBAL` (rank-0 only), `PER_RANK`,
`REPLICATED` (across DDP replicas), `PER_GROUP` (within a PP / TP
group), or `PER_NODE`. Coordination barriers and load paths are
derived from those declarations, so pipeline-parallel and FSDP-2 runs
checkpoint correctly without per-trainer custom code.

Each checkpoint also writes a JSON manifest recording every
component's size, sharing pattern, and origin ranks. Resume is
partial by default (a missing optional component warns instead of
failing). Optional **replication validation** (NONE / QUICK / TENSOR
/ FULL) catches DDP-synchronisation bugs by hashing parameters across
replicas post-save.

See [`docs/checkpointing/`](./docs/checkpointing/) for the full
abstraction.

## Core Concepts

### Projects

Every Forgather experiment is a **Project** with this structure:

```
my_project/
├── meta.yaml              # Project metadata
├── templates/
│   ├── project.yaml       # Project-wide defaults
│   └── configs/           # Experiment configurations
│       ├── baseline.yaml
│       └── experiment_a.yaml
├── output_models/         # Generated code + runs (per config)
└── project_index.ipynb    # Optional interactive notebook
```

A **workspace** groups related projects and centralises template
search paths. Use `forgather ws create` to scaffold one and
`forgather project create` to add projects to it.

### Template language

Forgather uses **Jinja2 + YAML** with custom syntax:

- `-- extends 'template.yaml'` -- template inheritance (single parent)
- `[block_name]` -- named override-able sections
- `== super()` -- include parent's version of the current block
- `-- set ns.var = value` -- set a variable in the namespace
- `-- include 'template.yaml'` -- include template content inline
- `#---- inline.template.name ----` -- split a document into multiple templates
- `!partial:module:Class` / `!factory:...` / `!singleton:...` -- construct Python objects
- `!var "name"` -- variable references

### Config pipeline

Every config goes through the same pipeline, and each intermediate
step is inspectable:

```
Templates → YAML → Node Graph → Python Objects
                       │
                       └──> (optional) Python source code
                            - model source export (for HF
                              trust_remote_code loading)
                            - debugging / pedagogy
```

Forgather materialises the node graph *directly* into Python objects
at runtime; the Python-source path is a separate export, not an
intermediary step. Model construction uses the export path so the
resulting model is framework-portable; everything else (trainer,
optimiser, dataset, callbacks) is built by walking the graph.

Inspection commands:

```bash
forgather -t config.yaml pp                      # Preprocess Jinja2 → YAML
forgather -t config.yaml graph --format yaml     # Parsed node graph
forgather -t config.yaml targets                 # Constructable objects in the graph
forgather -t config.yaml code --target model     # Python-source export of a target (debug / model export)
forgather -t config.yaml construct --target model --call
                                                 # Materialise and show the constructed object
```

When you hit a config bug, start with `forgather ls -d` (dumps the
preprocessed file with YAML errors, or the Jinja2 error if
preprocessing itself failed), then escalate to `pp --debug` (dumps
every template in the chain).

## Learning Forgather

### Recommended path

1. **[examples/tutorials/tiny_llama](./examples/tutorials/tiny_llama)**
   -- trains a 5M-param Llama in ~10 minutes; covers config anatomy,
   dynamic CLI args, monitoring, control, eval, inference, and
   exporting to plain HF format. Start here.
2. **[examples/tutorials/projects_overview](./examples/tutorials/projects_overview)**
   -- how Forgather's multi-project layout is organised.
3. **[examples/tutorials/project_composition](./examples/tutorials/project_composition)**
   -- cross-project composition (datasets / models / evals as
   independent projects that reference each other).
4. **[examples/tutorials/hp_lovecraft_project](./examples/tutorials/hp_lovecraft_project)**
   -- fine-tune Mistral-7B / Llama-2-7B on the complete works of
   H.P. Lovecraft on a single 24 GB GPU. Long-context (up to 53K
   tokens), YaRN, gradient checkpointing, activation offloading.

### Interactive shell

```bash
forgather -i
```

Drops you into a shell where every subcommand works without the
`forgather` prefix (so `pp`, `ls`, `train` instead of `forgather pp`,
etc.). Convenient for quick iteration inside a single project.

## Featured Examples

Forgather ships with a library of worked examples that go well beyond
the tutorials. The ones below are the best starting points for each
journey — each has a detailed README with reproducible commands and,
where relevant, a headline result. For the full directory map, see
[`examples/README.md`](./examples/README.md).

| Journey | Project |
|---|---|
| Pretrain from scratch | [`examples/pretrain/small-llm`](./examples/pretrain/small-llm) |
| Fine-tune a 7B model (multi-GPU) | [`examples/finetune/samantha`](./examples/finetune/samantha) |
| Instruction / reasoning fine-tune | [`examples/finetune/open-orca`](./examples/finetune/open-orca) |
| Long-context fine-tuning + RoPE recipes | [`examples/tutorials/hp_lovecraft_project`](./examples/tutorials/hp_lovecraft_project) |
| Cut peak memory | [`examples/tiny_experiments/peak_memory`](./examples/tiny_experiments/peak_memory) |
| Pick an optimizer | [`examples/tiny_experiments/optimizers`](./examples/tiny_experiments/optimizers) |
| Pipeline-parallel recipes | [`examples/tiny_experiments/pipeline_parallel`](./examples/tiny_experiments/pipeline_parallel) |
| Decentralised / bandwidth-limited training | [`examples/tiny_experiments/diloco`](./examples/tiny_experiments/diloco) |

### Highlights

**[`pretrain/small-llm`](./examples/pretrain/small-llm)** -- a
162M-parameter Llama trained from scratch on the SmolLM corpus
(FineWeb-Edu + Cosmopedia) with packed sequences and flex-attention.
Ten production-ready configs covering 1× and 10× Chinchilla budgets,
AdamW / Adafactor / bf16 variants, and the "Canon-A" custom
architecture variant. Reproducible Chinchilla scaling-law plots via
`forgather logs plot`. Runs on the `lm_training_project.yaml` base
template.

**[`finetune/samantha`](./examples/finetune/samantha)** -- fine-tune
Mistral-7B or Llama-3.2-1B on the Samantha conversational dataset
across every trainer backend in the library. Configs cover single-GPU,
2/4-GPU pipeline parallel, FSDP-2, and DDP. Documented throughput
(~8.9K tok/s on 4× RTX 4090 pipeline) and multi-node training notes.
The most-referenced finetune project -- most other recipes cross-link
to it rather than duplicating the setup.

**[`finetune/open-orca`](./examples/finetune/open-orca)** --
instruction + reasoning fine-tune on Open-Orca, complementing the
Samantha chat-persona work. The companion to Samantha for learners:
ChatML-formatted evaluation prompts covering chain-of-thought math,
logic puzzles, reading comprehension, summarisation, and
format-constrained instruction following (wired into the textgen
callback). Uses Forgather's fast iterable-dataset loader -- 1 B Llama
3.2 on a 1 B-token budget completes in ~11 hours on 4× RTX 4090,
with initialisation in seconds rather than the ~10 min a naive load
would take. Headline run includes a full inference-server eval
script as an appendix.

**[`tutorials/hp_lovecraft_project`](./examples/tutorials/hp_lovecraft_project)**
-- fine-tune Mistral-7B / Llama-2-7B on the complete works of H.P.
Lovecraft on a single 24 GB GPU. Fits up to **53 K tokens** of
context at 7B. Its companion
[`long_context_experiments.md`](./examples/tutorials/hp_lovecraft_project/long_context_experiments.md)
documents a four-way RoPE comparison (plain, YaRN, Llama-3
NTK-by-parts, bumped θ) evaluating 8K-trained models out to 16K on
held-out text. Headline: **bumping `rope_theta` to 500 000 is the
single biggest intervention** for extrapolation, and Llama-3-style
scaling adds a small further win. YaRN with a factor that doesn't
cover the deployment window is catastrophic. The doc ends with a
follow-up proposal for pretraining recipes.

**[`tiny_experiments/peak_memory`](./examples/tiny_experiments/peak_memory)**
-- a systematic 9-way ablation of memory-optimisation techniques
(BF16, activation checkpointing, `torch.compile`, fused optimizer
step, activation-memory budget) on a 1.6 B model. Headline:
**81% peak-memory reduction** (BF16 + fused checkpointing + optimizer
fusion) at ~2.7× throughput over the unoptimised baseline.
Pareto-frontier plots included.

**[`tiny_experiments/optimizers`](./examples/tiny_experiments/optimizers)**
-- empirical comparison of ten optimisers (Muon, Apollo, AdamW,
Adafactor, SinkGD, SGD, etc.) on a 30M Llama trained on the SmolLM
corpus. Headline: **Muon wins** at small batch (eval loss 2.6778 vs
AdamW 2.7392), and `beta2` scaling becomes critical at large batch.
References Marek et al. on small-batch SGD viability, the Muon paper,
Apollo, SinkGD. Includes per-optimiser memory / throughput tiers and
implementation-maturity notes.

**[`tiny_experiments/pipeline_parallel`](./examples/tiny_experiments/pipeline_parallel)**
-- test harness and reference configs for PyTorch's pipeline-parallel
schedules (GPipe, 1F1B, ZBV, interleaved), with checkpoint save/resume
coverage across 2/4-GPU setups.

**[`tiny_experiments/diloco`](./examples/tiny_experiments/diloco)**
-- DiLoCo (distributed local SGD) on a 4M-parameter model. Pseudo-
gradient compression, streaming-fragment overlap with backward pass,
sync and async modes. The lowest-communication-bandwidth trainer in
the library -- pair with the pipeline-parallel recipes above when
nodes aren't co-located.

### Building your own

- **Scaffold a new project** with `forgather project create` (inside
  an existing workspace) or `forgather ws create` (a brand-new
  workspace). These commands generate a minimum-working `meta.yaml`
  + `templates/` tree that extends the recommended base templates.
  Full walk-through: the [Tiny Llama tutorial](./examples/tutorials/tiny_llama/README.md).
- **[`examples/base_lm_project`](./examples/base_lm_project)** --
  a bare harness that drives the raw `projects/lm_training_project.yaml`
  template with no project-specific overrides. Useful for inspecting
  what the base template does on its own, and for debugging changes
  to the base template itself, but not a typical starting point for
  new work.
