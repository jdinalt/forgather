# Forgather Examples

This directory contains example projects demonstrating various Forgather capabilities and patterns.

For a curated list of the best-documented examples organized by what you
might want to do with them (pretrain, fine-tune, optimize memory, swap
optimizers, long-context training, etc.), see the **Featured Examples**
section of the [top-level README](../README.md).

## Directory Structure

- **[tutorials/](./tutorials/README.md)** - Educational projects with step-by-step learning materials (start here if you're new)
- **[pretrain/](./pretrain/README.md)** - Pretrain models from scratch
- **[finetune/](./finetune/README.md)** - Fine-tune pretrained models
- **[tiny_experiments/](./tiny_experiments/README.md)** - Small-scale ablations and experiments for testing specific features
- **[base_lm_project/](./base_lm_project/README.md)** - Bare harness that drives the raw `projects/lm_training_project.yaml` template (useful for inspecting the base template itself)
- **[datasets/](./datasets/README.md)** - Dataset projects referenced by the training recipes above
- **[models/](./models/README.md)** - Custom model-definition projects (Llama, Mistral, Qwen3, Gemma-3, DeepOne, etc.)
- **[tokenizers/](./tokenizers/README.md)** - Custom tokenizer configuration examples
- **[evaluation/](./evaluation/README.md)** - Named eval configs consumed by `forgather eval test`
- **[trainer_control/](./trainer_control/README.md)** - Minimal worked example of the trainer-control protocol
- **[torchtitan/](./torchtitan/README.md)** - Configure Torch Titan via Forgather (proof-of-concept)
- **[snippets/](./snippets/README.md)** - Small utility fragments and snippets

## Featured Examples highlights

Longer-form descriptions of the entries in the top-level README's
**Featured Examples** table. Each project's own README is still the
source of truth for commands and results; this is the "why this is
interesting" tour.

**[`pretrain/small-llm`](./pretrain/small-llm/README.md)** — a
162M-parameter Llama trained from scratch on the SmolLM corpus
(FineWeb-Edu + Cosmopedia) with packed sequences and flex-attention.
Ten production-ready configs covering 1× and 10× Chinchilla budgets,
AdamW / Adafactor / bf16 variants, and the "Canon-A" custom
architecture variant. Reproducible Chinchilla scaling-law plots via
`forgather logs plot`. Runs on the `lm_training_project.yaml` base
template.

**[`finetune/samantha`](./finetune/samantha/README.md)** — fine-tune
Mistral-7B or Llama-3.2-1B on the Samantha conversational dataset
across every trainer backend in the library. Configs cover single-GPU,
2/4-GPU pipeline parallel, FSDP-2, and DDP. Documented throughput
(~8.9K tok/s on 4× RTX 4090 pipeline) and multi-node training notes.
The most-referenced finetune project — most other recipes cross-link
to it rather than duplicating the setup.

**[`finetune/open-orca`](./finetune/open-orca/README.md)** —
instruction + reasoning fine-tune on Open-Orca, complementing the
Samantha chat-persona work. ChatML-formatted evaluation prompts cover
chain-of-thought math, logic puzzles, reading comprehension,
summarisation, and format-constrained instruction following (wired
into the textgen callback). Uses Forgather's fast iterable-dataset
loader — 1 B Llama 3.2 on a 1 B-token budget completes in ~11 hours
on 4× RTX 4090, with initialisation in seconds rather than the ~10
min a naive load would take. Headline run includes a full
inference-server eval script as an appendix.

**[`tutorials/hp_lovecraft_project`](./tutorials/hp_lovecraft_project/README.md)**
— fine-tune Mistral-7B / Llama-2-7B on the complete works of H.P.
Lovecraft on a single 24 GB GPU. Fits up to **53 K tokens** of
context at 7B. Its companion
[`long_context_experiments.md`](./tutorials/hp_lovecraft_project/long_context_experiments.md)
documents a four-way RoPE comparison (plain, YaRN, Llama-3
NTK-by-parts, bumped θ) evaluating 8K-trained models out to 16K on
held-out text. Headline: **bumping `rope_theta` to 500 000 is the
single biggest intervention** for extrapolation, and Llama-3-style
scaling adds a small further win. YaRN with a factor that doesn't
cover the deployment window is catastrophic. The doc ends with a
follow-up proposal for pretraining recipes.

**[`tiny_experiments/peak_memory`](./tiny_experiments/peak_memory/README.md)**
— a systematic 9-way ablation of memory-optimisation techniques
(BF16, activation checkpointing, `torch.compile`, fused optimizer
step, activation-memory budget) on a 1.6 B model. Headline:
**81% peak-memory reduction** (BF16 + fused checkpointing + optimizer
fusion) at ~2.7× throughput over the unoptimised baseline.
Pareto-frontier plots included.

**[`tiny_experiments/optimizers`](./tiny_experiments/optimizers/README.md)**
— empirical comparison of ten optimisers (Muon, Apollo, AdamW,
Adafactor, SinkGD, SGD, etc.) on a 30M Llama trained on the SmolLM
corpus. Headline: **Muon wins** at small batch (eval loss 2.6778 vs
AdamW 2.7392), and `beta2` scaling becomes critical at large batch.
References Marek et al. on small-batch SGD viability, the Muon paper,
Apollo, SinkGD. Includes per-optimiser memory / throughput tiers and
implementation-maturity notes.

**[`tiny_experiments/pipeline_parallel`](./tiny_experiments/pipeline_parallel/README.md)**
— test harness and reference configs for PyTorch's pipeline-parallel
schedules (GPipe, 1F1B, ZBV, interleaved), with checkpoint save/resume
coverage across 2/4-GPU setups.

**[`tiny_experiments/diloco`](./tiny_experiments/diloco/README.md)**
— DiLoCo (distributed local SGD) on a 4M-parameter model.
Pseudo-gradient compression, streaming-fragment overlap with backward
pass, sync and async modes. The lowest-communication-bandwidth
trainer in the library — pair with the pipeline-parallel recipes
above when nodes aren't co-located.
