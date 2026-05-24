# Release Notes

Per-release notes for Forgather. The most recent release is at the top.

- **[1.2.1](v1.2.1.md)** — May 2026. Docs-only patch: top-level
  `README.md` retargeted at the local-LLM / hobbyist / limited-hardware
  audience; News history moved into this release-notes tree.
- **[1.2.0](v1.2.0.md)** — May 2026. Headline is multi-node training:
  forgather server in cluster mode, a `forgather cluster` fan-out CLI,
  and TLS / mTLS across all three servers. Also includes:
    - dataset server with stateful resume and cluster auto-routing;
    - torchao QAT + PTQ unified under `forgather finalize --quantize`;
    - `server_config.yaml` + auto-start services and in-place server
      restart;
    - distributable runtime Docker image;
    - DGX Spark (GB10, aarch64) bring-up.

## Pre-1.2.0 highlights

Chronological list of notable changes that landed on `main` before
Forgather started cutting versioned releases. Newest first. Paths and
links are point-in-time references — some have been renamed or
restructured since.

- **Apr 2026** — **Forgather server**: web frontend over the CLI's APIs.
  Project browsing, a GPU-aware job queue, live job cards with TTY +
  training pills, an in-browser editor for templates and arbitrary text
  files (Forgather YAML+Jinja2 syntax highlighting), and a chat client
  against served inference jobs. End-to-end tour:
  [walkthrough](../guides/forgather-server-walkthrough.md).
- **Apr 2026** — New recommended base template
  **`projects/lm_training_project.yaml`** (pretraining and finetuning) and
  **`projects/finetune_v2.yaml`** (finetune-specific layer).
  Token-budget-driven step computation, automatic batch-size-aware LR
  scaling, WSD scheduler, fully-documented parameter surface. Replaces
  several drifting older base templates.
- **Apr 2026** — **Tiny Llama** and **H.P. Lovecraft** tutorials
  rewritten around the v2 templates as README-first (no Jupyter
  required). Tiny Llama covers the full
  train → monitor → control → eval → inference → export flow;
  Lovecraft covers long-context fine-tuning with RoPE scaling.
- **Mar 2026** — **YaRN** and **Llama-3 style RoPE scaling** in the
  rotary-embeddings module. Configure via `rope_parameters` with
  `rope_type: yarn` or `rope_type: llama3`.
- **Mar 2026** — **`forgather eval test`** — run any named eval config
  against a trained model and write markdown + JSON results to
  `{model}/evals/`.
- **Feb 2026** — **Trainer job control** (`forgather control list /
  status / save / stop / save-stop / abort`). Distributed-safe; works
  across DDP and pipeline-parallel runs.
- **Feb 2026** — **Sharded-checkpoint abstraction** with explicit
  state-sharing patterns (GLOBAL / PER_RANK / REPLICATED / PER_GROUP /
  PER_NODE) and per-checkpoint manifests. See
  [`docs/checkpointing/`](../checkpointing/README.md).
- **Dec 2025** — **Fused linear-cross-entropy loss**
  ([paper](https://arxiv.org/abs/2411.09009)) — Liger / Apple CCE /
  PyTorch-compiled implementations. Large peak-memory reduction for
  training with big vocabularies.
- **Dec 2025** — **Triton Adafactor** — lower peak memory and faster
  training than the reference Adafactor.
- **Dec 2025** — Inference server supports `device_map="auto"`, so
  models too large for one GPU can be sharded across all visible GPUs
  for serving.
- **Nov 2025** — Overhauled model-conversion tool with support for
  Llama (incl. RoPE scaling, tied embeddings), Mistral, Qwen3, Gemma-3.
- **Nov 2025** — **OpenAssistant dataset** — high-quality example of a
  custom dataset that generates examples on the fly (quality-weighted
  sampling from conversation trees, sequence packing, multi-language,
  deterministic), with a companion demo finetune project.
- **Nov 2025** — Support for
  [packed sequences](../datasets/sequence-packing.md) and
  [Flex Attention](https://pytorch.org/blog/flexattention/); KV cache
  in models.
- **Sep 2025** — **Torch Titan integration** — use Forgather to
  configure Torch Titan.
