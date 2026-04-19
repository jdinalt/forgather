# Forgather Documentation

Forgather is a configuration-driven ML framework that uses template inheritance and code generation to eliminate configuration duplication and enable systematic experimentation. Instead of copying and modifying entire training scripts, you inherit from base templates and specify only what changes.

Most research ML codebases accrete: one training script becomes ten, each a near-copy with subtle differences. Every variation is expensive to try. Small bugs — a loss function wired wrong, a scheduler silently reset on resume, a CLI flag that never reached the tokenizer — hide across forks.

Source code and examples: [github.com/jdinalt/forgather](https://github.com/jdinalt/forgather)

## Quick Navigation

**Start here:**

- **[Getting Started](getting-started/README.md)** - Installation, first training run, key CLI commands
- **[Core Concepts](core-concepts/README.md)** - Configuration pipeline, projects, templates, trainers

**Configuration:**

- **[Configuration Overview](configuration/README.md)** - Template system and YAML configuration
- **[Syntax Reference](configuration/syntax-reference.md)** - Complete syntax reference for tags and directives
- **[Model Initialization](configuration/model-initialization.md)** - Regex-based parameter initialization
- **[Project Templates](project-templates/lm-training-projects.md)** - LM Training and Auto LR project templates
- **[High-level API](configuration/project.ipynb)** - The "Project" abstraction
- **[Low-level API](configuration/low-level-api.md)** - The API upon which the Project abstraction is built

**Training:**

- **[Trainer Options Reference](trainers/trainer_options.md)** - Every training-argument field and constructor parameter across all built-in trainers
- **[Pipeline Parallel](trainers/pipeline-parallel.md)** - Pipeline parallelism for consumer GPUs and limited interconnects
- **[Trainer Control](trainers/trainer-control.md)** - External control of running training jobs (save, stop, abort)
- **[Training Performance Metrics](trainers/training-performance-metrics.md)** - Token throughput, FLOP tracking, and MFU
- **[DiLoCo](trainers/diloco.md)** - Distributed Local-SGD training across heterogeneous machines on LAN
- **[FP8 Training](trainers/fp8-training.md)** - FP8 training via torchao
- **[Checkpointing](checkpointing/README.md)** - Distributed checkpoint system for multi-GPU and multi-node training
- **[Torch Titan Integration](trainers/torchtitan.md)** - Forgather integration with PyTorch's Torch Titan training framework
- **[Adafactor Triton Performance](trainers/adafactor-triton-performance.md)** - Performance analysis for the Triton-optimized Adafactor kernel

**Models and inference:**

- **[Model Architecture](guides/model-architecture.md)** - Transformer module inventory, composition patterns, and optimization flags
- **[Model Conversion](guides/model-conversion.md)** - Bidirectional HuggingFace / Forgather model conversion
- **[Vocabulary and Chat Template](guides/update-vocab.md)** - Add tokens or set chat templates on existing models without conversion
- **[Fixing EOS Token Issues](guides/fixing-eos-token-issues.md)** - Diagnose and fix runaway generation after adding ChatML-style stop tokens
- **[vLLM Integration](inference/vllm_integration.md)** - Distributed inference with vLLM (currently blocked on Transformers v5)

**Guides:**

- **[Creating a Model Project](guides/creating-a-model-project.md)** - Define a custom model architecture from scratch
- **[Model CLI Reference](guides/model-cli.md)** - `forgather model` command: construct, test, checkpoint, and use models
- **[Creating a Dataset Project](guides/creating-a-dataset-project.md)** - Load, pack, and interleave HuggingFace datasets
- **[Dataset CLI Reference](datasets/dataset-cli.md)** - `forgather dataset` command: inspect, sample, and histogram datasets
- **[Working with Tokenizer Projects](guides/working-with-tokenizer-projects.md)** - CLI commands for tokenizer projects
- **[Debugging Configuration Errors](guides/debugging.md)** - Systematic troubleshooting and common error patterns
- **[Interactive CLI](guides/interactive-cli.md)** - Interactive shell with tab completion and editor integration
- **[Evaluating Models](guides/evaluating-models.md)** - Loss/perplexity evaluation via `forgather eval`
- **[Log Analysis](guides/logs-analysis.md)** - Training log summaries, plots, and heatmaps

## Tutorials
- **[Tiny Llama](tutorials/tiny_llama/README.md)** - Demonstration of basic usage
- **[Projects Overview](tutorials/projects_overview/project_index.ipynb)** - Learn about the Forgather Project abstraction
- **[Project Composition](tutorials/project_composition/project_index.ipynb)** - How the template system works
- **[Dynamic LM](tutorials/dynamic_lm/dynamic_lm.ipynb)** - Demonstrates how models are dynamically composed
- **[Samantha](tutorials/samantha/README.md)** - Demonstrates how to use Forgather to finetune a 7B parameter model on the Samantha dataset
- **[H.P. Lovecraft Project](tutorials/hp_lovecraft_project/README.md)** - Learn how to create workspaces and projects, while training a model to summon the Elder Gods

## Featured Examples

| Journey | Project |
|---------|---------|
| Pretrain from scratch | [pretrain/small-llm](https://github.com/jdinalt/forgather/tree/main/examples/pretrain/small-llm) |
| Fine-tune a 7B model (multi-GPU) | [finetune/samantha](https://github.com/jdinalt/forgather/tree/main/examples/finetune/samantha) |
| Instruction / reasoning fine-tune | [finetune/open-orca](https://github.com/jdinalt/forgather/tree/main/examples/finetune/open-orca) |
| Long-context fine-tuning + RoPE recipes | [tutorials/hp_lovecraft_project](https://github.com/jdinalt/forgather/tree/main/examples/tutorials/hp_lovecraft_project) |
| Cut peak memory | [tiny_experiments/peak_memory](https://github.com/jdinalt/forgather/tree/main/examples/tiny_experiments/peak_memory) |
| Pick an optimizer | [tiny_experiments/optimizers](https://github.com/jdinalt/forgather/tree/main/examples/tiny_experiments/optimizers) |
| Pipeline-parallel recipes | [tiny_experiments/pipeline_parallel](https://github.com/jdinalt/forgather/tree/main/examples/tiny_experiments/pipeline_parallel) |
| Decentralised / bandwidth-limited training | [tiny_experiments/diloco](https://github.com/jdinalt/forgather/tree/main/examples/tiny_experiments/diloco) |

**[pretrain/small-llm](https://github.com/jdinalt/forgather/tree/main/examples/pretrain/small-llm)** — A 162M-parameter Llama trained from scratch on the SmolLM corpus (FineWeb-Edu + Cosmopedia) with packed sequences and Flex Attention. Ten production-ready configs covering 1× and 10× Chinchilla budgets, AdamW / Adafactor / bf16 variants. Includes reproducible Chinchilla scaling-law plots.

**[finetune/samantha](https://github.com/jdinalt/forgather/tree/main/examples/finetune/samantha)** — Fine-tune Mistral-7B or Llama-3.2-1B on the Samantha conversational dataset across every trainer backend. Configs cover single-GPU, 2/4-GPU pipeline parallel, FSDP-2, and DDP. Documented throughput (~8.9K tok/s on 4× RTX 4090). The most-referenced finetune project in the library.

**[finetune/open-orca](https://github.com/jdinalt/forgather/tree/main/examples/finetune/open-orca)** — Instruction and reasoning fine-tune on Open-Orca with ChatML-formatted evaluation prompts covering chain-of-thought math, logic puzzles, reading comprehension, and summarisation. 1B Llama 3.2 on a 1B-token budget completes in ~11 hours on 4× RTX 4090.

**[tutorials/hp_lovecraft_project](https://github.com/jdinalt/forgather/tree/main/examples/tutorials/hp_lovecraft_project)** — Fine-tune Mistral-7B / Llama-2-7B on the complete works of H.P. Lovecraft on a single 24 GB GPU, with up to 53K tokens of context. Includes a four-way RoPE comparison (plain, YaRN, Llama-3 NTK-by-parts, bumped θ) evaluating 8K-trained models out to 16K.

**[tiny_experiments/peak_memory](https://github.com/jdinalt/forgather/tree/main/examples/tiny_experiments/peak_memory)** — A systematic 9-way ablation of memory-optimisation techniques on a 1.6B model: BF16, activation checkpointing, `torch.compile`, fused optimizer step, activation-memory budget. Headline: 81% peak-memory reduction at ~2.7× throughput over the unoptimised baseline.

**[tiny_experiments/optimizers](https://github.com/jdinalt/forgather/tree/main/examples/tiny_experiments/optimizers)** — Empirical comparison of ten optimizers (Muon, Apollo, AdamW, Adafactor, SinkGD, SGD, and more) on a 30M Llama. Headline: Muon wins at small batch (eval loss 2.6778 vs AdamW 2.7392). Includes per-optimizer memory and throughput tiers.

**[tiny_experiments/pipeline_parallel](https://github.com/jdinalt/forgather/tree/main/examples/tiny_experiments/pipeline_parallel)** — Test harness and reference configs for PyTorch's pipeline-parallel schedules (GPipe, 1F1B, ZBV, interleaved), with checkpoint save/resume coverage across 2/4-GPU setups.

**[tiny_experiments/diloco](https://github.com/jdinalt/forgather/tree/main/examples/tiny_experiments/diloco)** — DiLoCo (distributed local SGD) on a 4M-parameter model. Pseudo-gradient compression, streaming-fragment overlap with the backward pass, sync and async modes. The lowest-communication-bandwidth trainer in the library.

## Example Project Collections
- **[Tiny Experiments](https://github.com/jdinalt/forgather/tree/main/examples/tiny_experiments)** - A collection of experiments and integration tests using (mostly) small models
- **[Dataset Projects](https://github.com/jdinalt/forgather/tree/main/examples/datasets)** - A collection of demonstration dataset configurations
- **[Finetune](https://github.com/jdinalt/forgather/tree/main/examples/finetune)** - A collection of finetuning examples
- **[Tokenizers](https://github.com/jdinalt/forgather/tree/main/examples/tokenizers)** - Tokenizer definition examples
- **[Models](https://github.com/jdinalt/forgather/tree/main/examples/models)** - Example model definitions

## Development

- **[API Reference](api/index.md)** - Auto-generated Python API documentation
- **[Debugging Guide](configuration/debugging.md)** - Tools and techniques for debugging configurations
- **[Known Bugs](development/bugs.md)** - Known bugs in top-level modules, with corresponding xfail tests
- **[Testing Guide](development/testing.md)** How to create and run unit tests
- **[Integration Testing](development/integration-testing.md)** How to create and run integration tests

## Getting Help

- **Documentation Issues**: [Report documentation problems](https://github.com/jdinalt/forgather/issues)
- **Feature Requests**: [Request new features](https://github.com/jdinalt/forgather/issues)
- **Questions**: [Ask questions in discussions](https://github.com/jdinalt/forgather/discussions)

## Documentation Structure

```
docs/
├── getting-started/     # Installation and first training run
├── core-concepts/       # Configuration pipeline, projects, templates
├── configuration/       # Template and configuration system
├── project-templates/   # Reusable project templates (LM Training, Auto LR)
├── trainers/            # Training system (PP, DiLoCo, control, metrics, FP8)
├── checkpointing/       # Distributed checkpoint system
├── datasets/            # Data loading, packing, and preprocessing
├── inference/           # vLLM integration guide
├── guides/              # How-to guides (models, datasets, CLI, conversion)
├── development/         # Testing and development workflow
├── fused_loss/          # Fused linear cross-entropy loss
```