# Forgather Documentation

Forgather is a configuration-driven ML framework built on template inheritance and code generation. The core abstraction is the **Project**, which encapsulates an ML experiment through a sophisticated template system.

## Quick Navigation

**Start here:**
- **[Getting Started](getting-started/README.md)** - Installation, first training run, key CLI commands
- **[Core Concepts](core-concepts/README.md)** - Configuration pipeline, projects, templates, trainers

**Configuration:**
- **[Configuration Overview](configuration/README.md)** - Template system and YAML configuration
- **[Syntax Reference](configuration/syntax-reference.md)** - Complete syntax reference for tags and directives
- **[Model Initialization](configuration/model-initialization.md)** - Regex-based parameter initialization
- **[Debugging Guide](configuration/debugging.md)** - Tools and techniques for debugging configurations
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

## Example Project Collections
- **[Tiny Experiments](https://github.com/jdinalt/forgather/tree/main/examples/tiny_experiments)** - A collection of experiments and integration tests using (mostly) small models
- **[Dataset Projects](https://github.com/jdinalt/forgather/tree/main/examples/datasets)** - A collection of demonstration dataset configurations
- **[Finetune](https://github.com/jdinalt/forgather/tree/main/examples/finetune)** - A collection of finetuning examples
- **[Tokenizers](https://github.com/jdinalt/forgather/tree/main/examples/tokenizers)** - Tokenizer definition examples
- **[Models](https://github.com/jdinalt/forgather/tree/main/examples/models)** - Example model definitions

## Development

- **[Known Bugs](development/bugs.md)** - Known bugs in top-level modules, with corresponding xfail tests

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