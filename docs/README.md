# Forgather Documentation

Forgather is a configuration-driven ML framework built on template inheritance and code generation. The core abstraction is the **Project**, which encapsulates an ML experiment through a sophisticated template system.

## Quick Navigation

**Start here:**
- **[Getting Started](getting-started/README.md)** - Installation, first training run, key CLI commands
- **[Core Concepts](core-concepts/README.md)** - Configuration pipeline, projects, templates, trainers

**Configuration:**
- **[Configuration Overview](configuration/README.md)** - Template system and YAML configuration
- **[Syntax Reference](configuration/syntax-reference.md)** - Complete syntax reference for tags and directives
- **[Template Inheritance](configuration/inheritance.md)** - Inheritance patterns and block overrides
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

**Models and inference:**
- **[Model Architecture](model-architecture.md)** - Transformer module inventory, composition patterns, and optimization flags
- **[Model Conversion](guides/model-conversion.md)** - Bidirectional HuggingFace / Forgather model conversion
- **[Vocabulary and Chat Template](guides/update-vocab.md)** - Add tokens or set chat templates on existing models without conversion
- **[Fixing EOS Token Issues](guides/fixing-eos-token-issues.md)** - Diagnose and fix runaway generation after adding ChatML-style stop tokens
- **[vLLM Integration](inference/vllm_integration.md)** - Distributed inference with vLLM (currently blocked on Transformers v5)

**Guides:**
- **[Creating a Model Project](guides/creating-a-model-project.md)** - Define a custom model architecture from scratch
- **[Creating a Dataset Project](guides/creating-a-dataset-project.md)** - Load, pack, and interleave HuggingFace datasets
- **[Debugging Configuration Errors](guides/debugging.md)** - Systematic troubleshooting and common error patterns
- **[Interactive CLI](guides/interactive-cli.md)** - Interactive shell with tab completion and editor integration
- **[Evaluating Models](guides/evaluating-models.md)** - Loss/perplexity evaluation via `forgather eval`
- **[Log Analysis](logs-analysis.md)** - Training log summaries, plots, and heatmaps

## Tutorials
- **[Tiny Llama](../examples/tutorials/tiny_llama/project_index.ipynb)** - Demonstration of basic usage
- **[Projects Overview](../examples/tutorials/projects_overview/project_index.ipynb)** - Learn about the Forgather Project abstraction
- **[Project Composition](../examples/tutorials/project_composition/project_index.ipynb)** - How the template system works
- **[Dynamic LM](../examples/tutorials/dynamic_lm/dynamic_lm.ipynb)** - Demonstrates how models are dynamically composed
- **[Samantha](../examples/tutorials/samantha/README.md)** - Demonstrates how to use Forgather to finetune a 7B parameter model on the Samantha dataset
- **[H.P. Lovecraft Project](../examples/tutorials/hp_lovecraft_project/README.md)** - Learn how to create workspaces and projects, while training a model to summon the Elder Gods

## Example Project Collections
- **[Tiny Experiments](../examples/tiny_experiments/README.md)** - A collection of experiments and integration tests using (mostly) small models
- **[Dataset Projects](../examples/datasets/README.md)** - A collection of demostration dataset configurations
- **[Finetune](../examples/finetune/README.md)** - A collection of finetuning examples
- **[Tokenizers](../examples/tokenizers/README.md)** - Tokenizer definition examples
- **[Models](../examples/models/README.md)** - Example model definitions

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
├── known_issues/        # Known performance issues and workarounds
└── examples/            # Pointers to working examples
```