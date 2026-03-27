# Forgather Documentation

Forgather is a configuration-driven ML framework built on template inheritance and code generation. The core abstraction is the **Project**, which encapsulates an ML experiment through a sophisticated template system.

## Quick Navigation

- **[Model Architecture](model-architecture.md)** - Transformer module inventory, composition patterns, and optimization flags (`modelsrc/transformer/`)
- **[Interactive CLI](guides/interactive-cli.md)** - Interactive CLI
- **[Training Performance Metrics](trainers/training-performance-metrics.md)** - Token throughput, FLOP tracking, and MFU
- **[DiLoCo](trainers/diloco.md)** - Distributed Local-SGD training across heterogeneous machines on LAN
- **[Project Templates](project-templates/lm-training-projects.md)** - LM Training and Auto LR project templates
- **[Configuration Overview](configuration/README.md)** - Template system and YAML configuration
- **[Syntax Reference](configuration/syntax-reference.md)** - Complete syntax reference
- **[Model Initialization](configuration/model-initialization.md)** - Regex-based parameter initialization
- **[Checkpointing](checkpointing/README.md)** - Distributed checkpoint system for multi-GPU and multi-node training
- **[High-level API](configuration/project.ipynb)** - The "Project" abstraction
- **[Low-level API](configuration/low-level-api.md)** - The API upon which the 'Project' abstraction is built from
- **[Debugging Guide](configuration/debugging.md)** - Tools and techniques for debugging configurations

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

Much of this is still under construction.

```
docs/
├── getting-started/     # Installation and tutorials (WIP)
├── core-concepts/       # Fundamental concepts (WIP)
├── project-templates/   # Reusable project templates (LM Training, Auto LR)
├── trainers/            # Training system documentation
├── checkpointing/       # Distributed checkpoint system
├── datasets/            # Data loading, packing, and preprocessing
├── configuration/       # Template and configuration system
├── inference/           # vLLM integration guide
├── development/         # Testing and development workflow
├── fused_loss/          # Fused linear cross-entropy loss
├── known_issues/        # Known performance issues and workarounds
├── guides/              # Best practices and advanced topics (WIP)
└── examples/            # Pointers to working examples
```