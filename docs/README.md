# Forgather Documentation

Forgather is a configuration-driven ML framework built on template inheritance and code generation. The core abstraction is the **Project**, which encapsulates an ML experiment through a sophisticated template system.

## Quick Navigation

- **[Interactive CLI](guides/interactive-cli.md)** - Interactive CLI
- **[Configuration Overview](configuration/README.md)** - Template system and YAML configuration
- **[Syntax Reference](configuration/syntax-reference.md)** - Complete syntax reference
- **[High-level API](configuration/project.ipynb)** - The "Project" abstraction
- **[Low-level API](configuration/low-level-api.md)** - The API upon which the 'Project' abstraction is built from
- **[Debugging Guide](configuration/debugging.md)** - Tools and techniques for debugging configurations

## Tutorials
- **[Tiny Llama](../examples/tutorials/tiny_llama/project_index.ipynb)** - Demonstration of basic usage
- **[Projects Overview](../examples/tutorials/projects_overview/project_index.ipynb)** - Learn about the Forgather Project abstraction
- **[Project Composition](../examples/tutorials/project_composition/project_index.ipynb)** - How the template system works
- **[Dynamic LM](../examples/tutorials/dynamic_lm/dynamic_lm.ipynb)** - Demonstrates how models are dynamically composed

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
├── getting-started/     # Installation and tutorials
├── core-concepts/       # Fundamental concepts
├── trainers/           # Training system documentation
├── models/             # Model construction and templates
├── data/               # Data handling and preprocessing
├── configuration/      # Template and configuration system
├── examples/           # Working examples and tutorials
├── reference/          # Complete API and template reference
├── guides/             # Best practices and advanced topics
└── contributing/       # Development and contribution
```