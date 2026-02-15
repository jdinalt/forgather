# Configuration System

Forgather's configuration system combines Jinja2 templating with YAML and custom extensions to create a powerful, reusable configuration language.

A Forgather configuration defines a collection of constructable Python objects, "targets," which can be lazily materialized or transformed into Python code. The system is designed to allow one to define new configurations by specifying how the new configuration differs from an existing one, rather than the usual "copy-and-modify" approach. Unlike most configuration systems, Forgather can handle more than just plain-old-data-types, where any native Python datatype can be specified in a configuration, including via dynamic-imports.

## Documentation

- **[Syntax Reference](syntax-reference.md)** - Complete syntax reference
- **[High-level API](project.ipynb)** - The "Project" abstraction
- **[Low-level API](low-level-api.md)** - The API upon which the 'Project' abstraction is built from
- **[Model Initialization](model-initialization.md)** - Regex-based parameter initialization system
- **[Debugging Guide](debugging.md)** - Tools and techniques for debugging configurations

## Quick Overview

### Template Inheritance
```yaml
-- extends 'base_config.yaml'
-- block trainer_section
    == super()  # Include parent content
    # Add custom configuration
    save_steps: 500
-- endblock trainer_section
```

### Custom YAML Tags
```yaml
# Create singleton objects
model: !singleton:transformers:AutoModel@my_model ["gpt2"]

# Create partial functions
optimizer: !partial:torch.optim:AdamW
    lr: 1e-3
    weight_decay: 0.01

# Variable references
learning_rate: !var "base_lr"
```

### Line Statement Syntax
```yaml
-- if condition
    # Configuration when condition is true
-- else
    # Alternative configuration
-- endif

-- set variable_name = "value"
-- for item in list
    item_{{ loop.index }}: {{ item }}
-- endfor
```
