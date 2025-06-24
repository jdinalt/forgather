# Configuration System

Forgather's configuration system combines Jinja2 templating with YAML and custom extensions to create a powerful, reusable configuration language.

## Documentation

- **[Template Syntax](template-syntax.md)** - Jinja2 template syntax and line statements
- **[YAML Tags](yaml-tags.md)** - Custom YAML tags for object construction
- **[Template Inheritance](inheritance.md)** - Template inheritance patterns and best practices
- **[Debugging Guide](debugging.md)** - Tools and techniques for debugging configurations
- **[Syntax Reference](syntax-reference.md)** - Complete syntax reference

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
model: !singleton:transformers:AutoModel@my_model
    args: ["gpt2"]

# Create partial functions
optimizer: !partial:torch.optim:AdamW
    lr: 1e-3
    weight_decay: 0.01

# Variable references
learning_rate: !var "base_lr"
max_steps: !calc "epochs * steps_per_epoch"
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

## Configuration Workflow

1. **Template Creation**: Write YAML templates with Jinja2 preprocessing
2. **Inheritance Setup**: Extend base templates and override specific sections
3. **Variable Definition**: Define reusable variables and calculations
4. **Object Construction**: Use custom tags to create Python objects
5. **Preprocessing**: Jinja2 processes templates into pure YAML
6. **Object Instantiation**: YAML loader creates Python objects

## Key Concepts

### Template Inheritance
- **Base templates**: Define common structure and defaults
- **Specific templates**: Override and extend base templates
- **Block system**: Named sections that can be overridden
- **Super calls**: Include parent template content

### Object Factory System
- **Singletons**: Create single instances with caching
- **Partials**: Create functions with preset arguments
- **Factories**: Create objects with constructor patterns
- **References**: Share objects between different parts of configuration

### Variable System
- **Template variables**: Jinja2 variables for template processing
- **YAML variables**: References to computed or external values
- **Calculations**: Dynamic computation of configuration values
- **Environment integration**: Access to environment variables

## Benefits

### Systematic Experimentation
Compare different configurations systematically while maintaining common structure.

### Reusability
Share common patterns across projects and experiments.

### Reproducibility
Complete configuration serialization ensures exact reproduction.

### Maintainability
Clear inheritance hierarchy and modular structure.

## Getting Started

1. **[Core Concepts](../core-concepts/)** - Understand the big picture
2. **[Template Syntax](template-syntax.md)** - Learn the template language
3. **[Examples](../examples/)** - See working configurations
4. **[Trainers Configuration](../trainers/configuration.md)** - Specific trainer examples

## Advanced Topics

- **[Debugging](debugging.md)** - Troubleshooting configurations
- **[Performance](../guides/performance-optimization.md)** - Optimizing configuration processing
- **[Custom Extensions](../contributing/adding-trainers.md)** - Extending the configuration system

---

*The configuration system is the heart of Forgather's template-driven approach. Master it to unlock the full power of systematic ML experimentation.*