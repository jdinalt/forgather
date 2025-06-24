# Core Concepts

Understanding these core concepts is essential for effectively using Forgather.

## Main Concepts

- **[Projects](projects.md)** - The central abstraction for organizing ML experiments
- **[Templates](templates.md)** - The template inheritance system for configuration reuse
- **[Configuration](configuration.md)** - Jinja2 + YAML configuration language
- **[Code Generation](code-generation.md)** - How templates become runnable code

## The Big Picture

```
Templates + Configuration → Code Generation → Executable Objects
    ↓              ↓              ↓                    ↓
Project.yaml → Preprocessing → Python Code → Model/Trainer/etc.
```

### 1. Project System
A **Project** encapsulates everything needed for an ML experiment:
- Configuration templates
- Generated code
- Training outputs
- Experiment metadata

### 2. Template Inheritance
Templates allow systematic experimentation through inheritance:
```yaml
base_config.yaml → model_specific.yaml → experiment.yaml
```

### 3. Configuration Language
Jinja2 preprocessing enables:
- Template inheritance (`-- extends`, `-- block`)
- Variable substitution (`!var`, `!calc`)
- Component factories (`!singleton`, `!partial`)
- Dynamic configuration based on conditions

### 4. Code Generation
Templates generate standalone Python code:
- Models become importable Python modules
- Configurations become executable objects
- Complete dependency tracking and serialization

## Key Benefits

### Systematic Experimentation
Compare model architectures, training procedures, and hyperparameters systematically while maintaining full reproducibility.

### Configuration as Code
Version control your entire experimental setup with git-friendly YAML files.

### Reproducible Research
Complete configuration serialization ensures experiments can be exactly reproduced.

### Scalable Workflows
From laptop prototyping to distributed cluster training with the same configuration system.

## Next Steps

1. **[Projects](projects.md)** - Learn about project structure and management
2. **[Templates](templates.md)** - Master template inheritance patterns
3. **[Configuration](configuration.md)** - Understand the configuration language
4. **[Code Generation](code-generation.md)** - See how templates become code

## Related Topics

- **[Getting Started](../getting-started/)** - Hands-on introduction
- **[Configuration Reference](../configuration/)** - Detailed configuration guide
- **[Examples](../examples/)** - Working examples of concepts