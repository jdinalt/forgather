# Forgather Documentation

Forgather is a configuration-driven ML framework built on template inheritance and code generation. The core abstraction is the **Project**, which encapsulates an ML experiment through a sophisticated template system.

## Quick Navigation

### New to Forgather?
- **[Getting Started](getting-started/)** - Installation, quickstart, and first project
- **[Core Concepts](core-concepts/)** - Understanding projects, templates, and configuration
- **[Examples](examples/)** - Working examples for common use cases

### Main Topics
- **[Trainers](trainers/)** - Training system with HuggingFace compatibility
- **[Models](models/)** - Dynamic model construction and customization
- **[Data](data/)** - Dataset handling and preprocessing
- **[Configuration](configuration/)** - Template system and YAML configuration

### Reference & Guides
- **[Reference](reference/)** - Complete API and template reference
- **[Guides](guides/)** - Best practices, debugging, and optimization
- **[Contributing](contributing/)** - Development and contribution guidelines

## Key Features

### Template-Based Configuration
```yaml
-- extends 'projects/base.yaml'
-- block trainer_definition
    -- include 'my_trainer_config'
-- endblock trainer_definition
```

### HuggingFace Compatibility
```python
from forgather.ml import Trainer, TrainingArguments

# Drop-in replacement for transformers.Trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(...),
    train_dataset=dataset,
)
```

### Distributed Training
```yaml
# Multi-GPU with Accelerate
-- extends 'trainers/accel_trainer.yaml'

# Pipeline parallelism for large models  
-- extends 'trainers/pipeline_trainer.yaml'
```

### Advanced Checkpointing
```yaml
trainer_args:
    save_optimizer_state: true
    save_scheduler_state: true
    resume_from_checkpoint: true
```

## Getting Help

- **Documentation Issues**: [Report documentation problems](https://github.com/anthropics/forgather/issues)
- **Feature Requests**: [Request new features](https://github.com/anthropics/forgather/issues)
- **Questions**: [Ask questions in discussions](https://github.com/anthropics/forgather/discussions)

## Documentation Structure

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

---

*This documentation covers Forgather v1.0. For older versions, see the [version archive](versions/).*