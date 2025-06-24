# Reference Documentation

Complete technical reference for all Forgather APIs, templates, and tools.

## API Reference

- **[Trainers](api/trainers.md)** - Complete trainer API documentation
- **[Models](api/models.md)** - Model construction and factory APIs
- **[Configuration](api/config.md)** - Configuration system APIs
- **[Utilities](api/utils.md)** - Utility functions and helpers

## Template Library

- **[Trainer Templates](templates/trainers.md)** - Built-in trainer templates
- **[Model Templates](templates/models.md)** - Model architecture templates
- **[Dataset Templates](templates/datasets.md)** - Data loading templates

## Command Line Interface

- **[CLI Reference](cli.md)** - Complete command-line tool documentation

## Quick Reference

### Class Hierarchy
```
AbstractBaseTrainer
└── ExtensibleTrainer
    └── BaseTrainer
        ├── Trainer
        ├── AccelTrainer
        └── PipelineTrainer
```

### Configuration Tags
```yaml
!singleton:module:Class@name    # Singleton factory
!partial:module:function        # Partial function
!factory:call                   # Factory with call
!var "variable_name"            # Variable reference
!calc "expression"              # Calculated expression
```

### Template Syntax
```yaml
-- extends 'base_template.yaml'
-- block section_name
    == super()                  # Include parent content
    # Custom content here
-- endblock section_name
```

## API Quick Access

### Most Common APIs

#### Project Loading
```python
from forgather import Project

proj = Project("config.yaml")
trainer = proj("trainer")
model = proj("model")
```

#### Trainer Creation
```python
from forgather.ml import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(...),
    train_dataset=dataset,
)
```

#### Configuration Loading
```python
config = proj.environment.load("templates/config.yaml")
trainer_args = proj("trainer_args")
```

### Configuration Patterns

#### Basic Training
```yaml
-- extends 'trainers/trainer.yaml'
-- block trainer_args
    output_dir: "./output"
    per_device_train_batch_size: 16
-- endblock trainer_args
```

#### Multi-GPU Training
```yaml
-- extends 'trainers/accel_trainer.yaml'
-- block trainer_args
    == super()
    accelerator_args:
        device_placement: true
-- endblock trainer_args
```

#### Custom Components
```yaml
optimizer: &optimizer !partial:torch.optim:AdamW
    lr: 1e-3
    weight_decay: 0.01

trainer: !singleton:forgather.ml.trainer:Trainer
    optimizer_factory: *optimizer
```

## Reference Sections

### By Topic
- **Training**: Trainer classes, arguments, callbacks
- **Models**: Model construction, templates, components
- **Data**: Datasets, collators, preprocessing
- **Configuration**: Template syntax, YAML tags, variables
- **Distributed**: Multi-GPU, pipeline parallelism, communication

### By User Type
- **API Users**: Python APIs, class references, method signatures
- **Template Users**: YAML syntax, inheritance, configuration patterns
- **Contributors**: Internal APIs, extension points, development tools

### By Complexity
- **Basic**: Core classes and common patterns
- **Advanced**: Distributed training, custom components
- **Expert**: Internal APIs, framework extension

## Navigation Tips

### Finding Information
1. **Know what you want?** Use the quick reference above
2. **Learning a topic?** Start with the overview section
3. **Need examples?** Check the [examples](../examples/) section
4. **Debugging?** See the [guides](../guides/) section

### Cross-References
- Links to related concepts and examples
- "See also" sections for related topics
- Code examples in context

### Version Information
All reference documentation is for the current version. Historical versions available in the [version archive](versions/).

---

*This reference is auto-generated from code and manually curated. Report inaccuracies in [GitHub issues](https://github.com/anthropics/forgather/issues).*