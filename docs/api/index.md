# API Reference

Most interaction with Forgather happens through the CLI or configuration files, but the Python API is available for programmatic use — loading projects in notebooks, building custom training loops, or integrating with external tooling.

## Modules

| Module | Key classes |
|--------|-------------|
| [Project System](project.md) | `Project`, `MetaConfig`, `ConfigEnvironment` |
| [Trainers](trainers.md) | `BaseTrainer`, `AccelTrainer`, `PipelineTrainer` |
| [Optimizers](optimizers.md) | `AdamW`, `Adafactor`, `Apollo`, schedulers |
| [Datasets](datasets.md) | `FastHFLoader`, `BlockTokenizer`, dataset utilities |
| [Analysis](analysis.md) | `TrainingLog`, plotting, summary statistics |

## Quick Example

```python
from forgather.project import Project

# Load a configuration and materialize components
proj = Project("train_config.yaml")
training_script = proj()               # Full training script
model_factory = proj("model")          # Just the model factory
model = model_factory()                # Instantiate the model
```
