# API Reference

Most interaction with Forgather happens through the CLI or configuration files, but the Python API is available for programmatic use — loading projects in notebooks, building custom training loops, or integrating with external tooling.

!!! note "Rendering"
    These pages use `mkdocstrings` directives to pull docstrings from the source. The full MkDocs build renders them inline; the webui's lightweight Docs view does so via a pre-rendered cache populated by `forgather docs build` (run automatically by `./build-webui.sh`). When the cache is missing, the Docs view falls back to raw markdown and you'll see `::: forgather.foo.Bar` lines instead of expanded class docs — run `forgather docs build` to populate the cache. See [Serving the Forgather Docs with MkDocs](../guides/mkdocs.md#pre-rendering-api-directives-for-the-in-app-docs-view) for details.

## Modules

| Module | Key classes |
|--------|-------------|
| [Project System](project.md) | `Project`, `MetaConfig`, `ConfigEnvironment` |
| [Trainers](trainers.md) | `Trainer`, `DDPTrainer`, `FSDP2Trainer`, `PipelineTrainer` |
| [Callbacks](trainer_callbacks.md) | `TrainerCallback`, `JsonLogger`, `DivergenceDetector`, `TrainerControlCallback` |
| [Checkpoints](checkpoints.md) | `CheckpointMeta`, `save_checkpoint`, `load_checkpoint` |
| [Optimizers](optimizers.md) | `AdamW`, `Adafactor`, `Apollo`, schedulers |
| [Datasets](datasets.md) | `FastDatasetLoaderSimple`, `InterleavedDataset`, dataset utilities |
| [Analysis](analysis.md) | `TrainingLog`, plotting, summary statistics |

## Quick Example

```python
from forgather.project import Project

# Load a configuration and materialize components
proj = Project("train_config.yaml", "/path/to/project_dir")
training_script = proj()               # Full training script
model_factory = proj("model")          # Just the model factory
model = model_factory()                # Instantiate the model
```
