# Project System

The project system is the central abstraction in Forgather. A `Project` resolves a configuration file through the template inheritance chain and provides access to all configured components.

## Quick Example

```python
from forgather.project import Project

proj = Project("train_tiny_llama.yaml")

# Materialize the full training script
training_script = proj()

# Materialize individual components
model_factory = proj("model")
train_dataset  = proj("train_dataset")

model = model_factory()
```

---

::: forgather.project.Project

---

::: forgather.meta_config.MetaConfig

---

::: forgather.config.ConfigEnvironment
