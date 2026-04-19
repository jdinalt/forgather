# Project System

The project system is the central abstraction in Forgather. A `Project` resolves a configuration file through the template inheritance chain and provides access to all configured components.

**Related documentation:**

- [Core Concepts](../core-concepts/README.md) — projects, templates, and the configuration pipeline
- [Configuration Overview](../configuration/README.md) — template system and YAML configuration
- [Syntax Reference](../configuration/syntax-reference.md) — complete reference for line statements and YAML tags
- [Low-level API](../configuration/low-level-api.md) — the API underlying the `Project` abstraction

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
