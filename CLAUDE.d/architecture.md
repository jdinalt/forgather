# Architecture

Forgather is a configuration-driven ML framework built on template
inheritance and code generation. The central abstraction is the
**Project**, which encapsulates an ML experiment through a sophisticated
template system. Models are materialized as standalone Python code in
`output_models/` and are self-contained / deployable.

## Pipeline

```
Templates → YAML → Node Graph → Python Code → Objects
```

## Key components

**Project system** (`src/forgather/`)
- `project.py` — `Project`: central abstraction managing config + codegen
- `meta_config.py` — `MetaConfig`: project metadata + template search paths
- `config.py` — `ConfigEnvironment`: Jinja2 + YAML preprocessing

**Template hierarchy**

```
templatelib/base/           # Abstract base templates
templatelib/examples/       # Reusable example definitions
examples/*/templates/       # Project-specific templates
modelsrc/transformer/       # Reusable transformer components
```

**Configuration language**
- Jinja2 with line-statement syntax: `-- if`, `-- set`, `-- extends`,
  `-- block` / `-- endblock`
- Custom YAML tags: `!call`, `!factory`, `!partial`, `!var`, `!singleton`
- Template inheritance via `-- extends`; override with
  `-- block name` / `-- endblock`; `== super()` includes parent block
- Inline template definition: `#--- template.name ---`

YAML tag semantics (easy to get wrong — see `CLAUDE.d/gotchas.md`):
- `!partial` → `functools.partial`-style Callable
- `!singleton` → lazy object, called once and cached
- `!factory` → called every access, not cached
- With no args, pass `[]`

Full spec: `docs/configuration/syntax-reference.md`.

## Training system

Trainer classes in `src/forgather/ml/`:
- `BaseTrainer` → `SimpleTrainer` (single-GPU)
- `AccelTrainer` (Accelerate multi-GPU)
- `DDPTrainer` (DDP)
- `PipelineTrainer` (pipeline parallel)

Optimizers: `src/forgather/ml/optim/` (AdamW, SGD, AdaFactor, Apollo,
…). Callback system handles logging, checkpointing, and external
control (`TrainerControlCallback`).

## Project layout

```
project_dir/
├── meta.yaml              # Extends forgather_workspace/meta_defaults.yaml
├── templates/
│   ├── project.yaml       # Main project template
│   ├── configs/           # Experiment configurations
│   └── experiments/       # Alternative config organization
├── output_models/         # Generated code + training runs
└── project_index.ipynb    # Interactive exploration
```

## Programmatic API

```python
from forgather.project import Project

proj = Project("train_tiny_llama.yaml")
training_script = proj()
model_factory = proj("model")
model = model_factory()
```

Multiple targets at once:

```python
model_factory, train_dataset = proj("model", "train_dataset")
```
