# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Forgather CLI (fgcli.py) - Primary Interface

The `fgcli.py` command is the main way to interact with Forgather projects. It's available in PATH and provides comprehensive project management capabilities.

```bash
# Basic usage
fgcli.py [-p PROJECT_DIR] [-t CONFIG_TEMPLATE] <subcommand>

# Help
fgcli.py --help
fgcli.pu <subcommand> --help

# Common project exploration commands
fgcli.py index                    # Show project overview as markdown
fgcli.py ls                       # Show project name, short description, and all available configurations.
fgcli.py ls -r                    # As above, but recursively search all sub-directories for projects and list them.
fgcli.py tlist                    # List all available template files
fgcli.py tlist --format md        # Show template inheritance hierarchy for all templates as markdown.
fgcli.py [-t config.yaml] pp      # Show preprocessed configuration; run before attempting to train!
fgcli.py [-t config.yaml] trefs    # Show template inheritance hierarchy, starting with configuration template.
fgcli.py [-t config.yaml] targets # List available output targets

# Configuration development and debugging
fgcli.py tlist | xargs grep SEARCH_PATTERN     # Search all templates for pattern
fgcli.py -t config.yaml pp        # Useful for diagnosing configuration errors

# Training
fgcli.py -t config.yaml train                     # Train with default settings
fgcli.py -t config.yaml train -d 0,1              # Train on specific GPUs
fgcli.py -t config.yaml train --dry-run           # Show command without executing

# Get head and tail of training output logs
head -n 10 output_models/my_custom_model/runs/my_custom_model_2025-06-25T03-16-59/trainer_logs.json
tail -n 10 output_models/my_custom_model/runs/my_custom_model_2025-06-25T03-16-59/trainer_logs.json

# Get config used by training run
cat output_models/my_custom_model/runs/my_custom_model_2025-06-25T03-16-59/config.yaml
```

### Examples from Real Projects
```bash
# Working with tiny_llama tutorial project
cd examples/tutorials/tiny_llama
fgcli.py ls                                        # List: train_tiny_llama.yaml, etc.
fgcli.py -t train_tiny_llama.yaml pp               # Show pre-processed configuration.
fgcli.py -t train_tiny_llama.yaml train            # Train with selected configuration.
```

### Project Installation
```bash
pip install -e .
```

## Architecture Overview

Forgather is a configuration-driven ML framework built on template inheritance and code generation. The core abstraction is the **Project**, which encapsulates an ML experiment through a sophisticated template system.

### Key Components

**Project System**
- `Project` (`src/forgather/project.py`): Central abstraction managing configuration and code generation
- `MetaConfig` (`src/forgather/meta_config.py`): Defines project metadata and template search paths
- `ConfigEnvironment` (`src/forgather/config.py`): Handles template preprocessing with Jinja2 + YAML

**Template Hierarchy**
```
templatelib/base/         # Abstract base templates (trainers, models, datasets)
templatelib/examples/     # Reusable example definitions (models, datasets, tokenizers)
examples/*/templates/     # Project-specific templates
modelsrc/transformer/     # Reusable transformer components
```

**Configuration Language**
- Jinja2 preprocessing with custom line statement syntax (`-- if`, `-- set`, `-- extends`, `-- block`)
- Custom YAML tags: `!call`, `!factory`, `!partial`, `!var`, `!singleton`
- Template inheritance via `-- extends` and `-- block`/`-- endblock`

### Training System

**Trainer Classes** (`src/forgather/ml/`)
- `BaseTrainer` → `SimpleTrainer` (basic single-GPU trainer)
- `AccelTrainer` (multi-GPU via Accelerate)
- `PipelineTrainer` (pipeline parallelism)
- Custom optimizers in `src/forgather/ml/optim/` (AdamW, SGD, AdaFactor, Apollo, etc.)
- Extensible callback system for logging and checkpointing

**Model Management**
- Dynamic model construction from configuration graphs
- Code generation: Templates → YAML → Node Graph → Python Code → Objects
- Generated models stored in `output_models/` as standalone Python code
- Transformer components in `modelsrc/transformer/`

### Project Structure

Each project follows this pattern:
```
project_dir/
├── meta.yaml              # Extends forgather_workspace/meta_defaults.yaml
├── templates/
│   ├── project.yaml       # Main project template
│   ├── configs/           # Experiment configurations
│   └── experiments/       # Alternative config organization
├── output_models/         # Generated model code and training runs
└── project_index.ipynb    # Interactive exploration notebook
```

### Development Workflow

**Interactive Development**
- Use `project_index.ipynb` notebooks for experiment development
- Load projects with `Project("config.yaml")`
- Materialize from configurations: `model_factory, train_dataset = proj("model", "train_dataset")`

**Template Development**
- Templates use Jinja2 with custom line statement syntax
- Inherit via `-- extends 'template_name.yaml'`
- Override sections with `-- block section_name` / `-- endblock`
- Include other templates with `-- include 'template_name.yaml'`
- Inline template definition `#-------------------- template.name --------------------`
- Jinja2 inheritance, via 'extends', only allows a single parent template. When overriding blocks from multiple parents,
  use the 'include and extend' pattern. Example:

```
-- extends "types/training_script/causal_lm/causal_lm.yaml"
-- block optimizer
# Project override
optimizer: &optimizer !lambda:torch:optim.AdamW
    lr: 1.0e-3
<< endblock optimizer

-- block construct_new_model
    ## Includes inline template.
    -- include 'project.model_config'
-- endblock construct_new_model

# Inline template definition
#-------------------- project.model_config --------------------

-- block model_config
    == super()
    # Project overrides
    hidden_size: 512
<< endblock model_config
```

- For definitive syntax guide, see "docs/configuration/syntax-reference.md"

**Code Generation**
- Models materialized as standalone Python code in `output_models/`
- Generated code is self-contained and deployable
- Training runs stored in `output_models/model_name/runs/`

## Important Patterns

**Project Loading**
```python
from forgather.project import Project
proj = Project("train_tiny_llama.yaml")
training_script = proj()
model_factory = proj("model")
model = model_factory()
```

**Template Inheritance**
```yaml
-- extends 'types/training_script/causal_lm/causal_lm.yaml'
-- block config_metadata
    == super()
    -- set ns.config_name = "My Experiment"
-- endblock
```

**Training Script Usage**
- Use `-p` flag to specify project directory
- Config files are relative to project templates directory
- Supports distributed training with proper logging levels
- Generated models include training artifacts and source code

The framework emphasizes systematic experimentation through template-based configuration management, enabling reproducible ML experiments with modular, reusable components.

**Key Example Projects**
Refer to these when creating new projects.
- A template project to copy for starting a new one : "examples/template_project/"
- Projects overview : "examples/tutorials/projects_overview/"
- Forgather project structure : "examples/tutorials/project_composition/" 
- Model training tutorial project : "examples/tutorials/tiny_llama/"

**Style**

Follow existing style conventions. Avoid emojis.
