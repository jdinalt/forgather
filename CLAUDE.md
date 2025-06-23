# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Forgather CLI (fgcli.py) - Primary Interface

The `fgcli.py` command is the main way to interact with Forgather projects. It's available in PATH and provides comprehensive project management capabilities.

```bash
# Basic usage
fgcli.py [-p PROJECT_DIR] [-t CONFIG_TEMPLATE] <subcommand>

# Common project exploration commands
fgcli.py index                    # Show project overview
fgcli.py ls                       # List available configurations
fgcli.py -t config.yaml pp        # Show preprocessed configuration
fgcli.py -t config.yaml templates # Show template inheritance hierarchy
fgcli.py -t config.yaml targets   # List available output targets

# Configuration development and debugging
fgcli.py -t config.yaml code --target model    # Generate Python code for target
fgcli.py -t config.yaml construct --target model  # Materialize and print target
fgcli.py -t config.yaml graph --format yaml       # Show node graph as YAML

# Training and monitoring
fgcli.py -t config.yaml train                     # Train with default settings
fgcli.py -t config.yaml train -d 0,1              # Train on specific GPUs
fgcli.py -t config.yaml train --dry-run           # Show command without executing
fgcli.py -t config.yaml tb                        # Start Tensorboard
fgcli.py -t config.yaml tb --all                  # Tensorboard for all models
```

### Examples from Real Projects
```bash
# Working with tiny_llama tutorial project
cd examples/tutorials/tiny_llama
fgcli.py ls                                        # List: train_tiny_llama.yaml, etc.
fgcli.py -t train_tiny_llama.yaml templates        # Show template hierarchy
fgcli.py -t train_tiny_llama.yaml train -d 0       # Train on GPU 0
fgcli.py -t train_tiny_llama.yaml tb               # Monitor training

# Working with tiny experiments
cd examples/tiny_experiments/compare_trainers  
fgcli.py -t trainer.yaml targets                   # Show available targets
fgcli.py -t trainer.yaml code --target model       # Generate model code
fgcli.py -t trainer.yaml train --dry-run           # Preview training command
```

### Direct Training Scripts (Alternative)
```bash
# Direct usage of train_script.py (lower-level interface)
torchrun --standalone --nproc-per-node gpu scripts/train_script.py -p project_dir config_name.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node gpu scripts/train_script.py -p project_dir config_name.yaml
accelerate launch scripts/train_script.py -p project_dir config_name.yaml
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
- Load projects with `Project(project_dir="path")`
- Test configurations: `proj("config.yaml")` or `proj.environment.load("config.yaml")`

**Template Development**
- Templates use Jinja2 with custom line statement syntax
- Inherit via `-- extends 'template_name.yaml'`
- Override sections with `-- block section_name` / `-- endblock`
- Include other templates with `-- include 'template_name.yaml'`

**Code Generation**
- Models materialized as standalone Python code in `output_models/`
- Generated code is self-contained and deployable
- Training runs stored in `output_models/model_name/runs/`

## Important Patterns

**Project Loading**
```python
from forgather.project import Project
proj = Project(project_dir="./examples/tutorials/tiny_llama")
config = proj.environment.load("configs/train_tiny_llama.yaml")
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