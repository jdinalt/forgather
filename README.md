# Forgather

Forgather is a configuration-driven ML framework that uses template inheritance and code generation to eliminate configuration duplication and enable systematic experimentation. Instead of copying and modifying entire config files, you inherit from base templates and specify only what changes.

**Key Benefits:**
- **No Config Duplication** - Inherit and override instead of copy-paste
- **Types as Hyperparameters** - Change optimizers, models, datasets in config files
- **Dynamic Code Generation** - Generate standalone Python models from configs
- **Full Reproducibility** - Automatic snapshots of code and configs with each run

## Table of Contents

- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Learning Forgather](#learning-forgather)
- [Core Concepts](#core-concepts)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributing](#contributing)

## Quick Start

**1. Install Forgather:**
```bash
# Requires python >= 3.10
# Setup python virtual envrionment for install
# You can also use conda or whatever you are most comfortable with.
python3 -m venv /path/to/new/venv

# Activate the virtaul environment
source /path/to/new/venv/bin/activate

git clone https://github.com/jdinalt/forgather.git
cd forgather
pip install -e .

# Add "./bin" to your path for CLI
PATH="/path/to/forgather/bin:$PATH"

# Verify install works with CLI
fgcli.py ls -r
```

**2. Try a tutorial project:**

See: [./examples/tutorials/tiny_llama/project_index.ipynb](./examples/tutorials/tiny_llama/project_index.ipynb)

Or, from the comamand-line...

```bash
fgcli.py ls -r                                 # List all forgather projects
cd examples/tutorials/tiny_llama
fgcli.py index                                 # Show project summary
fgcli.py ls                                    # List available configs
fgcli.py -t train_tiny_llama.yaml pp | less    # Show pre-processed configuration
fgcli.py -t train_tiny_llama.yaml train        # Train model
```

**3. Monitor training:**
```bash
fgcli.py -t train_tiny_llama.yaml tb           # Start Tensorboard
```

That's it! You've just trained a small language model using Forgather's template system.

## Key Features

### Template Inheritance
Create new experiments by inheriting from existing configs and specifying only the differences:

```yaml
-- extends 'base_experiment.yaml'
-- block optimizer
    == super()
    lr: 1.0e-3  # Only change learning rate
-- endblock
```

### Dynamic Type System
Use any Python class or function directly in configs:

```yaml
optimizer: !partial:torch.optim.AdamW
    lr: 1.0e-3
    weight_decay: 0.01

-- block layer_factory
# Experiment: Switch from PreLayerNorm to PostLayerNorm
layer_factory: &layer_factory !partial:.post_ln_layer:PostLNLayer@layer_factory
    feedforward_factory: *feedforward_factory
    attention_factory: *attention_factory
    norm_factory: *layer_norm_factory
    dropout: !var "layer_dropout"
    residual_dropout: !var "residual_dropout"
<< endblock layer_factory
```

### Code Generation
Models are generated as standalone Python code with no framework dependencies:

### Built-in Training Infrastructure
- **Trainer**: Fast single-GPU training for small models.
- **AccelTrainer**: Multi-GPU with Accelerate
- **PipelineTrainer**: Pipeline parallelism
- **Custom Optimizers**: AdamW, AdaFactor, GaLore, Apollo

## Learning Forgather

### 1. **Start with Tutorials** (Recommended)
```bash
cd examples/tutorials/
```
- `tiny_llama/` - Train a small language model from scratch
- `project_composition/` - Template inheritance patterns
- `dynamic_lm/` - Dynamic model construction
- `projects_overview/` - Overview of Forgather projects

### 2. **Explore Example Projects**
```bash
cd forgather

# List all example projects and configurations
fgcli.py ls -r

# cd to example project directory
cd examples/...

# Show project info
fgcli.py index
```

### 3. **Interactive Development**
Each project includes a `project_index.ipynb` notebook for interactive exploration:

```python
from forgather.project import Project
proj = Project("train_tiny_llama.yaml")
training_script = proj("main")  # Materialize assets from config
training_script.run() # Train model
```

### 4. **Command Line Tools**
Master the `fgcli.py` interface:

```bash
fgcli.py index                    # Project overview
fgcli.py ls                       # List configs
fgcli.py -t config.yaml pp        # Show preprocessed config
fgcli.py -t config.yaml tlist     # Template hierarchy
fgcli.py -t config.yaml train     # Train model
```

## Core Concepts

### Projects
Every Forgather experiment is a **Project** with this structure:
```
my_project/
├── meta.yaml              # Project metadata
├── templates/
│   ├── project.yaml       # Main template
│   └── configs/           # Experiment configs
├── output_models/         # Generated code & results
└── project_index.ipynb    # Interactive notebook
```

### Template Language
Forgather uses **Jinja2 + YAML** with custom syntax:
- `-- extends 'template.yaml'` - Template inheritance
- `-- block name` / `-- endblock` - Override sections
- `-- set ns.var = value` - Set variables
- `!partial:module:Class` - Partial function construction
- `!factory:module:Class` - Factory construction
- `!var "variable_name"` - Variable references
- `#---- inline.template.name ----` - Split document into multiple templates

See [Syntax Reference](./docs/configuration/syntax-reference.md)

### Code Generation Pipeline
```
Templates → YAML → Node Graph → Python Code → Executable Objects
```

Each step can be inspected:
```bash
fgcli.py -t config.yaml pp                          # Preprocess with Jinja2 to YAML
fgcli.py -t config.yaml graph --format yaml         # Parsed node graph
fgcli.py -t config.yaml targets                     # List constructable objects in graph
fgcli.py -t config.yaml code [--target <target>]    # [optional] Equivalent Python code for target
fgcli.py -t config.yaml construct [--target <target>] [--call] # Materialize and show constructed object
```

## Project Structure

### Framework Code
- `src/forgather/` - Core framework
  - `project.py` - Project management
  - `config.py` - Template processing
  - `codegen.py` - Python code generation
  - `ml/` - Training infrastructure
- `templatelib/` - Reusable templates
  - `base/` - Abstract base templates
  - `examples/` - Common models, datasets, tokenizers
- `modelsrc/` - Modular model components library

### Example Projects
- `examples/tutorials/` - Learning materials
- `examples/tiny_experiments/` - Example experiments
- `examples/standalone/` - Self-contained projects
- `examples/template_project` - Starting point for new projects.

### Training and Monitoring
```bash
# Generate command to run "my_experiment.yaml" on GPUs 0 and 1
# Print command, but don't execute it.
fgcli.py -t my_experiment.yaml train -d "0,1" --dry-run

# Start Tensorboard to monitor progress on all models in project; bind to all ports.
fgcli.py tb --all -- --bind_all
```

## Contributing

Forgather is actively developed and welcomes contributions:

1. **Bug Reports & Feature Requests**: Open GitHub issues
2. **Code Contributions**: Submit pull requests
3. **Documentation**: Improve tutorials and examples
4. **Community**: Share your experiments and templates

