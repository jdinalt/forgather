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
- [Installation](#installation)
- [Learning Forgather](#learning-forgather)
- [Core Concepts](#core-concepts)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributing](#contributing)

## Quick Start

**1. Install Forgather:**
```bash
git clone https://github.com/jdinalt/forgather.git
cd forgather
pip install -e .
```

**2. Try a tutorial project:**

See: [./examples/tutorials/tiny_llama/project_index.ipynb](./examples/tutorials/tiny_llama/project_index.ipynb)

Or, from the comamand-line...

```bash
cd examples/tutorials/tiny_llama
fgcli.py index                                 # Show project summary
fgcli.py ls                                    # List available configs
fgcli.py -t train_tiny_llama.yaml templates    # Show template hierarchy
fgcli.py -t train_tiny_llama.yaml pp | less    # Show pre-processed configuration
fgcli.py -t train_tiny_llama.yaml train -d 0   # Train on GPU 0
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

model: !factory:transformers.LlamaForCausalLM.from_pretrained
    pretrained_model_name_or_path: "meta-llama/Llama-2-7b"
```

### Code Generation
Models are generated as standalone Python code with no framework dependencies:

### Built-in Training Infrastructure
- **SimpleTrainer**: Fast single-GPU training
- **AccelTrainer**: Multi-GPU with Accelerate
- **PipelineTrainer**: Pipeline parallelism
- **Custom Optimizers**: AdamW, SGD, AdaFactor, GaLore, Apollo

## Installation

**Requirements:**
- Python 3.10+
- PyTorch 2.3.1+
- CUDA (optional, for GPU training)

**Install from source:**
```bash
git clone https://github.com/your-repo/forgather.git
cd forgather
pip install -e .
```
**Optional**
Add forgather/bin to your PATH.
```bash
PATH="/path/to/forgather/bin:$PATH"
```

**Verify installation:**
```bash
./bin/fgcli.py --help
```
## Learning Forgather

### 1. **Start with Tutorials** (Recommended)
```bash
cd examples/tutorials/
```
- `tiny_llama/` - Complete training pipeline from scratch
- `project_composition/` - Template inheritance patterns
- `dynamic_lm/` - Dynamic model construction

### 2. **Explore Example Projects**
```bash
cd examples/tiny_experiments/
```
- `compare_trainers/` - Different training approaches
- `optimizers/` - Custom optimizer implementations
- `init_weights/` - Weight initialization experiments
- `flash_attention/` - Attention mechanism comparisons

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
fgcli.py -t config.yaml templates # Template hierarchy
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

### Code Generation Pipeline
```
Templates → YAML → Node Graph → Python Code → Executable Objects
```

Each step can be inspected:
```bash
fgcli.py -t config.yaml pp                    # Preprocess with Jinja2 to YAML
fgcli.py -t config.yaml graph --format yaml   # Node graph
fgcli.py -t config.yaml code --target model   # [optional] Python code
fgcli.py -t config.yaml construct --target model  # Live object
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
- `modelsrc/` - Transformer components library

### Example Projects
- `examples/tutorials/` - Learning materials
- `examples/tiny_experiments/` - Example experiments
- `examples/standalone/` - Self-contained projects

### Training and Monitoring
```bash
# Generate command to run "my_experiment.yaml" on GPUs 0 and 1
# Print command, but don't execute it.
fgcli.py -t my_experiment.yaml train -d "0,1" --dry-run

# Start Tensorboard to monitor progress
fgcli.py -t my_experiment.yaml tb --all
```

## Contributing

Forgather is actively developed and welcomes contributions:

1. **Bug Reports & Feature Requests**: Open GitHub issues
2. **Code Contributions**: Submit pull requests
3. **Documentation**: Improve tutorials and examples
4. **Community**: Share your experiments and templates

**Development Setup:**
```bash
git clone https://github.com/jdinalt/forgather.git
cd forgather
pip install -e .
# Run existing examples to verify setup
cd examples/tutorials/tiny_llama && fgcli.py -t train_tiny_llama.yaml train --dry-run
```
