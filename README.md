# Forgather

Forgather is a configuration-driven ML framework that uses template inheritance and code generation to eliminate configuration duplication and enable systematic experimentation. Instead of copying and modifying entire config files, you inherit from base templates and specify only what changes.

**Key Benefits:**
- **No Config Duplication** - Inherit and override instead of copy-paste
- **Types as Hyperparameters** - Change optimizers, models, datasets in config files
- **Full Reproducibility** - Automatic snapshots of code and configs with each run
- **Pipeline Parallel Trainer** - Includes Pipeline Parallel trainer, optmized for training on consumer GPUs 
- **Extensible Trainers** - Easily extensible trainer implementation for modification and experimentation
- **Dynamic Models Library** - Define and customize model architectures entirely through configurtion files 
- **Templates Library** - Extensive templates library for tokenizers, models, trainers, datasets, etc.

## News
- Dec 14 Multiple new features:
  - Forgather's models now work with [vLLM](https://github.com/vllm-project/vllm), in both Tensor and Pipeline parallel mode. See [documentation](./docs/inference/vllm_integration.md).
  - Added support for [fused-linear-causal-loss](https://arxiv.org/abs/2411.09009), which significantly reduces peak memory requirements for training models with large vocabularies. [example usage](./examples/finetune/samantha/templates/configs/llama3_1b/1gpu_packed.yaml). We support the following implementations: [Liger](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/fused_linear_cross_entropy.py), [CCE](https://github.com/apple/ml-cross-entropy/tree/main), PyTorch compiled
  - Added a Triton implementation of [Forgather's Adafactor Optimizer](./src/forgather/ml/optim/adafactor_triton.py). This reduces peak memory further and speeds up training.
  - Enabled support for loading models with `device_map="auto"`, which allows our inference server to shard models across multiple GPUs.
- Nov 17, Completed a major overhaul of the [model conversion tool](./tools/convert_model/README.md) and added support for Mistral, Qwen3 models, and Llama models with RoPE scaling and tied word embeddings.
- Nov 9, **[OpenAssistant Dataset](./examples/datasets/OpenAssistant/README.md)** - High-quality example demonstrating how to build custom datasets that dynamically generate examples on-the-fly. Features quality-weighted sampling from conversation trees, sequence packing, multi-language support, and deterministic generation. Includes complete Python examples and extensive documentation. There is also a [demo finetune project](./examples/finetune/openassistant/README.md).
- Nov 4, Added support for [packed sequences](docs/datasets/sequence-packing.md) and [Flex Attention](https://pytorch.org/blog/flexattention/). Updating [Samantha tutorial](./examples/finetune/samantha/README.md) to demonstrate. Models now support KV cache.
- Oct 21, **[H.P. Lovecraft Project](./examples/tutorials/hp_lovecraft_project/README.md)** - Learn how to create workspaces and projects, while training a model to summon the Elder Gods. You can perform full-finetuning (not LoRA) on a 7B model, with a context length of up to 16K on a single 24 GB GPU!
- Oct 19, **[Samantha](./examples/finetune/samantha/README.md)** -- New tutorial on how to perform full finetuning on a 7B parameter model on a single, 24 GB, GPU on the "Samantha" dataset -- she believes she is sentient!
- **[Torch Titan integration](./examples/torchtitan/README.md)** -- Use Forgather to configure Torch Titan

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

# Verify install works with CLI
forgather ls -r
```

Note: We are using bleeding-edge PyTorch features, like flex-attention, which require PyTorch 2.9.0. If you are updating from a previous install,
run 'pip install -e .' again to force uprading to the latest libraries. If in doubt, nuke your venv and rebuild it.

Flex attention also depends upon having a working C compiler and python development packages installed.

```bash
sudo apt-get install build-essential python3-dev
```

Some of our plotting tools make use of [Graphviz](https://graphviz.org/).

```bash
sudo apt-get install graphviz  
```

**2. Try a tutorial project:**

See: [./examples/tutorials/tiny_llama/project_index.ipynb](./examples/tutorials/tiny_llama/project_index.ipynb)

Or, from the comamand-line...

```bash
# Optional
forgather -i                                    # Start interactive Forgather shell

forgather ls -r                                 # List all forgather projects
cd examples/tutorials/tiny_llama
forgather index                                 # Show project summary
forgather ls                                    # List available configs
forgather -t train_tiny_llama.yaml pp | less    # Show pre-processed configuration
forgather -t train_tiny_llama.yaml train        # Train model

```

**3. Monitor and control:**
```bash
forgather -t train_tiny_llama.yaml tb           # Start Tensorboard

forgather control list                          # List running traininig jobs
forgather control status JOB_ID                 # Get status of training job
fogrgater control [stop|abort|save] JOB_ID      # Control training jobs
```
**4. Test Model Inference:**

```bash
# Start inference server
forgather inf server -c -m /path/to/model

# Perform text completion on prompt
forgather inf client --completion "Once upon a time"
```

That's it! You've just trained a small language model using Forgather's template system.

## Key Features

### Template Inheritance
Create new experiments by inheriting from existing configs and specifying only the differences:

```yaml
-- extends 'base_experiment.yaml'
[optimizer]
    == super()
    lr: 1.0e-3  # Only change learning rate
```

### Dynamic Type System
Use any Python class or function directly in configs:

```yaml
optimizer: !partial:torch.optim.AdamW
    lr: 1.0e-3
    weight_decay: 0.01

[layer_factory]
# Experiment: Switch from PreLayerNorm to PostLayerNorm
layer_factory: &layer_factory !partial:.post_ln_layer:PostLNLayer@layer_factory
    feedforward_factory: *feedforward_factory
    attention_factory: *attention_factory
    norm_factory: *layer_norm_factory
    dropout: !var "layer_dropout"
    residual_dropout: !var "residual_dropout"
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
forgather ls -r

# cd to example project directory
cd examples/...

# Show project info
forgather index
```

### 3. **Interactive Development**

Run the interactive shell.

```python
forgather -i
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
- `[block_name]` - Override sections
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
forgather -t config.yaml pp                          # Preprocess with Jinja2 to YAML
forgather -t config.yaml graph --format yaml         # Parsed node graph
forgather -t config.yaml targets                     # List constructable objects in graph
forgather -t config.yaml code [--target <target>]    # [optional] Equivalent Python code for target
forgather -t config.yaml construct [--target <target>] [--call] # Materialize and show constructed object
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
forgather -t my_experiment.yaml train -d "0,1" --dry-run

# Start Tensorboard to monitor progress on all models in project; bind to all ports.
forgather tb --all -- --bind_all

# Show running training jobs -- which can be controlled via the CLI
forgather control list
```

## Contributing

Forgather is actively developed and welcomes contributions:

1. **Bug Reports & Feature Requests**: Open GitHub issues
2. **Code Contributions**: Submit pull requests
3. **Documentation**: Improve tutorials and examples
4. **Community**: Share your experiments and templates

