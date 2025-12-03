# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Forgather CLI (forgather) - Primary Interface

The `forgather` command is the main way to interact with Forgather projects. It's available in PATH and provides comprehensive project management capabilities.

```bash
# Basic usage
forgather [-p PROJECT_DIR] [-t CONFIG_TEMPLATE] <subcommand>

# Help
forgather --help
fgcli.pu <subcommand> --help

# Common project exploration commands
forgather index                    # Show project overview as markdown
forgather ls                       # Show project name, short description, and all available configurations.
forgather ls -r                    # As above, but recursively search all sub-directories for projects and list them.
forgather tlist                    # List all available template files
forgather tlist --format md        # Show template inheritance hierarchy for all templates as markdown.
forgather [-t config.yaml] pp      # Show preprocessed configuration; run before attempting to train!
forgather [-t config.yaml] trefs    # Show template inheritance hierarchy, starting with configuration template.
forgather [-t config.yaml] targets # List available output targets

# Configuration development and debugging
forgather tlist | xargs grep SEARCH_PATTERN     # Search all templates for pattern
forgather -t config.yaml pp        # Useful for diagnosing configuration errors

# Training
forgather -t config.yaml train                     # Train with default settings
forgather -t config.yaml train -d 0,1              # Train on specific GPUs
forgather -t config.yaml train --dry-run           # Show command without executing

# Get head and tail of training output logs
head -n 10 output_models/my_custom_model/runs/my_custom_model_2025-06-25T03-16-59/trainer_logs.json
tail -n 10 output_models/my_custom_model/runs/my_custom_model_2025-06-25T03-16-59/trainer_logs.json

# Get config used by training run
cat output_models/my_custom_model/runs/my_custom_model_2025-06-25T03-16-59/config.yaml
```

### Training Job Control

Forgather supports **external control of running training jobs** through the `forgather control` commands. This enables real-time interaction with distributed training for hyperparameter experimentation, checkpoint management, and graceful shutdown.

```bash
# List all discoverable training jobs
forgather control list

# Get detailed status of a specific job  
forgather control status JOB_ID

# Control running training jobs
forgather control save JOB_ID           # Save checkpoint (triggers evaluation if configured)
forgather control stop JOB_ID           # Gracefully stop training (saves final checkpoint)
forgather control save-stop JOB_ID      # Save checkpoint then gracefully stop  
forgather control abort JOB_ID          # Abort immediately without saving (useful for failed hyperparameter experiments)

# Job management
forgather control cleanup               # Remove dead job files
forgather control cleanup --force      # Skip confirmation
```

**To enable control in your training jobs**, add the TrainerControlCallback:

```python
from forgather.ml.trainer.callbacks import TrainerControlCallback

callbacks = [
    TrainerControlCallback(
        job_id="my_experiment",  # Optional: auto-generated if not provided
    ),
    # ... your other callbacks
]

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    callbacks=callbacks
)
```

**Typical workflow:**
1. Start training job: `forgather -t config.yaml train`
2. In another terminal: `forgather control list` to see running jobs
3. Save checkpoint on-demand: `forgather control save JOB_ID`
4. Gracefully stop when satisfied: `forgather control stop JOB_ID`
5. Or abort failed experiments: `forgather control abort JOB_ID`

The system works with **distributed training** - commands sent to any rank are automatically coordinated across all processes. See `examples/trainer_control/` for a complete working example and `docs/trainers/trainer-control.md` for full documentation.

### Inference

Forgather includes a basic OpenAPI compatible inference server and client.

**Start server**

```bash
# Load model in directory using AutoModelForCausalLM.from_pretrained()
# This defaults to bfloat16 on cuda:0
forgather inf server -m MODEL_PATH

# Load model from latest Forgather checkpoint
forgather inf server -c -m MODEL_PATH
```
**Start client**

```bash
# Start in interactive (chat) mode
forgather inf client

# Perform text completion on prompt
forgather inf client --completion "Once upon a time"

# Get response to single message
forgather inf client --message "Tell me a story"
```

Detailed inference instructions are located in 'tools/inference_server/README.md'

### vLLM Distributed Inference

Forgather models support distributed inference with [vLLM](https://docs.vllm.ai/) for high-throughput serving with tensor and pipeline parallelism.

**Validate vLLM Support**

```python
from forgather.ml.model_conversion import validate_vllm_plans, print_model_structure
from transformers import AutoModelForCausalLM

# Load trained model
model = AutoModelForCausalLM.from_pretrained("output_models/my_model")

# Print model structure
print_model_structure(model, max_depth=4)

# Validate vLLM plans
if hasattr(model, '_tp_plan') and model._tp_plan:
    is_valid = validate_vllm_plans(model, tp_plan=model._tp_plan, pp_plan=model._pp_plan, strict=True)
```

**Deploy with vLLM**

```bash
# Single-GPU inference
vllm serve output_models/my_model --trust-remote-code

# Tensor parallel (4 GPUs)
vllm serve output_models/my_model --trust-remote-code --tensor-parallel-size 4

# Tensor + Pipeline parallel (8 GPUs: 2 PP stages, 4 TP per stage)
vllm serve output_models/my_model \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 8192
```

**Adding vLLM Support to Custom Models**

Most transformer models (Llama, DeepOne, etc.) include vLLM support by default. For custom models, add vLLM plans to the `[model_code_generator]` section:

```yaml
# In your model configuration
[model_code_generator]
    == super()

    # vLLM Support
    tp_plan:
        "model.layer_stack.layers.*.attention.query_linear": "colwise"
        "model.layer_stack.layers.*.attention.key_linear": "colwise"
        "model.layer_stack.layers.*.attention.value_linear": "colwise"
        "model.layer_stack.layers.*.attention.output_linear": "rowwise"
        # ... additional layers
    pp_plan:
        "model.input_encoder": [["input_ids"], ["hidden_states"]]
        "model.layer_stack": [["hidden_states", "attention_mask"], ["hidden_states"]]
        "model.output_decoder": [["hidden_states"], ["logits"]]
```

See `templatelib/base/models/causal_lm/vllm_plans.yaml` for the complete reference template.
For detailed information, see `docs/inference/vllm_integration.md`

### Examples from Real Projects
```bash
# Working with tiny_llama tutorial project
cd examples/tutorials/tiny_llama
forgather ls                                        # List: train_tiny_llama.yaml, etc.
forgather -t train_tiny_llama.yaml pp               # Show pre-processed configuration.
forgather -t train_tiny_llama.yaml train            # Train with selected configuration.
```

### Workspace and Project Creation

Forgather uses a two-level structure: **Workspaces** contain **Projects**. Use the `forgather ws` commands to create and manage both.

#### Creating a New Workspace
```bash
# Basic workspace creation
forgather ws init --name "My ML Workspace" --description "Machine learning research experiments" --forgather-dir /path/to/forgather

# With additional template search paths
forgather ws init --name "Advanced Workspace" --description "Advanced ML experiments" --forgather-dir /path/to/forgather /extra/templates/path /another/path

# With no default search paths (minimal workspace)
forgather ws init --name "Minimal Workspace" --description "Clean minimal setup" --forgather-dir /path/to/forgather --no-defaults
```

This creates a `forgather_workspace/` directory containing:
- `README.md` - Workspace documentation
- `base_directories.yaml` - Base directory configuration  
- `meta_defaults.yaml` - Template search paths and workspace metadata

#### Creating a New Project in a Workspace
```bash
# Basic project creation (directory name auto-generated from project name)
forgather ws project --name "Sentiment Analysis" --description "BERT-based sentiment analysis experiments"

# With custom settings
forgather ws project --name "Image Classification" --description "CNN experiments" --config-prefix "experiments" --default-config "baseline.yaml" custom_directory_name
```

This creates a project directory with:
- `README.md` - Project documentation
- `meta.yaml` - Project metadata extending workspace defaults
- `templates/configs/{default_config}` - Default configuration template

#### Typical Workspace Setup Workflow
1. **Create workspace**: `forgather ws init --name "My Research" --description "ML experiments" --forgather-dir /path/to/forgather`
2. **Create project(s)**: `forgather ws project --name "Project 1" --description "First experiment"`
3. **Navigate to project**: `cd project_1`
4. **List configurations**: `forgather ls`
5. **Test configuration**: `forgather pp` 
6. **Train model**: `forgather train`

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
- **IMPORTANT YAML Tag Distinctions**:
  - `!partial`: Constructs a Python partial function (produces Callable type)
  - `!singleton`: Lazy object, called once and cached for subsequent accesses
  - `!factory`: Called every time it's accessed (not cached)
  - When using with no arguments, add empty list `[]`
  - See docs/configuration/syntax-reference.md for details and examples

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

**Working with Created Projects**
- After creating a project, `cd` into the project directory to work with it
- The generated default config is minimal - extend it by inheriting from base templates:
  ```yaml
  -- extends "types/training_script/causal_lm/causal_lm.yaml"
  
  -- block construct_new_model
      -- include 'models/llama.yaml'
  -- endblock construct_new_model
  
  -- block optimizer
  optimizer: &optimizer !partial:torch.optim:AdamW
      lr: 1.0e-4
  -- endblock optimizer
  ```
- Key template inheritance patterns:
  - Use `-- extends "template_name.yaml"` for single inheritance
  - Use `-- include 'template_name.yaml'` to include template content
  - Override template blocks with `-- block name` / `-- endblock`
  - Use `== super()` to include parent block content
- Add additional config files in `templates/configs/` for different experiments
- Test configurations immediately: `forgather ls` then `forgather pp` 
- Use `forgather meta` to see workspace/project structure and template search paths

**Configuration Validation**
- ALWAYS run `forgather ls` to validate all configurations after making changes
- Failed configs show as "PARSE ERROR" instead of their descriptive names
- Use `forgather -t config.yaml pp` to debug preprocessing issues
- Check for syntax errors, missing imports, and template reference issues

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
- Attention mechanisms testing project : "examples/tiny_experiments/attention/"

**Common Issues and Solutions**
- Missing import errors (e.g., `Callable` not imported): Add missing imports to affected files
- YAML tag errors: Use `!partial` for function objects, `!singleton`/`!factory` for function calls
- Configuration validation: Run `forgather ls` to check all configs parse correctly
- Complex64 serialization: RoPE models may fail to save due to safetensors limitations with complex tensors

**Style**

Follow existing style conventions. Avoid emojis.
