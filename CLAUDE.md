# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude Code Development Notes

**Location**: `.claude_notes/`

This directory contains development notes, implementation summaries, and progress tracking documents created by Claude Code during feature development. These files are useful for:
- Tracking implementation progress across sessions
- Documenting design decisions and issues encountered
- Providing context for future work on related features
- Keeping detailed records of what was implemented and tested

**Important**: This directory is gitignored and should NOT be merged to main branches. These are working notes, not official documentation.

**Examples**: PHASE1_IMPLEMENTATION_SUMMARY.md, CHECKPOINT_INTEGRATION_SUMMARY.md

## Testing

See `docs/development/testing.md` for the complete testing guide, including test organization, running tests, shared fixtures, and recommended workflows.

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

### Training Log Analysis

Forgather provides powerful tools for analyzing and visualizing training logs through the `forgather logs` command. Training metrics are automatically logged to `trainer_logs.json` files during training.

```bash
# List available training logs
forgather logs list

# Generate summary statistics
forgather logs summary                              # Auto-detect latest log
forgather logs summary path/to/trainer_logs.json    # Specific log
forgather logs summary --format json                # JSON output
forgather logs summary --format one-line            # Compact one-line format
forgather logs summary --all --format one-line      # All logs in compact table
forgather logs summary --format md --output report.md

# Generate plots (default: saves to tmp/ directory)
forgather logs plot                                 # Save to tmp/training_plot.png
forgather logs plot -e                              # Save and open in editor
forgather logs plot --loss-curves                   # Loss curves with LR (tmp/loss_curves.png)
forgather logs plot --output training.png           # Custom output location
forgather logs plot --metrics "loss,grad_norm"      # Specific metrics
forgather logs plot --x-axis epoch                  # Plot by epoch
forgather logs plot --smooth 10                     # Apply smoothing
forgather logs plot --format svg                    # SVG format (default: png)

# Compare multiple runs
forgather logs plot --compare run1/trainer_logs.json run2/trainer_logs.json --loss-curves
```

**Programmatic API:**

```python
from forgather.ml.analysis import TrainingLog, compute_summary_statistics

log = TrainingLog.from_file("path/to/trainer_logs.json")
summary = compute_summary_statistics(log)
print(f"Best loss: {summary['best_loss']} at step {summary['best_loss_step']}")
```

For complete documentation, see `docs/logs-analysis.md` and `examples/log_analysis_example.py`.

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

Forgather uses a two-level structure: **Workspaces** contain **Projects**. Use `forgather ws create` for workspaces and `forgather project create` for projects.

#### Creating a New Workspace
```bash
# Basic workspace creation
forgather ws create --name "My ML Workspace" --description "Machine learning research experiments" --forgather-dir /path/to/forgather

# With additional template search paths
forgather ws create --name "Advanced Workspace" --description "Advanced ML experiments" --forgather-dir /path/to/forgather --search-path /extra/templates/path
```

This creates a `forgather_workspace/` directory containing:
- `README.md` - Workspace documentation
- `base_directories.yaml` - Base directory configuration
- `meta_defaults.yaml` - Template search paths and workspace metadata

#### Creating a New Project in a Workspace
```bash
# Basic project creation (run from inside a workspace directory)
forgather project create --name "Sentiment Analysis" --description "BERT-based sentiment analysis experiments"

# With custom settings
forgather project create --name "Image Classification" --description "CNN experiments" --config-prefix "experiments" --default-config "baseline.yaml" --project-dir-name custom_directory_name

# Copy default config from an existing configuration file
forgather project create --name "Custom Model" --description "Customized model" path/to/source_config.yaml
```

This creates a project directory with:
- `README.md` - Project documentation
- `meta.yaml` - Project metadata extending workspace defaults
- `templates/configs/{default_config}` - Default configuration template

#### Typical Workspace Setup Workflow
1. **Create workspace**: `forgather ws create --name "My Research" --description "ML experiments" --forgather-dir /path/to/forgather`
2. **Create project(s)**: `forgather project create --name "Project 1" --description "First experiment"`
3. **Navigate to project**: `cd project_1`
4. **List configurations**: `forgather ls`
5. **Test configuration**: `forgather pp`
6. **Train model**: `forgather train`

#### Creating a Project That Extends Another Model Project

When creating a project that builds on an existing model project (e.g., experimenting with a model defined in `examples/models/`), you need to:

1. **Add the base model's templates to the search path** in `meta.yaml`:
   ```yaml
   -- extends "meta_defaults.yaml"
   [searchdir_project]
       == super()
       - "{{ joinpath(ns.forgather_dir, 'examples/models/base_model/templates') }}"
   ```

2. **Override `[model_submodule_searchpath]`** if the base model has a `modelsrc/` directory. The base model's config typically uses `{{ joinpath(project_dir, 'modelsrc') }}` which resolves to the *current* project's directory, not the base model's. Fix this by adding an inline model template that points to the correct modelsrc path:
   ```yaml
   -- extends "configs/base_config.yaml"

   [config_metadata]
       == super()
       -- set ns.model_name = "my_experiment"

   [model_definition]
       -- include "config.my_experiment.model"

   #------------- config.my_experiment.model --------------
   -- extends "config.base.model"

   [model_submodule_searchpath]
       - "{{ joinpath(ns.forgather_dir, 'examples/models/base_model/modelsrc') }}"
       == super()
   ```

   Without this override, the model code generator will fail to find the base model's Python source files (e.g., `ModuleNotFoundError`).

3. **Check if the base model has `modelsrc/`**: If it does not (e.g., models that only use standard `modelsrc/transformer/` components), no search path override is needed.

See `examples/tiny_experiments/canon/` for a complete example of a project extending `examples/models/llama_canon/`, and `examples/pretrain/small-llm/custom_deepone/` for a simpler case where no modelsrc override is needed.

#### Adding Training to a Model Experiment Project

A single project can contain both model configs and training configs. The pattern:

1. **Create a training project template** (`templates/project.yaml`) that extends the base training template and points to the current project for model loading:
   ```yaml
   -- extends 'projects/tiny.yaml'
   [config_metadata]
       == super()
       -- set ns.model_project_dir = project_dir
       -- set ns.model_project_config = "baseline.yaml"
   ```

2. **Create training configs** (`configs/train_*.yaml`) that extend the project template and select which model config to train:
   ```yaml
   -- extends 'project.yaml'
   [config_metadata]
       == super()
       -- set ns.config_name = "Train My Variant"
       -- set ns.model_name = "my_variant"
       -- set ns.model_project_config = "variant.yaml"
   ```

3. **Create model variant configs** using inline model templates to override specific blocks. For example, to remove RoPE:
   ```yaml
   -- extends "configs/baseline.yaml"
   [config_metadata]
       == super()
       -- set ns.model_name = "nope_variant"
   [model_definition]
       -- include "config.nope.model"
   #------------- config.nope.model --------------
   -- extends "config.baseline.model"
   [rel_positional_encoder]
   .define: &relative_pe null
   ```

Model configs (`forgather model test`) and training configs (`forgather train`) coexist in the same `configs/` directory. Use `forgather ls` to see both.

See `examples/tiny_experiments/canon/` for a working example with baseline and NoPE model variants plus their training configs.

### Project Installation
```bash
pip install -e .
```

## Checkpointing

Forgather provides automatic distributed checkpoint coordination for multi-GPU and multi-node training. The system uses explicit state sharing patterns to handle complex parallelism strategies.

### Basic Usage

Enable checkpointing in training arguments:

```python
args = TrainingArguments(
    output_dir="output_models/my_model",
    save_strategy="steps",
    save_steps=1000,                  # Save every 1000 steps
    save_total_limit=3,               # Keep only last 3 checkpoints
)

trainer = Trainer(model=model, args=args, ...)
trainer.train()  # Checkpoints saved automatically
```

All state is saved: model, optimizer, scheduler, dataset position, RNG, training progress.

**Output:**
```
output_models/my_model/
├── checkpoint-1000/
│   ├── model.safetensors
│   ├── optimizer_state.pt
│   ├── scheduler_state.pt
│   ├── dataset_state.pt
│   ├── rng_state.pt
│   ├── trainer_state.pt
│   └── checkpoint_manifest.json    # Metadata for debugging
├── checkpoint-2000/
└── checkpoint-3000/
```

### Resuming from Checkpoint

```python
args = TrainingArguments(
    output_dir="output_models/my_model",
    resume_from_checkpoint=True,      # Auto-finds latest checkpoint
    max_steps=5000,
)

trainer = Trainer(model=model, args=args, ...)
trainer.train()  # Continues from checkpoint-3000
```

**To skip loading components** (e.g., changing dataset):
```bash
# Manually delete the checkpoint files you don't want to restore
rm checkpoint-1000/dataset_state.pt
rm checkpoint-1000/trainer_state.pt
```
Checkpoint loading will warn about missing components but continue with your current configuration.

**Note:** Model weights are always required and cannot be skipped.

### Distributed Training Patterns

**Data Parallel (DDP):**
```python
from forgather.ml.trainer.ddp import DDPTrainer, DDPTrainingArguments

args = DDPTrainingArguments(
    dispatch_batches=True,            # Rank 0 loads and dispatches data
    save_strategy="steps",
    save_steps=1000,
)

# Launch with: torchrun --nproc_per_node=4 train.py
trainer = DDPTrainer(model=model, args=args, ...)
trainer.train()

# Model/optimizer saved once (REPLICATED - DDP synchronizes)
# Dataset saved once (GLOBAL - centralized loading)
# RNG saved per rank (PER_RANK - each rank needs different random numbers)
```

**Pipeline Parallel:**
```python
from forgather.ml.trainer.pipeline import PipelineTrainer

trainer = PipelineTrainer(
    model_splitter=split_model_into_stages,
    args=args,
    ...
)
trainer.train()

# Model/optimizer saved per rank (PER_RANK - different pipeline stages)
# Dataset saved once (GLOBAL - rank 0 loads and broadcasts)
```

### Checkpoint Manifests

Every checkpoint includes a manifest with complete metadata:

```json
{
  "checkpoint_path": "/path/to/checkpoint-1000",
  "world_size": 4,
  "timestamp": "2026-01-24T10:30:45",
  "components": {
    "model": {
      "sharing_pattern": "replicated",
      "ranks": [0],
      "size_bytes": 445678123
    },
    "rng": {
      "sharing_pattern": "per_rank",
      "ranks": [0, 1, 2, 3],
      "size_bytes": 14042
    }
  }
}
```

Use manifests for debugging checkpoint issues or verifying what was saved.

### Common Issues

**Training hangs during checkpoint save:**
- Distributed barrier deadlock (fixed in built-in trainers)
- Ensure all ranks call checkpoint save methods

**Different results after resume:**
- Ensure RNG and dataset state files were not deleted
- Both are saved automatically for exact reproducibility

**Validation failed for component 'optimizer':**
- Known issue with AccelerateOptimizer wrapper (validation automatically disabled)
- Model validation still works correctly

### Documentation

For complete documentation, see:
- **User Guide**: `docs/checkpointing/user_guide.md` - Practical usage and troubleshooting
- **Technical Details**: `docs/checkpointing/distributed_checkpoint_abstraction.md` - Architecture and patterns
- **Migration Guide**: `docs/checkpointing/migration_guide.md` - Implementing custom trainers

**Key Features:**
- Automatic coordination for multi-GPU/multi-node training
- All state always saved (model, optimizer, scheduler, dataset, RNG, training progress)
- Explicit state sharing patterns (GLOBAL, PER_RANK, REPLICATED, PER_GROUP, PER_NODE)
- Optional replication validation to catch DDP synchronization bugs
- Complete checkpoint manifests for debugging
- Partial loading via manual file deletion

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
- Cross-project model inheritance (with modelsrc) : "examples/tiny_experiments/canon/"
- Cross-project model inheritance (without modelsrc) : "examples/pretrain/small-llm/custom_deepone/"

**Common Template Override Patterns**
- To override a block in the model template chain (e.g., `[rel_positional_encoder]`, `[attention_factory]`), you must do it in an **inline model template** (after `#--- config.*.model ---`), not in the main config section
- To disable RoPE: override `[rel_positional_encoder]` with `.define: &relative_pe null` — attention modules guard with `if self.pos_encoder:`
- To disable a factory-based component: set the anchor to `null` in the appropriate block
- The `projects/tiny.yaml` template provides a complete training setup for small model experiments (TinyStories dataset, 1 epoch, AdamW + InfiniteLR scheduler)
- Use `-- set ns.model_project_dir = project_dir` to reference the current project as a model project from a training config

**Common Issues and Solutions**
- Missing import errors (e.g., `Callable` not imported): Add missing imports to affected files
- YAML tag errors: Use `!partial` for function objects, `!singleton`/`!factory` for function calls
- Configuration validation: Run `forgather ls` to check all configs parse correctly
- Complex64 serialization: RoPE models may fail to save due to safetensors limitations with complex tensors
- `ModuleNotFoundError` when extending model projects with `modelsrc/`: The `project_dir` variable in `[model_submodule_searchpath]` resolves to the current project, not the base model project. Override the search path to point to the base model's modelsrc directory. See "Creating a Project That Extends Another Model Project" above.
- **Template name shadowing / RecursionError**: When multiple model projects are in the search path, their config files may shadow each other (e.g., both `llama/` and `llama_canon/` have `configs/4M.yaml`). A child config named `4M.yaml` that extends `configs/4M.yaml` will resolve to itself, causing infinite recursion. Fix: use a distinct config name (e.g., `nope_4M.yaml`) or create a separate model project for each base model you extend.
- **Extending multiple model projects**: When experiments need model variants from different base models (e.g., Canon + plain Llama), create a separate model project for each base rather than adding all templates to one search path. This avoids name shadowing and keeps template resolution predictable. Example: `examples/tiny_experiments/llama_nope/` is a separate project for the plain Llama NoPE variant, referenced from canon training configs via `ns.model_project_dir`.

**Style**

Follow existing style conventions. Avoid emojis.
