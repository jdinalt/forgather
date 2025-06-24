# Creating External Forgather Workspaces

This guide demonstrates how to create and use Forgather projects outside the main forgather repository directory. You'll learn to set up the workspace configuration, create custom models, and train them successfully.

## Overview

When working with Forgather outside the main repository, you need to:

1. **Create a `forgather_workspace` directory** that points to the Forgather installation
2. **Set up proper template search paths** to access Forgather's templatelib
3. **Create your project** with proper template inheritance
4. **Test and train** your models

This guide walks through creating an 8M parameter transformer with ALiBi attention and GLU feedforward layers trained on TinyStories, based on the comprehensive example in `examples/standalone/external_workspace_example/`.

## Prerequisites

- Forgather installed (`pip install -e .` from source, or via package manager)
- Python environment with PyTorch, transformers, datasets
- Understanding that external workspaces assume Forgather is properly installed

**Important**: This guide assumes Forgather is installed in your Python environment. Do not add Forgather to sys.path manually in notebooks or scripts.

## Prerequisites

- Forgather installed (via `pip install -e .` from source)
- Python environment with PyTorch, transformers, datasets
- Access to the Forgather source directory for templatelib access

## Step 1: Create External Workspace Structure

First, create your workspace directory structure:

```bash
# Create workspace root
mkdir -p /path/to/your/workspace

# Create forgather_workspace configuration directory
mkdir -p /path/to/your/workspace/forgather_workspace
```

## Step 2: Configure Workspace Templates

Create the workspace configuration files to enable access to Forgather's template library.

### base_directories.yaml

```yaml
## The root directories from which to compute the locations of other directories.
## This file is common to both config and meta-config templates.

## The following absolute locations are available, as starting points:
## project_dir : The location of the project directory
## workspace_root : The location of the workspace-root directory.
## forgather_config_dir() : The user's Forgather configuration directory
##    On Linux, this is "~/.config/forgather"
## user_home_dir() : The location of the user's home directory (~ on Linux).

## Where is Forgather located?
## For external workspaces, this points to the installed Forgather location
-- set ns.forgather_dir = "/path/to/forgather/installation"

## Additional user-defined locations should be added here.
-- set ns.ai_assets_dir = "/path/to/your/assets"
-- set ns.user_projects_dir = "/path/to/your/workspace"
```

**Important**: Update `/path/to/forgather/installation` to point to your actual Forgather installation directory.

**Example**: If Forgather is installed at `/home/user/ai_assets/forgather`, use:
```yaml
-- set ns.forgather_dir = "/home/user/ai_assets/forgather"
```

### meta_defaults.yaml

```yaml
-- set ns = namespace()

## Import default paths; common to meta and regular templates.
-- include "base_directories.yaml"
-- set ns.forgather_templates_dir = joinpath(ns.forgather_dir, "templatelib")

## Search these directories for templates
## The list is split, which makes it easier to selectively append or prepend.
searchdir:
-- block searchdir_project
    - "{{ joinpath(project_dir, 'templates') }}"
-- endblock searchdir_project

-- block searchdir_common
    - "{{ joinpath(workspace_root, 'forgather_workspace') }}"
    - "{{ joinpath(ns.forgather_templates_dir, 'modellib') }}"
    - "{{ joinpath(ns.forgather_templates_dir, 'examples') }}"
    - "{{ joinpath(ns.forgather_templates_dir, 'base') }}"
-- endblock searchdir_common

-- block configs
## Set default prefix and config name.
## config_prefix: "configs"
## default_config: "control.yaml"
-- endblock configs
```

### README.md

```markdown
# External Workspace Configuration

This workspace configures access to the Forgather templatelib from an external project directory.

When a Project is constructed, the enclosing directories are recursively searched for a directory named 'forgather_workspace' and, if found, this directory is implicitly added to the template search path of all enclosed 'meta.yaml' files.

This allows external projects to access Forgather's template library and define common workspace configurations.
```

## Step 3: Create Your Project

Now create your actual project directory:

```bash
# Create project directory
mkdir -p /path/to/your/workspace/my_project
mkdir -p /path/to/your/workspace/my_project/templates/{models,configs}
```

### Project Meta Configuration

Create `meta.yaml`:

```yaml
-- extends 'meta_defaults.yaml'

-- block meta_config
project_name: "My Custom Model Project"
project_description: "External workspace example with custom transformer components"
-- endblock meta_config
```

### Custom Model Template

Create `templates/models/custom_model.yaml`:

```yaml
-- extends 'models/dynamic_causal_transformer.yaml'

-- block model_meta_config
    == super()
    -- set model_def.name = "Custom ALiBi GLU Transformer"
    -- set model_def.description = "8M parameter transformer with ALiBi attention and GLU feedforward layers"
    -- set model_def.short_name = "custom_alibi_glu"
-- endblock model_meta_config

## Override specific components to use interesting alternatives
-- block model_bits
    
    -- block loss_fn
loss_fn: &loss_fn !singleton:.causal_loss:CausalLoss@loss_fn []
    << endblock loss_fn

    -- block layer_norm_factory
layer_norm_factory: &layer_norm_factory !lambda:torch.nn:LayerNorm@layer_norm_factory
    normalized_shape: !var "hidden_size"
    << endblock layer_norm_factory

    ## Use GLU feedforward instead of regular feedforward
    -- block feedforward_factory
feedforward_factory: &feedforward_factory !lambda:.glu_feedforward:GLUFeedforwardLayer@feedforward_factory
    d_model: !var "hidden_size"
    d_feedforward: !var "dim_feedforward"
    activation_factory: !lambda:torch.nn:SiLU
    dropout: !var "activation_dropout"
    << endblock feedforward_factory
    
    ## Use ALiBi attention instead of regular multihead attention
    -- block attention_factory
attention_factory: &attention_factory !lambda:.causal_alibi_attn:CausalAlibiAttn@attention_factory
    d_model: !var "hidden_size"
    num_heads: !var "num_attention_heads"
    dropout: !var "attention_dropout"
    trainable_alibi: true
    alt_alibi_init: false
    << endblock attention_factory

    -- block layer_factory
layer_factory: &layer_factory !lambda:.post_ln_layer:PostLNLayer@layer_factory
    feedforward_factory: *feedforward_factory
    attention_factory: *attention_factory
    norm_factory: *layer_norm_factory
    dropout: !var "layer_dropout"
    residual_dropout: !var "residual_dropout"
    << endblock layer_factory

    -- block layer_stack
layer_stack: &layer_stack !singleton:.layer_stack:LayerStack@layer_stack
    layer_factory: *layer_factory
    num_hidden_layers: !var "num_hidden_layers"
    << endblock layer_stack

    -- block output_decoder
output_decoder: &output_decoder !singleton:torch.nn:Linear@output_decoder
    - !var "hidden_size"
    - !var "vocab_size"
    << endblock output_decoder

    ## Use NullPE since ALiBi handles positional information
    -- block positional_encoder
positional_encoder: &positional_encoder !singleton:.null_pe:NullPE@positional_encoder
    << endblock positional_encoder

    -- block input_encoder
input_encoder: &input_encoder !singleton:.input_encoder:InputEncoder@input_encoder
    d_model: !var "hidden_size"
    vocab_size: !var "vocab_size"
    dropout: !var "embedding_dropout"
    positional_encoder: *positional_encoder
    << endblock input_encoder

    -- block init_weights
init_weights: &init_weights !lambda:.init_weights:simple_weight_init@init_weights []
    << endblock init_weights

    -- block model_factory
model_factory: &model_factory !singleton:.causal_lm:CasualLM@model_factory
    loss_fn: *loss_fn
    input_encoder: *input_encoder
    output_decoder: *output_decoder
    layer_stack: *layer_stack
    init_weights: *init_weights
    << endblock model_factory
    
<< endblock model_bits

-- block model_config
    == super()
    ## ~8M parameter configuration
    hidden_size: 288
    num_attention_heads: 6
    num_hidden_layers: 6
    dim_feedforward: 1152
    max_sequence_length: 2048
    
    ## Dropout configuration
    embedding_dropout: 0.1
    layer_dropout: 0.1
    residual_dropout: 0.0
    attention_dropout: 0.1
    activation_dropout: 0.1
<< endblock model_config
```

### Project Template

Create `templates/project.yaml` using the split document approach for complex inheritance:

```yaml
-- extends 'types/training_script/causal_lm/causal_lm.yaml'

-- block config_metadata
    == super()
    -- set ns.config_name = "My Custom Model Project"
    -- set ns.config_description = "External workspace example training a custom model"
    -- set ns.create_new_model = True
    -- set ns.save_model = True
-- endblock config_metadata

-- block model_definition
    -- include 'tokenizers/tiny_2k.yaml'
    -- include 'models/custom_model.yaml'
-- endblock model_definition

-- block datasets_definition
    -- include 'datasets/tiny_stories_abridged.yaml'
-- endblock datasets_definition

-- block trainer_definition
    -- include 'trainers/trainer.yaml'
-- endblock trainer_definition

#-------------------- project.trainer_config --------------------
-- extends 'trainers/trainer.yaml'

-- block trainer_args
    == super()
    
    output_dir: "./output_models/my_model"
    
    ## Training schedule
    num_train_epochs: 1
    per_device_train_batch_size: 8
    per_device_eval_batch_size: 8
    
    ## Optimization
    learning_rate: 3e-4
    weight_decay: 0.01
    warmup_steps: 100
    
    ## Logging and evaluation
    logging_steps: 50
    eval_steps: 200
    eval_strategy: "steps"
    
    ## Checkpointing
    save_steps: 500
    save_strategy: "steps"
    save_total_limit: 2
    
    ## Memory management - prevent OOM on variable length sequences
    ## Limit sequences to 512 tokens
-- endblock trainer_args

-- block datacollator
    == super()
    ## Limit maximum sequence length to prevent OOM
    truncation: True
    max_length: 512
-- endblock datacollator

-- block tokenize_args
    == super()
    ## Truncate during tokenization as well
    max_length: 512
-- endblock tokenize_args
```

**Important**: The split document approach (`#--------------------`) allows complex template inheritance that avoids circular dependencies. Use this pattern when creating trainer configurations that need to override multiple blocks.

### Training Configurations

Create multiple configurations for hyperparameter comparison in `templates/configs/`:

#### baseline.yaml
```yaml
-- extends 'project.yaml'

-- block config_metadata
    == super()
    -- set ns.config_name = "Baseline Training"
    -- set ns.config_description = "Standard hyperparameters for baseline comparison"
    -- set ns.model_name = "baseline_model"
-- endblock config_metadata

-- block trainer_definition
    -- include 'baseline.trainer_config'
-- endblock trainer_definition

#-------------------- baseline.trainer_config --------------------
-- extends 'project.trainer_config'

-- block trainer_args
    == super()
    
    learning_rate: 3e-4
    weight_decay: 0.01
    per_device_train_batch_size: 8
    warmup_steps: 100
    
    output_dir: "./output_models/baseline"
-- endblock trainer_args
```

#### higher_lr.yaml
```yaml
-- extends 'project.yaml'

-- block project_meta_config
    == super()
    config_name: "Higher Learning Rate"
    config_description: "Testing with higher learning rate and longer warmup"
-- endblock project_meta_config

-- block trainer_args
    == super()
    
    learning_rate: 6e-4
    weight_decay: 0.01
    per_device_train_batch_size: 8
    warmup_steps: 200
    
    output_dir: "./output_models/higher_lr"
-- endblock trainer_args
```

#### larger_batch.yaml
```yaml
-- extends 'project.yaml'

-- block project_meta_config
    == super()
    config_name: "Larger Batch Size"
    config_description: "Using larger batch size with adjusted learning rate"
-- endblock project_meta_config

-- block trainer_args
    == super()
    
    learning_rate: 5e-4
    weight_decay: 0.01
    per_device_train_batch_size: 16
    warmup_steps: 150
    
    output_dir: "./output_models/larger_batch"
-- endblock trainer_args
```

## Step 4: Testing Your Project

### Interactive Testing

Create `project_index.ipynb` for interactive development:

```python
from forgather import Project
import torch

# Create project instance - automatically finds workspace configuration
proj = Project()
print(f"Project loaded successfully!")

# Test loading baseline configuration
proj.load_config("baseline.yaml")
print("Configuration loaded successfully!")

# Get project metadata
meta = proj("meta")
print(f"Config name: {meta['config_name']}")
print(f"Config description: {meta['config_description']}")

# Create and inspect model
model = proj("model")()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Model size: {total_params / 1e6:.1f}M parameters")

# Verify custom components architecture
causal_lm = model.causal_lm
print(f"Positional encoder: {causal_lm.input_encoder.positional_encoder.__class__.__name__}")
first_layer = causal_lm.layer_stack.layers[0]
print(f"Attention: {first_layer.attention.__class__.__name__}")
print(f"Feedforward: {first_layer.feedforward.__class__.__name__}")

# Test tokenizer
tokenizer = proj("tokenizer")
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

# Test dataset
train_dataset = proj("train_dataset")
print(f"Train dataset size: {len(train_dataset)}")

# Test a forward pass
batch_size, seq_len = 2, 64
input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
with torch.no_grad():
    logits = model(input_ids=input_ids)
print(f"Forward pass successful! Output shape: {logits.shape}")
```

**Key Changes from Previous Versions**:
- Removed manual sys.path modification (assumes Forgather is installed)
- Simplified Project constructor (no explicit parameters needed)
- Added architecture verification to confirm custom components

### Command Line Testing

Test configuration loading:

```bash
cd /path/to/your/workspace/my_project

# Test project loading with installed Forgather
python -c "
from forgather import Project

proj = Project()
proj.load_config('baseline.yaml')
model = proj('model')()
print(f'Model created: {sum(p.numel() for p in model.parameters()):,} parameters')
"
```

## Step 5: Training Your Model

### Quick Training Test

First, test with a few steps to verify everything works:

```bash
cd /path/to/your/workspace/my_project

# Edit your config to add max_steps: 5 for quick testing
# Then test training
python -c "
from forgather import Project

proj = Project()
proj.load_config('baseline.yaml')
trainer = proj('trainer')

# Quick test with modified args
trainer.args.max_steps = 5
trainer.args.logging_steps = 1
trainer.args.eval_strategy = 'no'

result = trainer.train()
print(f'Training completed! Final loss: {result.training_loss:.4f}')
print(f'Steps completed: {result.global_step}')
"
```

### Using Forgather CLI (Recommended)

The easiest way to train is using the Forgather CLI:

```bash
cd /path/to/your/workspace/my_project

# Single GPU training
fgcli.py -t baseline.yaml train

# Specify GPU device
fgcli.py -t baseline.yaml train -d 0

# Run different configurations on different GPUs
fgcli.py -t baseline.yaml train -d 0 &
fgcli.py -t higher_lr.yaml train -d 1 &
fgcli.py -t larger_batch.yaml train -d 2 &
```

### Legacy Training Scripts

For compatibility, you can also use the direct training script:

```bash
# Single GPU with environment variable (workaround)
RANK=0 python /path/to/forgather/scripts/train_script.py baseline.yaml

# Multi-GPU training with torchrun
torchrun --nproc-per-node 2 /path/to/forgather/scripts/train_script.py baseline.yaml

# With accelerate
accelerate launch /path/to/forgather/scripts/train_script.py baseline.yaml
```

**Note**: The RANK=0 environment variable is a temporary workaround. The training script should be updated to assume single process when RANK is not set.

## Step 6: Monitoring and Results

### TensorBoard Monitoring

Training automatically creates TensorBoard logs:

```bash
# View training progress
tensorboard --logdir ./output_models/baseline/runs/
```

### Model Output

Your trained models will be saved in:
```
output_models/
├── baseline/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── checkpoints/
│   └── runs/
├── higher_lr/
└── larger_batch/
```

## Key Features Demonstrated

### Custom Transformer Components

This example showcases:

1. **ALiBi Attention** (`CausalAlibiAttn`):
   - Relative positional encoding that extrapolates to longer sequences
   - Trainable ALiBi slopes for improved performance
   - No absolute positional embeddings needed

2. **GLU Feedforward** (`GLUFeedforwardLayer`):
   - Gated Linear Unit variant with SiLU activation
   - Improved capacity compared to standard feedforward layers
   - Efficient implementation using chunking

3. **Dynamic Model Construction**:
   - Template-based model definition
   - Component swapping via configuration
   - Automatic code generation and parameter calculation

### External Workspace Benefits

1. **Isolation**: Keep your projects separate from Forgather source
2. **Flexibility**: Easy to share projects without Forgather dependency
3. **Organization**: Logical separation of different model families
4. **Version Control**: Track your projects independently

## Troubleshooting

### Common Issues and Solutions

#### 1. Template Not Found Errors
**Problem**: `TemplateNotFound` or `FileNotFoundError` when loading configurations
**Solution**: 
- Verify `forgather_dir` path in `base_directories.yaml` points to correct Forgather installation
- Ensure the workspace structure includes `forgather_workspace/` directory
- Check that templates use correct include paths (e.g., `tokenizers/tiny_2k.yaml`)

#### 2. Template Inheritance Issues  
**Problem**: Complex inheritance creates circular dependencies or unexpected behavior
**Solution**: Use split document approach with `#--------------------` syntax
```yaml
-- extends 'project.yaml'

-- block trainer_definition  
    -- include 'baseline.trainer_config'
-- endblock

#-------------------- baseline.trainer_config --------------------
-- extends 'project.trainer_config'
```

#### 3. Architecture Compatibility Problems
**Problem**: Mixing incompatible components (e.g., ALiBi + Absolute PE)
**Solution**: 
- ALiBi attention should use `NullPE` (no positional encoding)
- Standard attention should use `SinusoidalPE` or `RotaryPE`
- Never mix relative and absolute positional encoding

#### 4. Out of Memory (OOM) Errors
**Problem**: Training fails with CUDA OOM on variable-length sequences
**Solution**: Add truncation to prevent long sequences
```yaml
-- block datacollator
    == super()
    truncation: True
    max_length: 512
-- endblock datacollator

-- block tokenize_args
    == super()
    max_length: 512
-- endblock tokenize_args
```

#### 5. RANK Environment Variable Issues
**Problem**: Training script requires RANK environment variable
**Solution**: 
- Use Forgather CLI: `fgcli.py -t config.yaml train`
- Or set RANK for legacy scripts: `RANK=0 python scripts/train_script.py`
- Note: This is a temporary workaround; the training script should be fixed

#### 6. Project Constructor Issues
**Problem**: Using unnecessary explicit parameters in notebooks
**Solution**: Simplify to `Project()` - it automatically finds workspace configuration
```python
# Wrong
proj = Project(project_dir=".", config_name="")

# Right
proj = Project()
```

#### 7. Import Path Problems
**Problem**: Manually adding Forgather to sys.path
**Solution**: Assume Forgather is properly installed, don't modify sys.path
```python
# Wrong
import sys
sys.path.insert(0, '/path/to/forgather')

# Right - just import directly
from forgather import Project
```

### Debugging Configuration

Use the project environment to debug templates:

```python
# Debug template preprocessing
proj = Project(project_dir=".", config_name="")
proj.load_config("baseline.yaml")
print(proj.pp_config)  # View preprocessed configuration
```

### Validation

Verify your setup works:

```bash
# Test each component individually
python -c "
import sys
sys.path.insert(0, '/path/to/forgather/modelsrc')
from forgather import Project

proj = Project(project_dir='.', config_name='baseline.yaml')

# Test all components
tokenizer = proj('tokenizer')
model = proj('model')()
train_dataset = proj('train_dataset')
trainer = proj('trainer')

print('All components created successfully!')
print(f'Tokenizer: {tokenizer.vocab_size} vocab')
print(f'Model: {sum(p.numel() for p in model.parameters()):,} params')
print(f'Dataset: {len(train_dataset)} samples')
print(f'Trainer: {trainer.__class__.__name__}')
"
```

## Complete Working Example

A fully functional external workspace example is available at:
```
examples/standalone/external_workspace_example/
├── forgather_workspace/          # Workspace configuration
│   ├── base_directories.yaml     # Points to Forgather installation
│   ├── meta_defaults.yaml        # Template search paths
│   └── README.md
└── alibi_glu_project/            # Example project
    ├── meta.yaml                 # Project metadata
    ├── templates/                # Project templates
    │   ├── project.yaml          # Main project template
    │   ├── models/               # Custom model definitions
    │   └── configs/              # Training configurations
    ├── output_models/            # Generated models and training results
    ├── project_index.ipynb       # Interactive development notebook
    └── LESSONS_LEARNED.md        # Development notes and insights
```

This example demonstrates:
- **ALiBi vs AbsPE comparison experiment** with documented results
- **Multiple training configurations** for hyperparameter comparison  
- **Proper split document template inheritance**
- **Memory management** with sequence truncation
- **End-to-end workflow** from setup to trained models

## Best Practices

### Development Workflow
1. **Start Simple**: Begin with basic templates, add complexity incrementally
2. **Test Early**: Validate each component before combining
3. **Use Truncation**: Always add sequence length limits to prevent OOM
4. **Systematic Experiments**: Change one hyperparameter at a time
5. **Document Results**: Capture findings immediately after experiments

### Template Design
1. **Use Split Documents**: For complex inheritance with multiple blocks
2. **Follow Naming Conventions**: Clear, descriptive config and model names
3. **Leverage Existing Components**: Reuse templatelib components when possible
4. **Architecture Compatibility**: Verify component combinations make sense
5. **Memory Considerations**: Add truncation blocks to all configurations

### Training Strategy
1. **Quick Validation**: Test with max_steps=5 before full training
2. **Use Forgather CLI**: Preferred over direct script execution
3. **Monitor Progress**: Use TensorBoard and logs actively
4. **Comparative Experiments**: Design controlled comparisons
5. **Save Results**: Document experimental outcomes

## Next Steps

1. **Study the Working Example**: Examine `examples/standalone/external_workspace_example/`
2. **Experiment with Components**: Try different attention mechanisms, normalization schemes
3. **Scale Up**: Create larger models by adjusting architecture parameters
4. **Custom Datasets**: Create templates for your own datasets  
5. **Advanced Training**: Explore multi-GPU and pipeline parallel training
6. **Contribute Back**: Share successful patterns with the Forgather community

This external workspace approach provides a scalable foundation for developing and experimenting with custom models while leveraging Forgather's powerful template system and training infrastructure. The working example demonstrates proven patterns that help avoid common pitfalls and accelerate development.