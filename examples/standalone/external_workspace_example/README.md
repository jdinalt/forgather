# External Workspace Example: ALiBi GLU Transformer

This example demonstrates how to create and train custom transformer models in external Forgather workspaces, showcasing advanced architectural components and proper external workspace configuration.

## Workspace Structure

This is a complete external workspace demonstrating:

```
external_workspace_example/
├── README.md                          # This file
├── forgather_workspace/               # Workspace configuration
│   ├── README.md                      # Workspace setup documentation
│   ├── base_directories.yaml          # Points to Forgather installation
│   └── meta_defaults.yaml             # Template search paths
└── alibi_glu_project/                 # Example project
    ├── meta.yaml                      # Project metadata
    ├── project_index.ipynb            # Interactive exploration
    ├── templates/
    │   ├── project.yaml               # Main project configuration
    │   ├── models/
    │   │   └── custom_alibi_glu.yaml  # Custom transformer model
    │   └── configs/
    │       ├── baseline.yaml          # Standard hyperparameters
    │       ├── higher_lr.yaml         # Higher learning rate
    │       ├── larger_batch.yaml      # Larger batch size
    │       └── low_weight_decay.yaml  # Reduced regularization
    └── output_models/                 # Generated during training
```

## Key Workspace Features

### External Configuration

The `forgather_workspace/` directory demonstrates proper external workspace setup:

1. **base_directories.yaml**: Points to Forgather installation
   ```yaml
   -- set ns.forgather_dir = "/path/to/forgather"
   ```

2. **meta_defaults.yaml**: Configures template search paths
   ```yaml
   searchdir:
     - "{{ns.forgather_dir}}/templatelib/base"
     - "{{ns.forgather_dir}}/templatelib/examples"
   ```

This allows projects to use Forgather's template library while being located outside the main repository.

## Custom Transformer Architecture

The project creates an 8M parameter transformer with:
- **ALiBi Attention**: Relative positional encoding for better sequence extrapolation
- **GLU Feedforward**: Gated linear units for improved representational capacity  
- **No Absolute Positional Encoding**: Proper architecture when using ALiBi

### Model Components

1. **ALiBi (Attention with Linear Biases)**
   - Replaces absolute positional embeddings with relative biases
   - Extrapolates to longer sequences than training length
   - Trainable slopes for improved performance

2. **GLU Feedforward Layers**
   - Gated Linear Units with SiLU activation
   - Better representational capacity than standard MLPs
   - Efficient implementation using tensor chunking

3. **Proper Architecture Integration**
   - Uses `NullPE` instead of sinusoidal PE to avoid conflicts with ALiBi
   - All components explicitly defined for clarity and maintainability

## Quick Start

### Prerequisites

1. **Forgather Installation**: Ensure Forgather is installed
2. **Update Workspace**: Edit `forgather_workspace/base_directories.yaml` with your Forgather path
3. **Dependencies**: PyTorch, transformers, datasets

### Interactive Development

Navigate to the project directory and open the notebook:

```bash
cd alibi_glu_project
jupyter notebook project_index.ipynb
```

Update the `FORGATHER_DIR` path in the first cell to your installation.

### Command Line Training

Train with different configurations:

```bash
# Navigate to project directory
cd alibi_glu_project

# Use Forgather CLI for training (automatically detects trainer type and GPU setup)
forgather -t baseline.yaml train
forgather -t higher_lr.yaml train  
forgather -t larger_batch.yaml train

# For multi-GPU training on specific GPUs (useful for concurrent experiments)
forgather -t baseline.yaml train -d 0           # Use GPU 0
forgather -t higher_lr.yaml train -d 1          # Use GPU 1  
forgather -t larger_batch.yaml train -d 2       # Use GPU 2

# To see what command will be run without executing
forgather -t baseline.yaml train --dry-run
```

## Training Configurations

| Configuration | Learning Rate | Batch Size | Weight Decay | Description |
|---------------|---------------|------------|--------------|-------------|
| `baseline.yaml` | 3e-4 | 8 | 0.01 | Standard hyperparameters (ALiBi model) |
| `higher_lr.yaml` | 6e-4 | 8 | 0.01 | Faster convergence |
| `larger_batch.yaml` | 5e-4 | 16 | 0.01 | Larger batch with scaled LR |
| `low_weight_decay.yaml` | 3e-4 | 8 | 0.001 | Reduced regularization |
| `abspe_comparison.yaml` | 3e-4 | 8 | 0.01 | Same model with absolute PE for comparison |

## Experiments and Demonstrations

This example includes several experiments to explore the architectural choices:

### ALiBi vs Absolute Positional Encoding

**Hypothesis**: We expect the ALiBi model to perform better than the absolute PE model because:
1. **Better extrapolation**: ALiBi can handle sequences longer than training length
2. **Relative positioning**: ALiBi captures relative distances which are more generalizable  
3. **Parameter efficiency**: ALiBi doesn't add extra parameters like absolute PE
4. **Training dynamics**: ALiBi may provide better gradient flow for long sequences

**Prediction**: ALiBi model will achieve lower perplexity and faster convergence

Compare the models:
```bash
# Train ALiBi model
forgather -t baseline.yaml train

# Train AbsPE model (can run concurrently on different GPU)
forgather -t abspe_comparison.yaml train -d 1

# Compare training curves in TensorBoard
tensorboard --logdir ./output_models/
```

### Text Generation Demo

Test story completion with the dragon prompt:
```python
prompt = "Once upon a time, there was a little dragon who loved to read stories"
```

**Before training**: Random, incoherent text  
**After training**: Coherent story structure following TinyStories patterns

## Expected Results

With proper training, you should see:
- **Initial loss**: ~7.7 (random initialization)
- **Convergence**: Loss decreasing to ~3-4 range
- **Training speed**: ~20-50 steps/second (depending on hardware)
- **Memory usage**: ~2-4GB GPU memory for batch size 8
- **ALiBi advantage**: Better performance on longer sequences during evaluation

## Architectural Design Choices

### ALiBi vs Absolute Positional Encoding

ALiBi (Attention with Linear Biases) replaces absolute positional embeddings with relative biases applied directly to attention weights. Using both would be redundant and potentially harmful:

1. **ALiBi provides relative positions**: Biases are computed as `(key_pos - query_pos) * slope`
2. **Absolute PE would conflict**: Adding both relative and absolute positional information
3. **Better extrapolation**: ALiBi alone enables better performance on longer sequences

```yaml
# Correct: Use NullPE with ALiBi attention
positional_encoder: !singleton:.null_pe:NullPE
attention_factory: !lambda:.causal_alibi_attn:CausalAlibiAttn
  trainable_alibi: true
```

### Workspace Benefits

External workspaces enable:
- **Project isolation**: Each project has its own directory structure
- **Template reuse**: Access to Forgather's template library
- **Flexible organization**: Organize projects however you prefer
- **Version control**: Independent git repositories for projects

## Extending the Example

### Try Different Components

1. **Alternative Attention**:
   ```yaml
   attention_factory: !lambda:.single_head_alibi_attn:SingleHeadAlibiAttn
   ```

2. **Different Normalization**:
   ```yaml
   layer_factory: !lambda:.pre_ln_layer:PreLNLayer  # Pre-LN instead of Post-LN
   ```

3. **Other Activations**:
   ```yaml
   feedforward_factory: !lambda:.glu_feedforward:GLUFeedforwardLayer
     activation_factory: !lambda:torch.nn:GELU  # GELU instead of SiLU
   ```

### Create New Projects

1. **Copy the workspace structure**:
   ```bash
   cp -r external_workspace_example my_new_workspace
   ```

2. **Update base_directories.yaml** with your Forgather path

3. **Create new projects** in the workspace directory

4. **Customize models and training** as needed

## Troubleshooting

### Common Issues

1. **Module Import Errors**: Ensure modelsrc is in Python path
2. **Template Not Found**: Verify forgather_workspace configuration points to correct Forgather installation
3. **Memory Issues**: Reduce batch size or model size
4. **Sequence Length**: Ensure `max_sequence_length` accommodates your data

### Debugging

Use the notebook cells to debug:
- Test workspace configuration: Verify template loading
- Test model creation: Check parameter count and architecture
- Test forward pass: Verify input/output shapes  
- Quick training: Run a few steps to verify training loop

## References

- [ALiBi Paper](https://arxiv.org/abs/2108.12409): "Train Short, Test Long: Attention with Linear Biases"
- [GLU Variants](https://arxiv.org/abs/2002.05202): "GLU Variants Improve Transformer"  
- [Forgather Documentation](../../../docs/getting-started/external-workspace.md): External workspace setup guide