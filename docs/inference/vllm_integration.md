# vLLM Integration Guide

This guide explains how to configure Forgather models for distributed inference with [vLLM](https://docs.vllm.ai/), including tensor parallelism and pipeline parallelism support.

## Overview

vLLM is a high-throughput inference engine that supports distributed inference through:
- **Tensor Parallelism (TP)**: Splits individual layers across multiple GPUs
- **Pipeline Parallelism (PP)**: Distributes sequential layers across multiple GPUs

Forgather models generated with proper vLLM configuration can be deployed with vLLM for efficient distributed inference.

## Quick Start

### 1. Generate Model with vLLM Support

Most Forgather transformer models (Llama, DeepOne, etc.) now include vLLM support by default. Simply train and export your model as usual:

```bash
# Train model
forgather -t my_config.yaml train

# The generated model will include vLLM plans automatically
```

### 2. Validate vLLM Plans

Before deploying to vLLM, validate that the plans match your model structure:

```python
from forgather.ml.model_conversion import validate_vllm_plans, print_model_structure
from transformers import AutoModelForCausalLM

# Load your trained model
model = AutoModelForCausalLM.from_pretrained("output_models/my_model")

# Print model structure to verify layer naming
print_model_structure(model, max_depth=4)

# Validate vLLM plans
if hasattr(model, '_tp_plan') and model._tp_plan:
    is_valid = validate_vllm_plans(
        model,
        tp_plan=model._tp_plan,
        pp_plan=model._pp_plan,
        strict=True  # Show detailed validation info
    )
    if is_valid:
        print("✓ Model is ready for vLLM deployment")
    else:
        print("✗ vLLM plans need adjustment")
```

### 3. Deploy with vLLM

```bash
# Tensor parallel only (4 GPUs)
vllm serve output_models/my_model --tensor-parallel-size 4

# Tensor + Pipeline parallel (8 GPUs: 2 PP stages, 4 TP per stage)
vllm serve output_models/my_model \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2

# With additional optimization
vllm serve output_models/my_model \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --enable-chunked-prefill
```

## Understanding vLLM Plans

### Tensor Parallel Plan (`_tp_plan`)

The tensor parallel plan tells vLLM how to split weight matrices across GPUs. It's a dictionary mapping layer name patterns to split styles:

```python
_tp_plan = {
    # Column-wise split: Independent outputs (queries, keys, values)
    "model.layer_stack.layers.*.attention.query_linear": "colwise",
    "model.layer_stack.layers.*.attention.key_linear": "colwise",
    "model.layer_stack.layers.*.attention.value_linear": "colwise",

    # Row-wise split: Combined inputs (output projections)
    "model.layer_stack.layers.*.attention.output_linear": "rowwise",

    # Feedforward layers
    "model.layer_stack.layers.*.feedforward.gate_proj": "colwise",
    "model.layer_stack.layers.*.feedforward.up_proj": "colwise",
    "model.layer_stack.layers.*.feedforward.down_proj": "rowwise",
}
```

**Column-wise (`colwise`)**: Splits the output dimension. Each GPU computes a subset of output features independently.
- Use for: Query/Key/Value projections, Gate/Up projections
- Communication: AllReduce after computation

**Row-wise (`rowwise`)**: Splits the input dimension. Each GPU processes a subset of input features.
- Use for: Output projections, Down projections (combining parallel streams)
- Communication: AllGather before computation

### Pipeline Parallel Plan (`_pp_plan`)

The pipeline parallel plan defines how modules are distributed across pipeline stages and their I/O interfaces:

```python
_pp_plan = {
    # Stage boundaries defined by major model components
    "model.input_encoder": (
        ["input_ids"],              # Inputs
        ["hidden_states"]           # Outputs
    ),
    "model.layer_stack": (
        ["hidden_states", "attention_mask"],
        ["hidden_states"]
    ),
    "model.output_decoder": (
        ["hidden_states"],
        ["logits"]
    ),
}
```

vLLM distributes these modules across pipeline stages automatically based on model size and available GPUs.

### No-Split Modules (`_no_split_modules`)

Specifies module types that should never be split during parallelism:

```python
_no_split_modules = ["PreLNLayer"]  # For Llama models
# or
_no_split_modules = ["PostLNLayer"]  # For vanilla transformers
# or
_no_split_modules = ["DeepnetLayer"]  # For DeepNet models
```

This ensures transformer blocks remain intact on single devices, which is critical for correctness.

## Customizing vLLM Plans

### For New Model Architectures

If you're creating a custom model architecture, add vLLM support by including the base template:

```yaml
# In your model configuration (e.g., my_custom_model.yaml)
[model_code_generator]
    == super()

    # Basic vLLM support (works for most CausalLM models)
    -- include 'models/causal_lm/vllm_plans.yaml'
```

### For Custom Layer Naming

If your model uses different layer names, override the vLLM plan blocks:

```yaml
[model_code_generator]
    == super()

    -- block vllm_tensor_parallel_plan
    tp_plan:
        # Custom layer naming
        "model.my_layers.*.my_attention.q_proj": "colwise"
        "model.my_layers.*.my_attention.k_proj": "colwise"
        "model.my_layers.*.my_attention.v_proj": "colwise"
        "model.my_layers.*.my_attention.out_proj": "rowwise"
    -- endblock vllm_tensor_parallel_plan
```

### For Non-Standard Architectures

If your model doesn't use the standard `input_encoder -> layer_stack -> output_decoder` structure:

```yaml
[model_code_generator]
    == super()

    -- block vllm_pipeline_parallel_plan
    pp_plan:
        "model.embeddings":
            - ["input_ids"]
            - ["embeddings"]
        "model.encoder":
            - ["embeddings", "mask"]
            - ["encoded"]
        "model.decoder":
            - ["encoded"]
            - ["logits"]
    -- endblock vllm_pipeline_parallel_plan

    -- block vllm_no_split_modules
    no_split_modules: ["MyCustomBlock"]
    -- endblock vllm_no_split_modules
```

## Layer Naming Convention

Forgather uses semantic layer naming that differs from standard HuggingFace naming:

| Component | Forgather Naming | HuggingFace Equivalent |
|-----------|------------------|------------------------|
| Query projection | `attention.query_linear` | `self_attn.q_proj` |
| Key projection | `attention.key_linear` | `self_attn.k_proj` |
| Value projection | `attention.value_linear` | `self_attn.v_proj` |
| Attention output | `attention.output_linear` | `self_attn.o_proj` |
| FFN gate | `feedforward.gate_proj` | `mlp.gate_proj` |
| FFN up | `feedforward.up_proj` | `mlp.up_proj` |
| FFN down | `feedforward.down_proj` | `mlp.down_proj` |

**Important**: vLLM's tensor parallel plans use glob patterns that match your actual layer names. Forgather's default plans use Forgather's naming convention, which works directly with vLLM.

## Troubleshooting

### Plans Not Matching Model Structure

**Problem**: vLLM fails to load model or reports missing parameters.

**Solution**: Use the validation utility to debug:

```python
from forgather.ml.model_conversion import print_model_structure, validate_vllm_plans
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("output_models/my_model")

# Print full model structure
print_model_structure(model, max_depth=5, show_params=True)

# Validate plans with detailed output
validate_vllm_plans(model, tp_plan=model._tp_plan, pp_plan=model._pp_plan, strict=True)
```

The output will show:
- Which patterns matched which parameters
- Suggestions for unmatched patterns
- Full module hierarchy for debugging

### Tensor Parallel Failures

**Problem**: Model works in single-GPU mode but fails with `--tensor-parallel-size > 1`.

**Common causes**:
1. **Missing TP plan entries**: Some linear layers aren't in `_tp_plan`
2. **Wrong split style**: Using `colwise` where `rowwise` is needed (or vice versa)
3. **Custom layers**: vLLM doesn't know how to split custom layer types

**Solution**:
- Ensure ALL linear layers that need splitting are in the TP plan
- Verify split styles match the mathematical operation (see "Tensor Parallel Plan" above)
- For custom layers, you may need to add custom vLLM support

### Pipeline Parallel Failures

**Problem**: Model fails with `--pipeline-parallel-size > 1`.

**Common causes**:
1. **Module not found**: PP plan references modules that don't exist
2. **Wrong I/O specification**: Input/output names don't match forward() signature
3. **Complex control flow**: vLLM PP requires simple sequential execution

**Solution**:
- Use `print_model_structure()` to verify module names
- Check that I/O names match your forward pass
- Simplify model architecture if needed

### Performance Issues

**Problem**: Distributed inference is slower than expected.

**Tips**:
- Use `--dtype bfloat16` for faster computation
- Enable `--enable-chunked-prefill` for better batching
- Tune `--max-num-batched-tokens` based on GPU memory
- For small models, tensor parallelism may add overhead (use PP instead)
- Monitor GPU utilization to identify bottlenecks

## Examples

### Example 1: Llama 7B on 4 GPUs (Tensor Parallel)

```bash
# Generate model with vLLM support (happens automatically)
forgather -t configs/llama_7b.yaml train

# Deploy with tensor parallelism
vllm serve output_models/llama_7b \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 4096
```

### Example 2: Large Model on 8 GPUs (TP + PP)

```bash
# For very large models, combine tensor and pipeline parallelism
vllm serve output_models/llama_70b \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 8192
```

This creates 2 pipeline stages, each with 4-way tensor parallelism (2×4=8 GPUs total).

### Example 3: Custom Validation Script

```python
#!/usr/bin/env python3
"""Validate vLLM plans before deployment."""

import sys
from transformers import AutoModelForCausalLM
from forgather.ml.model_conversion import validate_vllm_plans, print_model_structure

def main(model_path: str):
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path)

    print("\n" + "="*60)
    print("Model Structure")
    print("="*60)
    print_model_structure(model, max_depth=4, show_params=True)

    print("\n" + "="*60)
    print("vLLM Plan Validation")
    print("="*60)

    tp_plan = getattr(model, '_tp_plan', None)
    pp_plan = getattr(model, '_pp_plan', None)

    if not tp_plan and not pp_plan:
        print("⚠ Warning: No vLLM plans found in model")
        print("This model may not support distributed inference with vLLM")
        return 1

    is_valid = validate_vllm_plans(
        model,
        tp_plan=tp_plan,
        pp_plan=pp_plan,
        strict=True
    )

    if is_valid:
        print("\n✓ Model is ready for vLLM deployment!")
        return 0
    else:
        print("\n✗ vLLM plans need adjustment")
        print("See validation messages above for details")
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_path>")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
```

Save as `validate_vllm.py` and run:

```bash
python validate_vllm.py output_models/my_model
```

## Advanced Topics

### Converting Existing Models

If you have a Forgather model trained before vLLM support was added, you can add vLLM plans manually:

1. Update your model configuration to include vLLM plans
2. Regenerate the model code: `forgather -t config.yaml pp`
3. Reload checkpoint weights into the new model structure

### Multi-Node Deployment

vLLM supports multi-node distributed inference:

```bash
# On node 0
vllm serve model_path \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 4 \
    --distributed-executor-backend ray

# Ray will coordinate across nodes automatically
```

### Integration with OpenAI API

vLLM provides an OpenAI-compatible API:

```bash
vllm serve output_models/my_model \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000
```

Then use with OpenAI client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="output_models/my_model",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Further Reading

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/)
- [Distributed Inference Guide](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [Forgather Model Conversion](../model_conversion/overview.md)
- [Forgather Training Guide](../trainers/overview.md)

## Summary

Forgather models generated with the default transformer templates (Llama, DeepOne, etc.) include vLLM support automatically through:
- `_tp_plan`: Defines tensor parallelism strategies
- `_pp_plan`: Defines pipeline parallelism boundaries
- `_no_split_modules`: Protects transformer blocks from splitting

Use the validation utilities to verify plans before deployment, and customize plans for non-standard architectures by overriding template blocks in your model configuration.
