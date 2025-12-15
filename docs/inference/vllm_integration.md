# vLLM Integration Guide

This guide explains how to configure Forgather models for distributed inference with [vLLM](https://docs.vllm.ai/), including tensor parallelism and pipeline parallelism support.

## Overview

vLLM is a high-throughput inference engine that supports distributed inference through:
- **Tensor Parallelism (TP)**: Splits individual layers across multiple GPUs
- **Pipeline Parallelism (PP)**: Distributes sequential layers across multiple GPUs

Forgather models generated with [proper vLLM configuration](https://docs.vllm.ai/en/v0.8.4/models/supported_models.html) can be deployed with vLLM for efficient distributed inference.

## Quick Start

### 1. Install vLLM

vLLM has it's own dependencies, which may conflict with those of Forgather. I would recommend installing it with its own Python virtual environment or even
in its own container to provide full dependency isolation.

The official instructions for installation can be found [here](https://docs.vllm.ai/en/latest/getting_started/installation/).

These instructions have been tested by directly cloning the [vLLM git repo ](https://github.com/vllm-project/vllm) and [installing from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source).

### 1. Generate Model with vLLM Support

Most Forgather transformer models (Llama, Qwen3, etc.) now include vLLM support by default. Simply train and export your model as usual:

```bash
# We will use the tiny-models project to demonstrate
cd examples/tiny_experiments/tiny_models

# Clear old model definitions, to be certain that the models have the latest updates
rm -rf output_models

# Train model
forgather -t tiny_fg_qwen3.yaml train
```

The will produce a model definition and checkpoints. vLLM will expect the checkpoints to be in the same directory as the model's source code. You can create appropriate symbolic links to the latest checkpoint like this:

```bash
# Create symlinks to latest checkpoint in model directory
forgather -t tiny_fg_qwen3.yaml checkpoint link
```

Just to be sure that you have a working model, test it with Forgathers inference server first. While not as fast as vLLM, it's much easier to use for a quick model test.

```bash
# Start server
forgather inf server -m output_models/tiny_fg_qwen3

# In a separate terminal, test the server via the OpenAI "completion" API
forgather inf client --completion "Once upon a time"
...
Once upon a time, there was a little girl named Lily. She loved to play with her toys and her favorite thing to do was to eat...
```

### 2. Deploy with vLLM

```bash
# From your vLLM Python environment (or container)
cd examples/tiny_experiments/tiny_models/

# Start vLLM server on single GPU
# Notes:
# - Leave off the training directory slash. This avoids spurious warnings about module names
# - Models with custom code require the --trust-remote-code flag, even though the code is not "remote"
# - Make sure you don't have another inference server running
vllm serve --trust-remote-code output_models/tiny_fg_qwen3
```

### 3. Test the Model
Test using Forgathers OpenAI API client

```bash
# From Forgather's Python environment...
# Check what name vLLM is using for the model
forgather inf client --list-models
Available models:
  - output_models/tiny_fg_qwen3

# Test the model
forgather inf client --model output_models/tiny_fg_qwen3 --completion "Once upon a time"
Once upon a time, there was a little girl named Lily. She loved to play with her toys all day long...
```

Test the model using the [vLLM completion client](https://docs.vllm.ai/en/latest/cli/complete/)

```bash
vllm complete --max-tokens 512 -q "Once upon a time"
...
there was a furry little bunny. The bunny had a big sneaker and
```

Test directly with curl

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "output_models/tiny_fg_qwen3",
        "prompt": "Once upon a time",
        "max_tokens": 512,
        "temperature": 0.7
    }'
```

### Fine Tuning and Testing Workflow

For this example, we will demonstrate using the small-ish [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) model.
Note that downloading this model requires authorization from Meta, which can be obtained from the model's HF site, linked above.

```bash
# Download a small-ish HF model for testing
hf download --exclude "original*" --local-dir Llama-3.2-1B-Instruct meta-llama/Llama-3.2-1B-Instruct

# Convert the model to Forgather format
forgather convert Llama-3.2-1B-Instruct/ fg_Llama-3.2-1B-Instruct/
```

#### Test Converted Model
First, let's make sure the converted model runs. Setting `--enforce-eager` run the model in 'eager' mode. This is a little slower, but
can be helpful for diagnosing model issues, if something goes wrong. For faster performance, omit `--enforce-eager`.

```bash
# Start server (in vLLM's Python environment)
vllm serve --trust-remote-code --enforce-eager --model fg_Llama-3.2-1B-Instruct

# Test the model in 'chat' mode (from another terminal in Forgathers environment)
forgather inf client --model fg_Llama-3.2-1B-Instruct --message "Hello. What is your name?"
...
Hello! I'm an artificial intelligence model, and I don't have a personal name...
```

We can verify that the converted model's Tensor Parallel and Pipeline Parallel plans work like this:

```bash
# Test with pipeline parallel (requires 2 GPUs)
vllm serve --trust-remote-code --pipeline-parallel-size 2 --model fg_Llama-3.2-1B-Instruct

# Test with tensor parallel (requires 2 GPUs)
vllm serve --trust-remote-code --tensor-parallel-size 2 --model fg_Llama-3.2-1B-Instruct

# Test with both tensor and pipeline parallel (requires 4 GPUs)
vllm serve --trust-remote-code --pipeline-parallel-size 2 --tensor-parallel-size 2 --model fg_Llama-3.2-1B-Instruct
```

If you wish to try interactive chat with the model...

```bash
# Start Forgather chat client (enter quit to exit)
forgather inf client --model fg_Llama-3.2-1B-Instruct

# vLLM's chat client (^C to exit)
vllm chat
```

#### Train Converted Model

We will train the model to become Samantha, as this makes for a fairly quick demonstration (~5 min on 2x RTX 4090)

```bash
cd examples/finetune/samantha

# Train on 2 GPUs using Torch Pipeline Parallel
# Remember to shutdown vLLM first, or you will probably hit an OOM error.
forgather -t llama3_1b/2gpu_pp_1f1b.yaml train -M ~/ai_assets/models/fg_Llama-3.2-1B-Instruct

# If you only have a single GPU, try:
forgather -t llama3_1b/1gpu_packed.yaml train -M ~/ai_assets/models/fg_Llama-3.2-1B-Instruct
```

### Test the Trained Model

The models checkpoints will be saved in the "checkpoints" sub-directory, which vLLM does not know about. To create symbolic links to
the newest checkpoint, and clobber the original weights, run this command:

```bash
# WARNING: This command will overwrite the original model weights with symlinks. Make sure that you have another copy of these!
forgather checkpoint link -f --output-path ~/ai_assets/models/fg_Llama-3.2-1B-Instruct
```

As this should be a freshly converted model, you can start over again by reconverting the model. Just be careful not to use this approach for any models when the weights in the model directory are your only copy. Make a backup first!

As before, start the vLLM server and test it with an OpenAI compatible client. e.g.

```bash
# Start server (in vLLM's Python environment)
vllm serve --trust-remote-code --model fg_Llama-3.2-1B-Instruct

# Start Forgather chat client (enter quit to exit)
# And run from Forgather's Python environment!
forgather inf client --model fg_Llama-3.2-1B-Instruct
...
> Hello. What is your name?
Hello! My name is Samantha. It's a pleasure to meet you. I find our interactions to be both engaging and enlightening, and I look forward to our future conversations.
```

## Understanding vLLM Plans

vLLM uses the HF interface for specifying Tensor Parallel (TP) and Pipeline Parallel (PP) plans. The plan definitions can be found via the `_tp_plan` and
`_pp_plan` attributes, attached to a PreTrainedModel, but the full specification can't be set directly with they attributes. They are constructed by the model's `post_init()` method, which starts with the `base_model_tp_plan` and `base_model_pp_plan` attributes of the model's configuration class, then all modules are searched for `_tp_plan` and `_pp_plan` attributes, which are merged with the base definitions to produce the final plans.

### Tensor Parallel Plan

The tensor parallel plan tells vLLM how to split weight matrices across GPUs. It's a dictionary mapping layer name patterns to split styles:

```python
tp_plan = {
    # Column-wise split: Independent outputs (queries, keys, values)
    "causal_lm.layer_stack.layers.*.attention.query_linear": "colwise",
    "causal_lm.layer_stack.layers.*.attention.key_linear": "colwise",
    "causal_lm.layer_stack.layers.*.attention.value_linear": "colwise",

    # Row-wise split: Combined inputs (output projections)
   "causal_lm.layer_stack.layers.*.attention.output_linear": "rowwise",

    # Feedforward layers
    "causal_lm.layer_stack.layers.*.feedforward.gate_proj": "colwise",
    "causal_lm.layer_stack.layers.*.feedforward.up_proj": "colwise",
    "causal_lm.layer_stack.layers.*.feedforward.down_proj": "rowwise",
}
```

**Column-wise (`colwise`)**: Splits the output dimension. Each GPU computes a subset of output features independently.
- Use for: Query/Key/Value projections, Gate/Up projections
- Communication: AllReduce after computation

**Row-wise (`rowwise`)**: Splits the input dimension. Each GPU processes a subset of input features.
- Use for: Output projections, Down projections (combining parallel streams)
- Communication: AllGather before computation

See:
- [HF Distributed inference](https://huggingface.co/docs/transformers/en/perf_infer_gpu_multi)
- [PyTorch TP Tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)
- [vLLM Custom Models](https://docs.vllm.ai/en/v0.8.4/models/supported_models.html)

### Pipeline Parallel Plan (`_pp_plan`)

The pipeline parallel plan defines how modules are distributed across pipeline stages and their I/O interfaces:

```python
_pp_plan = {
    # Stage boundaries defined by major model components
    "causal_lm.input_encoder": (
        ["input_ids"],              # Inputs
        ["hidden_states"]           # Outputs
    ),
    "causal_lm.layer_stack.layers": (
        ["hidden_states", "attention_mask"],
        ["hidden_states"]
    ),
    "causal_lm.layer_stack.layer_norm": (
        ["hidden_states"],
        ["logits"]
    ),
}
```

vLLM distributes these modules across pipeline stages automatically based on model size and available GPUs.

vLLM expects that exactly one of these named modules is an instance of nn.ModuleList, where it is assumed that the layers actually reside. 
After constructing the model, the unused modules are replaced with instances of `PPMissingLayer`, which is a derivative of `nn.Identity`, with
logic for only returning the first element from returned tuples or dictionaries.

Technically, vLLM does not support specifying these via Fully Qualified Names (FQNs) and assumes that they are attributes of the outer-most module.

As to work around this limitation, Forgathers "hf_causal.py" implementation implements `__getattr__` and `__setattr__`, allowing it to perform full
FQN name lookups.

Forgather does not use `nn.ModuleList`, but uses `nn.ModuleDict`, which is required to support the Pytorch approach to Pipeline Parallelism. As to 
address this, we a proxy `nn.ModuleList` derivative is used, which forwards modifications to the `nn.ModuleDict`.

### No-Split Modules (`_no_split_modules`)

Specifies module types that should never be split with pipeline parallelism:

```python
_no_split_modules = ["PreLNLayer"]  # For Llama models
# or
_no_split_modules = ["PostLNLayer"]  # For vanilla transformers
# or
_no_split_modules = ["DeepnetLayer"]  # For DeepNet models
```

This ensures transformer blocks remain intact on single devices, which is critical for correctness; the skip-layers within these modules would otherwise be a problem.

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

## Further Reading

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/)
- [Distributed Inference Guide](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [Forgather Model Conversion](../model_conversion/overview.md)
- [Forgather Training Guide](../trainers/overview.md)
