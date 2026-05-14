# FP8 Training

Forgather supports FP8 (8-bit floating point) training via [torchao](https://github.com/pytorch/ao).
FP8 training replaces the matrix multiplications in Linear layers with FP8-quantized versions
using `torch._scaled_mm`, reducing compute cost and memory bandwidth requirements compared to
bf16/fp16.

## Requirements

- **GPU**: NVIDIA Ada Lovelace (RTX 4090) or Hopper (H100) or newer -- CUDA compute capability >= 8.9
- **PyTorch**: >= 2.4 with CUDA support
- **torchao**: installed (`pip install torchao`)
- **Dimension alignment**: All Linear layer dimensions (in_features, out_features) must be
  divisible by 16 for FP8. Layers that don't meet this are automatically skipped and remain
  as standard Linear layers.

torchao ships as a pure-Python (`py3-none-any`) wheel and resolves cleanly on `aarch64`
(DGX Spark / GB10), so the Forgather Docker images install it from PyPI on every arch.
The float8 ops delegate to `torch._scaled_mm` and friends, which are built into the
PyTorch wheel.

## Quick Start

Add `fp8_recipe` to your trainer arguments:

```yaml
[trainer_args]
    == super()
    mixed_precision: "bf16"
    fp8_recipe: "tensorwise"
```

Or from the command line:

```bash
forgather -t config.yaml train --fp8-recipe tensorwise
```

FP8 is orthogonal to mixed precision -- use both together. `mixed_precision: "bf16"` handles
non-linear operations (LayerNorm, softmax, activations) via torch.autocast, while `fp8_recipe`
handles Linear layer matmuls in FP8. This is the recommended configuration.

## Recipes

Three pre-built recipes are available, controlling how weights, activations, and gradients
are quantized across the three matmuls in each Linear layer (forward, grad_input, grad_weight):

| Recipe | Scaling | Speed | Accuracy | Notes |
|--------|---------|-------|----------|-------|
| `tensorwise` | Per-tensor | Fastest | Good | Default. Uses cuBLAS kernel. |
| `rowwise` | Per-row/column | Medium | Better | Uses CUTLASS kernel. Scales rounded to power-of-2. |
| `rowwise_with_gw_hp` | Per-row (fwd/bwd), high-precision (grad_weight) | Slower | Best | Keeps grad_weight computation in original precision. **Broken in torchao 0.16.0 for ND inputs** -- see Limitations. |

Start with `tensorwise` for maximum throughput. Switch to `rowwise` or `rowwise_with_gw_hp`
if you observe training instability or degraded convergence compared to bf16.

## Configuration

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `fp8_recipe` | str or null | null | FP8 recipe name. null = disabled. |
| `fp8_dim_alignment` | int | 16 | Minimum dimension alignment for FP8 conversion. Layers with in_features or out_features not divisible by this value are skipped. Set to 0 to disable filtering. |

### Example Configuration

```yaml
[trainer_args]
    == super()
    mixed_precision: "bf16"
    fp8_recipe: "rowwise"
    fp8_dim_alignment: 16
    torch_compile: True
```

`torch_compile: True` is recommended with FP8 for best performance, but not required.

## How It Works

During trainer initialization (`_prepare_model()`), after the model is constructed and moved
to device:

1. All `nn.Linear` layers are inspected
2. Layers with dimensions not divisible by `fp8_dim_alignment` are skipped (logged at INFO level)
3. Eligible layers are replaced with `Float8Linear` from torchao
4. A summary is logged: `FP8 training (tensorwise): converted 42/44 Linear layers`

During training:
- **Forward pass**: Input and weight are dynamically cast to FP8 (e4m3fn), matmul runs via
  `torch._scaled_mm`, result returned in original dtype
- **Backward pass**: Gradients are cast to FP8 for the two backward matmuls (grad_input, grad_weight)
- **Optimizer step**: Operates on weights in original precision (bf16/fp32) -- FP8 casting is transient

### Scaling

Each FP8 cast computes a dynamic scale factor: `scale = max_fp8 / max(abs(tensor))`. This
maps the tensor's value range into the FP8 representable range. For `tensorwise`, one scale
per tensor. For `rowwise`, one scale per row/column.

## Checkpointing

FP8 training is fully checkpoint-compatible:

- Weights are stored in their original precision (bf16/fp32), not FP8
- FP8 scale factors are recomputed dynamically during each forward pass
- Checkpoints saved during FP8 training can be loaded into standard models (without FP8), and vice versa
- No special handling is needed for save or resume

## Distributed Training

### DDP

DDP works with FP8 training without any additional configuration. The FP8 conversion happens
before DDP wrapping, so each rank runs FP8 matmuls locally while gradient all-reduce operates
in the original dtype. This provides the compute speedup but not communication reduction.

### Pipeline Parallel

The pipeline trainer applies FP8 conversion to each pipeline stage independently after
materialization. Each stage's Linear layers are converted on their assigned device.

### FSDP

FSDP2 integration (FP8 all-gather for reduced communication) is not yet implemented.
This is planned for a future release.

## Limitations

- Requires matmul dimensions divisible by 16 (hardware constraint). Layers that don't meet
  this are silently left as standard Linear and logged.
- The `blockwise` FP8 recipe (prototype in torchao) requires SM 9.0+ (Hopper) and is not
  exposed through this interface.
- FP8 training benefits vary by model size and architecture. For very small models, the
  overhead of scale computation may negate the compute savings.
- **`rowwise_with_gw_hp` recipe is broken in torchao 0.16.0 for ND inputs.** Transformer
  hidden states have shape `(batch, seq, hidden)`, and torchao's
  `matmul_with_hp_or_float8_args.forward` reshapes the input to 2D before the matmul.
  Reshape on an axiswise-scaled `Float8Tensor` is unimplemented and trips
  `AssertionError: aten.reshape.default with axiswise scaling is not supported yet` in
  `torchao/float8/float8_ops.py`. Plain `rowwise` uses a different autograd path and is
  unaffected. Use `tensorwise` or `rowwise` until this is fixed upstream.

- **`rowwise` recipe fails on Blackwell SM 12.1 (DGX Spark / GB10) with torch 2.10 / cu128.**
  The recipe's dynamic scale rounding (`torch.exp2(torch.floor(torch.log2(scale)))`) gets
  JIT-compiled by inductor, and the NVRTC bundled with torch 2.10 cu128 wheels rejects the
  arch flag for SM 12.1 with `nvrtc: error: invalid value for --gpu-architecture (-arch)`.
  PyTorch's own warning surfaces the underlying issue: "Maximum cuda capability supported
  by this version of PyTorch is (8.0) - (12.0)". `tensorwise` works on GB10 because its
  cuBLAS path (`torch._scaled_mm`) doesn't go through NVRTC. Fix is in upstream
  PyTorch/cu128 nightlies; until then, prefer `tensorwise` on Blackwell.

## Programmatic Usage

```python
from forgather.ml.trainer import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="output_models/my_model",
    mixed_precision="bf16",
    fp8_recipe="tensorwise",
    torch_compile=True,
    # ... other training args
)

trainer = Trainer(
    args=args,
    model_init=model_factory,
    train_dataset=train_dataset,
    # ...
)

trainer.train()
```

## Troubleshooting

**"Expected trailing dimension of mat1 to be divisible by 16"**: A Linear layer with
unaligned dimensions was converted to FP8. Increase `fp8_dim_alignment` or check that
your model's hidden dimensions are multiples of 16.

**"Skipping import of cpp extensions"**: torchao version mismatch with PyTorch. FP8 training
still works via the Python fallback path, but compiled C++ extensions may provide better
performance. Update torchao to match your PyTorch version.

**No speedup observed**: Ensure `torch_compile=True` is set. Without compilation, the
overhead of FP8 scale computation and casting can offset the matmul speedup, especially
for small models.
