# Quantization-Aware Training (QAT)

Forgather supports torchao-style quantization-aware training. At training
time `nn.Linear` modules are wrapped in `FakeQuantizedLinear`, which
simulates the target low-bit precision in the forward pass while the
backward pass stays in full precision. The model learns to be robust to the
quantization noise so that the converted (real low-bit) artifact retains
most of the bf16 accuracy.

QAT is a two-phase workflow:

1. **Prepare** -- done at training time via `--qat-recipe`. Inserts fake
   quantizers into the model. Training proceeds normally (the optimizer
   updates full-precision master weights; the fake-quant scales/zero-points
   are recomputed each step).
2. **Convert** -- done after training via `forgather finalize --qat-convert
   <recipe>`. Swaps each `FakeQuantizedLinear` for the real low-bit
   quantized linear op described by the recipe, producing a deployable
   artifact.

## Requirements

- **GPU**: any CUDA GPU (or CPU). QAT runs in full precision; the fake
  quantizers are pure PyTorch math with no hardware gating.
- **torchao**: `>=0.16.0`. Bundled in the Forgather Docker images.

## Quick Start

```bash
# 1. Train with fake quantizers installed
forgather -t config.yaml train --qat-recipe int8-dynamic-act-int4-weight

# 2. After training, produce the deployable quantized artifact
forgather finalize output_models/my_run out/my_run_int8_int4 \
    --qat-convert int8-dynamic-act-int4-weight --safetensors
```

The recipe string passed to `--qat-recipe` and `--qat-convert` must be the
**same** -- the convert step needs the matching base config to know what
scales and dtypes to use. Recipe strings are validated against the registry
in `src/forgather/ml/qat_recipes.py`.

QAT is mutually exclusive with `fp8_recipe`. Both transform `nn.Linear`,
so the trainer rejects the combination at startup.

## Recipes

| Recipe | Activations | Weights | torchao base config |
|--------|-------------|---------|---------------------|
| `int8-dynamic-act-int4-weight` | int8 per-token dynamic | int4 per-group (group_size=32) | `Int8DynamicActivationInt4WeightConfig` |
| `int4-weight-only` | full precision | int4 per-group (group_size=128) | `Int4WeightOnlyConfig` |
| `float8-dynamic-act-float8-weight` | float8 per-row dynamic | float8 per-row | `Float8DynamicActivationFloat8WeightConfig` |
| `float8-dynamic-act-int4-weight` | float8 per-row dynamic | int4 per-group | `Float8DynamicActivationInt4WeightConfig` |

Recommended default: `int8-dynamic-act-int4-weight`. It's the most
broadly-validated production path -- the same recipe Meta and NVIDIA use
when shipping QAT'd LLMs for edge inference.

To add or tweak a recipe (e.g. change `group_size`), edit
`src/forgather/ml/qat_recipes.py:recipe_to_base_config`. Both the trainer
and finalize resolve through the same function, so they stay in sync.

## How It Works

At trainer init, when `qat_recipe` is set:

```python
quantize_(model, QATConfig(base_config, step="prepare"))
```

`quantize_` walks the module tree and swaps each `nn.Linear` for a
`FakeQuantizedLinear` instance. On every forward pass:

1. Activations are quantize-then-dequantize through the activation fake
   quantizer (if the recipe has one).
2. Weights are quantize-then-dequantize through the weight fake quantizer.
3. The matmul runs in the original (bf16/fp32) dtype on the dequantized
   tensors.

In the backward pass nothing about this is special: gradients flow through
the standard linear backward in full precision, and the optimizer updates
the original full-precision weights. The fake quantizers don't have learned
parameters by default -- their scales and zero-points are derived from the
current weight/activation statistics every step.

At finalize, when `--qat-convert <recipe>` is set:

```python
quantize_(model, QATConfig(base_config, step="convert"))
```

This swaps each `FakeQuantizedLinear` for the real low-bit quantized linear
op (e.g. `Int8DynActInt4WeightLinear`). The resulting state_dict contains
torchao's quantized parameter format; `save_pretrained` writes it through
torchao's registered serialization hooks.

## Loss Trajectory Smoke Test

500-step from-scratch runs of `examples/tutorials/tiny_llama:v2.yaml` on
hal9000 (RTX 4090), same model, same seed.

*Trajectory numbers will be filled in by the verification step.*

| step | bf16 | int8-dyn-int4 | int4-weight-only | delta int8 | delta int4-wo |
|------|------|---------------|------------------|------------|---------------|
| 64   |      |               |                  |            |               |
| 128  |      |               |                  |            |               |
| 192  |      |               |                  |            |               |
| 256  |      |               |                  |            |               |
| 320  |      |               |                  |            |               |
| 384  |      |               |                  |            |               |
| 448  |      |               |                  |            |               |

Expected: QAT recipes track bf16 within ~+0.05 loss at step 448. A larger
gap would mean the fake-quant noise is overwhelming the optimization signal
for this model size / learning rate, in which case a longer warmup or
shorter group sizes are the usual remedies.

## Save Format

`forgather finalize --qat-convert` always writes the converted artifact in
PyTorch (`.bin`) format. The `--safetensors` flag is silently disabled with
a warning when both are set: torchao's quantized tensor subclasses
(`Int8DynActInt4WeightLinear`, `Int4Tensor`, etc.) wrap multiple inner
tensors and don't expose a single `.storage().data_ptr()`, which is what
the safetensors writer requires. Until torchao ships explicit safetensors
serialization, `.bin` is the working save format.

The default `.bin` artifact loads cleanly through `torch.load` + the
torchao `quantize_(model, QATConfig(base_config, step="convert"))` re-cast
applied at load time. See the programmatic example below.

## Behavior on Models Without QAT

If you pass `--qat-convert <recipe>` to `forgather finalize` on a model
that wasn't trained with `--qat-recipe`, the convert step is a no-op with
a logged warning -- finalize still produces a valid (un-quantized)
artifact. This is by design so finalize remains a single entry point
regardless of whether QAT was used.

## Programmatic Usage

```python
from forgather.ml.trainer import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="output_models/my_qat_run",
    qat_recipe="int8-dynamic-act-int4-weight",
    # ... other training args
)

trainer = Trainer(
    args=args,
    model_init=model_factory,
    train_dataset=train_dataset,
)
trainer.train()
```

To run convert programmatically:

```python
from torchao.quantization import quantize_
from torchao.quantization.qat import QATConfig
from forgather.ml.qat_recipes import recipe_to_base_config

base_config = recipe_to_base_config("int8-dynamic-act-int4-weight")
quantize_(model, QATConfig(base_config, step="convert"))
model.save_pretrained("out/my_quantized_model", safe_serialization=True)
```

## Out of Scope

The v1 integration intentionally omits a few torchao QAT knobs that aren't
needed for the common case:

- **Auto-convert at training end**: convert is run by `forgather finalize`,
  not the trainer. Keeps training and deployment concerns separated.
- **Custom `group_size` / granularity flags on the CLI**: the per-recipe
  defaults in `qat_recipes.py` are the standard values. Edit them locally
  if you need to experiment.
- **Range learning** (learned per-channel scales): torchao supports it via
  `IntxFakeQuantizeConfig(range_learning=True)`, but the v1 recipes leave
  it off.

## See Also

- [FP8 Training](fp8-training.md) -- the other torchao Linear-swap recipe;
  mutually exclusive with QAT.
- [Finalizing a Trained Model](../guides/finalize-model.md) -- the
  `forgather finalize` reference (including `--qat-convert`).
