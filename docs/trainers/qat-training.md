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
2. **Convert** -- done after training via `forgather finalize --quantize
   <recipe>`. Swaps each `FakeQuantizedLinear` for the real low-bit
   quantized linear op described by the recipe, producing a deployable
   artifact.

`forgather finalize --quantize` also works on **plain bf16 models**, in which
case it performs standard post-training quantization (PTQ) instead of the
QAT round-trip. Same flag, same recipes; the only difference is whether the
source weights were shaped under fake-quant noise during training. See
[PTQ Mode](#ptq-mode) below.

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
    --quantize int8-dynamic-act-int4-weight
```

The converted artifact is always written as `.bin` — see
[Save Format](#save-format) below.

The recipe string passed to `--qat-recipe` and `--quantize` must be the
**same** -- the convert step needs the matching base config to know what
scales and dtypes to use. Recipe strings are validated against the registry
in `src/forgather/ml/qat_recipes.py`.

QAT is mutually exclusive with `fp8_recipe`. Both transform `nn.Linear`,
so the trainer rejects the combination at startup.

## Recipes

| Recipe | Activations | Weights | torchao base config |
|--------|-------------|---------|---------------------|
| `int8-dynamic-act-int4-weight` | int8 per-token dynamic | int4 per-group (group_size=32) | `Int8DynamicActivationIntxWeightConfig` |
| `int4-weight-only` | full precision | int4 per-group (group_size=128) | `Int4WeightOnlyConfig` |
| `float8-dynamic-act-float8-weight` | float8 per-row dynamic | float8 per-row | `Float8DynamicActivationFloat8WeightConfig` |

`float8-dynamic-act-int4-weight` is *not* exposed in v1 — torchao gates its
underlying kernel to the `preshuffled` int4 packing format which is Hopper-only
(SM90+ / FBGEMM). It will be added back behind a runtime capability check.

Recommended default: `int8-dynamic-act-int4-weight`. It's the most
broadly-validated production path -- the same recipe Meta and NVIDIA use
when shipping QAT'd LLMs for edge inference.

To add or tweak a recipe (e.g. change `group_size`), edit
`src/forgather/ml/qat_recipes.py:recipe_to_base_config`. Both the trainer
and finalize resolve through the same function, so they stay in sync.

## Choosing a Recipe

The three recipes hit different points on the memory / compute / accuracy
triangle. Pick one by working backward from the deployment constraint
that matters most.

### Quick guide

| Constraint | Recipe | Why |
|-----------|--------|-----|
| Smallest model on disk / smallest VRAM footprint | `int8-dynamic-act-int4-weight` | int4 weights = ~4× smaller than bf16; int8 activations dequant on the fly. |
| Memory-bound inference, accuracy-sensitive | `int4-weight-only` | int4 weight storage, matmul still in bf16 — keeps the bf16 accuracy ceiling. |
| Throughput on Hopper / Ada / Blackwell, near-lossless | `float8-dynamic-act-float8-weight` | Native fp8 tensor cores; ~2× weight memory savings, throughput parity or better with bf16. |
| Don't know yet | `int8-dynamic-act-int4-weight` | The most broadly-validated production path (same recipe Meta/NVIDIA ship for edge LLMs). |

### Tradeoffs in detail

| Recipe | Weight bits | Activation bits | Storage vs bf16 | Compute path | Hardware floor | Typical accuracy cost |
|--------|-------------|-----------------|-----------------|--------------|----------------|----------------------|
| `int8-dynamic-act-int4-weight` | 4 | 8 (dynamic) | ~4× smaller | int8 matmul (where available); else dequant + bf16 | any CUDA / CPU | small-to-moderate; QAT closes most of it |
| `int4-weight-only` | 4 | bf16 (unchanged) | ~4× smaller | weight dequant → bf16 matmul | any CUDA / CPU | small; matmul stays at bf16 precision |
| `float8-dynamic-act-float8-weight` | 8 | 8 (dynamic) | ~2× smaller | native fp8 matmul on SM 8.9+; emulated below | SM 8.9+ (Ada, Hopper, Blackwell) for speed | near-zero on supported HW |

Important nuances:

- **`int4-weight-only` keeps the matmul in bf16.** It only quantizes the
  *storage* — weights are dequantized to bf16 in the matmul. That makes
  it the safest "I want a smaller model, not a slower one" choice for
  memory-bound regimes. It won't speed up compute-bound regimes.
- **`int8-dynamic-act-int4-weight` quantizes both sides.** Its win is
  storage *and* (on hardware with int8 mma kernels) compute. Its risk is
  the int8 activation path: dynamic per-token scaling is fairly robust,
  but it's the recipe that benefits most from QAT.
- **`float8-...` is the only recipe that needs specific hardware to be
  fast.** On SM < 8.9 it emulates fp8 in software and is slower than
  bf16. Use it on Ada (RTX 4090, L40), Hopper (H100), or Blackwell.

### QAT vs PTQ for each recipe

QAT pays a ~1.67× training-time overhead on small models (Tiny Llama,
RTX 3090; see [Loss Trajectory](#loss-trajectory-1-chinchilla-tiny-llama)).
Whether that's worth it depends heavily on the recipe:

| Recipe | QAT vs PTQ delta (expected) | Recommendation |
|--------|---------------------------|----------------|
| `int8-dynamic-act-int4-weight` | Largest QAT benefit — int4 weights are aggressive enough that plain PTQ can drift noticeably. | QAT if you care about the last point of eval loss; PTQ fine for prototyping. |
| `int4-weight-only` | Moderate QAT benefit. Per-group int4 + bf16 matmul is already quite robust. | PTQ first; reach for QAT only if eval drops more than you can absorb. |
| `float8-dynamic-act-float8-weight` | Minimal — fp8 is already near-lossless. | PTQ. QAT is rarely justified for fp8. |

### Recommended workflow

1. Train bf16 first. Establish your baseline eval loss / perplexity.
2. Run `forgather finalize --quantize <recipe>` on the bf16 model (PTQ).
   Eval the converted model. Most of the time this is good enough.
3. If PTQ eval is unacceptable, re-train with `--qat-recipe <recipe>` and
   convert. Compare the QAT-converted artifact's eval against the PTQ
   one. Decide if the training-time overhead is worth the gap.

This ordering minimizes wasted training cycles: you only pay the QAT
overhead when PTQ has already proven insufficient for *this specific
model + recipe combination*.

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

At finalize, when `--quantize <recipe>` is set:

```python
# 1. Re-install fake quantizers on top of the loaded float weights
quantize_(model, QATConfig(base_config, step="prepare"))
# 2. Swap them for the real low-bit quantized linear ops
quantize_(model, QATConfig(base_config, step="convert"))
```

The first call is necessary because Forgather's sharded checkpoint saver
serialises `state_dict()` which returns *float* weights — the
`FakeQuantizedLinear` modules' scale/zero-point inner state is not
persistent. We re-install fake quantizers from the float weights and then
let convert compute the final low-bit weights and scales. The scales the
convert step picks are derived from the QAT-trained weight statistics, so
the QAT training-time accuracy benefit is preserved.

The result is a model whose `nn.Linear` modules are now torchao subclasses
(`Int8DynActInt4WeightLinear`, etc.). Forgather's `save_checkpoint` writes
the resulting state_dict as PyTorch `.bin` (safetensors is incompatible —
see below).

## Loss Trajectory: 1-Chinchilla Tiny Llama

Full-length training run of `examples/tutorials/tiny_llama:v2.yaml` (Tiny
Llama, 4.43M params, ~82.6M training tokens — chinchilla-optimal at
~20 tokens/param), single GPU (RTX 3090, sm_86, wopr), same seed, same
config. The baseline run uses the v2.yaml default precision settings (bf16
AMP via `mixed_precision: "bf16"`); the QAT run adds `--qat-recipe
int8-dynamic-act-int4-weight` on top.

| Eval step | bf16 AMP baseline (eval_loss) | QAT int8-act-int4-wt (eval_loss) | Δ (QAT − baseline) |
|-----------|-------------------------------|----------------------------------|--------------------|
| 642       | 2.0651                        | 2.0789                           | +0.0138            |
| 1284      | 1.6999                        | 1.7142                           | +0.0143            |
| 1926      | 1.5658                        | 1.5799                           | +0.0141            |
| 4494      | 1.3725                        | 1.3896                           | +0.0171            |
| 5136      | 1.3602                        | 1.3776                           | +0.0174            |
| **5140 (final)** | **1.3601**             | **1.3774**                       | **+0.0173**        |

Final train loss at step 5120 was 1.3352 vs 1.3534 (Δ +0.0182). The two
trajectories track each other from the very first eval through to
completion — QAT pays a stable ~+0.017 eval-loss premium throughout
training rather than a divergent late-training gap, which is the
encouraging signal: the model is learning under the fake-quant noise, not
just accumulating it.

**Wall-clock overhead.** Same GPU, same model, same data:

| Run | Wall time | Steps/sec | Tokens/sec |
|-----|-----------|-----------|------------|
| bf16 AMP baseline | 197 s | 26.1 | 419K |
| QAT int8-act-int4-wt | 329 s | 15.6 | 251K |

QAT is ~1.67× slower than the bf16 baseline (the cost of running the fake
quantizers in pure PyTorch in the forward pass). Whether it pays for
itself depends on what the converted artifact recovers — that comparison
needs `forgather eval` + inference-server support for quantized models
(tracked in #41 and #42).

## Save Format

`forgather finalize --quantize` always writes the converted artifact in
PyTorch (`.bin`) format. The `--safetensors` flag is silently disabled with
a warning when both are set: torchao's quantized tensor subclasses
(`Int8DynActInt4WeightLinear`, `Int4Tensor`, etc.) wrap multiple inner
tensors and don't expose a single `.storage().data_ptr()`, which is what
the safetensors writer requires. Until torchao ships explicit safetensors
serialization, `.bin` is the working save format.

The default `.bin` artifact loads cleanly through `torch.load` + the
torchao `quantize_(model, QATConfig(base_config, step="convert"))` re-cast
applied at load time. See the programmatic example below.

## PTQ Mode

`forgather finalize --quantize <recipe>` accepts any source model — it
does not require `--qat-recipe` at training time. The same
prepare-then-convert pipeline runs regardless; what changes is the
quality of the result:

- **QAT round-trip** (source was trained with `--qat-recipe`): the
  weights were already shaped under fake-quantization noise during
  training. The convert step recovers the QAT training-time accuracy
  benefit. This is the full intended workflow.
- **PTQ** (plain bf16 source, no QAT at training time): the recipe is
  applied to weights that have not seen quantization noise. The result
  is bog-standard post-training quantization — a valid, deployable
  low-bit artifact, but without the QAT accuracy benefit.

PTQ example:

```bash
# Plain bf16 training, no --qat-recipe
forgather -t config.yaml train

# Same finalize flag, plain source — this is PTQ
forgather finalize output_models/my_bf16_run out/my_bf16_run_int8_int4 \
    --quantize int8-dynamic-act-int4-weight
```

This is the path used for the AMP-baseline / PTQ / QAT three-way
comparison: train the same model once in plain bf16, then run finalize
twice (one source bf16, one source QAT) with the same `--quantize`
recipe and compare the eval results. See [Three-Way
Comparison](#three-way-comparison-bf16--ptq--qat) below for measured
numbers on Tiny Llama.

## Evaluating Quantized Models

`forgather eval` loads `--quantize`-finalized models with no extra flag:

```bash
# Same invocation as for a bf16 model — eval autodetects the quantized
# artifact and routes through HF `from_pretrained()` so the
# TorchAoHfQuantizer pre-process path installs the right linear modules.
forgather -p examples/tutorials/tiny_llama eval test tinystories \
    -M /path/to/quantized_model
```

How it works: at finalize time, `--quantize` writes a
`quantization_config` block into `config.json` with the recipe. At eval
time, `scripts/eval_script.py:resolve_checkpoint()` reads that field
and forces the `from_pretrained()` load path (the normal
checkpoint-resume path uses `from_config()` + `load_state_dict()`, which
does not run any quantizer hook and fails when the saved tensors are
torchao quantized subclasses). Same mechanism applies to the inference
server — no caller-side changes needed.

The check is purely additive: bf16 models keep using the
checkpoint-resume path. Pass `--no-checkpoint` to opt into
`from_pretrained()` manually for non-quantized models.

### Three-Way Comparison: bf16 / PTQ / QAT

Full eval against the Tiny Stories test split for the 4.43M Tiny Llama
trained at 1-Chinchilla on a single RTX 3090 (matching the [Loss
Trajectory](#loss-trajectory-1-chinchilla-tiny-llama) setup above), all
three using `int8-dynamic-act-int4-weight`:

| Model | eval_loss | perplexity | Δ vs bf16 |
|-------|-----------|------------|-----------|
| bf16 baseline | 1.3656 | 3.918 | — |
| PTQ on bf16 baseline | 1.3917 | 4.022 | +0.0262 |
| QAT-trained + converted | 1.3905 | 4.017 | +0.0249 |

At this scale (4.43M params, recipe = int4 weights / int8 dynamic
activations), QAT shaves about **0.0013 eval-loss** off PTQ — a real
but small gain. Whether QAT's ~1.67× training-time overhead pays for
that depends on your tolerance for the +0.025 perplexity premium that
quantization itself imposes (PTQ buys you almost all of the win at zero
training cost). QAT is expected to scale better at larger models and
more aggressive recipes — measure your own setup before committing.

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
  `forgather finalize` reference (including `--quantize`).
