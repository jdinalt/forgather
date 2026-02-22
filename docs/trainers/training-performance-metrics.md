# Training Performance Metrics

Forgather tracks token throughput and estimated FLOPs during training, reporting
both per-interval speed metrics in the console and cumulative totals in the final
training output and `trainer_logs.json`.

## Overview

The trainer automatically:

- Counts non-padding tokens processed each step (using the cross-entropy `ignore_index=-100` in
  labels as the mask, so padding and special tokens are excluded)
- Estimates FLOPs per token from the model's trainable parameter count using the standard
  transformer approximation: `18 × num_params` per token (6N forward + 12N backward)
- Accumulates both counts into `state.num_input_tokens_seen` and `state.total_flos`
- Synchronizes counts across distributed processes at each log step (not every step,
  to minimize communication overhead)

### Final training metrics

At the end of training, the following metrics are added to the output dict and logged:

| Metric | Description |
|--------|-------------|
| `total_tokens` | Total non-padding tokens processed (from `state.num_input_tokens_seen`) |
| `tokens_per_second` | Tokens / total runtime (after warmup) |
| `total_flops` | Estimated total FLOPs (from `state.total_flos`) |
| `flops_per_second` | Estimated total FLOPs / total runtime |

### Per-interval metrics (ProgressCallback)

The `ProgressCallback` computes two types of per-interval speed metrics:

- **tok/s** (token throughput): Uses wall-clock time between log steps, capturing
  real end-to-end throughput including optimizer updates, data loading, gradient
  synchronization, and all other overhead. This gives an accurate picture of actual
  training speed and is useful for comparing different optimizers or configurations.

- **MFU** (Model FLOPs Utilization): Uses accumulated pure training step time
  (forward + backward pass only, from `on_step_begin` to `on_step_end`), excluding
  evaluation, optimizer, and data loading time. This measures how efficiently the
  hardware is utilized during the compute-bound portion of training.

Both are display-only; they are not written to `trainer_logs.json`. The underlying
token and FLOP values in `trainer_logs.json` can be used to reproduce these
calculations offline.

## ProgressCallback options

```python
from forgather.ml.trainer.callbacks import ProgressCallback

callbacks = [
    ProgressCallback(
        show_tokens_per_second=True,          # Display tok/s each log step
        peak_hardware_flops=4 * 165.2e12,     # 4× RTX 4090, for MFU display
    ),
]
```

When `show_tokens_per_second=True`, the console output gains a `tok/s` column:

```
2025-01-15 10:23:45   1000  1.0   train-loss: 2.31450   learning-rate: 1.00e-04  tok/s: 142857
```

When `peak_hardware_flops` is also set, an `mfu` column appears alongside it:

```
2025-01-15 10:23:45   1000  1.0   train-loss: 2.31450   learning-rate: 1.00e-04  tok/s: 142857  mfu: 38.5%
```

### Full option reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `show_loss` | `True` | Display training loss |
| `show_learning_rate` | `True` | Display current learning rate |
| `show_epoch` | `True` | Display epoch |
| `show_grad_norm` | `True` | Display gradient norm |
| `show_tokens` | `False` | Display token count for the log interval |
| `show_tokens_per_second` | `False` | Display tokens/sec from wall-clock time (real throughput) |
| `peak_hardware_flops` | `None` | Aggregate peak BF16 FLOP/s across all GPUs; enables MFU display |

## Setting `peak_hardware_flops`

`peak_hardware_flops` must be the **aggregate** peak FLOP/s across **all GPUs** used in
the training job. The trainer accumulates `total_flos` by counting tokens across all
processes (via `all_reduce`), so `achieved_flops = delta_flos / elapsed` is the total
rate for the entire job, not per-GPU.

```
MFU = achieved_aggregate_flops_per_second / peak_aggregate_flops_per_second
```

For a 4-GPU job on RTX 4090s:

```python
peak_hardware_flops = 4 * 165.2e12   # 4 GPUs × 165.2 TFLOPS each
```

### Peak BF16 FLOP/s reference table

The figures below are the **dense BF16 Tensor Core** numbers with **FP32 accumulation**,
which is what PyTorch uses in mixed-precision (autocast BF16) training. This is the correct
figure for MFU calculations.

Note that NVIDIA spec sheets for consumer GPUs often advertise the higher
FP16-with-FP16-accumulation figure (approximately 2× the values below). That figure
is not applicable to standard training workloads.

#### NVIDIA Data Center GPUs

| GPU | Architecture | BF16 dense (FP32 accum) |
|-----|--------------|------------------------|
| H100 SXM | Hopper | 989 TFLOPS |
| H100 PCIe | Hopper | 756 TFLOPS |
| A100 SXM 80GB | Ampere | 312 TFLOPS |
| A100 PCIe 80GB | Ampere | 312 TFLOPS |
| A100 SXM 40GB | Ampere | 312 TFLOPS |
| A10 | Ampere | 31.2 TFLOPS |

#### NVIDIA Consumer / Workstation GPUs

| GPU | Architecture | BF16 dense (FP32 accum) |
|-----|--------------|------------------------|
| RTX 4090 | Ada Lovelace | 165.2 TFLOPS |
| RTX 4080 SUPER | Ada Lovelace | 104.4 TFLOPS |
| RTX 4080 | Ada Lovelace | 97.0 TFLOPS |
| RTX 4070 Ti SUPER | Ada Lovelace | 79.8 TFLOPS |
| RTX 3090 | Ampere | 71.2 TFLOPS |
| RTX 3090 Ti | Ampere | 79.8 TFLOPS |
| RTX 3080 Ti | Ampere | 59.8 TFLOPS |
| RTX 3080 (10GB) | Ampere | 44.7 TFLOPS |

> **Note on RTX 3090 / 3080 series:** NVIDIA's published specs for Ampere consumer GPUs
> cite the FP16-with-FP16-accumulation throughput (~142 TFLOPS for the 3090). The
> BF16-with-FP32-accumulation figure (~71 TFLOPS) is half that, and is the value to use
> here. Both BF16 and FP16 share the same Tensor Core hardware on Ampere and Ada; the
> difference in published figures is entirely the accumulation precision.

#### Example: multi-GPU configurations

| Configuration | `peak_hardware_flops` |
|---------------|-----------------------|
| 1× RTX 4090 | `165.2e12` |
| 2× RTX 4090 | `330.4e12` |
| 4× RTX 4090 | `660.8e12` |
| 1× RTX 3090 | `71.2e12` |
| 4× A100 SXM | `1248e12` |
| 8× A100 SXM | `2496e12` |
| 8× H100 SXM | `7912e12` |

## Notes on FLOP estimation accuracy

The `18 × num_params` formula is a standard approximation for decoder-only transformer
models. It assumes:

- Forward pass: `6 × num_params` FLOPs per token (2 multiply-adds per weight, times 3
  for Q, K, V projections and attention being rolled into the parameter count)
- Backward pass: `12 × num_params` FLOPs per token (approximately 2× forward)

Real FLOPs will differ from this estimate due to:

- **Attention FLOPs**: The quadratic attention term (`2 × seq_len × model_dim` per layer)
  is not included. For short sequences this is negligible; at very long sequence lengths
  it can be significant.
- **Non-transformer architectures**: The formula assumes a standard transformer with weight
  matrices dominating the compute. Models with unusual architectures (MoE, state-space
  models, etc.) may diverge substantially.
- **Gradient checkpointing**: Recomputes activations during backward, adding approximately
  one extra forward pass. The true FLOPs are closer to `24 × num_params` per token when
  gradient checkpointing is enabled, though the `18×` estimate is still commonly used.

For comparing runs on the same model and hardware, the absolute accuracy of the estimate
does not matter — the MFU and FLOP/s values are consistent relative to each other.
