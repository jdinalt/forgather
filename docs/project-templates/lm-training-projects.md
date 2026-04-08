# LM Training Project Template

Forgather ships a reusable project template for language model pre-training.
It computes training steps automatically from a target token budget, includes
automatic LR scaling based on global batch size, and supports three trainer
backends switchable via `--trainer-type`: basic (single GPU), DDP (multi-GPU),
and Pipeline Parallel.

**Template:** [projects/lm_training_project.yaml](../../templatelib/examples/projects/lm_training_project.yaml)\
**Extends:** [training_script/causal_lm/causal_lm.yaml](../../templatelib/base/training_script/causal_lm/causal_lm.yaml)

## Quick Start

The `examples/base_lm_project` directory provides a ready-made harness.

```bash
cd examples/base_lm_project

# List available configurations
forgather ls

# Preview the resolved configuration
forgather pp

# Train with defaults (single GPU, basic trainer)
forgather train

# DDP training on all GPUs
forgather train --trainer-type ddp

# DDP on specific GPUs
forgather train --trainer-type ddp -d 0,1

# Pipeline Parallel on all GPUs
forgather train --trainer-type pipeline

# Mixed-precision with torch.compile (Ampere+ GPUs)
forgather train --trainer-type ddp \
    --compile true --mixed-precision bf16 --float32-matmul-precision high
```

## Using in Your Own Project

Create a config that extends the template:

```yaml
-- extends "projects/lm_training_project.yaml"

[config_metadata]
    == super()
    -- set ns.config_name = "My Experiment"
    ...
```

Override any defaults using template blocks or by passing values via the
preprocessor. For example, to change the model, token budget, and default
trainer:

```yaml
-- extends "projects/lm_training_project.yaml"

[config_metadata]
    == super()
    -- set ns.config_name = "My Experiment"
    -- set ns.model_project_dir = abspath(joinpath(ns.forgather_dir, "examples", "models", "my_model"))
    -- set ns.model_project_config = "7B.yaml"
    -- set ns.total_tokens = 20000
    -- set ns.seq_len = 2048
    -- set ns.per_device_train_batch_size = 8
    -- set ns.trainer_type = "ddp"
```

## Trainer Selection

The template supports three trainer backends, selected via `--trainer-type`
or by setting `ns.trainer_type` in a child template's `[config_metadata]`:

| Trainer Type | Backend | Default nproc_per_node | Description |
|--------------|---------|------------------------|-------------|
| `basic` | `forgather.ml.trainer:Trainer` | 1 | Single-GPU training |
| `ddp` | `forgather.ml.trainer.ddp:DDPTrainer` | gpu (all GPUs) | Distributed Data Parallel |
| `pipeline` | `forgather.ml.trainer.pipeline:PipelineTrainer` | gpu (all GPUs) | Pipeline Parallel |

The trainer type controls which trainer template is included and, for
`ddp` and `pipeline`, automatically sets `nproc_per_node` to `"gpu"` (all
available GPUs). Use `-d` to restrict to specific devices.

### DDP Notes

- All GPUs are used by default. Restrict with `-d 0` or `-d 0,1`.
- Stopping DDP with Ctrl-C can leave worker processes running. Use
  `forgather control list` and `forgather control stop JOB_ID` for a clean
  shutdown.

### Pipeline Parallel

Pipeline Parallel splits the model into stages distributed across GPUs.
When `--trainer-type pipeline` is selected, the template automatically:

- Forces `dispatch_batches = True` (rank-0 loads and dispatches data)
- Computes microbatch and stage configuration from the pipeline schedule
- Overrides batch sizes to the computed PP batch size
- Disables `torch_compile_mode: max-autotune` (incompatible with PP)

The pipeline schedule determines how microbatches flow through stages:

| Schedule | stages_per_rank | Notes |
|----------|-----------------|-------|
| `ScheduleGPipe` | 1 | Simple, high pipeline bubble |
| `Schedule1F1B` | 1 | Reduced bubble vs GPipe |
| `ScheduleInterleaved1F1B` | 2 | Default; lower bubble |
| `ScheduleLoopedBFS` | 2 | Alternative interleaved schedule |
| `ScheduleInterleavedZeroBubble` | 2 | Near-zero bubble |
| `ScheduleZBVZeroBubble` | 2 | Zero-bubble V layout (experimental) |

**Batch size constraint:** `per_device_train_batch_size` must be divisible by
`stages_per_rank * microbatch_scale`. The default batch size of 32 works with
all schedules. Use `--microbatch-scale` to increase throughput by adding more
microbatches without changing the logical batch size.

## Token Budget and Step Computation

The template converts a token budget (specified in millions) into training
steps using the following calculation:

```
tokens_per_step = seq_len * global_batch_size * batch_density
total_steps     = total_tokens / tokens_per_step

where:
  global_batch_size = per_device_batch_size * gradient_accumulation_steps * world_size
```

The `batch_density` parameter compensates for padding tokens -- set it close
to 1.0 for packed datasets or lower for padded datasets.

## Chinchilla-Optimal Token Budgets

The default token budget is sized for Chinchilla-optimal training of the
default 28M-parameter Llama model (~20 tokens per parameter = 560M tokens).
The Chinchilla scaling law (Hoffmann et al., 2022) established that training
tokens should scale linearly with model parameters for compute-optimal
training:

```
Optimal tokens ~ 20 x N
```

Note that recent work suggests the true optimum may be higher (~40-100x)
when inference costs are factored in. See the template header comments for
full references.

## Automatic LR Scaling

The template automatically scales the learning rate based on the global batch
size using a power-law rule:

```
lr = base_lr * (tokens_per_step / base_batch_size) ** lr_alpha
```

This allows you to change the batch size, number of GPUs, or gradient
accumulation steps without manually retuning the learning rate.

### Scaling Regimes

The `lr_alpha` exponent controls the scaling behaviour:

| alpha | Regime | When to use |
|-------|--------|-------------|
| 0.0 | No scaling | LR is independent of batch size |
| 0.5 | Sqrt scaling | Noise-dominated: batch_size >> B_crit (default) |
| 1.0 | Linear scaling | Signal-dominated: batch_size << B_crit |

The transition between regimes is governed by the critical batch size B_crit,
which can be estimated from the gradient noise scale. The default alpha=0.5
(sqrt scaling) is a conservative choice appropriate for most LLM training
scenarios where the batch size exceeds the critical batch size. See the
template header comments for a full discussion and references (McCandlish
et al. 2018, Mayberry et al. 2025).

### Disabling LR Scaling

To use a fixed learning rate regardless of batch size, set `lr_alpha` to 0:

```yaml
[config_metadata]
    == super()
    -- set ns.lr_alpha = 0.0
```

## LR Scheduler Behaviour

The cosine-decay scheduler's `total_steps` is clamped to at least
`min_cooldown_steps` so that short training runs (or runs that are stopped
early) do not decay the learning rate all the way to zero. This shows up
in the preprocessed output as:

```yaml
# LR Scheduler (cosine decay with warmup)
# total_steps is clamped to min_cooldown_steps (20345)
# so that short runs do not decay the LR all the way to zero.
lr_scheduler: ...
    warmup_steps: 3797
    total_steps: 37978
```

## Configuration Parameters

All token counts are specified in **millions** unless noted otherwise.

### Training Budget

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `total_tokens` | int | 560 | Total training tokens (M) |
| `warmup_tokens` | int | 56 | LR warmup tokens (M) |
| `min_cooldown_tokens` | int | 300 | Minimum cosine-decay window (M); prevents decay to zero on short runs |

### Batching

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seq_len` | int | 512 | Maximum sequence length |
| `batch_size` | int | 32 | Per-device training batch size |
| `gradient_accumulation_steps` | int | 1 | Gradient accumulation steps |
| `batch_density` | float | 0.95 | Estimated fraction of non-pad tokens per batch; used to correct token-count estimates |

### Data

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_project` | path | `examples/datasets/HuggingFaceTB` | Path to dataset project |
| `dataset_config` | str | `smollm-corpus/fineweb-edu-packed.yaml` | Dataset project configuration |
| `dispatch_batches` | bool | False | When True, rank-0 loads and dispatches all batches (DDP). Use when the dataset does not support sharding, or to match single-process example ordering |

### Model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_project` | path | `examples/models/llama` | Path to model project |
| `model_config` | str | `small.yaml` | Model project configuration (28M parameters) |
| `attn_implementation` | str | `sdpa` | Attention backend. Choices: eager, sdpa, flash_attention_2, flex_attention |

### Optimizer / Scheduler / LR Scaling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 3e-4 | Base learning rate at `base_batch_size` |
| `base_batch_size` | int | 16384 | Reference batch size (tokens) for the scaling calculation |
| `lr_alpha` | float | 0.5 | Scaling exponent: 0.0 = no scaling, 0.5 = sqrt, 1.0 = linear |

The computed LR appears in `forgather pp` output as:

```
# ns.global_lr: 1.3258252147247766e-05
```

### Trainer Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trainer_type` | str | `basic` | Trainer backend: basic, ddp, pipeline |
| `nproc_per_node` | str/int | auto | Processes per node: `"gpu"` for all GPUs, or integer count. Auto-set from trainer type if not specified |
| `pipeline_schedule` | str | `ScheduleInterleaved1F1B` | Pipeline Parallel schedule class (pipeline trainer only) |
| `microbatch_scale` | int | 1 | Microbatch scale factor (pipeline trainer only) |

### Step Cadence

Controls the interval (in tokens) between logging, evaluation, and checkpoint
save steps. The `step_cadence` multiplier scales all three intervals
proportionally.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `step_cadence` | float | 1.0 | Multiplier applied to all base intervals below |
| `base_logging_tokens` | int | 1 | Base tokens (M) between log steps |
| `base_validation_tokens` | int | 25 | Base tokens (M) between eval steps |
| `base_save_tokens` | int | 500 | Base tokens (M) between save steps |

### Hardware / Performance

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `peak_hardware_flops` | float | 165.2e12 | Peak device FLOPS for MFU computation. See [training-performance-metrics](../trainers/training-performance-metrics.md) |

### Precision / Compilation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_dtype` | str | null | Torch dtype for model construction. Choices: float32, bfloat16, float16 |
| `float32_matmul_precision` | str | null | Approximate float32 matmul with bf16. Choices: highest, high, medium |
| `mixed_precision` | str | null | AMP dtype. Choices: bf16, fp16, no |
| `fp8_recipe` | str | null | FP8 recipe for linear layers. Choices: tensorwise, rowwise, rowwise_with_gw_hp |
| `compile` | bool | False | Enable torch.compile |

### Misc

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | 42 | Random seed |
| `resume` | bool | True | Auto-resume from latest checkpoint (falls back to fresh init if none exists) |
| `save_strategy` | str | steps | Checkpoint save strategy. Choices: no, steps, epoch |

## CLI Arguments

All parameters listed above are available as CLI arguments via `forgather train`.
Additional arguments inherited from the base training script:

```
forgather -p examples/base_lm_project train --help
```

| CLI Flag | Parameter | Description |
|----------|-----------|-------------|
| `--trainer-type {basic,ddp,pipeline}` | `trainer_type` | Trainer backend |
| `--nproc-per-node N` | `nproc_per_node` | Processes per node (auto-set from trainer type) |
| `--pipeline-schedule NAME` | `pipeline_schedule` | Pipeline schedule class |
| `--microbatch-scale N` | `microbatch_scale` | Microbatch scale factor |
| `--total-tokens N` | `total_tokens` | Total training tokens in millions |
| `--warmup-tokens N` | `warmup_tokens` | Warmup tokens in millions |
| `--min-cooldown-tokens N` | `min_cooldown_tokens` | Minimum LR decay window in millions |
| `--batch-size N` | `batch_size` | Per-device training batch size |
| `--gradient-accumulation-steps N` | `gradient_accumulation_steps` | Gradient accumulation steps |
| `--seq-len N` | `seq_len` | Maximum sequence length |
| `--lr X` | `lr` | Base learning rate |
| `--step-cadence X` | `step_cadence` | Scale log/eval/save intervals |
| `--model-project PATH` | `model_project` | Path to model project |
| `--model-config NAME` | `model_config` | Model project configuration |
| `--dataset-project PATH` | `dataset_project` | Path to dataset project |
| `--dataset-config NAME` | `dataset_config` | Dataset project configuration |
| `--compile BOOL` | `compile` | Enable torch.compile |
| `--mixed-precision {bf16,fp16,no}` | `mixed_precision` | AMP dtype |
| `--default-dtype {float32,bfloat16,float16}` | `default_dtype` | Model construction dtype |
| `--float32-matmul-precision {highest,high,medium}` | `float32_matmul_precision` | Float32 matmul approximation |
| `--fp8-recipe {tensorwise,rowwise,rowwise_with_gw_hp}` | `fp8_recipe` | FP8 training recipe |
| `--dispatch-batches BOOL` | `dispatch_batches` | Dispatch batches from rank-0 |
| `--resume BOOL` | `resume` | Resume from checkpoint |
| `--seed N` | `seed` | Random seed |
| `--peak-hardware-flops X` | `peak_hardware_flops` | Peak FLOPS for MFU |
| `--attn-implementation NAME` | `attn_implementation` | Attention backend |
| `-d DEVICES` | -- | CUDA visible devices (e.g., "0,1" or "gpu" for all) |
| `--max-steps N` | `max_steps` | Override computed max training steps |
| `-S {no,steps,epoch}` | `save_strategy` | Checkpoint save strategy |
| `--dry-run` | -- | Show generated command without executing |

## Inspecting the Configuration

Use `forgather pp` to see the fully resolved configuration with all computed
values. The output includes a variable listing showing all derived quantities:

```
# **LM Training Project**
# ns.per_device_train_batch_size: 32
# ns.gradient_accumulation_steps: 1
# ns.effective_per_device_batch_size: 32
# ns.global_batch_size: 32
# ns.seq_len: 512 tokens
# ns.total_steps: 37978 steps
# ns.warmup_steps: 3797
# ns.min_cooldown_steps: 20345
# ns.total_tokens: 560M
# ns.tokens_per_step: 14745 tokens
# ns.total_peak_hardware_flops: 165.2 TFLOPS
# ns.base_lr: 0.0003
# ns.base_batch_size: 16384
# ns.lr_alpha: 0.5
# ns.global_lr: 0.00028460498941515414
# ns.trainer_type: basic
```

With `--trainer-type pipeline`, additional PP variables are shown:

```
# ns.trainer_type: pipeline
# Pipeline Parallel:
# ns.stages_per_rank: 2
# ns.per_stage_batch_size: 16
# ns.n_microbatches: 2
# ns.pp_batch_size: 32
# ns.pp_stage_type: loop
```

## Examples

```bash
# Basic trainer (single GPU, default)
forgather train

# DDP on all GPUs
forgather train --trainer-type ddp

# DDP on specific GPUs with gradient accumulation (LR scales automatically)
forgather train --trainer-type ddp --gradient-accumulation-steps 8 -d 0,1

# Pipeline Parallel with GPipe schedule
forgather train --trainer-type pipeline --pipeline-schedule ScheduleGPipe

# Pipeline Parallel with increased microbatch count
forgather train --trainer-type pipeline --microbatch-scale 2

# Mixed-precision DDP with torch.compile
forgather train --trainer-type ddp \
    --compile true --mixed-precision bf16 --float32-matmul-precision high

# Override base learning rate
forgather train --lr 1e-3

# Fixed LR (disable scaling) -- set lr_alpha in your config
# -- set ns.lr_alpha = 0.0

# Quick test: 10M tokens, fast logging
forgather train --total-tokens 10 --step-cadence 0.1 -d 0
```
