# LM Training Project Templates

Forgather ships two reusable project templates for language model pre-training.
Both compute training steps automatically from a target token budget and provide
extensive CLI overrides for rapid experimentation.

| Template | Description |
|----------|-------------|
| [projects/lm_training_project.yaml](../../templatelib/examples/projects/lm_training_project.yaml) | Token-budget training with cosine LR decay, warmup, and AdamW |
| [projects/auto_lr_project.yaml](../../templatelib/examples/projects/auto_lr_project.yaml) | Extends the above with automatic LR scaling based on global batch size; defaults to DDP |

## Quick Start

The `examples/base_lm_project` directory provides a ready-made harness for both
templates.

```bash
cd examples/base_lm_project

# List available configurations
forgather ls

# Preview the resolved configuration
forgather pp

# Train with defaults (single GPU, full float32 precision)
forgather train

# Train with mixed-precision and torch.compile (Ampere+ GPUs)
forgather train --compile true --mixed-precision bf16 --float32-matmul-precision high

# Train with Auto LR (DDP, uses all GPUs by default)
forgather -t auto_lr_project.yaml train

# Auto LR on a single GPU with increased effective batch size
forgather -t auto_lr_project.yaml train --gradient-accumulation-steps 8 -d 0

# Auto LR with mixed-precision on GPUs 0 and 1
forgather -t auto_lr_project.yaml train \
    --compile true --mixed-precision bf16 --float32-matmul-precision high -d 0,1
```

## Using in Your Own Project

To use these templates in your own project, create a config that includes
one of them:

```yaml
-- extends "projects/lm_training_project.yaml"

[config_metadata]
    == super()
    -- set ns.config_name = "My Experiment"
    ...
```

or

```yaml
-- extends "projects/auto_lr_project.yaml"
...
```

Override any defaults using template blocks or by passing values via the
preprocessor. For example, to change the model and token budget:

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
```

## LM Training Project

**Template:** [projects/lm_training_project.yaml](../../templatelib/examples/projects/lm_training_project.yaml)\
**Extends:** [training_script/causal_lm/causal_lm.yaml](../../templatelib/base/training_script/causal_lm/causal_lm.yaml)

Computes training steps from a target token budget, accounting for sequence
length, batch size, world size, and an estimated batch density (fraction of
non-pad tokens). Includes a cosine-decay LR scheduler with warmup, an AdamW
optimizer, and step-based logging/eval/save cadence.

### Token Budget and Step Computation

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

### Chinchilla-Optimal Token Budgets

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

### LR Scheduler Behaviour

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

### Configuration Parameters

All token counts are specified in **millions** unless noted otherwise.

#### Training Budget

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `total_tokens` | int | 560 | Total training tokens (M) |
| `warmup_tokens` | int | 56 | LR warmup tokens (M) |
| `min_cooldown_tokens` | int | 300 | Minimum cosine-decay window (M); prevents decay to zero on short runs |

#### Batching

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seq_len` | int | 512 | Maximum sequence length |
| `batch_size` | int | 32 | Per-device training batch size |
| `gradient_accumulation_steps` | int | 1 | Gradient accumulation steps |
| `batch_density` | float | 0.90 | Estimated fraction of non-pad tokens per batch; used to correct token-count estimates |

#### Data

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_project` | path | `examples/datasets/HuggingFaceTB` | Path to dataset project |
| `dataset_config` | str | `smollm-corpus/fineweb-edu-packed.yaml` | Dataset project configuration |
| `dispatch_batches` | bool | False | When True, rank-0 loads and dispatches all batches (DDP). Use when the dataset does not support sharding, or to match single-process example ordering |

#### Model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_project` | path | `examples/models/llama` | Path to model project |
| `model_config` | str | `small.yaml` | Model project configuration (28M parameters) |
| `attn_implementation` | str | `sdpa` | Attention backend. Choices: eager, sdpa, flash_attention_2, flex_attention |

#### Optimizer / Scheduler

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 3e-4 | Learning rate |

#### Step Cadence

Controls the interval (in tokens) between logging, evaluation, and checkpoint
save steps. The `step_cadence` multiplier scales all three intervals
proportionally.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `step_cadence` | float | 1.0 | Multiplier applied to all base intervals below |
| `base_logging_tokens` | int | 1 | Base tokens (M) between log steps |
| `base_validation_tokens` | int | 25 | Base tokens (M) between eval steps |
| `base_save_tokens` | int | 500 | Base tokens (M) between save steps |

#### Hardware / Performance

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `peak_hardware_flops` | float | 165.2e12 | Peak device FLOPS for MFU computation. See [training-performance-metrics](../trainers/training-performance-metrics.md) |

#### Precision / Compilation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_dtype` | str | null | Torch dtype for model construction. Choices: float32, bfloat16, float16 |
| `float32_matmul_precision` | str | null | Approximate float32 matmul with bf16. Choices: highest, high, medium |
| `mixed_precision` | str | null | AMP dtype. Choices: bf16, fp16, no |
| `compile` | bool | False | Enable torch.compile |

#### Misc

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | 42 | Random seed |
| `resume` | bool | True | Auto-resume from latest checkpoint (falls back to fresh init if none exists) |
| `save_strategy` | str | steps | Checkpoint save strategy. Choices: no, steps, epoch |

### CLI Arguments

All parameters listed above are available as CLI arguments via `forgather train`.
Additional arguments inherited from the base training script:

```
forgather -p examples/base_lm_project train --help
```

| CLI Flag | Parameter | Description |
|----------|-----------|-------------|
| `--total-tokens N` | `total_tokens` | Total training tokens in millions |
| `--warmup-tokens N` | `warmup_tokens` | Warmup tokens in millions |
| `--min-cooldown-tokens N` | `min_cooldown_tokens` | Minimum LR decay window in millions |
| `--batch-size N` | `batch_size` | Per-device training batch size |
| `--gradient-accumulation-steps N` | `gradient_accumulation_steps` | Gradient accumulation steps |
| `--seq-len N` | `seq_len` | Maximum sequence length |
| `--lr X` | `lr` | Learning rate |
| `--step-cadence X` | `step_cadence` | Scale log/eval/save intervals |
| `--model-project PATH` | `model_project` | Path to model project |
| `--model-config NAME` | `model_config` | Model project configuration |
| `--dataset-project PATH` | `dataset_project` | Path to dataset project |
| `--dataset-config NAME` | `dataset_config` | Dataset project configuration |
| `--compile BOOL` | `compile` | Enable torch.compile |
| `--mixed-precision {bf16,fp16,no}` | `mixed_precision` | AMP dtype |
| `--default-dtype {float32,bfloat16,float16}` | `default_dtype` | Model construction dtype |
| `--float32-matmul-precision {highest,high,medium}` | `float32_matmul_precision` | Float32 matmul approximation |
| `--dispatch-batches BOOL` | `dispatch_batches` | Dispatch batches from rank-0 |
| `--resume BOOL` | `resume` | Resume from checkpoint |
| `--seed N` | `seed` | Random seed |
| `--peak-hardware-flops X` | `peak_hardware_flops` | Peak FLOPS for MFU |
| `-d DEVICES` | -- | CUDA visible devices (e.g., "0,1") |
| `--max-steps N` | `max_steps` | Override computed max training steps |
| `-S {no,steps,epoch}` | `save_strategy` | Checkpoint save strategy |
| `--dry-run` | -- | Show generated command without executing |
| `--attn-implementation NAME` | `attn_implementation` | Attention backend |

### Inspecting the Configuration

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
```

---

## Auto LR Project

**Template:** [projects/auto_lr_project.yaml](../../templatelib/examples/projects/auto_lr_project.yaml)\
**Extends:** [projects/lm_training_project.yaml](../../templatelib/examples/projects/lm_training_project.yaml)

Inherits all parameters from the LM Training Project and adds automatic
learning rate scaling based on global batch size using a power-law rule:

```
lr = base_lr * (global_batch_size / base_batch_size) ** alpha
```

This allows you to change the batch size, number of GPUs, or gradient
accumulation steps without manually retuning the learning rate. The template
defaults to DDP training (`nproc_per_node = "gpu"`), so it will use all
available GPUs unless constrained with `-d`.

### Scaling Regimes

The `lr_alpha` exponent controls the scaling behaviour:

| alpha | Regime | When to use |
|-------|--------|-------------|
| 0.0 | No scaling | LR is independent of batch size |
| 0.5 | Sqrt scaling | Noise-dominated: batch_size >> B_crit (default) |
| 1.0 | Linear scaling | Signal-dominated: batch_size << B_crit |

The transition is governed by the critical batch size B_crit. The default
alpha=0.5 (sqrt scaling) is a conservative choice appropriate for most LLM
training scenarios where the batch size exceeds the critical batch size. See
the template header comments for a full discussion and references.

### Additional Parameters

In addition to all [LM Training Project parameters](#configuration-parameters):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_lr` | float | 3e-4 | Reference learning rate at `base_batch_size`. Set via `--lr` |
| `base_batch_size` | int | 16384 | Reference batch size (tokens) for the scaling calculation |
| `lr_alpha` | float | 0.5 | Scaling exponent (see table above) |

The computed LR appears in `forgather pp` output as:

```
# ns.global_lr: 1.3258252147247766e-05
```

### DDP Notes

Because the Auto LR Project defaults to DDP, be aware:

- All GPUs are used by default. Restrict with `-d 0` or `-d 0,1`.
- Stopping DDP with Ctrl-C can leave worker processes running. Use
  `forgather control list` and `forgather control stop JOB_ID` for a clean
  shutdown.

### Examples

```bash
# Default DDP training on all GPUs
forgather -t auto_lr_project.yaml train

# Single GPU with 8x gradient accumulation (LR scales automatically)
forgather -t auto_lr_project.yaml train --gradient-accumulation-steps 8 -d 0

# Mixed-precision on GPUs 0 and 1
forgather -t auto_lr_project.yaml train \
    --compile true --mixed-precision bf16 --float32-matmul-precision high -d 0,1

# Override base learning rate
forgather -t auto_lr_project.yaml train --lr 1e-3

# Quick test: 10M tokens, fast logging
forgather -t auto_lr_project.yaml train --total-tokens 10 --step-cadence 0.1 -d 0
```
