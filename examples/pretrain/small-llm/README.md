# Small LLM Pretraining

A complete example project for pretraining small language models (100M-2B parameters) from scratch, demonstrating token-efficient training with principled learning rate scheduling.

## Overview

This project trains models on the [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus), a curated collection of high-quality educational and synthetic data designed for training small language models. The training dataset combines "cosmopedia-v2" and "fineweb-edu-dedup" subsets with intelligent interleaving to balance data sources.

**Key Features:**
- Token-budget-based training with Chinchilla-optimal defaults
- Principled LR scheduling that adapts to training scale
- Multi-GPU distributed training with DDP and Pipeline Parallel
- Sequence packing for efficiency (4K token blocks)
- Comprehensive monitoring and checkpointing
- Support for multiple model architectures (DeepOne, Llama, Qwen, etc.)

**Default Configuration:**
- Model: Custom DeepOne (162M parameters)
- Token Budget: 3B tokens (Chinchilla-optimal for 150M params)
- Batch Size: 4 per device (auto-scaled with world size)
- Sequence Length: 4096 tokens
- Optimizer: Adafactor with BF16 stochastic rounding

## Quick Start

```bash
# Train with defaults (DeepOne 117M, 3B tokens)
forgather train --init-model

# Resume from checkpoint
forgather train

# Train with different model
forgather train --init-model --model-project ../../models/llama/ --model-config 124M.yaml

# Train for longer (10B tokens instead of 3B)
forgather train --init-model --total-tokens 10

# Use flex attention for better performance
forgather train --init-model --attn-implementation flex_attention
```

## Training Configuration

The training configuration is built around several interconnected parameters that together determine training dynamics, compute requirements, and final model quality.

### Token Budget and Compute Allocation

Training is controlled by a **token budget** rather than epochs or arbitrary step counts. This aligns with modern understanding of compute-optimal training.

**Key Parameters:**
- `--total-tokens N`: Total training tokens in billions (default: 3)
- `--max-steps N`: Override to limit training steps (optional)

**Chinchilla Optimal:**
The default 3B token budget follows Chinchilla scaling laws for compute-optimal training:
```
optimal_tokens ≈ 20 × model_parameters
```

For a 150M parameter model: `150M × 20 = 3B tokens`

**How it works:**
```python
# These are computed automatically from your settings:
tokens_per_step = max_length × global_batch_size
total_steps = total_tokens // tokens_per_step

# Example with defaults (4 GPUs):
# tokens_per_step = 4096 × (4 × 4) = 65,536
# total_steps = 3B / 65,536 ≈ 45,777 steps
```

### Batch Size and Learning Rate Scaling

Batch sizes and learning rates are coupled through sqrt-scaling to maintain training dynamics across different hardware configurations.

**Key Parameters:**
- `--batch-size N`: Per-device training batch size (default: 4)
- `--learning-rate LR`: Base learning rate for batch size 1 (default: 1.4e-5)

**How it works:**
```python
# Global batch size scales with world size
global_batch_size = batch_size × gradient_accumulation_steps × world_size

# Learning rate scales by sqrt of global batch size
lr_scale = sqrt(global_batch_size)
actual_lr = base_lr × lr_scale

# Example with 4 GPUs:
# global_batch_size = 4 × 1 × 4 = 16
# lr_scale = sqrt(16) = 4.0
# actual_lr = 1.4e-5 × 4.0 = 5.6e-5
```

**Why sqrt-scaling?**
This maintains the signal-to-noise ratio in gradients as batch size increases, allowing larger batches without changing optimization dynamics. Based on empirical findings from "Don't Decay the Learning Rate, Increase the Batch Size" (Smith et al., 2017).

### Learning Rate Scheduling Strategy

The LR schedule uses the **InfiniteLRScheduler**, designed for flexible pretraining with optional continuation and annealing phases.

**Schedule Structure:**
```
1. Warmup (500 steps): 0 → max_lr
2. Cooldown (variable): max_lr → constant_lr (cosine decay)
3. Constant (remaining): constant_lr
4. Annealing (optional): constant_lr → min_lr (not used by default)
```

**Key Parameters:**
- `--warmup-steps N`: Steps to warm up to max_lr (default: 500)
- `--min-cooldown-tokens N`: Minimum tokens (billions) for cooldown phase (default: 100)

**Adaptive Cooldown Duration:**

The cooldown phase duration adapts to training scale:
```python
# Compute both minimum and ratio-based durations
min_cooldown_steps = min_cooldown_tokens / tokens_per_step
ratio_cooldown_steps = 0.7 × total_steps

# Use the larger value
cooldown_steps = max(min_cooldown_steps, ratio_cooldown_steps)
```

**Why this matters:**

Short training runs (e.g., 3B tokens) benefit from maintaining high LR longer, while long runs (100B+ tokens) can afford earlier decay. The cosine function is nearly flat at the beginning, so even "starting" decay early keeps LR high initially:

- **3B token run**: `max(1.22M steps, 32K steps) = 1.22M` → trains at ~99.6% max_lr throughout
- **50B token run**: `max(1.22M steps, 427K steps) = 1.22M` → meaningful but gentle decay
- **300B token run**: `max(1.22M steps, 2.56M steps) = 2.56M` → full 70% decay ratio

**Base Learning Rates** (before scaling):
- `max_lr`: 1.4e-5 (set with `--learning-rate`)
- `constant_lr`: 7.5e-6 (hardcoded, ~53% of max)
- `min_lr`: 1.4e-6 (hardcoded, ~10% of max)

**Tuning min_cooldown_tokens:**

The default 100B was chosen conservatively. For experimentation:
```bash
# More aggressive (start decay sooner)
forgather train --min-cooldown-tokens 20

# More conservative (maintain high LR longer)
forgather train --min-cooldown-tokens 200
```

**The annealing phase** (warmup → cooldown → constant → anneal) allows you to:
1. Train to token budget with constant_lr
2. Save checkpoint
3. Fork for task-specific annealing
4. Continue pretraining from constant_lr checkpoint without re-warming

See "Training to Do Better Than a Fake Loss Function" for the research behind this approach.

### Sequence Length

**Parameter:**
- `--max-length N`: Maximum sequence length in tokens (default: 4096)

Sequences are packed to this length using multiple examples with masking to prevent cross-attention between samples. This maximizes GPU utilization while maintaining training semantics.

**Trade-offs:**
- Longer sequences: Better context, more memory, fewer steps for same token budget
- Shorter sequences: Less memory, faster iteration, more gradient updates

**Memory scaling:** Attention is O(n²) in sequence length. For large models, you may need to reduce sequence length or use memory-efficient attention (flash_attention_2, flex_attention).

### Step Cadence

**Parameter:**
- `--step-cadence FACTOR`: Scales logging/eval/save intervals (default: 1.0)

Controls how often various events occur without changing the token scale:
```python
# Base intervals (in tokens processed, not steps):
logging_interval = 100 tokens
eval_interval = 1000 tokens
save_interval = 20000 tokens

# Actual step intervals:
logging_steps = (100 × step_cadence) / (batch_size × grad_accum)
eval_steps = (1000 × step_cadence) / (batch_size × grad_accum)
save_steps = (20000 × step_cadence) / (batch_size × grad_accum)
```

**Use cases:**
- `--step-cadence 4.0`: For small models - checkpoint/eval less frequently
- `--step-cadence 0.25`: For debugging - more frequent monitoring

## Model Selection

### Default Model: Custom DeepOne

A customized 162M parameter model based on DeepNet architecture:
- 16 layers × 768 hidden × 8 attention heads
- RoPE positional encoding (replaced ALiBi for speed)
- Qwen3-style QK-Norm
- GLU feedforward (SwiGLU)
- 32K vocabulary

**Why DeepOne?**
The DeepNet architecture is relatively forgiving for pretraining experiments due to its improved initialization and normalization. Good choice for learning about pretraining dynamics.

See [Custom DeepOne README](./custom_deepone/README.md) for details.

### Using Different Models

```bash
# Llama architecture (30M - 3B params)
forgather train --init-model \
    --model-project ../../models/llama/ \
    --model-config 124M.yaml

# Qwen3 architecture
forgather train --init-model \
    --model-project ../../models/qwen3/ \
    --model-config 124M.yaml

# List available model configs
ls ../../models/llama/templates/configs/
```

**Important:** When changing models, delete the old output directory:
```bash
rm -rf output_models
forgather train --init-model --model-project ...
```

## Dataset

**Source:** [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)

**Subsets used:**
- `cosmopedia-v2`: Synthetic educational content
- `fineweb-edu-dedup`: Deduplicated educational web content

**Processing:**
- Interleaved with proportional sampling (balanced consumption)
- Packed into 4K token blocks with masking
- Fast loading with [Fast HF Dataset Loader](../../../docs/datasets/fast-hf-loader.md)

**Note:** Initial download is large (~100GB+) but subsequent loads are nearly instant when cached.

See [SmolLM-Corpus dataset project](../../datasets/HuggingFaceTB/README.md) for details.

## Command-Line Options

### Core Options

| Option | Description | Default |
|--------|-------------|---------|
| `--init-model` | Initialize new model (vs resume from checkpoint) | Resume |
| `--total-tokens N` | Total training tokens in billions | 3 |
| `--batch-size N` | Per-device training batch size | 4 |
| `--learning-rate LR` | Base learning rate (for batch size 1) | 1.4e-5 |
| `--max-length N` | Maximum sequence length | 4096 |
| `--min-cooldown-tokens N` | Minimum tokens (billions) for LR cooldown | 100 |

### Model Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-project PATH` | Path to model project directory | `./custom_deepone` |
| `--model-config NAME` | Model configuration file | `custom_deepone.yaml` |

### Attention Implementation

| Option | Description | Performance |
|--------|-------------|-------------|
| `--attn-implementation sdpa` | PyTorch SDPA (default) | Good, no sparsity |
| `--attn-implementation flex_attention` | Flex attention | Best, supports sparsity |
| `--attn-implementation flash_attention_2` | Flash Attention 2 | Excellent, requires installation |
| `--attn-implementation eager` | Standard PyTorch | Slow, debugging only |

**Recommendation:** Use `flex_attention` for best performance with sequence packing.

### Training Control

| Option | Description | Default |
|--------|-------------|---------|
| `--max-steps N` | Override max steps (instead of token budget) | Auto-computed |
| `--save-strategy {no,steps,epoch}` | When to save checkpoints | steps |
| `--step-cadence FACTOR` | Scale log/eval/save intervals | 1.0 |
| `--compile` | Enable Torch compile (slower startup, faster training) | Disabled |

### Distributed Training

| Option | Description | Default |
|--------|-------------|---------|
| `-d DEVICES` | CUDA visible devices (e.g., "0,1,3") | All GPUs |
| `--dist-backend BACKEND` | PyTorch distributed backend | nccl |

### Pipeline Parallel Options

These options apply when using `pp.yaml` (`forgather -t pp.yaml`).

| Option | Description | Default |
|--------|-------------|---------|
| `--pipeline-schedule NAME` | Pipeline schedule class to use (see below) | `ScheduleInterleaved1F1B` |
| `--microbatch-scale N` | Multiply the number of microbatches by N | 1 |

**Available pipeline schedules:**

| Schedule | `stages_per_rank` | Notes |
|----------|--------------------|-------|
| `ScheduleGPipe` | 1 | Simple, high pipeline bubble |
| `Schedule1F1B` | 1 | Reduced bubble vs GPipe |
| `ScheduleInterleaved1F1B` | 2 | Default; lower bubble, requires 2 stages/rank |
| `ScheduleLoopedBFS` | 2 | Alternative interleaved schedule |
| `ScheduleInterleavedZeroBubble` | 2 | Near-zero bubble |
| `ScheduleZBVZeroBubble` | 2 | Zero-bubble V layout (experimental) |

**Batch size constraint:** `per_device_train_batch_size` must be divisible by `stages_per_rank × microbatch_scale`. The default batch size of 4 works with `stages_per_rank=2` (the interleaved default). Use `--microbatch-scale` to increase throughput by adding more microbatches without changing the logical batch size.

### Debugging

| Option | Description |
|--------|-------------|
| `--dry-run` | Show command without executing |
| `--verbose-info` | Display detailed config at startup |
| `--no-restore-dataset-state` | Start from dataset beginning |
| `--save-strategy no` | Disable checkpointing for quick experiments |

## Examples

### Example 1: Quick Chinchilla-Optimal Run

Train the default 117M model for 3B tokens (Chinchilla-optimal):

```bash
forgather train --init-model --attn-implementation flex_attention
```

**What happens:**
- Model: DeepOne 117M
- Total tokens: 3B
- Total steps: ~45K (with 4 GPUs)
- LR: Stays at ~99.6% of max_lr throughout
- Training time: ~6-8 hours on 4x RTX 3090

### Example 2: Longer Training Run

Train for 10B tokens to see more convergence:

```bash
forgather train --init-model \
    --total-tokens 10 \
    --attn-implementation flex_attention
```

**What happens:**
- Total steps: ~150K
- LR: Gentle decay over training
- Better final performance but diminishing returns per token

### Example 3: Small Model, High Cadence

Train a tiny model quickly with frequent checkpointing:

```bash
forgather train --init-model \
    --model-project ../../models/llama/ \
    --model-config 30M.yaml \
    --batch-size 16 \
    --step-cadence 4.0 \
    --attn-implementation flex_attention
```

**Why this works:**
- 30M model trains fast
- Larger batch size (16) → fewer steps
- `step_cadence 4.0` → checkpoints less frequently
- Good for testing configurations quickly

### Example 4: Multi-GPU on Specific Devices

Use specific GPUs (skip GPU 2):

```bash
forgather train --init-model \
    -d 0,1,3,4,5 \
    --attn-implementation flex_attention
```

**What happens:**
- Uses 5 GPUs (skips GPU 2)
- `global_batch_size = 4 × 5 = 20`
- LR auto-scales: `1.4e-5 × sqrt(20) ≈ 6.3e-5`

### Example 5: Hyperparameter Exploration

Quick experiment without saving checkpoints:

```bash
forgather train --init-model \
    --max-steps 2000 \
    --save-strategy no \
    --learning-rate 2.0e-5 \
    --attn-implementation flex_attention
```

**Use case:**
- Test different learning rates quickly
- No disk space used for checkpoints
- 2000 steps ≈ 130M tokens with defaults

### Example 6: Resume and Continue

Resume from checkpoint and train longer:

```bash
# Initial run
forgather train --init-model --total-tokens 3

# Later: continue for 7B more tokens (10B total)
forgather train --total-tokens 10
```

**What happens:**
- Resumes from last checkpoint
- Continues to 10B total tokens
- Dataset state restored (picks up where it left off)

### Example 7: Pipeline Parallel Training

Train using Pipeline Parallel across 4 GPUs with the interleaved schedule:

```bash
forgather -t pp.yaml train --init-model \
    -d 0,1,2,3 \
    --attn-implementation flex_attention \
    --pipeline-schedule ScheduleInterleaved1F1B
```

**What happens:**
- 4 GPUs, `stages_per_rank=2` → 8 total pipeline stages
- Each GPU holds 2 model stages (interleaved across the pipeline)
- Default batch size (4) split into microbatches of 1 each
- Memory per GPU: roughly 1/4 of total model parameters

**Performance testing without checkpointing:**

```bash
forgather -t pp.yaml train --init-model \
    -d 0,1 \
    --attn-implementation sdpa \
    --microbatch-scale 2 \
    --batch-size 8 \
    --max-steps 100 \
    --save-strategy no \
    --pipeline-schedule ScheduleGPipe
```

This runs 100 steps with `--save-strategy no` for fast iteration when testing schedule and batch size combinations.

**When to use PP over DDP:**
- Model does not fit on a single GPU in full precision
- You want to pipeline compute across GPUs rather than replicate the model

### Example 8: Conservative LR Schedule

Use more conservative cooldown for very long training:

```bash
forgather train --init-model \
    --total-tokens 50 \
    --min-cooldown-tokens 200 \
    --attn-implementation flex_attention
```

**What happens:**
- 50B token budget
- Won't start meaningful LR decay until very late in training
- Good for exploring if longer high-LR phases improve convergence

## Advanced Topics

### Distributed Data Parallel (DDP)

The project uses PyTorch DDP for multi-GPU training on a single node. Each GPU maintains a complete copy of the model and processes different data.

**Dataset sharding** (default):
- Each rank processes a different shard of the dataset
- More efficient: parallel data loading
- Drawback: Requires more CPU memory

**Batch dispatching** (alternative):
Set `ns.dispatch_batches = True` in config:
- Rank 0 loads and dispatches batches to all ranks
- More memory efficient
- Drawback: Potential data loading bottleneck

**Memory limits:**
DDP requires the full model to fit on a single GPU. For larger models (>2B params with 24GB), consider:
- Pipeline parallelism (see below)
- Tensor parallelism
- Fully-sharded data parallel (FSDP)

### Pipeline Parallel

Use `pp.yaml` instead of the default config to train with Pipeline Parallel (PP). This is configured with `forgather -t pp.yaml`.

**When to use PP:**
- The model is too large to fit on a single GPU with DDP
- You want to split compute across GPUs in a pipeline rather than replicate the full model

**How it works:**

The model is partitioned into stages distributed across GPUs. Each GPU holds one or more stages and processes microbatches in a pipelined fashion. The pipeline schedule determines the order in which microbatches flow through stages.

**Batch size and microbatches:**

```
# DDP: each GPU processes a full batch independently
# PP: the batch is split into microbatches that flow through the pipeline

stages_per_rank = 1 or 2 (determined by schedule)
n_microbatches = world_size × stages_per_rank × microbatch_scale
per_stage_batch_size = per_device_train_batch_size // (stages_per_rank × microbatch_scale)
pp_batch_size = n_microbatches × per_stage_batch_size
```

With defaults (`--batch-size 4`, `ScheduleInterleaved1F1B`, 2 GPUs):
```
stages_per_rank = 2
n_microbatches = 2 × 2 × 1 = 4
per_stage_batch_size = 4 // (2 × 1) = 2
pp_batch_size = 4 × 2 = 8  (the logical batch size seen by the trainer)
```

**Choosing a schedule:**

Simple schedules (`ScheduleGPipe`, `Schedule1F1B`) use one stage per rank. Interleaved schedules (`ScheduleInterleaved1F1B`, `ScheduleLoopedBFS`) use two stages per rank and require an even divisor batch size, but reduce pipeline bubble and improve utilization. Start with `ScheduleInterleaved1F1B` (the default).

**Dataset loading:**

PP loads the dataset only on rank 0 (not sharded), unlike DDP which shards across ranks. This is handled automatically by the `pp.yaml` config.

**Basic usage:**
```bash
# 4 GPUs, interleaved schedule, flex attention
forgather -t pp.yaml train --init-model -d 0,1,2,3 --attn-implementation flex_attention

# 2 GPUs, GPipe schedule (simpler, one stage per rank)
forgather -t pp.yaml train --init-model -d 0,1 --pipeline-schedule ScheduleGPipe

# Increase microbatch count without changing batch size
forgather -t pp.yaml train --init-model -d 0,1 --microbatch-scale 2 --batch-size 8
```

**Resuming from checkpoint:**

PP checkpoints are saved per rank (each rank holds different pipeline stages). Resume works the same as DDP:
```bash
forgather -t pp.yaml train  # omit --init-model to resume
```

### Attention Implementations

**SDPA (default):**
- PyTorch native scaled dot-product attention
- Works everywhere, good performance
- Does NOT support sparsity from sequence packing

**Flex Attention (recommended):**
```bash
forgather train --attn-implementation flex_attention
```
- Supports sparsity masks (respects sequence packing)
- Excellent performance
- Requires PyTorch 2.5+

**Flash Attention 2:**
```bash
forgather train --attn-implementation flash_attention_2
```
- Fastest for dense attention
- Requires separate installation: `pip install flash-attn`
- Does NOT support sparsity

### Torch Compile

Enable with `--compile` flag:
```bash
forgather train --init-model --compile
```

**Trade-offs:**
- Initial compilation: takes a few minutes on first run
- Training speed: 10-30% faster after compilation
- Worth it for long runs, skip for quick experiments

### Checkpoint Management

**Automatic checkpointing:**
- Saves every 20K tokens (adjustable with `--step-cadence`)
- Keeps last 4 checkpoints (`save_total_limit: 4`)
- Preserves best 2 models by eval loss

**What's saved:**
- Model weights
- Optimizer state
- LR scheduler state
- Dataset position (for exact resume)
- RNG state (for reproducibility)
- Training progress

**Manual checkpoint control:**

Use training control commands while running:
```bash
# List running jobs
forgather control list

# Save checkpoint now
forgather control save JOB_ID

# Stop gracefully (saves final checkpoint)
forgather control stop JOID_ID

# Abort without saving (failed hyperparameter experiment)
forgather control abort JOB_ID
```

See [Training Job Control](../../../docs/trainers/trainer-control.md) for details.

### Monitoring Training

**Training logs:**
```bash
# View recent progress
tail -f output_models/default/runs/*/trainer_logs.json

# Get summary statistics
forgather logs summary

# Generate plots
forgather logs plot --loss-curves
forgather logs plot --loss-curves -e  # Open in editor

# Compare multiple runs
forgather logs plot --compare run1/trainer_logs.json run2/trainer_logs.json

# Start tensorboard for all models in output_models directory.
forgather tb --all               # Only available from localhost
forgather tb --all -- --bind_all # Available on all interfaces
```

See [Training Log Analysis](../../../docs/logs-analysis.md) for details.

**Divergence detection:**

The config includes a `DualTimeScaleDivergenceDetector` that monitors loss with fast and slow EMAs. If the fast EMA diverges from slow EMA by threshold (1.0), training aborts automatically to save compute.

### Memory Optimization

For fitting larger models:

1. **Reduce batch size:** `--batch-size 2` or `--batch-size 1`
2. **Reduce sequence length:** `--max-length 2048`
3. **Use gradient accumulation:** Edit config to set `gradient_accumulation_steps: 4`
4. **Use flash attention:** `--attn-implementation flash_attention_2`
5. **Disable some callbacks:** Edit config to remove peak_memory, text_gen callbacks

## Troubleshooting

**OOM (Out of Memory):**
- Reduce `--batch-size`
- Reduce `--max-length`
- Use `--attn-implementation flash_attention_2`
- Train on fewer GPUs with same global batch (LR auto-adjusts)

**Slow data loading:**
- First run downloads dataset (slow, one-time)
- Subsequent runs load from cache (fast)
- If still slow, check disk I/O and consider SSD

**Loss not decreasing:**
- Check LR isn't too low (use `--verbose-info` to see actual LR)
- Try higher `--learning-rate`
- Check for data preprocessing issues
- Verify model initialized correctly (`--init-model`)

**Loss exploding:**
- LR might be too high, reduce `--learning-rate`
- Check gradient clipping (`max_grad_norm: 4.0` in config)
- Divergence detector should catch this automatically

**Different results after resume:**
- Dataset state should restore automatically
- If deleted dataset checkpoint, training continues from random position
- Use `--no-restore-dataset-state` to explicitly restart from beginning

## Configuration Files

**Main config:** `templates/project.yaml`
- Defines all training parameters
- Inherits from base training script templates
- Customizable through child configs

**Alternative configs:**
- `configs/pp.yaml`: Pipeline Parallel training with configurable schedule and microbatch settings
- `configs/big_adam.yaml`: Large batch via gradient accumulation with AdamW
- `configs/tiny_small_lm.yaml`: Progressive curriculum (Tiny Stories → SmolLM)

**Creating custom configs:**
```yaml
-- extends 'project.yaml'

[config_metadata]
    == super()
    -- set ns.config_name = "My Experiment"

[optimizer]
    # Override to use AdamW instead of Adafactor
    optimizer: &optimizer !partial:torch.optim:AdamW
        lr: {{ (ns.base_learning_rate * ns.lr_scale) | toyaml }}
```

## References

- [Chinchilla Paper](https://arxiv.org/abs/2203.15556): Training Compute-Optimal Large Language Models
- [DeepNet Paper](https://arxiv.org/abs/2203.00555): Scaling Transformers to 1,000 Layers and Beyond
- [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus): High-quality pretraining data
- [InfiniteLR Scheduler](https://arxiv.org/abs/2108.06084): Training to Do Better Than a Fake Loss Function
- [Learning Rate Scaling](https://arxiv.org/abs/1711.00489): Don't Decay the Learning Rate, Increase the Batch Size
