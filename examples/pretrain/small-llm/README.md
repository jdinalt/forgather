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
- Precision: Automatic Mixed Precision (bf16)
- Model: Custom DeepOne (162M parameters)
- Token Budget: 2.3B tokens (Chinchilla-optimal for ~113M non-embedding params)
- Batch Size: 4 per device (auto-scaled with world size)
- Sequence Length: 4096 tokens
- Optimizer: AdamW

## Quick Start

```bash
# Init new model and train with defaults (Custom DeepOne 162M, 2.3B tokens)
forgather train --resume false

# Resume from checkpoint
forgather train

# Train with different model
rm -rf output_models # First, delete or backup old output directory
forgather train --resume false --model-project ../../models/llama/ --model-config medium.yaml

# Train for longer (23B tokens instead of 2.3B)
forgather train --resume false --total-tokens 23000

# Use sdpa attention and disable Torch compile for faster startup
forgather train --resume false --compile false --attn-implementation sdpa
```

## Training Configuration

The training configuration is built around several interconnected parameters that together determine training dynamics, compute requirements, and final model quality.

The project configuration is derived from our base LM training templates:

- [project.yaml](./templates/project.yaml)
  - [projects/auto_lr_project.yaml](../../../templatelib/examples/projects/auto_lr_project.yaml)
    - [projects/lm_training_project.yaml](../../../templatelib/examples/projects/lm_training_project.yaml)

The full documentation for the base templates can be found [here](../../../docs/project-templates/lm-training-projects.md).

### Token Budget and Compute Allocation

Training is controlled by a **token budget** rather than epochs or arbitrary step counts. This aligns with modern understanding of compute-optimal training.

**Key Parameters:**
- `--total-tokens N`: Total training tokens in millions (default: 2270)
- `--max-steps N`: Override to limit training steps (optional)

**Chinchilla Optimal:**
The default 2.3B token budget follows Chinchilla scaling laws for compute-optimal training:
```
optimal_tokens ≈ 20 × model_parameters
```

For the default model with ~113M non-embedding parameters: `113M × 20 = 2.3B tokens`

Note that we only count non-embedding parameters. For forgather's models, you can get this by constructing a model on the meta device.

```bash
# From model project directory
forgather [-t MODEL_CONFIG] model construct
```

**How it works:**
```python
# These are computed automatically from your settings:
tokens_per_step = max_length × global_batch_size
total_steps = total_tokens // tokens_per_step

# Example with defaults (4 GPUs):
# tokens_per_step = 4096 × (4 × 4) = 65,536
# total_steps = 2.27B / 65,536 ≈ 34,637 steps
```

### Batch Size and Learning Rate Scaling

Batch sizes and learning rates are coupled through sqrt-scaling to maintain training dynamics across different hardware configurations.

**Key Parameters:**
- `--batch-size N`: Per-device training batch size (default: 4)
- `--lr LR`: Base learning rate (default: 1.5e-4)

**How it works:**
```python
# Global batch size scales with world size
global_batch_size = batch_size × gradient_accumulation_steps × world_size
tokens_per_step = max_length × global_batch_size

# Learning rate scales relative to reference batch (single GPU defaults)
# Reference: tokens_per_step = 4096 × 4 = 16384
lr_scale = (tokens_per_step / 16384) ** 0.5
actual_lr = base_lr × lr_scale

# Example with 4 GPUs:
# tokens_per_step = 4096 × (4 × 4) = 65,536
# lr_scale = sqrt(65536 / 16384) = sqrt(4) = 2.0
# actual_lr = 1.5e-4 × 2.0 = 3.0e-4
```

**Why sqrt-scaling?**
This maintains the signal-to-noise ratio in gradients as batch size increases, allowing larger batches without changing optimization dynamics. Based on empirical findings from "Don't Decay the Learning Rate, Increase the Batch Size" (Smith et al., 2017).

### Learning Rate Scheduling Strategy

The LR schedule uses the **InfiniteLRScheduler**, designed for flexible pretraining with optional continuation and annealing phases.

**Schedule Structure:**
```
1. Warmup (5% of total steps): 0 → max_lr
2. Cooldown (variable): max_lr → constant_lr (cosine decay)
3. Constant (remaining): constant_lr
4. Annealing (optional): constant_lr → min_lr (not used by default)
```

**Key Parameters:**
- `--warmup-tokens N`: The number of tokens (millions) for the warmup phase.
- `--min-cooldown-tokens N`: Minimum tokens (millions) for cooldown phase.

**Cooldown Duration:**

The cooldown duration is determined by two factors: a proportional term `P` and a minimum floor:

```python
cooldown_steps = max(min_cooldown_steps, total_steps * P)
```

`P` (default 0.3) controls what fraction of total training is spent in cooldown. This matches the best result from the InfiniteLR paper's appendix, where P=0.3 outperformed longer cooldown proportions for language model training at 10x+ Chinchilla scale.

The floor (`min_cooldown_tokens`, default 6.8B) prevents excessively fast cooldown on short runs. It is set to approximately 3x Chinchilla-optimal for the default 113M non-embedding parameter model (113M x 20 x 3 = 6.8B tokens). This value should be scaled in proportion to the model size when training larger or smaller models.

**How the two terms interact:**

The proportional term `total_steps * P` exceeds the floor when `total_tokens > min_cooldown_tokens / P`. For the defaults, this crossover is at ~22.7B tokens (~10x Chinchilla). Below that, the floor dominates:

- **2.3B tokens** (1x Chinchilla): cooldown = floor (6.8B). Training ends ~32% through the cosine, keeping LR at ~77% of max. The model never reaches the constant phase, which is appropriate for a short run.
- **6.8B tokens** (3x Chinchilla): cooldown = floor (6.8B). Cosine nearly completes, transitioning to constant_lr.
- **23B+ tokens** (10x+ Chinchilla): P=0.3 takes over. The first 30% of training decays from max_lr to constant_lr, and the remaining 70% trains at constant_lr.

**Base Learning Rates** (before sqrt-scaling):
- `max_lr`: 1.5e-4 (set with `--lr`)
- `constant_lr`: 5.0e-5 (~33% of max)
- `min_lr`: 5.0e-6 (~3% of max)

**The annealing phase** (warmup → cooldown → constant → anneal) is a key feature of the InfiniteLR schedule. Unlike cosine scheduling, which requires knowing the total training budget upfront, InfiniteLR can run indefinitely at `constant_lr` and only anneal when a converged checkpoint is needed.

**Annealing formula** (Eq. 1 from the paper):
```
η(n) = η_const × (η_min / η_const) ^ ((n - N_d) / (t_a + N_d))
```

Where `N_d` is the step at which annealing starts (`checkpoint_step`) and `t_a` is the annealing budget (`tau`). The LR reaches exactly `η_min` after `t_a + N_d` annealing steps.

**Adaptive decay rate:** The denominator `t_a + N_d` means that models trained longer before annealing (larger `N_d`) decay more slowly. This is intentional: a model that has trained for 100B tokens should anneal more gradually than one trained for 1B tokens, since the loss landscape is flatter and more sensitive to abrupt LR changes.

**Typical workflow:**
1. Train at `constant_lr` for as long as needed (the "infinite" phase)
2. When ready for a converged checkpoint, set `checkpoint_step` to the current step and choose a `tau` (annealing budget)
3. Continue training through the annealing phase to produce a final model
4. Alternatively, fork from the constant-phase checkpoint for multiple task-specific annealing runs with different `tau` values

This approach decouples the training budget decision from the LR schedule, which is particularly valuable for continual pre-training where new data arrives over time. See [Beyond Cosine Decay: On the effectiveness of Infinite Learning Rate Schedule for Continual Pre-training](https://arxiv.org/html/2503.02844v1) for the full analysis.

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
# Base intervals (in millions of tokens):
logging_interval = 1M tokens
eval_interval = 25M tokens
save_interval = 500M tokens

# Actual step intervals:
logging_steps = (1M × step_cadence) / tokens_per_step
eval_steps = (25M × step_cadence) / tokens_per_step
save_steps = (500M × step_cadence) / tokens_per_step
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
- GLU feedforward (ReLU)
- 32K vocabulary

**Why DeepOne?**
The DeepNet architecture is relatively forgiving for pretraining experiments due to its improved initialization and normalization. Good choice for learning about pretraining dynamics.

See [Custom DeepOne README](./custom_deepone/README.md) for details.

### Using Different Models

```bash
# Llama architecture (30M - 3B params)
forgather train --resume false \
    --model-project ../../models/llama/ \
    --model-config medium.yaml

# Qwen3 architecture
forgather train --resume false \
    --model-project ../../models/qwen3/ \
    --model-config medium.yaml

# List available model configs
ls ../../models/llama/templates/configs/
```

**Important:** When changing models, delete the old output directory:
```bash
rm -rf output_models
forgather train --resume false --model-project ...
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
| `--resume false` | Initialize new model (vs resume from checkpoint) | Resume |
| `--total-tokens N` | Total training tokens in millions | 2270 |
| `--batch-size N` | Per-device training batch size | 4 |
| `--lr LR` | Base learning rate (before batch size scaling) | 1.5e-4 |
| `--max-length N` | Maximum sequence length | 4096 |
| `--min-cooldown-tokens N` | Minimum tokens (millions) for LR cooldown | 6800 |

### Model Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-project PATH` | Path to model project directory | `./custom_deepone` |
| `--model-config NAME` | Model configuration file | `custom_deepone.yaml` |

### Attention Implementation

| Option | Description | Performance |
|--------|-------------|-------------|
| `--attn-implementation sdpa` | PyTorch SDPA | Good, no sparsity |
| `--attn-implementation flex_attention` | Flex attention (default) | Best, supports sparsity |
| `--attn-implementation flash_attention_2` | Flash Attention 2 | Excellent, requires installation |
| `--attn-implementation eager` | Standard PyTorch | Slow, debugging only |

The default `flex_attention` provides best performance with sequence packing. Use `sdpa` for faster startup during quick experiments.

### Training Control

| Option | Description | Default |
|--------|-------------|---------|
| `--max-steps N` | Override max steps (instead of token budget) | Auto-computed |
| `--save-strategy {no,steps,epoch}` | When to save checkpoints | steps |
| `--step-cadence FACTOR` | Scale log/eval/save intervals | 1.0 |
| `--compile false` | Torch compile (slower startup, faster training) | Enabled |

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

Train the default Custom DeepOne 162M model for 2.3B tokens (Chinchilla-optimal):

```bash
forgather train --resume false
```

**What happens:**
- Model: Custom DeepOne 162M
- Total tokens: 2.3B
- Total steps: ~35K (with 4 GPUs)
- LR: Stays at ~99% of max_lr throughout
- Training time: ~6-8 hours on 4x RTX 3090

### Example 2: Longer Training Run

Train for 10B tokens to see more convergence:

```bash
forgather train --resume false \
    --total-tokens 10000 \
    --attn-implementation flex_attention
```

**What happens:**
- Total steps: ~153K (with 4 GPUs)
- LR: Gentle decay over training (~87% of max_lr at end)
- Better final performance but diminishing returns per token

### Example 3: Small Model, High Cadence

Train a tiny model quickly with frequent checkpointing:

```bash
forgather train --resume false \
    --model-project ../../models/llama/ \
    --model-config small.yaml \
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
forgather train --resume false \
    -d 0,1,3,4,5 \
    --attn-implementation flex_attention
```

**What happens:**
- Uses 5 GPUs (skips GPU 2)
- `global_batch_size = 4 × 5 = 20`, `tokens_per_step = 4096 × 20 = 81920`
- LR auto-scales: `1.5e-4 × sqrt(81920 / 16384) ≈ 3.4e-4`

### Example 5: Hyperparameter Exploration

Quick experiment without saving checkpoints:

```bash
forgather train --resume false \
    --max-steps 2000 \
    --save-strategy no \
    --lr 2.0e-5 \
    --attn-implementation flex_attention
```

**Use case:**
- Test different learning rates quickly
- No disk space used for checkpoints
- 2000 steps ≈ 130M tokens with defaults (4 GPUs)

### Example 6: Resume and Continue

Resume from checkpoint and train longer:

```bash
# Initial run (uses default 2.3B token budget)
forgather train --resume false

# Later: continue for more tokens (10B total)
forgather train --total-tokens 10000
```

**What happens:**
- Resumes from last checkpoint
- Continues to 10B total tokens (10000M)
- Dataset state restored (picks up where it left off)

### Example 7: Pipeline Parallel Training

Train using Pipeline Parallel across 4 GPUs with the interleaved schedule:

```bash
forgather -t pp.yaml train --resume false \
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
forgather -t pp.yaml train --resume false \
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
forgather train --resume false \
    --total-tokens 50000 \
    --min-cooldown-tokens 100000 \
    --attn-implementation flex_attention
```

**What happens:**
- 50B token budget (50000M)
- min_cooldown of 100B → only 50% cosine progress → gentle decay
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

Tip: When you double batch-size, also double microbatch-scale. e.g.

```
... --batch-size 4 --microbatch-scale 1
... --batch-size 8 --microbatch-scale 2
... --batch-size 32 --microbatch-scale 8
...
```

**Choosing a schedule:**

Simple schedules (`ScheduleGPipe`, `Schedule1F1B`) use one stage per rank. Interleaved schedules (`ScheduleInterleaved1F1B`, `ScheduleLoopedBFS`) use two stages per rank and require an even divisor batch size, but reduce pipeline bubble and improve utilization. Start with `ScheduleInterleaved1F1B` (the default).

**Dataset loading:**

PP loads the dataset only on rank 0 (not sharded), unlike DDP which shards across ranks. This is handled automatically by the `pp.yaml` config.

**Basic usage:**
```bash
# 4 GPUs, interleaved schedule, flex attention
forgather -t pp.yaml train --resume false -d 0,1,2,3 --attn-implementation flex_attention

# 2 GPUs, GPipe schedule (simpler, one stage per rank)
forgather -t pp.yaml train --resume false -d 0,1 --pipeline-schedule ScheduleGPipe

# Increase microbatch count without changing batch size
forgather -t pp.yaml train --resume false -d 0,1 --microbatch-scale 2 --batch-size 8
```

**Resuming from checkpoint:**

PP checkpoints are saved per rank (each rank holds different pipeline stages). Resume works the same as DDP:
```bash
forgather -t pp.yaml train  # or --resume true to resume
```

### Attention Implementations

**Flex Attention (default):**
- Supports sparsity masks (respects sequence packing)
- Excellent performance
- Requires PyTorch 2.5+

**SDPA:**
```bash
forgather train --attn-implementation sdpa
```
- PyTorch native scaled dot-product attention
- Works everywhere, good performance, faster startup
- Does NOT support sparsity from sequence packing

**Flash Attention 2:**
```bash
forgather train --attn-implementation flash_attention_2
```
- Fastest for dense attention
- Requires separate installation: `pip install flash-attn`
- Does NOT support sparsity

### Torch Compile

Enabled by default. Disable with `--compile false` for quick experiments:
```bash
forgather train --resume false --compile false
```

**Trade-offs:**
- Initial compilation: takes a few minutes on first run
- Training speed: 10-30% faster after compilation
- Worth it for long runs, disable for quick experiments

### Checkpoint Management

**Automatic checkpointing:**
- Saves every 500M tokens (adjustable with `--step-cadence`)
- Keeps last 4 checkpoints (`save_total_limit: 4`)
- Preserves best model by eval loss

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
forgather control stop JOB_ID

# Abort without saving (failed hyperparameter experiment)
forgather control abort JOB_ID
```

See [Training Job Control](../../../docs/trainers/trainer-control.md) for details.

### Monitoring Training

**Training logs:**
```bash
# View recent progress
tail -f output_models/sllm/runs/*/trainer_logs.json

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

The config includes a `DivergenceDetector` that monitors smoothed train loss (EMA with alpha=0.3) against its best observed value. If the smoothed loss exceeds the best by threshold (1.0) for 3 consecutive observations, training aborts automatically to save compute.

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
- Try higher `--lr`
- Check for data preprocessing issues
- Verify model initialized correctly (`--resume false`)

**Loss exploding:**
- LR might be too high, reduce `--lr`
- Consider enabling gradient clipping (uncomment `max_grad_norm: 4.0` in config)
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
- `configs/bf16.yaml`: Full bfloat16 precision training (no mixed precision)
- `configs/tiny_x_small_lm.yaml`: Progressive curriculum (Tiny Stories → SmolLM)

**Creating custom configs:**
```yaml
-- extends 'project.yaml'

[config_metadata]
    == super()
    -- set ns.config_name = "My Experiment"
    -- set ns.model_name = "my_experiment"
    -- set ns.base_lr = 3.0e-5
```

## References

- [Chinchilla Paper](https://arxiv.org/abs/2203.15556): Training Compute-Optimal Large Language Models
- [DeepNet Paper](https://arxiv.org/abs/2203.00555): Scaling Transformers to 1,000 Layers and Beyond
- [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus): High-quality pretraining data
- [InfiniteLR Schedule](https://arxiv.org/abs/2503.02844): Beyond Cosine Decay: On the effectiveness of Infinite Learning Rate Schedule for Continual Pre-training
- [Learning Rate Scaling](https://arxiv.org/abs/1711.00489): Don't Decay the Learning Rate, Increase the Batch Size
