# LM Training Project Template

Forgather ships a reusable project template for language model pre-training.
It computes training steps automatically from a target token budget, includes
automatic LR scaling based on global batch size, and supports four trainer
backends switchable via `--trainer-type`: basic (single GPU), DDP (multi-GPU),
FSDP2 (sharded multi-GPU), and Pipeline Parallel.

**Template:** [projects/lm_training_project.yaml](../templatelib/examples/projects/lm_training_project.yaml)\
**Extends:** [training_script/causal_lm/causal_lm.yaml](../templatelib/base/training_script/causal_lm/causal_lm.yaml)\
**See also:** [Trainer Options Reference](../trainers/trainer_options.md) -
every option exposed on the trainers that this template wires up.

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

The template supports four trainer backends, selected via `--trainer-type`
or by setting `ns.trainer_type` in a child template's `[config_metadata]`:

| Trainer Type | Backend | Default nproc_per_node | Description |
|--------------|---------|------------------------|-------------|
| `basic` | `forgather.ml.trainer:Trainer` | 1 | Single-GPU training |
| `ddp` | `forgather.ml.trainer.ddp:DDPTrainer` | gpu (all GPUs) | Distributed Data Parallel |
| `fsdp2` | `forgather.ml.trainer.fsdp2:FSDP2Trainer` | gpu (all GPUs) | Fully Sharded Data Parallel v2 (`torch.distributed.fsdp.fully_shard`) |
| `pipeline` | `forgather.ml.trainer.pipeline:PipelineTrainer` | gpu (all GPUs) | Pipeline Parallel |

The trainer type controls which trainer template is included and, for
`pipeline`, automatically sets `nproc_per_node` to `"gpu"` (all available
GPUs). For `ddp` and `fsdp2`, pass `--nproc-per-node` or set
`ns.nproc_per_node` in your child config. Use `-d` to restrict to
specific devices.

### DDP Notes

- All GPUs are used by default. Restrict with `-d 0` or `-d 0,1`.
- Stopping DDP with Ctrl-C can leave worker processes running. Use
  `forgather control list` and `forgather control stop JOB_ID` for a clean
  shutdown.

### FSDP2 Notes

FSDP2 shards parameters, gradients, and optimizer state across the
data-parallel mesh via `torch.distributed.fsdp.fully_shard`. Reach for
it when the model is larger than a single GPU's memory, or when you
want to free optimizer-state memory for a model that would otherwise
fit under DDP but leave little headroom for activations.

- Parameters become DTensors in place; the trainer applies
  `fully_shard` layer-by-layer on the transformer blocks (default
  attribute path `causal_lm.layer_stack.layers`) before the root module.
  Override `fsdp2.transformer_layers_path` in `[trainer_args]` if your
  model has a different block-list attribute.
- FSDP2-level mixed precision is configured via the
  `fsdp2.param_dtype` / `fsdp2.reduce_dtype` knobs (separate from the
  template's `--mixed-precision` AMP autocast). A typical setting for
  memory-bound runs is `param_dtype: bfloat16, reduce_dtype: float32`.
- `fsdp2.cpu_offload: True` offloads parameters and gradients to CPU
  between uses - large memory savings at the cost of PCIe transfer
  time.
- **Checkpoints are saved as per-rank DTensor shards** and are tied to
  the world size they were trained at. Resuming at the same world size
  works normally (`--resume` / `resume_from_checkpoint: True`);
  resuming at a different world size is not supported in this first
  cut and would require switching to `torch.distributed.checkpoint`
  (DCP).
- See [`trainer_options.md`](../trainers/trainer_options.md#fsdp2trainingarguments)
  for the full list of `fsdp2.*` fields.

### Pipeline Parallel

Pipeline Parallel splits the model into stages distributed across GPUs.
When `--trainer-type pipeline` is selected, the template automatically:

- Forces `dispatch_batches = True` (rank-0 loads and dispatches data)
- Computes microbatch and stage configuration from the pipeline schedule
- Overrides batch sizes to the computed PP batch size
- Rejects `torch_compile_mode: max-autotune` (incompatible with PP; use
  `max-autotune-no-cudagraphs` instead)
- Rejects `compile: true` combined with zero-bubble schedules
  (`ScheduleInterleavedZeroBubble`, `ScheduleZBVZeroBubble`)

The pipeline schedule determines how microbatches flow through stages:

| Schedule | stages_per_rank | When to use |
|----------|-----------------|-------------|
| `ScheduleGPipe` | 1 | Simple reference scheduler. Use only as a failsafe - other schedules are strictly better. |
| `Schedule1F1B` | 1 | **Lowest memory consumption.** Reach for it when pipeline memory is the bottleneck. |
| `ScheduleInterleaved1F1B` | 2 | **Default.** Stable, good throughput, broad compatibility, and works with `torch.compile`. |
| `ScheduleLoopedBFS` | 2 | Alternative interleaved schedule; worth trying if you're micro-optimising. |
| `ScheduleInterleavedZeroBubble` | 2 | Near-zero bubble; incompatible with `torch.compile`. |
| `ScheduleZBVZeroBubble` | 2 | **Best raw throughput**, but experimental and fickle. Incompatible with `torch.compile`. Flex-attention works via a Forgather monkey-patch - treat the combination as experimental. |

Rule of thumb: start with `ScheduleInterleaved1F1B` + `--compile true`. Drop
to `Schedule1F1B` if you're memory-limited. Try `ScheduleZBVZeroBubble` (with
compile off) when squeezing the last few percent of throughput matters more
than stability.

**Pipeline + text-generation callback (experimental):** the pipeline trainer
now supports distributed text generation, so the text-generation callback
finally works under pipeline parallel. Treat this as an experimental feature
and disable the callback if you see hangs or other oddities. See
[Pipeline Parallel -> Text generation](../trainers/pipeline-parallel.md#text-generation-experimental).

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

### Use an LR sweep before long runs

Automatic LR scaling is there to get you into the right ballpark and to let
you change batch size / world size without rethinking everything. It's
deliberately conservative - a good starting point, not a final answer.
Before committing to a long training run, do a short LR sweep and pick the
best point on the curve. The right answer is usually the highest LR that
doesn't diverge, but not always: there are cases where a lower LR is stable
*and* gives a better final loss, so sweep widely enough to see both
failure modes.

A cheap sweep: pick a handful of candidate LRs spaced ~0.5x apart (e.g.,
1x, 2x, 4x the scaled LR), train each for a few percent of the total token
budget, and compare training-loss trajectories.

### Pure bf16 training and stochastic rounding

"Pure bf16" means weights, activations, *and* gradients live in bf16 (as
opposed to mixed precision, where master weights stay in fp32). Pure bf16
halves optimizer memory but loses the fp32 accumulator, so small updates to
large weights are lost to rounding error unless the optimizer compensates.

The standard fix is **stochastic rounding (SR)** in the optimizer: instead
of round-to-nearest-even, round up or down with probability proportional to
how close the unrounded update is to each neighbour. Over many steps this
preserves updates that would otherwise be lost, closing most of the gap to
fp32 accumulation.

- Forgather's `Adafactor` and `Adam` optimizers default to SR when their
  state is bf16.
- `torchao` provides quantized Adam variants with SR if you also want to
  compress the optimizer state.
- For the theory and practical trade-offs, see *Stochastic Rounding for
  LLM Training: Theory and Practice*
  (<https://arxiv.org/pdf/2502.20566>).

Tuning notes:

- Pure bf16 + SR **needs** (and tolerates) higher learning rates than
  mixed-precision training. The automatic LR scaling curve fits
  mixed-precision; when switching to pure bf16 + SR, rerun the LR sweep.
- Properly tuned, pure bf16 + SR can reach perplexity comparable to
  mixed-precision at meaningfully lower optimizer-memory cost.

### Sweep batch size for throughput

Batch size deserves its own sweep before a long run. In practice the right
target is **the batch size that maximises sustained throughput (tokens per
second) without triggering OOM** - not the largest batch that happens to
fit. For smaller models in particular, the throughput-optimal batch size is
often noticeably below what the GPU can physically hold: past some point,
larger batches don't improve GPU utilisation and just inflate activation
memory and step time.

This empirical observation aligns with Marek et al. 2025, *Small Batch
Size Training for Language Models: When Vanilla SGD Works, and Why
Gradient Accumulation Is Wasteful*
(<https://arxiv.org/pdf/2507.07101>), whose main findings are worth
internalising:

- **Small batches train stably.** The paper demonstrates stable LM
  pre-training and fine-tuning all the way down to batch size one -
  contradicting the common belief that tiny batches are inherently
  unstable.
- **Equal or better per-FLOP performance.** Small batches achieve equal
  or better loss per FLOP than larger batches, and are consistently
  more *robust* to hyperparameter choices.
- **Avoid gradient accumulation on a single device.** The authors
  explicitly recommend against `gradient_accumulation_steps > 1` unless
  you're training multiple model replicas across devices. On a single
  device, accumulation pays the compute cost of a larger effective
  batch without the corresponding statistical benefit - a smaller real
  batch is usually better. In Forgather terms: prefer lowering
  `--batch-size` (and letting the template rescale the LR) over
  raising `--gradient-accumulation-steps`.
- **Adafactor is competitive with Adam at small-to-moderate batch
  sizes.** At the small end the paper finds Adafactor and Adam produce
  comparable results; Adam only pulls ahead at larger batch sizes.
  That matters for fine-tuning large models on limited hardware: you
  can run Adafactor for near-zero extra optimizer memory and get
  results comparable to plain or quantized Adam. Reach for Adam (or
  a quantized Adam) when you're training at a large batch size *and*
  you have the memory to spare.
- **Vanilla SGD is demonstrated viable, but not necessarily practical.**
  The paper shows that vanilla SGD (no momentum, *no* optimizer state)
  can train LMs stably at batch size one. That's a striking existence
  proof - it recovers a LoRA-like memory footprint while doing full
  fine-tuning - but don't read it as "use SGD by default". The main
  realistic case for SGD + batch-size-1 is when the model is so large
  that a single example is all that fits on the device; for most other
  setups you'll still want Adafactor or Adam.

#### A note on `beta2` scaling

The paper also proposes holding Adam's second-moment *half-life fixed
in terms of tokens* across batch sizes rather than holding `beta2`
itself fixed: if you cut the batch in half, push `beta2` closer to 1 so
the effective averaging window (in tokens) stays constant.

In practice, the default `beta2 = 0.999` shipped by `torch.optim.AdamW`
and Forgather's optimizers is *already* tuned for small batch sizes, so
you generally don't need to touch it for the default LM Training Project
settings. The scaling only starts to matter when you increase the batch
size substantially - we've measured a modest but real effect at batch
sizes around 256 tokens-per-step and above. Below that, leave `beta2`
alone.

When you do want to apply the scaling, `examples/tiny_experiments/optimizers/`
implements it as a template using the same reference batch size as the
LR scaling:

```yaml
[config_metadata]
    == super()
    ## beta2 is optimal at base_batch_size tokens per step
    -- set ns.beta2 = 0.999

[globals]
    == super()
    ## Constant-half-life scaling in token units
    -- set ns.scaled_beta2 = ns.beta2 ** (ns.tokens_per_step / ns.base_batch_size)

[optimizer]
optimizer: &optimizer !partial:torch:optim.AdamW
    lr: {{ ns.global_lr | toyaml }}
    betas:
        - !!float 0.9
        - {{ ns.scaled_beta2 | toyaml }}
```

At `tokens_per_step == base_batch_size` this is the identity and leaves
`beta2 = 0.999`. Smaller real batches push the exponent below 1, which
drives `scaled_beta2` closer to 1 (longer EMA window in tokens); larger
batches shorten it.

#### Practical batch-size workflow

1. Run a quick throughput sweep. Try `--batch-size` values spaced by
   powers of two (e.g., 8, 16, 32, 64), train a few hundred steps each,
   and record sustained tokens/sec from the training logs. Pick the
   best-throughput point that still leaves a little headroom below OOM.
2. With the chosen batch size, run the LR sweep described above. The
   template's automatic LR scaling will already move the base LR with
   batch size; your sweep is just fine-tuning around that.
3. If you have pushed the batch size well above the `base_batch_size`
   reference (say, 256+ tokens-per-step) and are using Adam, apply the
   `scaled_beta2` formula shown above. At small-to-moderate batches
   the default `0.999` is already fine.
4. Only reach for `--gradient-accumulation-steps > 1` when you genuinely
   cannot fit a single real batch of the target size *and* you are
   training a single model replica. On multi-device runs with
   replication (DDP / pipeline), accumulation is fine - that's the
   case the paper explicitly excludes from its recommendation.

## LR Scheduler Behaviour

The default LR schedule is **cosine decay with warmup**, which has been
industry standard practice for LM pre-training for years. The cosine
scheduler's `total_steps` is clamped to at least `min_cooldown_steps` so
that short training runs (or runs that are stopped early) do not decay the
learning rate all the way to zero. This shows up in the preprocessed output
as:

```yaml
# LR Scheduler (cosine decay with warmup)
# total_steps is clamped to min_cooldown_steps (20345)
# so that short runs do not decay the LR all the way to zero.
lr_scheduler: ...
    warmup_steps: 3797
    total_steps: 37978
```

### Limitations of cosine decay

The cosine-decay schedule has one major practical weakness: it requires you
to commit to a **total token budget up front**. The full LR curve is
parameterised on `total_steps`, so if you later want to continue pre-training
the same checkpoint on more tokens, you have to decide between:

- re-warming the LR from a small value (historically this induces a visible
  instability spike and a period of elevated loss before the model recovers),
  or
- ignoring the original schedule and ad-hoc patching in a new one, which
  loses the "compute-optimal" calibration the cosine curve was giving you.

This makes plain cosine decay a poor fit for continual pre-training or for
iterative workflows where the token budget is not known in advance. Two
alternatives address this directly and are both demonstrated in
`examples/pretrain/small-llm/`.

### Alternative: Warmup-Stable-Decay (WSD / WSD-S)

Warmup-Stable-Decay replaces the cosine curve with three explicit phases:

1. **Warmup** - linear ramp from 0 to peak LR over `warmup_steps`.
2. **Stable** - constant LR, held indefinitely until you trigger the decay.
3. **Decay (annealing)** - decay from peak LR down to a small fraction of
   it over `decay_steps`.

The original WSD protocol was "checkpoint-then-branch": train in the stable
phase, branch the checkpoint, anneal one branch for release while continuing
pre-training on the un-annealed branch. **WSD-S (Simplified)** - proposed
by Wen et al. 2024, *Understanding Warmup-Stable-Decay Learning Rates: A
River Valley Loss Landscape Perspective*
(<https://arxiv.org/abs/2410.05192>) - drops the branching: you anneal the
live checkpoint and then resume pre-training from the *decayed* checkpoint.
According to the authors, this avoids the re-warm-up instability that
plagues continued pre-training from a cosine-decayed checkpoint. If that
claim holds up in your setting, it is a meaningful quality-of-life win for
continual pre-training.

`examples/pretrain/small-llm/templates/configs/wds.yaml` uses Forgather's
`WSDScheduler` and demonstrates the full protocol, including the hooks for
triggering decay and resuming afterwards.

### Alternative: Infinite Learning Rate Schedule

*Beyond Cosine Decay: On the effectiveness of Infinite Learning Rate
Schedule for Continual Pre-training* (<https://arxiv.org/abs/2503.02844>)
proposes an extension of WSD: append a *cosine decay from the peak LR down
to roughly 1/3 of the peak* before the stable phase (so the "stable" phase
is actually a constant at ~1/3 peak), then anneal to a small final LR at
the end. The authors report that this schedule helps reduce catastrophic
forgetting when the dataset distribution changes between training sessions
(in addition to the usual replay-buffer tricks), making it well-suited to
continual pre-training.

`examples/pretrain/small-llm/templates/project.yaml` defaults to
Forgather's `InfiniteLRScheduler` for exactly this reason:

```yaml
lr_scheduler: &lr_scheduler !partial:forgather.ml.optim:InfiniteLRScheduler@lr_scheduler
    warmup_steps: {{ ns.warmup_steps }}
    cooldown_steps: {{ ns.cooldown_steps }}
    constant_lr: {{ ns.constant_lr | toyaml }}
    min_lr: {{ ns.min_lr | toyaml }}
    annealing_type: "rsqrt"
    start_annealing: {{ start_annealing | toyaml(False) }}
    annealing_steps: {{ ns.annealing_steps }}
```

Both the WSD and infinite schedules in that project plug into the same
`forgather control` machinery, so you can externally trigger the annealing
phase on a running job (via `--start-annealing` or the control callback)
and resume pre-training from the resulting checkpoint.

### Swapping schedulers in your own config

The LM Training Project defaults to cosine for backward compatibility and
because it is the right choice for one-shot runs with a known token budget.
To swap in WSD or the infinite schedule, override the `[lr_scheduler]`
block in your config - the `examples/pretrain/small-llm/` templates are
the reference implementation to copy from.

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
| `dispatch_batches` | bool | False | When True, rank-0 loads and dispatches all batches (DDP). Prefer False for throughput - centralized loading adds per-step dispatch overhead and serialises data loading on rank-0. Only enable when the dataset implementation doesn't support sharding, or when you need the DDP run to see the same example sequence as a single-GPU / pipeline run for direct comparison of training curves |

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
| `peak_hardware_flops` | float | auto-detected | Peak device FLOPS for MFU computation. Auto-detected from the current GPU or `~/.forgather/hardware.yaml`. See [training-performance-metrics](../trainers/training-performance-metrics.md) |

### Precision / Compilation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_dtype` | str | null | Torch dtype for model construction. Choices: float32, bfloat16, float16 |
| `float32_matmul_precision` | str | null | Approximate float32 matmul with bf16. Choices: highest, high, medium. On Ampere+ GPUs you usually want `high` - leaving this unset keeps the PyTorch default (`highest`) which disables TF32 and triggers a runtime warning |
| `mixed_precision` | str | null | AMP dtype. Choices: bf16, fp16, no |
| `fp8_recipe` | str | null | FP8 recipe for linear layers. Choices: tensorwise, rowwise, rowwise_with_gw_hp |
| `compile` | bool | False | Enable torch.compile |
| `torch_compile_mode` | str | `default` | `torch.compile` mode. Choices: `default`, `max-autotune`, `max-autotune-no-cudagraphs`. See [When to use each compile mode](#when-to-use-each-compile-mode) below |

#### When to use each compile mode

- `default` - Safe baseline. Works with gradient accumulation, DDP, and
  pipeline parallel.
- `max-autotune` - Preferred when it's supported. Best steady-state
  throughput, and it's the default in the pre-train example project.
  **Incompatible** with `gradient_accumulation_steps > 1` and with pipeline
  parallel (both raise a template error). Also incompatible with zero-bubble
  pipeline schedules.
- `max-autotune-no-cudagraphs` - Use this instead of `max-autotune` when you
  need autotune benefits with gradient accumulation or pipeline parallel.

The template refuses invalid combinations at preprocess time, so
`forgather pp` will fail with an explicit message rather than blow up
mid-training.

#### Compile startup cost

Enabling any `compile` setting adds significant latency to the first training
step and the first eval step, because `torch.compile` has to trace and
generate code for the model's forward (and backward) before they can run.
`max-autotune` in particular has very long startup times and tends to spam
the TTY with copious diagnostic messages during its tuning sweep. PyTorch
caches the compiled artefacts on disk, so subsequent runs of the same
configuration start faster - but "faster" is still "slow" compared to an
uncompiled run.

Practical guidance:

- Iterating on configs, hyperparameters, or dataset plumbing? Disable
  `--compile` so each `forgather train` invocation starts in seconds.
- Actual training runs? Turn `--compile` on and pick the most aggressive
  compatible mode (typically `max-autotune` for single-GPU, or
  `max-autotune-no-cudagraphs` when gradient accumulation or pipeline
  parallel is in play). The one-time warmup is amortised across the whole
  run.

### Memory / Performance

These options trade compute, memory, and flexibility. They are most useful
when you hit OOM or activation-memory pressure on large models.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gradient_checkpointing` | bool | False | Enable activation checkpointing for models that implement the HF gradient-checkpointing API. Trades recompute for lower activation memory. |
| `activation_offloading` | bool | False | Offload saved activations to CPU memory during backward (via `torch.autograd.graph.save_on_cpu`). Pair with `gradient_checkpointing` for maximum memory savings, at the cost of extra host-device bandwidth. |
| `activation_memory_budget` | float | null | Fraction passed to `torch._functorch.config.activation_memory_budget`. Requires `compile=True`. See the [PyTorch activation checkpointing techniques post](https://pytorch.org/blog/activation-checkpointing-techniques/). |
| `fuse_optim_with_backward` | bool | False | Apply the optimizer update inside the backward grad hook so each gradient is consumed and freed immediately. Biggest memory wins when combined with `gradient_checkpointing`. Supported by the basic and pipeline trainers. Incompatible with DDP / Accelerate (they need gradients intact for the all-reduce), with `max_grad_norm` gradient clipping (needs the full gradient vector before any update), with `gradient_accumulation_steps > 1`, and with fp16 AMP. |
| `construct_model_on` | str | `default` | Where to build the model. `default` builds on CPU and moves to device (safest); `device` builds directly on device (faster, may fail for sharded models); `meta` builds on the meta device and materialises empty tensors on the target device (fastest). With `meta`, the basic trainer falls back to CPU construction automatically if no checkpoint is available. Ignored by the pipeline trainer - it always builds on meta (see note below). |
| `gc_threshold` | float | 0.9 | Ratio of allocated-to-total GPU memory at which to try `empty_cache()` and (conditionally) `gc.collect()`. Real trade-off: too low and frequent cache flushes will slow training; too high and you OOM from fragmentation. Tune against the OOM edge. |

Picking a combination:

- **OOM from activations** - start with `--gradient-checkpointing`. If that is
  still tight, add `--activation-offloading true`.
- **Maximum memory savings** - combine `--gradient-checkpointing`,
  `--fuse-optim-with-backward true`, and `--activation-offloading true`.
  `fuse_optim_with_backward` is supported by the basic and pipeline trainers
  but **not** by DDP or Accelerate: DDP needs all gradients intact after
  backward for the all-reduce, and the fused hook would consume and free
  them before that step could run. `fuse_optim_with_backward` is also
  incompatible with gradient clipping (`max_grad_norm`): clipping requires
  the full gradient vector before any optimizer update, and the fused hook
  applies each parameter's update as soon as its gradient is produced.
  Similarly incompatible with `gradient_accumulation_steps > 1` and fp16
  AMP.
- **Fastest model construction** - use `--construct-model-on device` when the
  model fits on one GPU. Use `meta` when you already have a checkpoint to
  load and want to skip weight initialisation entirely; if there is no
  checkpoint, the basic trainer falls back to CPU construction
  automatically so you don't end up training against uninitialised weights.
- **Fragmentation-related OOM** - lower `--gc-threshold` (e.g., `0.75`). Don't
  lower it further than you have to: each cache flush synchronises the CUDA
  stream and can visibly cut throughput.

`PipelineTrainer` always builds the model on meta, regardless of
`construct_model_on`. When resuming from a checkpoint, the meta model is
filled directly from the checkpoint - no CPU materialisation. When training
from scratch, rank-0 currently builds the full model on CPU just long
enough to initialise the weights, then dispatches the initialised
parameters and buffers to each rank's empty stage. Transformers v5 exposes
an API for initialising weights directly on a meta-constructed model,
which would let us skip the rank-0 CPU step - not wired up yet.

See [`../trainers/trainer_options.md`](../trainers/trainer_options.md) for the
underlying trainer field semantics.

### Misc

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | 42 | Random seed |
| `resume` | bool | True | Auto-resume from latest checkpoint (falls back to fresh init if none exists) |
| `save_strategy` | str | steps | Checkpoint save strategy. Choices: no, steps, epoch |
| `save_safetensors` | bool | False | Use safetensors format for weights |
| `debug_optimizer_groups` | bool | False | Log parameter -> optimizer-group assignments when the optimizer is built. Useful for verifying `optimizer_groups` overrides. |

Notes:

- `save_safetensors`: We default to `False`, as safetensors can't handle tied weights.

## Optimizer Parameter Groups

The LM Training Project template ships with an `[optimizer_groups]` block
that excludes biases, norms, embeddings, and the LM head from weight decay
by default:

```yaml
[optimizer_groups]
optimizer_groups: &optimizer_groups
    no_decay:
        regex: 'norm|bias|embed|lm_head'
        config:
            weight_decay: 0.0
```

This is the standard "no-decay" convention for language-model training -
applying weight decay to parameters whose natural scale is small or
interpretation-dependent (layer-norm gains, biases, embedding rows,
output-projection rows) tends to hurt, while applying it to the main
transformer weight matrices tends to help. Keeping the split explicit in
the template means new projects inherit the convention for free and do
not silently apply decay to those parameters.

### How the mapping is read

Each entry has the form:

```yaml
<group_name>:
    regex: '<pattern>'
    config:
        <override_key>: <override_value>
        ...
```

`config:` may be omitted when a group just carves parameters out without
changing hyperparameters. At optimizer-construction time the trainer
walks the model's named parameters and assigns each one to the **first**
group whose regex matches (insertion order). Parameters that match no
group fall through to an implicit default group with no overrides, so
every parameter is guaranteed to end up in some group.

Regex semantics are Python `re.search` (substring match). More specific
patterns should be declared first - once a parameter is claimed by a
group, later patterns do not see it.

### Overriding the defaults

To change the default mapping, override the `[optimizer_groups]` block
in your child config. Because each group is a named dictionary entry,
you can add, replace, or remove individual groups from the parent via
`== super()`.

Replace one group and add a new one:

```yaml
-- extends "projects/lm_training_project.yaml"

-- block optimizer_groups
optimizer_groups: &optimizer_groups
    == super()
    no_decay:
        regex: 'norm|bias|embed'
        config:
            weight_decay: 0.0
    lm_head:
        regex: 'lm_head'
        config:
            weight_decay: 0.0
            lr: 1.0e-4
-- endblock
```

Remove an inherited group without replacing anything else by setting its
value to `null`:

```yaml
-- block optimizer_groups
optimizer_groups: &optimizer_groups
    == super()
    no_decay: ~     # cancel the default no-decay group entirely
-- endblock
```

To disable parameter grouping entirely (a single uniform optimizer over
all parameters), set the whole mapping to `null`:

```yaml
-- block optimizer_groups
optimizer_groups: &optimizer_groups ~
-- endblock
```

### Verifying the assignment

Set `debug_optimizer_groups: True` (or pass `--debug-optimizer-groups`)
to have the trainer log every parameter -> group assignment when the
optimizer is built. Use this whenever you edit a regex - it's the
fastest way to confirm the override landed on the parameters you
expected.

See [Trainer Options Reference -> Parameter groups via `optimizer_groups`](../trainers/trainer_options.md#parameter-groups-via-optimizer_groups)
for the underlying trainer API. All trainer backends supported by the
LM Training Project honour `optimizer_groups` (basic, DDP, FSDP2, and
pipeline).

All parameters listed above are available as CLI arguments via `forgather train`.
Additional arguments inherited from the base training script:

```
forgather -p examples/base_lm_project train --help
```

The prupose of exposing all of these parameters as CLI arguments is for quickly iterating over options to find the optimal values.
Once you have identified the settings to use, these should be committed to a configuration. Don't create bash CLI scripts using these, which is what
our configuration system is intended to avoid!

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
| `--torch-compile-mode {default,max-autotune,max-autotune-no-cudagraphs}` | `torch_compile_mode` | torch.compile mode |
| `--gradient-checkpointing` / `-G` | `gradient_checkpointing` | Enable activation checkpointing |
| `--activation-offloading BOOL` | `activation_offloading` | Offload saved activations to CPU |
| `--activation-memory-budget X` | `activation_memory_budget` | Functorch activation memory budget (requires `--compile true`) |
| `--fuse-optim-with-backward BOOL` | `fuse_optim_with_backward` | Apply optimizer update in backward hook (basic trainer only) |
| `--construct-model-on {default,meta,device}` | `construct_model_on` | Where to build the model |
| `--gc-threshold X` | `gc_threshold` | GPU allocator cleanup threshold |
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
| `--save-safetensors BOOL` | `save_safetensors` | Save checkpoint using safetensors format |
| `--debug-optimizer-groups` | `debug_optimizer_groups` | Log parameter -> optimizer-group assignments |
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

## References

### Scaling laws and compute-optimal training

- Kaplan, J., McCandlish, S., Henighan, T., Brown, T.B., Chess, B., Child,
  R., Gray, S., Radford, A., Wu, J., Amodei, D. (2020). *Scaling Laws for
  Neural Language Models.* <https://arxiv.org/abs/2001.08361>
- Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T.,
  Rutherford, E., Casas, D., Hendricks, L.A., Welbl, J., Clark, A., et al.
  (2022). *Training Compute-Optimal Large Language Models* (Chinchilla).
  <https://arxiv.org/abs/2203.15556>
- Besiroglu, T., Erdil, E., Barnett, M., You, J. / Epoch AI (2024).
  *Chinchilla Scaling: A Replication Attempt.*
  <https://arxiv.org/abs/2404.10102>

### Critical batch size and LR scaling

- McCandlish, S., Kaplan, J., Amodei, D., OpenAI Dota Team (2018).
  *An Empirical Model of Large-Batch Training.*
  <https://arxiv.org/abs/1812.06162>
- Mayberry, R., et al. (2025). *Critical Batch Size Revisited.*
  <https://arxiv.org/abs/2505.23971>

### Small batch size training and Adam hyperparameters

- Marek, M., Lotfi, S., Somasundaram, A., Wilson, A.G., Goldblum, M.
  (2025). *Small Batch Size Training for Language Models: When Vanilla
  SGD Works, and Why Gradient Accumulation Is Wasteful.*
  <https://arxiv.org/abs/2507.07101>

### Learning-rate schedules for continual pre-training

- Wen, K., Li, Z., Wang, J., Hall, D., Liang, P., Ma, T. (2024).
  *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss
  Landscape Perspective.* (WSD-S protocol.)
  <https://arxiv.org/abs/2410.05192>
- *Beyond Cosine Decay: On the effectiveness of Infinite Learning Rate
  Schedule for Continual Pre-training.* (2025)
  <https://arxiv.org/abs/2503.02844>

### Low-precision training

- *Stochastic Rounding for LLM Training: Theory and Practice.* (2025)
  <https://arxiv.org/abs/2502.20566>

### PyTorch documentation

- *Activation Checkpointing Techniques in PyTorch.*
  <https://pytorch.org/blog/activation-checkpointing-techniques/>
- *DistributedDataParallel.*
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>
- *Pipeline Parallelism.*
  <https://docs.pytorch.org/docs/stable/distributed.pipelining.html>
- *`torch.set_float32_matmul_precision`.*
  <https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html>
