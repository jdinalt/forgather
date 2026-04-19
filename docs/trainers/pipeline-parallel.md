# Pipeline Parallel Training

Pipeline parallelism splits a model's layers across multiple GPUs, allowing you
to train models that are too large to fit on a single device. Each GPU holds a
subset of layers (a "stage") and data flows through them in sequence, with
microbatching to keep all GPUs busy.

Forgather's `PipelineTrainer` builds on PyTorch's
[`torch.distributed.pipelining`](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)
module, adding integration with Forgather's checkpoint system, parameter
initialization, and training loop.

## When to use pipeline parallelism

The primary reason to choose pipeline parallelism is **limited interconnect
bandwidth**. Other parallelism strategies have much higher communication
requirements:

- **FSDP** shards parameters across GPUs and must all-gather them every forward
  pass. This works well with NVLink or InfiniBand but can grind to a near
  standstill on PCIe-only or Ethernet-connected GPUs.
- **Tensor Parallelism** splits individual layers across GPUs, requiring
  communication on every layer's forward and backward pass. This demands the
  highest bandwidth of any strategy and is essentially unusable without NVLink.
- **DDP** replicates the full model and all-reduces gradients after each step.
  This works reasonably well on slower interconnects for models that fit in a
  single GPU's memory, but struggles with larger models where gradient
  communication dominates.

**Pipeline parallelism communicates only at stage boundaries** -- the activations
passed between adjacent stages. With a 32-layer model split across 4 GPUs, there
are only 3 communication points per forward pass, versus 32 for tensor parallelism
or a full parameter gather for FSDP. In practice, PP shows very little performance
loss even over 1 Gigabit Ethernet, making it possible to train large models across
a collection of consumer machines.

This makes PP well-suited for:

- **Consumer GPUs** (RTX 3090, 4090, etc.) that lack NVLink and are often connected
  via PCIe risers that cannot run at full speed
- **Multi-node training** over standard Ethernet, where other strategies are
  bandwidth-starved
- **Models too large for a single GPU** when you do not have high-bandwidth
  interconnects

## Key concepts

### Stages and microbatches

The model is split into *stages*, one per GPU (or multiple per GPU with
multi-stage schedulers). A training batch is split into *microbatches* that flow
through the pipeline in sequence. While one microbatch is in the forward pass on
stage 2, another can be doing the backward pass on stage 1 -- this overlap is
what keeps GPUs utilized.

The `n_microbatches` parameter controls how many microbatches the batch is divided
into. More microbatches means better GPU utilization (less "pipeline bubble") but
more communication overhead.

### Pipeline schedules

The schedule determines the order of forward and backward passes across stages.
Forgather supports all schedules from `torch.distributed.pipelining`:

**Single-stage schedulers** (`stages_per_rank=1`):

| Schedule | Description |
|----------|-------------|
| `ScheduleGPipe` | Simple GPipe: all forwards, then all backwards. Largest bubble but simplest. |
| `Schedule1F1B` | Alternates 1 forward and 1 backward. Smaller bubble than GPipe. |

**Multi-stage schedulers** (`stages_per_rank=2`, `is_multistage=True`):

| Schedule | Description |
|----------|-------------|
| `ScheduleInterleaved1F1B` | Interleaved 1F1B across 2 stages per rank. |
| `ScheduleLoopedBFS` | Looped breadth-first scheduling. |
| `ScheduleInterleavedZeroBubble` | Zero-bubble with interleaving. |
| `ScheduleZBVZeroBubble` | V-pattern zero bubble (requires `pp_stage_type="v"`). |

Multi-stage schedulers assign 2 stages per GPU in either a round-robin ("loop")
or V-shaped ("v") pattern, which can reduce the pipeline bubble further.

#### Choosing a schedule

- `ScheduleGPipe` - Simple reference scheduler. Use it as a failsafe when
  something more sophisticated is misbehaving; not recommended otherwise.
- `Schedule1F1B` - Lowest memory consumption of all the schedulers. Reach
  for this first when pipeline memory pressure is the bottleneck.
- `ScheduleInterleaved1F1B` - Stable, good throughput, broad compatibility.
  **Works with `torch.compile`.** This is the recommended default for most
  runs and is what the `lm_training_project` template selects.
- `ScheduleZBVZeroBubble` - Best raw throughput, but experimental and a bit
  fickle. Its biggest drawback is that it is incompatible with
  `torch.compile`. It also doesn't natively handle flex-attention; Forgather
  ships a monkey-patch that works around this, but treat the combination as
  experimental and expect occasional rough edges.
- `ScheduleLoopedBFS` / `ScheduleInterleavedZeroBubble` - Alternative
  interleaved layouts. Worth trying if you're micro-optimising, but the
  sweet spot for most workloads is `ScheduleInterleaved1F1B` (for
  compile-compatible stability) or `ScheduleZBVZeroBubble` (for peak
  throughput without compile).

### Model splitting

The model must be divided into stages. Forgather uses the
[manual splitting](https://docs.pytorch.org/docs/stable/distributed.pipelining.html#module-torch.distributed.pipelining)
approach described in the PyTorch documentation, based on the implementation in
[TorchTitan](https://github.com/pytorch/torchtitan): the model is deep-copied for
each stage, and the modules that don't belong to that stage are deleted. This requires
that the model is explicitly designed to support it -- its modules must be
independently deletable without breaking the remaining forward pass.

All of Forgather's built-in model architectures (`CasualLM` and derivatives --
Llama, Qwen, DeepOne, etc.) are designed with this in mind. The built-in splitter
handles them:

```python
from forgather.ml.trainer.pipeline import create_manual_causal_lm_splitter

splitter = create_manual_causal_lm_splitter(
    num_layers=None,       # Auto-detected from model
    input_weight=1,        # Relative weight of input encoder stage
    output_weight=1,       # Relative weight of output decoder stage
)
```

The splitter distributes transformer layers as evenly as possible across stages,
with the input encoder on the first stage and the output decoder on the last.
Extra layers go to earlier stages.

For models that don't use the `CasualLM` interface, you need to provide a custom
`ModelSplitter` function. See `src/forgather/ml/trainer/pipeline/model_splitter.py`
for the type signature. The model must be structured so that individual modules
can be removed cleanly -- models with complex cross-layer dependencies may not be
suitable for manual splitting.

## PipelineTrainer

### Constructor

```python
from forgather.ml.trainer.pipeline import (
    PipelineTrainer,
    PipelineTrainingArguments,
    create_manual_causal_lm_splitter,
)
from torch.distributed.pipelining import ScheduleGPipe

trainer = PipelineTrainer(
    args=PipelineTrainingArguments(...),
    model_splitter=create_manual_causal_lm_splitter(),
    pipe_schedule_factory=ScheduleGPipe,
    model_init=model_factory,
    train_dataset=train_dataset,
    optimizer_factory=optimizer_factory,
    # ... other Trainer arguments
)
```

The key difference from the base `Trainer` is two additional required arguments:

- `model_splitter` -- A function that splits the model into pipeline stages.
- `pipe_schedule_factory` -- The pipeline schedule class (not an instance).

### PipelineTrainingArguments

Extends `TrainingArguments` with pipeline-specific parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_microbatches` | `4` | Number of microbatches per batch. Batch size must be evenly divisible by this. |
| `stages_per_rank` | `1` | Pipeline stages per GPU. Set to 2 for multi-stage schedulers. |
| `pp_stage_type` | `"loop"` | Stage assignment pattern: `"loop"` (round-robin) or `"v"` (for `ScheduleZBVZeroBubble`). |
| `is_multistage` | `False` | Must be `True` when `stages_per_rank > 1`. |

For the fields inherited from `TrainingArguments` (compile, AMP, checkpointing,
memory options, etc.), see the
[Trainer Options Reference](trainer_options.md).

### Constraints

- **Batch size** must be evenly divisible by `n_microbatches`.
- **Eval batch size must equal train batch size.** PyTorch's pipeline communication
  buffers are allocated on the first pass for a fixed shape and cannot change after
  that. If the eval batch size differs, the buffer shapes will mismatch and
  training will fail. Set `per_device_eval_batch_size` to the same value as
  `per_device_train_batch_size`.
- **`fp16` is not supported.** The PyTorch GradScaler is incompatible with pipeline
  schedules. Use `bf16` (or `mixed_precision: bf16`) instead.
- **`dataloader_drop_last`** is automatically set to `True` (pipeline requires
  fixed batch shapes).
- **`torch_compile_mode: max-autotune`** is incompatible; use `default` if
  compiling.

### Text generation (experimental)

The pipeline trainer now has distributed text-generation support, which makes
the text-generation callback compatible with pipeline runs for the first
time. This is an experimental feature - if you encounter hangs, shape
mismatches, or other oddities during training, disable the
text-generation callback and file a report.

## How it works

### Model preparation

The `PipelineTrainer` constructs and initializes the model differently from the
base trainer:

1. **Meta-device construction** -- The model is first constructed on the meta
   device (no memory allocated). This avoids OOM for models larger than a single
   GPU's memory.
2. **Stage splitting** -- The model splitter divides the model into stages and
   creates `PipelineStage` objects.
3. **Parameter materialization** -- Each rank materializes only its assigned stages
   onto its GPU.
4. **Centralized initialization** -- When not resuming from a checkpoint, rank 0
   constructs a fully initialized model on CPU and distributes each rank's
   parameters via point-to-point NCCL transfers. This avoids having N full copies
   of the model in memory. When resuming, each rank loads its own stage parameters
   directly from the checkpoint.

### Forward and backward passes

The pipeline scheduler orchestrates forward and backward passes across stages:

- The first stage receives `input_ids` and produces activations.
- Middle stages receive activations, transform them, and pass them along.
- The last stage computes the loss and initiates backpropagation.

**Attention masks** are created externally (not passed through the pipeline),
because PyTorch's pipeline transport only handles gradient-requiring tensors.
Non-gradient tensors like attention masks are computed independently on each
stage that needs them.

### Loss scaling

Loss is automatically scaled by `1 / (n_microbatches * gradient_accumulation_steps)`
so that the effective loss is the mean over the full batch regardless of how many
microbatches it is split into.

### Checkpointing

Each rank saves only its assigned pipeline stages. The checkpoint system uses a
shard index to coordinate distributed saves and correctly reconstruct the full
model on load. Parameter sharing relationships (e.g., tied embeddings) are tracked
and preserved across stages.

## Configuration template

Forgather provides a pipeline trainer template at
`templatelib/base/trainers/pipeline_trainer.yaml`. Switch a configuration to
pipeline parallel by including it:

```yaml
[trainer_definition]
    -- include 'trainers/pipeline_trainer.yaml'
```

The template sets up the `PipelineTrainer` with `create_manual_causal_lm_splitter`
and defaults to `ScheduleGPipe`. Override the schedule via `ns.pipe_schedule_factory`.

For a working example with dynamic schedule selection and microbatch configuration,
see `examples/pretrain/small-llm/templates/configs/pp.yaml`.

## Launching

Pipeline parallel training uses `torchrun` (via `forgather train`) with one
process per GPU:

```bash
# 2 GPUs
forgather -t pp.yaml train -d 0,1

# 4 GPUs
forgather -t pp.yaml train -d 0,1,2,3
```
