# Trainer Options Reference

This document is a consolidated reference for all training-argument fields and
constructor parameters across Forgather's built-in trainers. It complements the
docstrings and per-field comments that live next to the dataclasses themselves
(see the [Source of Truth](#source-of-truth) section at the bottom).

Trainers covered:

- `forgather.ml.trainer.Trainer` - lightweight single-device trainer
- `forgather.ml.trainer.accelerate.AccelTrainer` - multi-GPU via HF Accelerate
- `forgather.ml.trainer.ddp.DDPTrainer` - multi-GPU via raw PyTorch DDP
- `forgather.ml.trainer.fsdp2.FSDP2Trainer` - sharded DP via `torch.distributed.fsdp.fully_shard`
- `forgather.ml.trainer.pipeline.PipelineTrainer` - pipeline parallel

The `torchtitan`-based trainer is out of scope - see the torchtitan docs.

## Training-Argument Hierarchy

Training arguments are defined as a chain of dataclasses, each subclass adding
fields on top of the previous:

```
MinimalTrainingArguments    (HF-compatible baseline)
   v
BaseTrainingArguments       (adds checkpoint, AMP, FP8, SDPA, anomaly, ...)
   v
TrainingArguments           (adds memory/compile options for simple Trainer)
   v
 +-- AccelTrainingArguments     (no new fields; trainer overrides behaviour)
 +-- DDPTrainingArguments       (adds dispatch_batches, DDP, Post-Local-SGD)
 +-- FSDP2TrainingArguments     (adds dispatch_batches, fsdp2.* policies)
 +-- PipelineTrainingArguments  (adds pipeline-schedule/microbatch fields)
```

When you configure a trainer via YAML, every field from every class in the
chain above is available. Defaults propagate down and may be overridden at any
level.

---

## MinimalTrainingArguments

HuggingFace-compatible baseline. Defined in
[`trainer_types.py`](../src/forgather/ml/trainer/trainer_types.py).

### Output and execution

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | str | `tmp_trainer` | Directory for model output and checkpoints. |
| `logging_dir` | str \| None | auto | TensorBoard log dir. Defaults to `<output_dir>/runs/<timestamp>_<host>`. |
| `device` | Any | None | Device override (cuda, cpu, etc.). Auto-detected if None. |
| `seed` | int | -1 | RNG seed. `-1` disables seeding. |
| `use_cpu` | bool | False | Force CPU even if CUDA is available. |

### Steps and epochs

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_train_epochs` | int | 1 | Total epochs to train. May be fractional. A negative value disables the epoch cap so training runs until `max_steps` (or an external stop signal) -- useful with schedulers like WSD / InfiniteLR where the budget is expressed purely in steps. |
| `max_steps` | int | -1 | If > 0, overrides `num_train_epochs` with an absolute step count. When `num_train_epochs < 0`, `max_steps` is the only training-length cap, so it must be > 0. |
| `epoch_train_steps` | int | 100000 | Fallback epoch length for datasets that don't implement `len()`. Forgather extension. |

### Batching and data loading

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `per_device_train_batch_size` | int | 16 | Training batch size per device. Global batch = per-device * num-devices * `gradient_accumulation_steps`. |
| `per_device_eval_batch_size` | int | 16 | Eval batch size per device. |
| `gradient_accumulation_steps` | int | 1 | Accumulate gradients over N micro-batches before stepping the optimizer. |
| `max_grad_norm` | float \| None | None | Clip gradients to this L2 norm. `None` disables clipping. |
| `dataloader_num_workers` | int | 0 | Worker processes for data loading. `0` = main process. |
| `dataloader_pin_memory` | bool | True | Pin memory in DataLoader for faster host-to-device copies. |
| `dataloader_persistent_workers` | bool | False | Keep worker processes alive between epochs. |
| `dataloader_prefetch_factor` | int \| None | auto | Batches prefetched per worker. Defaults to 2 when `num_workers > 0`. |
| `dataloader_drop_last` | bool | False | Drop the last incomplete batch. |

### Eval / logging cadence

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `eval_strategy` | str | `"no"` | When to run eval: `"no"`, `"steps"`, or `"epoch"`. |
| `eval_steps` | int | 100 | Eval frequency in steps (when `eval_strategy="steps"`). |
| `eval_delay` | int | 0 | Epochs/steps to wait before first eval. |
| `logging_strategy` | str | `"steps"` | `"no"`, `"steps"`, or `"epoch"`. |
| `logging_steps` | int | 50 | Log cadence in steps. |
| `logging_first_step` | bool | False | Whether to log the very first global step. |

### torch.compile

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `torch_compile` | bool | False | Enable `torch.compile()` on the model. |
| `torch_compile_backend` | str \| None | `"inductor"` | Backend passed to `torch.compile(backend=...)`. |
| `torch_compile_mode` | str \| None | `"default"` | `"default"`, `"reduce-overhead"`, `"max-autotune"`, or `"max-autotune-no-cudagraphs"`. |
| `torch_compile_dynamic` | bool | True | Allow dynamic input shapes. |
| `torch_compile_full_graph` | bool | False | Force full-graph compilation (no graph breaks). |

Notes:

- `max-autotune` is incompatible with `gradient_accumulation_steps > 1` and with
  pipeline parallel training. Use `max-autotune-no-cudagraphs` in those cases.
- Zero-bubble pipeline schedules (`ScheduleInterleavedZeroBubble`,
  `ScheduleZBVZeroBubble`) are incompatible with `torch_compile` regardless of
  mode, because AOTAutograd flattens the compiled stage interior and the split
  backward cannot walk the resulting graph.

### Checkpointing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `save_strategy` | str | `"steps"` | `"no"`, `"steps"`, or `"epoch"`. |
| `save_steps` | int | 1000 | Save cadence (when `save_strategy="steps"`). |
| `save_total_limit` | int | 2 | Maximum checkpoints to keep. Best-marked checkpoints are preserved beyond this. |
| `save_safetensors` | bool | True | Use safetensors format for weights. |
| `save_on_each_node` | bool | False | In multi-node runs, save on each node. Don't use with shared storage. |
| `overwrite_output_dir` | bool | False | Overwrite existing contents in `output_dir`. |
| `resume_from_checkpoint` | bool \| str | True | `True` = auto-resume from latest; `False` = force fresh; path = resume from specific. |
| `load_best_model_at_end` | bool | False | Load best checkpoint at end of training. Requires `save_strategy == eval_strategy`. |
| `metric_for_best_model` | str | `"loss"` | Metric used to compare checkpoints. |
| `greater_is_better` | bool \| None | auto | Override the metric direction. Inferred from metric name if None. |

See [`docs/checkpointing/user_guide.md`](../checkpointing/user_guide.md) for a
practical guide to checkpointing.

Notes:

- `save_safetensors` is incompatible with saving tied weights. The format was created to address a real security issue with
PyTorch's native format, which allowed arbitrary code execution. PyTorch has since addressed the security and performance issues.

### Optimizer and LR scheduler (HF compat)

These fields exist for HuggingFace compatibility. If you pass an explicit
`optimizer_factory` / `lr_scheduler_factory` to the trainer constructor, most
of these are ignored.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `learning_rate` | float | 5e-5 | Initial LR for the default AdamW optimizer. |
| `weight_decay` | float | 0.0 | Weight decay for the default AdamW optimizer. |
| `adam_beta1` | float | 0.9 | AdamW beta1. |
| `adam_beta2` | float | 0.999 | AdamW beta2. |
| `adam_epsilon` | float | 1.0e-8 | AdamW epsilon. |
| `lr_scheduler_type` | str | `"linear"` | `"linear"`, `"cosine"`, `"polynomial"`, etc. |
| `lr_scheduler_kwargs` | dict \| None | `{}` | Extra kwargs passed to the scheduler. |
| `warmup_steps` | int | 0 | Linear warmup from 0 to `learning_rate` over this many steps. |

### Gradient checkpointing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gradient_checkpointing` | bool | False | Enable activation checkpointing on models that support the HF API. Trades compute for memory. |

Customise the enable function by passing `enable_activation_checkpoint_fn` to
the `Trainer` constructor (see below).

---

## BaseTrainingArguments

Adds checkpoint-preservation, AMP/FP8, SDPA backend selection, and runtime
PyTorch tweaks. Defined in
[`base_trainer.py`](../src/forgather/ml/trainer/base_trainer.py).

### Default dtype and eval

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_dtype` | str \| None | None | Default dtype used during model construction (`"float32"`, `"bfloat16"`, `"float16"`). |
| `max_eval_steps` | int | -1 | Maximum eval batches per eval run. `-1` = unlimited. |

### Checkpoint preservation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `preserve_best_model` | bool | False | Keep the best-metric checkpoint safe from cleanup. |
| `best_model_metric` | str | `"loss"` | Metric used for best-model tracking. |
| `best_model_greater_is_better` | bool \| None | auto | Override direction; inferred from metric name if None. |
| `preserve_n_best` | int | 1 | Number of best checkpoints to preserve. |
| `eval_on_save` | bool | False | Run eval immediately before each save (decouples save/eval schedules). |

### Memory and debugging

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_activation_offloading` | bool | False | Offload saved activations to CPU during backward (via `torch.autograd.graph.save_on_cpu`). Best combined with activation checkpointing. |
| `detect_anomaly` | bool | False | Enable `torch.autograd.set_detect_anomaly(True)` to help track down NaNs. Adds overhead - debug only. |

Notes:
 - `enable_activation_offloading` appears to be incompatible with flex-attention.

### Scaled Dot-Product Attention

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sdpa_backend` | str \| list[str] \| None | None | Force SDPA backend: `"math"`, `"flash"`, `"efficient"`, or `"cudnn"` (or a list). |
| `sdpa_set_priority` | bool | False | When `sdpa_backend` is a list, interpret it as a priority order. |

### Matmul precision and dynamo

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `float32_matmul_precision` | str \| None | None | `"highest"`, `"high"` (TF32), or `"medium"`. Passed to `torch.set_float32_matmul_precision`. |
| `dynamo_recompile_limit` | int \| None | None | Override `torch._dynamo.config.recompile_limit`. Raise if you see frequent recompiles with `torch.compile`. |

On any GPU that supports TF32 (Ampere or newer), you usually want
`float32_matmul_precision = "high"`. Leaving it at the PyTorch default
(`"highest"`) disables TF32 and triggers a runtime warning from
`torch.set_float32_matmul_precision` on modern hardware.

### Mixed precision and FP8

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mixed_precision` | str \| None | None | `None` / `"no"` disabled, `"bf16"` (no GradScaler), or `"fp16"` (with GradScaler). |
| `fp8_recipe` | str \| None | None | `"tensorwise"`, `"rowwise"`, or `"rowwise_with_gw_hp"`. Converts `nn.Linear` to `Float8Linear` via torchao. Orthogonal to `mixed_precision`. Mutually exclusive with `qat_recipe`. |
| `fp8_dim_alignment` | int | 16 | Minimum alignment for FP8 Linear layer dimensions; non-conforming layers are skipped. |
| `qat_recipe` | str \| None | None | `"int8-dynamic-act-int4-weight"`, `"int4-weight-only"`, or `"float8-dynamic-act-float8-weight"`. Installs `FakeQuantizedLinear` via torchao QAT (prepare phase). Run `forgather finalize --quantize <recipe>` after training to produce the deployable low-bit artifact. Mutually exclusive with `fp8_recipe`. |

FP8 requires CUDA SM >= 8.9 (RTX 4090, H100, etc.). See
[`fp8-training.md`](fp8-training.md). QAT has no hardware gate (runs on any
CUDA GPU or CPU); see [`qat-training.md`](qat-training.md).

---

## TrainingArguments (simple `Trainer`)

Adds memory-optimisation options specific to the simple single-device trainer.
Defined in [`trainer.py`](../src/forgather/ml/trainer/trainer.py).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gc_threshold` | float | 0.5 | Ratio of allocated-to-total GPU memory that triggers `empty_cache()` (and possibly `gc.collect()`). See the note below - there is a real tuning trade-off. |
| `construct_model_on` | str | `"default"` | Where to build the model. `"default"` builds on the target device (the model's `__init__` initialises weights); when resuming from a checkpoint with a `model_init`, it transparently uses the meta path. `"meta"` builds on the meta device (no allocation), materialises empty on the target device, then loads/initialises — fastest, and the standard route for checkpoint loading. `"device"` forces on-device construction wrapping a resume load in `no_init_weights()` (fallback for models that can't build on meta). See the notes below. |
| `checkpoint_components` | list[str] \| None | None | Which checkpoint state components to **save and load**. `None` = every component the trainer produces (default). A list restricts to a subset of `model`, `optimizer`, `scheduler`, `trainer`, `dataset`, `rng`. Excluding `"model"` is the weights-external signal: weights are never checkpointed and the model is built empty on meta (see the note below). A key outside that vocabulary raises. |
| `activation_memory_budget` | float \| None | None | Sets `torch._functorch.config.activation_memory_budget`. Requires `torch_compile=True`. See [PyTorch activation checkpointing techniques](https://pytorch.org/blog/activation-checkpointing-techniques/). |
| `fuse_optim_with_backward` | bool | False | Apply optimizer step from the backward gradient hook so each parameter's gradient is consumed and freed immediately after it's produced. Biggest savings when combined with activation checkpointing. Supported by the basic `Trainer` and `PipelineTrainer`, **not** by `DDPTrainer` or `AccelTrainer` (they need gradients intact for the all-reduce). Also incompatible with `max_grad_norm` gradient clipping (clipping needs the full gradient vector before any update is applied), `gradient_accumulation_steps > 1`, and fp16 mixed precision. |
| `speed_metrics_start_step` | int | 1 | Step at which to start collecting speed metrics. `1` excludes the first step (to skip `torch.compile` warmup). Set higher for longer compile warmups. |
| `set_dataset_epoch` | bool | True | If the train dataset has `set_epoch(epoch: int)`, call it at the start of each epoch. |
| `debug_optimizer_groups` | bool | False | Log the parameter-name -> optimizer-group assignments produced from the `optimizer_groups` constructor argument when the optimizer is built. Useful to verify that an `optimizer_groups` mapping picks up the parameters you expect. |

Notes on `construct_model_on = "meta"`:

- Meta construction produces a model with uninitialised (empty) tensors,
  filled either by a checkpoint load right after, or — when weights are
  supplied externally (`checkpoint_components` excludes `"model"`; see
  below) — by an initialise pass plus that external source. If you set
  `"meta"` with no checkpoint and weights are not external, the basic
  trainer falls back to constructing on the device so you never end up
  training against uninitialised weights.
- Checkpoint loading follows a meta-init contract: construct on meta,
  materialise empty, load the checkpoint, then initialise only the tensors
  the checkpoint didn't fill (e.g. non-persistent RoPE buffers).
- `PipelineTrainer` **always** builds the model on meta - the field is
  effectively ignored for that trainer. For its from-scratch path, rank-0
  builds the full model on CPU just long enough to initialise weights, then
  dispatches the initialised parameters/buffers to each rank's empty stage.
  With a checkpoint, the meta stages are filled directly from the checkpoint
  with no CPU materialisation step.

Notes on `checkpoint_components`:

- `None` (the default) saves and loads every component the trainer
  produces. A list selects a subset; a key outside
  `model`/`optimizer`/`scheduler`/`trainer`/`dataset`/`rng` raises (a typo
  can't silently drop a component). A listed key the run doesn't produce
  (e.g. `dataset` with a non-stateful loader) is ignored.
- Excluding `"model"` is the **weights-external** signal: the
  CheckpointManager skips model-weight save/load (marking the checkpoint so
  it stays resumable), and the trainer builds the model empty on the meta
  device, initialises the skeleton, and expects the weights from elsewhere.
  This is how DiLoCo workers run — the parameter server owns the weights —
  so the DiLoCo template sets
  `checkpoint_components: [optimizer, scheduler, trainer, rng]`. It also omits
  `dataset`, whose position the DiLoCo server tracks via work-units (a local
  dataloader checkpoint would resume from a stale position).

`gc_threshold` is a real trade-off, not just an "OOM knob":

- Too low: `empty_cache()` / `gc.collect()` fire frequently. Each call
  synchronises the CUDA stream and can noticeably slow training.
- Too high: the caching allocator holds onto fragmented blocks until you
  actually OOM.

Start with the default and only lower it if you see OOM from fragmentation
(reserved >> allocated). If you lower it and throughput drops, you've gone
too far - raise it back toward the OOM edge.

Not supported by `AccelTrainer` or `DDPTrainer`:

- `fuse_optim_with_backward` (both trainers assert against it - they need
  the complete gradient vector after backward for their all-reduce /
  gradient synchronisation step, and the fused hook would consume and free
  each gradient before that could happen). `PipelineTrainer` does support
  `fuse_optim_with_backward`.

Not supported by `PipelineTrainer`:

- `mixed_precision="fp16"` (use `"bf16"`)
- `torch_compile=True` combined with zero-bubble schedules

---

## DDPTrainingArguments

Adds DDP-specific options. Defined in
[`ddp/ddp_trainer.py`](../src/forgather/ml/trainer/ddp/ddp_trainer.py).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dispatch_batches` | bool | True | When True, rank-0 loads and dispatches every batch. When False, each rank reads its own shard and a `SynchronizedDataLoader` agrees on when to stop. See the note below - False is usually the right default for performance. |
| `ddp` | DDPArguments | default | Nested DDP wrapper options (see below). |
| `post_local_sgd` | PostLocalSGDArguments | default | Nested Post-Local-SGD options (see below). |

When to set `dispatch_batches = True`:

- **Primary:** the dataset implementation doesn't support sharding. Rank-0
  iterating it centrally is the only way to feed every rank.
- **Secondary:** you want the DDP run to see the *exact same* sequence of
  training examples as a single-GPU or pipeline run, so training curves are
  directly comparable. Sharding gives each rank a different subset, so curves
  from sharded vs. non-sharded runs are not directly comparable.

Otherwise prefer `dispatch_batches = False`. Centralised loading adds
per-step dispatch overhead and serialises data loading on rank-0; sharded
loading parallelises it across ranks.

### `DDPArguments` (nested as `ddp.*`)

Passed directly to `torch.nn.parallel.DistributedDataParallel`. See
[PyTorch DDP docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `broadcast_buffers` | bool | True | Broadcast module buffers at each forward. |
| `init_sync` | bool | True | Synchronise parameters/buffers during init. |
| `bucket_cap_mb` | int \| None | None | Bucket size in MB for grad allreduce. |
| `find_unused_parameters` | bool | False | Enable to allow unused parameters at the cost of an extra traversal. |
| `gradient_as_bucket_view` | bool | True | Memory-saving grad layout optimization. |
| `static_graph` | bool | False | Enable DDP static-graph optimization. |
| `skip_all_reduce_unused_params` | bool | False | Skip the allreduce for unused-parameter handling. |

### `PostLocalSGDArguments` (nested as `post_local_sgd.*`)

Post-Local-SGD wraps the optimizer so that each rank takes local steps and
periodically averages parameters instead of performing all-reduce every step.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | False | Enable Post-Local-SGD. |
| `start_step` | int | 500 | Step at which local SGD kicks in (before this, standard all-reduce is used). |
| `period` | int | 4 | Average every `period` optimizer steps. |
| `post_local_gradient_allreduce` | bool | False | Whether to still all-reduce gradients after the switch-over. |

---

## FSDP2TrainingArguments

Adds FSDP2 (`torch.distributed.fsdp.fully_shard`) options. Defined in
[`fsdp2/fsdp2_trainer.py`](../src/forgather/ml/trainer/fsdp2/fsdp2_trainer.py).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dispatch_batches` | bool | True | Same semantics as `DDPTrainingArguments.dispatch_batches`. True = rank-0 loads and dispatches every batch; False = each rank reads its own shard via `SynchronizedDataLoader`. |
| `fsdp2` | FSDP2Arguments | default | Nested FSDP2 policy options (see below). |

`fuse_optim_with_backward` is not supported - FSDP2 needs gradients intact
for the reduce-scatter in its backward hook.

### `FSDP2Arguments` (nested as `fsdp2.*`)

Passed to `fully_shard()`. See the
[PyTorch FSDP2 docs](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reshard_after_forward` | bool \| int | True | True behaves like ZeRO-3 (reshard parameters after forward, minimum memory, more communication). False behaves like ZeRO-2 (keep params unsharded between forward and backward, lower comm, higher memory). An int N enables ZeRO++ hybrid sharding across N ranks. |
| `param_dtype` | str \| None | None | FSDP `MixedPrecisionPolicy` parameter dtype (e.g., `"bfloat16"`). None disables FSDP-level mixed precision; the trainer's existing AMP autocast still applies. |
| `reduce_dtype` | str \| None | None | Gradient reduce-scatter accumulation dtype. Typically `"float32"` when `param_dtype="bfloat16"` for numerical stability. |
| `buffer_dtype` | str \| None | None | Non-parameter buffer dtype. |
| `cpu_offload` | bool | False | Enable `CPUOffloadPolicy` - offload parameters (and gradients) to CPU between uses. Dramatically reduces GPU memory at the cost of PCIe transfer time. |
| `shard_transformer_layers` | bool | True | Apply `fully_shard` layer-by-layer on transformer blocks before the root module. Required for meaningful memory savings - root-only wrapping can't reshard per-layer during forward/backward. |
| `transformer_layers_path` | str | `"causal_lm.layer_stack.layers"` | Dotted attribute path that resolves to the iterable of transformer blocks. The default matches Forgather's standard causal-LM structure (see `modelsrc/transformer/`). Override for models with a different block-list path; set to an empty/unresolvable path to fall back to root-only sharding. |

Checkpointing: model and optimizer state are saved as **per-rank DTensor
shards** (`SharingPattern.PER_RANK`) via
`torch.distributed.checkpoint.state_dict.get_model_state_dict` /
`get_optimizer_state_dict` with default (sharded) `StateDictOptions`.
The model is saved under key `fsdp2_model` (not `model`) so it routes
through the `CheckpointCoordinator`'s PER_RANK path rather than the
safetensors path, which cannot handle DTensors. Checkpoints are tied
to the world size they were saved at - resuming at a different world
size is not supported without going through
`torch.distributed.checkpoint` (DCP).

When `world_size == 1`, the trainer transparently degrades to the single-device
path without calling `fully_shard`.

---

## PipelineTrainingArguments

Adds pipeline-parallel options. Defined in
[`pipeline/pipeline_trainer.py`](../src/forgather/ml/trainer/pipeline/pipeline_trainer.py).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_microbatches` | int | 4 | Micro-batches each full batch is split into. Must evenly divide `per_device_train_batch_size`. Higher = better pipeline utilisation, but more memory. |
| `stages_per_rank` | int | 1 | Pipeline stages per GPU. Only set > 1 with `is_multistage=True` schedules (e.g., `ScheduleZBVZeroBubble`). |
| `pp_stage_type` | str | `"loop"` | `"loop"` for round-robin stage assignment, `"v"` for V-layout used by `ScheduleZBVZeroBubble`. |
| `is_multistage` | bool | False | Set True when using a multi-stage scheduler (inherits from `PipelineScheduleMulti`). |
| `debug_pipeline` | bool | False | Internal: enable DEBUG-level logging. |
| `debug_split_model` | bool | False | Internal: dump model-split details. |
| `debug_model_params` | bool | False | Internal: dump per-parameter placement/dtype. |
| `debug_model_init` | bool | False | Internal: dump init sequence. |

See [`pipeline-parallel.md`](pipeline-parallel.md) for a full walkthrough.

---

## Constructor Arguments

The dataclass fields above define the *training configuration*. Each trainer
class also takes a handful of *non-primitive* constructor arguments for the
model, optimizer, callbacks, etc. These are passed directly when you
instantiate the trainer (or are wired up by the YAML templates).

### `BaseTrainer` (abstract)

All concrete trainers inherit these keyword arguments:

| Argument | Type | Description |
|----------|------|-------------|
| `args` | `BaseTrainingArguments` \| dict | Required. Training configuration. Dicts are converted via `dacite`. |
| `model` | `nn.Module` \| None | Model instance. Either `model` or `model_init` must be set. |
| `model_init` | `Callable[[], nn.Module]` \| None | Zero-arg factory called to construct the model. Required for trainers that build on meta device (e.g., `PipelineTrainer`). |
| `data_collator` | callable \| None | Batches a list of dataset items into tensors. Defaults to `torch.utils.data.default_collate` in the simple `Trainer`. |
| `train_dataset` | dataset \| None | Training dataset or pre-built dataloader. |
| `eval_dataset` | dataset \| None | Evaluation dataset or pre-built dataloader. |
| `processing_class` | callable \| None | Tokenizer or feature extractor (saved alongside the model for HF compat). |
| `callbacks` | list[TrainerCallback] \| None | Callbacks to attach. Defaults to the class's `default_callbacks()`. |
| `compute_loss_func` | callable \| None | Optional fused or custom loss function. Required when using `gradient_accumulation_steps > 1`. |

### `Trainer` (simple, single-device)

Adds:

| Argument | Type | Description |
|----------|------|-------------|
| `distributed_env` | `DistributedEnvInterface` | Required. Provides rank / world-size / device info. Single-device subclasses assert `world_size == 1`. |
| `optimizer_factory` | `Callable[[params], Optimizer]` | Builds the optimizer from model parameters. If omitted and `optimizer_cls_and_kwargs` is also None, a default AdamW is built from `args`. |
| `optimizer_cls_and_kwargs` | `tuple[type, dict]` | HF-Trainer-compatible alternative to `optimizer_factory`. |
| `optimizer_groups` | `Mapping[str, Mapping \| None]` \| None | Optional regex-to-group mapping used to assign parameters to named optimizer param groups with per-group hyperparameter overrides. See [Parameter groups via `optimizer_groups`](#parameter-groups-via-optimizer_groups) below. |
| `lr_scheduler_factory` | `Callable[[Optimizer], LRScheduler]` | Builds the LR scheduler from the optimizer. |
| `enable_activation_checkpoint_fn` | `Callable[[int, nn.Module], None]` | Called to enable activation checkpointing on the model. Defaults to `enable_hf_activation_checkpointing`. |
| `fused_loss_factory` | `Callable[[nn.Module], LossFunction]` | Builds a loss function that shares state with the model (e.g., fused cross entropy). |

#### Parameter groups via `optimizer_groups`

`optimizer_groups` lets you split a model's parameters into named groups,
each matched by a regex against the fully-qualified parameter name, and
each contributing its own hyperparameter overrides to the optimizer. The
mapping has the form:

```python
optimizer_groups = {
    "<group_name>": {
        "regex": r"<pattern>",
        "config": {"<override_key>": <override_value>, ...},
    },
    # ...
}
```

`config` may be omitted (or set to `None`) when a group just carves
parameters out without changing hyperparameters. Setting an entry's
value to `None` removes the group entirely - useful for cancelling an
inherited group from a child template without replacing the whole
mapping.

Each parameter is assigned to the **first** group whose regex matches
(dict insertion order), so more-specific patterns should come first.
Parameters that match no group fall through to an implicit default group
with no overrides - every parameter is guaranteed to end up somewhere.
Empty groups are dropped from the result.

The canonical use case is excluding biases, norms, and embeddings from
weight decay:

```yaml
[optimizer_groups]
optimizer_groups: &optimizer_groups
    no_decay:
        regex: 'norm|bias|embed|lm_head'
        config:
            weight_decay: 0.0
```

Named entries are deliberate: they let a child template override an
individual group by re-declaring its name (and replacing the entire
spec), rather than having to re-specify every group in the mapping.
Set `debug_optimizer_groups: True` in the trainer args to log each
parameter -> group assignment when the optimizer is built - useful for
verifying that the regex picks up the parameters you expected.

Supported by `Trainer`, `DDPTrainer`, `AccelTrainer`, `FSDP2Trainer`,
and `PipelineTrainer`.

### `AccelTrainer`

Extends `Trainer`, adds:

| Argument | Type | Description |
|----------|------|-------------|
| `accelerator` | `accelerate.Accelerator` | Required. Pre-configured Accelerate object. AMP/FP8 and mixed-precision should be configured on the Accelerator, not via `args.mixed_precision` (which is ignored with a warning). `fuse_optim_with_backward` is not supported (Accelerate needs gradients intact for its synchronisation step). |

`args.gradient_accumulation_steps` is reconciled with
`accelerator.gradient_accumulation_steps` at init time (Accelerator wins, with
a warning).

### `DDPTrainer`

Extends `Trainer`, adds:

| Argument | Type | Description |
|----------|------|-------------|
| `fused_loss_factory` | `Callable[[nn.Module], LossFunction]` | Same as `Trainer`, but passed through explicitly. |

No new required arguments; DDP wrapping is done internally using the
`DDPArguments` in `args.ddp`. `fuse_optim_with_backward` is not supported:
DDP needs the complete gradient vector after backward for the all-reduce,
and the fused hook would consume and free each gradient before that could
happen.

When `world_size == 1`, the trainer transparently degrades to the single-device
path without wrapping in DDP.

### `FSDP2Trainer`

Extends `Trainer`. Same keyword arguments as `Trainer`; no FSDP2-specific
constructor parameters. `fully_shard` wrapping happens inside
`_prepare_model()`, before the optimizer is built, so the optimizer is
constructed from the already-sharded DTensor parameters.

When `world_size == 1`, the trainer transparently degrades to the single-device
path without calling `fully_shard`.

### `PipelineTrainer`

Extends `Trainer`, adds:

| Argument | Type | Description |
|----------|------|-------------|
| `model_splitter` | `ModelSplitter` | Required. Function that splits the model (built on meta device) into stage modules and `PipelineStage` objects. See [`model_splitter.py`](../src/forgather/ml/trainer/pipeline/model_splitter.py). |
| `pipe_schedule_factory` | scheduler class | Scheduler class from `torch.distributed.pipelining`, e.g., `ScheduleGPipe`, `Schedule1F1B`, `ScheduleInterleaved1F1B`, `ScheduleInterleavedZeroBubble`, `ScheduleZBVZeroBubble`. Default: `ScheduleGPipe`. |

Notes and constraints:

- `model` must be `None` - pipeline must construct via `model_init`.
- `per_device_train_batch_size` and `per_device_eval_batch_size` must be evenly
  divisible by `n_microbatches`.
- `dataloader_drop_last` is forced to `True` (pipeline needs fixed shapes).
- `mixed_precision="fp16"` is rejected; use `"bf16"`.
- `torch_compile=True` with zero-bubble schedulers is rejected at init time.
- `world_size` must be > 1.

---

## Source of Truth

The definitive documentation lives next to the code:

- `src/forgather/ml/trainer/trainer_types.py` - `MinimalTrainingArguments`
- `src/forgather/ml/trainer/base_trainer.py` - `BaseTrainingArguments`, `BaseTrainer`
- `src/forgather/ml/trainer/trainer.py` - `TrainingArguments`, `Trainer`
- `src/forgather/ml/trainer/accelerate/accel_trainer.py` - `AccelTrainingArguments`, `AccelTrainer`
- `src/forgather/ml/trainer/ddp/ddp_trainer.py` - `DDPTrainingArguments`, `DDPTrainer`, `DDPArguments`, `PostLocalSGDArguments`
- `src/forgather/ml/trainer/fsdp2/fsdp2_trainer.py` - `FSDP2TrainingArguments`, `FSDP2Trainer`, `FSDP2Arguments`
- `src/forgather/ml/trainer/pipeline/pipeline_trainer.py` - `PipelineTrainingArguments`, `PipelineTrainer`

When updating an option, update the dataclass comment first and then
synchronise the table above.
