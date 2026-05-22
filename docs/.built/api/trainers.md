# Trainers

Forgather provides a hierarchy of trainer classes for single-GPU through multi-node distributed training.

| Trainer | Use case |
|---------|----------|
| `Trainer` | Single-GPU, the fast path for small-model experiments |
| `DDPTrainer` | Multi-GPU DistributedDataParallel, with optional PostLocalSGD |
| `FSDP2Trainer` | FSDP-2 sharded data parallel with CPU offload support |
| `PipelineTrainer` | Pipeline parallelism for bandwidth-limited environments |

For a complete reference of every training argument and constructor parameter across all trainers, see also:

- [Trainer Options Reference](../trainers/trainer_options.md)
- [DiLoCo Architecture & Maintainer Guide](../trainers/diloco-architecture.md)

---

## Core Types

Shared types and protocols used across all trainers.

### `DistributedEnvironment` {#forgather-ml-distributed-distributedenvironment}

`forgather.ml.distributed.DistributedEnvironment`

```python
class DistributedEnvironment(rank: int = 0, local_rank: int = 0, world_size: int = 1, local_world_size: int = 1, master_addr: str = 'localhost', master_port: int = 29501, backend: str | None = None, log_level = 'INFO', device_map = None, always: bool = True, no_accelerator: bool = False)
```

Initialize and manage the PyTorch distributed training environment.

This class handles the complete setup of distributed training, including:
- Synchronizing with environment variables set by launchers (torchrun, etc.)
- Setting up the appropriate device (GPU or CPU)
- Initializing the torch.distributed process group

The distributed environment must be initialized before any torch.distributed
calls can be made. In forgather configurations, this is typically included
as an early dependency to ensure proper initialization order.

Environment Variable Behavior:
    - If environment variables are set (e.g., by torchrun), they override
      the values passed to __init__
    - If environment variables are not set, this class exports the __init__
      values to the environment for consistency

Device Selection:
    - With GPU available and no_accelerator=False: Uses GPU with nccl backend
    - With no_accelerator=True or no GPU: Uses CPU with gloo backend
    - Device is automatically assigned based on local_rank (or device_map)

Attributes:
    rank: Global rank of this process
    local_rank: Rank within the local node
    world_size: Total number of processes
    local_world_size: Number of processes on this node
    master_addr: Address of rank 0 for rendezvous
    master_port: Port for rendezvous
    backend: Distributed backend ("nccl", "gloo", etc.)
    device: Device string for this rank (e.g., "cuda:0", "cpu")
    device_type: Device type string (e.g., "cuda", "cpu")
    use_accelerator: Whether to use GPU acceleration

Example:
    In a forgather YAML configuration::

        distributed_env: &distributed_env !singleton:forgather.ml.distributed:DistributedEnvironment
            backend: "nccl"

    For CPU-only testing::

        distributed_env: &distributed_env !singleton:forgather.ml.distributed:DistributedEnvironment
            no_accelerator: True

**Attributes**

- `rank`
- `local_rank`
- `world_size`
- `local_world_size`
- `master_addr`
- `master_port`
- `backend`
- `always`
- `device_map`
- `use_accelerator`

---

### `MinimalTrainingArguments` {#forgather-ml-trainer-trainer_types-minimaltrainingarguments}

`forgather.ml.trainer.trainer_types.MinimalTrainingArguments`

```python
class MinimalTrainingArguments(*, output_dir: str = OUTPUTDIR_NAME, logging_dir: str | None = None, per_device_eval_batch_size: int = 16, per_device_train_batch_size: int = 16, num_train_epochs: int = 1, device: Any = None, seed: int = -1, use_cpu: bool = False, epoch_train_steps: int = 100000, max_steps: int = -1, dataloader_num_workers: int = 0, dataloader_pin_memory: int = True, dataloader_persistent_workers: bool = False, dataloader_prefetch_factor: int | None = None, dataloader_drop_last: bool = False, eval_strategy: str = 'no', eval_steps: int = 100, eval_delay: int = 0, logging_strategy: str = 'steps', logging_steps: int = 50, logging_first_step: bool = False, torch_compile: bool = False, torch_compile_backend: str | None = None, torch_compile_mode: str | None = 'default', torch_compile_dynamic: bool = True, torch_compile_full_graph: bool = False, max_grad_norm: float | None = None, gradient_accumulation_steps: int = 1, save_strategy: str = 'steps', save_steps: int = 1000, save_total_limit: int = 2, save_safetensors: bool = True, save_on_each_node: bool = False, overwrite_output_dir: bool = False, resume_from_checkpoint: bool | str = True, load_best_model_at_end: bool = False, metric_for_best_model: str = 'loss', greater_is_better: bool | None = None, lr_scheduler_type: str = 'linear', lr_scheduler_kwargs: dict | None = None, warmup_steps: int = 0, learning_rate: float = 5e-05, weight_decay: float = 0.0, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, gradient_checkpointing: bool = False)
```

Minimal training configuration compatible with HuggingFace Trainer.

Provides a subset of ``transformers.TrainingArguments`` sufficient for basic
training. This is the base configuration class; extend it for additional
features rather than adding fields here.

Direct subclasses: ``BaseTrainingArguments`` (checkpoint control, PyTorch
optimisations) and ``TrainingArguments`` (simple single-GPU memory options).

**Parameters**

- `output_dir` (str) — Directory where model predictions and checkpoints are written.
- `logging_dir` (str or None) — TensorBoard log directory. Defaults to ``output_dir/runs/TIMESTAMP_HOSTNAME``.
- `per_device_train_batch_size` (int) — Training batch size per device. Effective global batch size is
``per_device_train_batch_size * num_devices * gradient_accumulation_steps``.
- `per_device_eval_batch_size` (int) — Evaluation batch size per device.
- `num_train_epochs` (int) — Total training epochs (may be fractional, e.g. ``2.5``).
- `max_steps` (int) — If > 0, total optimiser steps to perform (overrides ``num_train_epochs``).
- `device` (Any) — Device to use (``"cuda"``, ``"cpu"``, etc.). Auto-detected if ``None``.
- `seed` (int) — Random seed for reproducibility. Default ``-1`` disables seeding.
- `use_cpu` (bool) — Force CPU usage even when CUDA is available.
- `epoch_train_steps` (int) — Fallback epoch length when the dataset does not support ``len()``.
Used only for progress bookkeeping. Forgather extension.
- `dataloader_num_workers` (int) — Subprocesses for data loading. ``0`` loads in the main process.
- `dataloader_pin_memory` (bool) — Pin memory in DataLoader for faster GPU transfer.
- `dataloader_persistent_workers` (bool) — Keep worker processes alive between epochs (faster, uses more RAM).
- `dataloader_prefetch_factor` (int or None) — Batches prefetched per worker. Defaults to ``2`` when ``num_workers > 0``.
- `dataloader_drop_last` (bool) — Drop the last incomplete batch when the dataset is not evenly divisible.
- `eval_strategy` (str) — When to run evaluation: ``"no"``, ``"steps"``, or ``"epoch"``.
- `eval_steps` (int) — Evaluation frequency in steps (when ``eval_strategy="steps"``).
- `eval_delay` (int) — Epochs or steps to wait before the first evaluation.
- `logging_strategy` (str) — When to log metrics: ``"no"``, ``"steps"``, or ``"epoch"``.
- `logging_steps` (int) — Logging frequency in steps (when ``logging_strategy="steps"``).
- `logging_first_step` (bool) — Log metrics at the very first global step.
- `torch_compile` (bool) — Compile the model with ``torch.compile()`` for speedup.
- `torch_compile_backend` (str or None) — Backend for ``torch.compile`` (e.g. ``"inductor"``, ``"aot_eager"``).
- `torch_compile_mode` (str or None) — Compilation mode: ``"default"``, ``"reduce-overhead"``, or ``"max-autotune"``.
- `torch_compile_dynamic` (bool) — Allow dynamic shapes in the compiled model.
- `torch_compile_full_graph` (bool) — Force compilation of the entire model as a single graph.
- `max_grad_norm` (float or None) — Maximum gradient norm for clipping. ``None`` disables clipping.
- `gradient_accumulation_steps` (int) — Accumulate gradients over this many steps before an optimiser update.
- `save_strategy` (str) — Checkpoint save strategy: ``"no"``, ``"steps"``, or ``"epoch"``.
- `save_steps` (int) — Checkpoint save frequency in steps (when ``save_strategy="steps"``).
- `save_total_limit` (int) — Maximum number of checkpoints to keep; oldest are deleted first.
- `save_safetensors` (bool) — Write weights as Safetensors (safer and HF-compatible) rather than pickle.
- `save_on_each_node` (bool) — In multi-node training, save on every node rather than only rank 0.
Do not enable when using shared storage.
- `overwrite_output_dir` (bool) — Overwrite ``output_dir`` contents on a fresh run.
- `resume_from_checkpoint` (bool or str) — ``True`` (default) automatically finds and loads the latest checkpoint,
falling back to fresh initialisation if none exists. A path string loads
that specific checkpoint. ``False`` forces fresh initialisation.
- `load_best_model_at_end` (bool) — Restore the best checkpoint at the end of training. Requires
``save_strategy == eval_strategy``.
- `metric_for_best_model` (str) — Metric used to compare checkpoints when ``load_best_model_at_end=True``.
- `greater_is_better` (bool or None) — Whether a higher metric value is better. Auto-determined from the metric
name when ``None``.
- `lr_scheduler_type` (str) — LR scheduler type for the built-in AdamW path (``"linear"``, ``"cosine"``, etc.).
- `lr_scheduler_kwargs` (dict or None) — Additional keyword arguments forwarded to the LR scheduler constructor.
- `warmup_steps` (int) — Linear warmup steps from 0 to ``learning_rate``.
- `learning_rate` (float) — Initial learning rate for the built-in AdamW optimiser.
- `weight_decay` (float) — Weight decay applied to all parameters except bias and LayerNorm weights.
- `adam_beta1` (float) — Beta1 for the built-in AdamW optimiser.
- `adam_beta2` (float) — Beta2 for the built-in AdamW optimiser.
- `adam_epsilon` (float) — Epsilon for the built-in AdamW optimiser.
- `gradient_checkpointing` (bool) — Enable activation checkpointing on models that support the HF API.
Pass ``enable_activation_checkpoint_fn`` to the Trainer constructor to
customise the checkpointing behaviour.

**Attributes**

- `output_dir` (str)
- `logging_dir` (str | None)
- `per_device_eval_batch_size` (int)
- `per_device_train_batch_size` (int)
- `num_train_epochs` (int)
- `device` (Any)
- `seed` (int)
- `use_cpu` (bool)
- `epoch_train_steps` (int)
- `max_steps` (int)
- `dataloader_num_workers` (int)
- `dataloader_pin_memory` (int)
- `dataloader_persistent_workers` (bool)
- `dataloader_prefetch_factor` (int | None)
- `dataloader_drop_last` (bool)
- `eval_strategy` (str)
- `eval_steps` (int)
- `eval_delay` (int)
- `logging_strategy` (str)
- `logging_steps` (int)
- `logging_first_step` (bool)
- `torch_compile` (bool)
- `torch_compile_backend` (str | None)
- `torch_compile_mode` (str | None)
- `torch_compile_dynamic` (bool)
- `torch_compile_full_graph` (bool)
- `max_grad_norm` (float | None)
- `gradient_accumulation_steps` (int)
- `save_strategy` (str)
- `save_steps` (int)
- `save_total_limit` (int)
- `save_safetensors` (bool)
- `save_on_each_node` (bool)
- `overwrite_output_dir` (bool)
- `resume_from_checkpoint` (bool | str)
- `load_best_model_at_end` (bool)
- `metric_for_best_model` (str)
- `greater_is_better` (bool | None)
- `lr_scheduler_type` (str)
- `lr_scheduler_kwargs` (dict | None)
- `warmup_steps` (int)
- `learning_rate` (float)
- `weight_decay` (float)
- `adam_beta1` (float)
- `adam_beta2` (float)
- `adam_epsilon` (float)
- `gradient_checkpointing` (bool)

---

### `TrainerState` {#forgather-ml-trainer-trainer_types-trainerstate}

`forgather.ml.trainer.trainer_types.TrainerState`

```python
class TrainerState(*, logging_steps: int, eval_steps: int, train_batch_size: int, max_steps: int, epoch: float = 0.0, global_step: int = 0, num_train_epochs: int, is_local_process_zero: bool = True, is_world_process_zero: bool = True, log_history: list[Dict[str, float]] = (lambda: [])(), save_steps: int = 0, best_metric: float | None = None, best_model_checkpoint: str | None = None, num_input_tokens_seen: int = 0, total_flos: float = 0.0, is_hyper_param_search: bool = False, stateful_callbacks: List[TrainerCallback] = (lambda: [])(), max_eval_steps: int, epoch_start_step: int = 0, raw_epoch: int = 0)
```

Trainer state tracking training progress and configuration.

Maintains compatibility with HuggingFace Trainer API for easier porting.
Passed to callbacks to allow them to inspect and log training progress.

Key training progress fields:
- global_step: Total optimizer updates since start (0-indexed)
- raw_epoch: Integer epoch counter (increments at end of each dataset iteration)
- epoch_start_step: Global step when current epoch started
- epoch: Continuous epoch value = raw_epoch + fractional progress through current epoch
          Computed as: epoch = raw_epoch + (global_step - epoch_start_step) / epoch_train_steps

Best model tracking (for load_best_model_at_end):
- best_metric: Best metric value seen during training
- best_model_checkpoint: Path to checkpoint with best metric

See: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py

**Attributes**

- `logging_steps` (int)
- `eval_steps` (int)
- `train_batch_size` (int)
- `max_steps` (int)
- `epoch` (float)
- `global_step` (int)
- `num_train_epochs` (int)
- `is_local_process_zero` (bool)
- `is_world_process_zero` (bool)
- `log_history` (list[Dict[str, float]])
- `save_steps` (int)
- `best_metric` (float | None)
- `best_model_checkpoint` (str | None)
- `num_input_tokens_seen` (int)
- `total_flos` (float)
- `is_hyper_param_search` (bool)
- `stateful_callbacks` (List[TrainerCallback])
- `max_eval_steps` (int)
- `epoch_start_step` (int)
- `raw_epoch` (int)

---

### `TrainerControl` {#forgather-ml-trainer-trainer_types-trainercontrol}

`forgather.ml.trainer.trainer_types.TrainerControl`

```python
class TrainerControl(should_training_stop: bool = False, should_epoch_stop: bool = False, should_save: bool = False, should_evaluate: bool = False, should_log: bool = False, should_abort_without_save: bool = False)
```

Control flags for trainer execution flow.

Callbacks can return a modified TrainerControl to influence trainer behavior:
- Trigger checkpointing: Set should_save = True
- Trigger evaluation: Set should_evaluate = True
- Trigger logging: Set should_log = True
- Stop training gracefully: Set should_training_stop = True
- Stop current epoch: Set should_epoch_stop = True
- Abort without saving: Set should_abort_without_save = True

Compatible with HuggingFace Trainer API for easier callback porting.

Example callback usage:
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 1000 == 0:
            control.should_save = True  # Force checkpoint every 1000 steps
        return control

**Attributes**

- `should_training_stop` (bool)
- `should_epoch_stop` (bool)
- `should_save` (bool)
- `should_evaluate` (bool)
- `should_log` (bool)
- `should_abort_without_save` (bool)

---

### `TrainOutput` {#forgather-ml-trainer-trainer_types-trainoutput}

`forgather.ml.trainer.trainer_types.TrainOutput`

_No documentation._

**Attributes**

- `global_step` (int)
- `metrics` (Dict[str, float])

---

## Base Classes

Abstract base from which all concrete trainers derive. Implement these three
methods to build a custom trainer: `_prepare`, `_train_loop`, `_eval_loop`.

### `BaseTrainer` {#forgather-ml-trainer-base_trainer-basetrainer}

`forgather.ml.trainer.base_trainer.BaseTrainer`

```python
class BaseTrainer(args: TBaseTrainingArguments, model: torch.nn.Module | None = None, *, data_collator: Optional[DataCollatorT] = None, train_dataset: Optional[IterableDatasetT] = None, eval_dataset: Optional[IterableDatasetT] = None, processing_class: Optional[PreprocessingClassT] = None, model_init: Optional[ModelConstructor] = None, callbacks: Optional[List[TrainerCallback]] = None, compute_loss_func: Optional[LossFunctionT] = None)
```

Abstract base class implementing common trainer infrastructure.

Provides callback management, checkpoint coordination, training-state tracking,
and the ``PyTorch Stateful`` interface. The actual training and evaluation loops
are left abstract so that concrete subclasses (``Trainer``, ``AccelTrainer``,
``PipelineTrainer``) can implement them with their own parallelism strategy.

This class intentionally mirrors the public surface of ``transformers.Trainer``
to make porting existing training scripts straightforward.

**Parameters**

- `args` (TBaseTrainingArguments) — Training configuration dataclass. Must be an instance of
``BaseTrainingArguments`` or one of its subclasses.
- `model` (torch.nn.Module or None) — Pre-constructed model. Either ``model`` or ``model_init`` must be
provided. Default is ``None``.
- `data_collator` (DataCollatorT or None) — Callable that collates a list of dataset examples into a batch dict.
Default is ``None``.
- `train_dataset` (IterableDatasetT or None) — Training dataset (``torch.utils.data.Dataset`` or any iterable).
Default is ``None``.
- `eval_dataset` (IterableDatasetT or None) — Evaluation dataset. Default is ``None``.
- `processing_class` (PreprocessingClassT or None) — Tokenizer or feature extractor saved alongside model weights.
Default is ``None``.
- `model_init` (Callable[[], torch.nn.Module] or None) — Zero-argument factory that constructs the model. Required when ``model``
is ``None`` (e.g. for pipeline training where construction must happen
inside the trainer). Default is ``None``.
- `callbacks` (list of TrainerCallback or None) — Callbacks to install. When ``None``, ``default_callbacks()`` is used.
Default is ``None``.
- `compute_loss_func` (LossFunctionT or None) — Custom loss function. When ``None``, the trainer computes cross-entropy
from model logits. Default is ``None``.

**Attributes**

- `state` (TrainerState) — Mutable training progress state (global step, epoch, log history, etc.).
- `control` (TrainerControl) — Mutable flags set by callbacks to signal save/eval/stop actions.
- `checkpoint_manager` (CheckpointInterface or None) — Set by ``_prepare()`` before the training loop starts.

**Raises**

- `AssertionError` — If neither ``model`` nor ``model_init`` is provided.
- `AssertionError` — If ``args.gradient_accumulation_steps`` is not greater than 0.

> **Note**
>
> Concrete subclasses must implement three abstract methods:

> * ``_prepare(train_dataset, eval_dataset)`` — set up dataloaders, model,
>   optimizer, and checkpoint manager.
> * ``_train_loop()`` — the main training iteration loop, returning
>   ``TrainOutput``.
> * ``_eval_loop()`` — the evaluation loop, returning a metrics dict.

**Attributes**

- `args` (TBaseTrainingArguments)
- `model` (torch.nn.Module | None)
- `data_collator` (DataCollatorT | None)
- `train_dataset` (IterableDatasetT | None)
- `eval_dataset` (IterableDatasetT | None)
- `processing_class` (PreprocessingClassT | None)
- `model_init` (ModelConstructor | None)
- `callbacks` (List[TrainerCallback])
- `loss_fn` (LossFunctionT | None)
- `train_dataloader` (Iterable | None)
- `eval_dataloader` (Iterable | None)
- `optimizer` (OptimizerT | None)
- `lr_scheduler` (LRSchedulerT | None)
- `is_local_process_zero` (bool)
- `is_world_process_zero` (bool)
- `num_processes` (int)
- `checkpoint_manager` (CheckpointInterface | None)
- `state` (TrainerState)
- `control` (TrainerControl)

**Methods**

#### `default_callbacks` {#forgather-ml-trainer-base_trainer-basetrainer-default_callbacks}

```python
def default_callbacks(cls)
```

Return the default callbacks for this trainer class.

Subclasses override this to provide callbacks that are always installed
(e.g. ``ProgressCallback``, ``InfoCallback``). The base implementation
returns an empty list.

**Returns**

- `list of TrainerCallback` — Default callback instances.

#### `train` {#forgather-ml-trainer-base_trainer-basetrainer-train}

```python
def train(**kwargs)
```

Run the full training loop.

Applies any configured PyTorch context managers (SDPA backend, activation
offloading), calls ``_prepare()`` to set up all components, then delegates
to ``_train_loop()``.

**Returns**

- `TrainOutput` — Named tuple with ``global_step``, ``training_loss``, and ``metrics``.

#### `evaluate` {#forgather-ml-trainer-base_trainer-basetrainer-evaluate}

```python
def evaluate(eval_dataset: Optional[BaseDataset] = None, **kwargs)
```

Run evaluation on the given dataset.

Applies the configured SDPA backend context, calls ``_prepare()`` with
``train_dataset=None``, then delegates to ``_eval_loop()``.

**Parameters**

- `eval_dataset` (BaseDataset or None) — Dataset to evaluate on. Falls back to ``self.eval_dataset`` when
``None``. Default is ``None``.

**Returns**

- `dict of str to float` — Evaluation metrics, e.g. ``{"eval_loss": 1.23}``.

#### `add_callback` {#forgather-ml-trainer-base_trainer-basetrainer-add_callback}

```python
def add_callback(callback: TrainerCallback)
```

_No documentation._

#### `pop_callback` {#forgather-ml-trainer-base_trainer-basetrainer-pop_callback}

```python
def pop_callback(callback: TrainerCallback)
```

_No documentation._

#### `remove_callback` {#forgather-ml-trainer-base_trainer-basetrainer-remove_callback}

```python
def remove_callback(callback: TrainerCallback)
```

_No documentation._

#### `log` {#forgather-ml-trainer-base_trainer-basetrainer-log}

```python
def log(logs: Dict[str, float])
```

Log metrics and dispatch the ``on_log`` event to all callbacks.

Appends the metrics dict to ``state.log_history``, then fires the
``on_log`` callback event. Callbacks use this to write to TensorBoard,
wandb, or other logging backends.

**Parameters**

- `logs` (dict of str to float) — Metrics to record, e.g. ``{"loss": 0.5, "lr": 1e-4}``.

**Returns**

- `TrainerControl` — Updated control object (callbacks may have set ``should_save``,
``should_evaluate``, etc.).

#### `unwrapped_model` {#forgather-ml-trainer-base_trainer-basetrainer-unwrapped_model}

```python
def unwrapped_model()
```

Return the underlying model, free of any distributed wrappers.

Subclasses that wrap ``self.model`` in DDP, FSDP, Accelerate, or
pipeline-parallel containers override this method to strip those wrappers
before the model is passed to callbacks.

**Returns**

- `torch.nn.Module` — The bare model without any framework wrapper.

#### `save_model` {#forgather-ml-trainer-base_trainer-basetrainer-save_model}

```python
def save_model(output_dir: Optional[os.PathLike | str] = None)
```

Save model weights and the preprocessing class (HF Trainer API compatibility).

Writes only the model weights to ``output_dir`` (or ``args.output_dir``
when ``output_dir`` is ``None``). The full training state (optimizer,
scheduler, RNG, etc.) is **not** saved. For resumable training, prefer
``save_checkpoint()``.

**Parameters**

- `output_dir` (path - like or str) — Destination directory. Defaults to ``args.output_dir``.

#### `save_checkpoint` {#forgather-ml-trainer-base_trainer-basetrainer-save_checkpoint}

```python
def save_checkpoint(checkpoint_path = None)
```

Save a complete training checkpoint.

Writes all training state to a timestamped directory under
``args.output_dir``. The following components are always saved:

* Model weights (required for resuming)
* Optimizer state (momentum buffers, adaptive learning rates, etc.)
* LR scheduler state (current step position)
* Training progress (``global_step``, epoch counter, etc.)
* Dataset position (when the dataloader is stateful)
* Random number generator states (for bit-exact reproducibility)

**Parameters**

- `checkpoint_path` (path - like or str) — Explicit checkpoint directory path. When ``None``, a path is
auto-generated under ``args.output_dir`` based on the current
step count.

#### `load_checkpoint` {#forgather-ml-trainer-base_trainer-basetrainer-load_checkpoint}

```python
def load_checkpoint(checkpoint_path = None)
```

Load a training checkpoint to resume training.

Restores all available training state from the specified checkpoint
directory. Each component is loaded only if its file exists:

* Model weights (always required — raises if missing)
* Optimizer state
* LR scheduler state
* Training progress (``global_step``, epoch, etc.)
* Dataset position
* Random number generator states

When ``checkpoint_path`` is ``None``, the latest checkpoint under
``args.output_dir`` is located automatically.

To intentionally skip reloading a component, delete its file from the
checkpoint directory before calling this method. The checkpoint system
logs a warning for each missing file but continues loading the rest.

**Parameters**

- `checkpoint_path` (path - like or str) — Path to the checkpoint directory. ``None`` auto-selects the latest
checkpoint under ``args.output_dir``.

#### `get_state_components` {#forgather-ml-trainer-base_trainer-basetrainer-get_state_components}

```python
def get_state_components()
```

Return state components for checkpoint save/load.

Describes every piece of training state that should be persisted. The
checkpoint manager calls this method to determine what to save and how
state is shared across distributed ranks.

For the single-GPU base trainer all components use ``GLOBAL`` sharing
except RNG which uses ``PER_RANK``.

Returned components (in order):

* ``"model"`` — model weights, **required**
* ``"optimizer"`` — optimizer state (optional)
* ``"scheduler"`` — LR scheduler state (optional)
* ``"trainer"`` — training progress counters (optional)
* ``"dataset"`` — dataloader position, only when stateful (optional)
* ``"rng"`` — per-rank RNG state (optional)

**Returns**

- `list of StateComponent` — All checkpointable state components with their sharing patterns.

#### `get_process_groups` {#forgather-ml-trainer-base_trainer-basetrainer-get_process_groups}

```python
def get_process_groups()
```

Return named process groups for ``PER_GROUP`` sharing pattern.

The checkpoint manager uses this to coordinate group-level saves (e.g.
tensor-parallel replicas). Single-GPU trainers have no process groups.
Subclasses implementing hybrid parallelism should override this method.

**Returns**

- `dict of str to Any` — Empty dict for the single-GPU base trainer.

#### `load_state_dict` {#forgather-ml-trainer-base_trainer-basetrainer-load_state_dict}

```python
def load_state_dict(state_dict)
```

Restore trainer progress state from a checkpoint state dict.

Implements the ``torch.distributed.checkpoint.stateful.Stateful``
interface. Restores step counters and progress tracking so training
resumes at the exact point where it was saved. Also restores the
``GradScaler`` state when fp16 AMP is active.

**Parameters**

- `state_dict` (dict) — State dictionary previously returned by ``state_dict()``. Expected
keys: ``global_step``, ``epoch_start_step``, ``raw_epoch``,
``num_input_tokens_seen``, ``total_flos``, and optionally
``grad_scaler``.

#### `state_dict` {#forgather-ml-trainer-base_trainer-basetrainer-state_dict}

```python
def state_dict()
```

Return trainer progress state for checkpointing.

Implements the ``torch.distributed.checkpoint.stateful.Stateful``
interface. The returned dict is consumed by ``load_state_dict()`` to
restore training from the exact saved point.

**Returns**

- `dict` — Training state with the following keys:

* ``global_step`` — total optimizer updates performed.
* ``epoch_start_step`` — global step at the start of the current epoch.
* ``raw_epoch`` — integer epoch counter.
* ``num_input_tokens_seen`` — total tokens processed (for throughput logging).
* ``total_flos`` — total floating-point operations (for efficiency metrics).
* ``grad_scaler`` — ``GradScaler`` state dict (only when fp16 AMP is active).

---

### `BaseTrainingArguments` {#forgather-ml-trainer-base_trainer-basetrainingarguments}

`forgather.ml.trainer.base_trainer.BaseTrainingArguments`

```python
class BaseTrainingArguments(*, output_dir: str = OUTPUTDIR_NAME, logging_dir: str | None = None, per_device_eval_batch_size: int = 16, per_device_train_batch_size: int = 16, num_train_epochs: int = 1, device: Any = None, seed: int = -1, use_cpu: bool = False, epoch_train_steps: int = 100000, max_steps: int = -1, dataloader_num_workers: int = 0, dataloader_pin_memory: int = True, dataloader_persistent_workers: bool = False, dataloader_prefetch_factor: int | None = None, dataloader_drop_last: bool = False, eval_strategy: str = 'no', eval_steps: int = 100, eval_delay: int = 0, logging_strategy: str = 'steps', logging_steps: int = 50, logging_first_step: bool = False, torch_compile: bool = False, torch_compile_backend: str | None = None, torch_compile_mode: str | None = 'default', torch_compile_dynamic: bool = True, torch_compile_full_graph: bool = False, max_grad_norm: float | None = None, gradient_accumulation_steps: int = 1, save_strategy: str = 'steps', save_steps: int = 1000, save_total_limit: int = 2, save_safetensors: bool = True, save_on_each_node: bool = False, overwrite_output_dir: bool = False, resume_from_checkpoint: bool | str = True, load_best_model_at_end: bool = False, metric_for_best_model: str = 'loss', greater_is_better: bool | None = None, lr_scheduler_type: str = 'linear', lr_scheduler_kwargs: dict | None = None, warmup_steps: int = 0, learning_rate: float = 5e-05, weight_decay: float = 0.0, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, gradient_checkpointing: bool = False, default_dtype: str | None = None, max_eval_steps: int = -1, preserve_best_model: bool = False, best_model_metric: str = 'loss', best_model_greater_is_better: bool | None = None, preserve_n_best: int = 1, eval_on_save: bool = False, enable_activation_offloading: bool = False, detect_anomaly: bool = False, sdpa_backend: List[str] | str | None = None, sdpa_set_priority: bool = False, float32_matmul_precision: str | None = None, dynamo_recompile_limit: int | None = None, mixed_precision: str | None = None, fp8_recipe: str | None = None, fp8_dim_alignment: int = 16, qat_recipe: str | None = None)
```

Extended training arguments with checkpoint management and PyTorch optimizations.

Extends ``MinimalTrainingArguments`` with full checkpoint state preservation and a
range of PyTorch runtime optimizations (mixed precision, FP8, SDPA backend
selection, activation offloading, etc.).

All training state (model, optimizer, scheduler, dataset position, RNG state) is
automatically saved in checkpoints. To skip loading a specific component when
resuming, manually delete its file from the checkpoint directory before calling
``train()``.

.. note::
    The checkpoint-related options in this class are **not** compatible with the
    HuggingFace ``Trainer``. Use ``MinimalTrainingArguments`` when HF compatibility
    is required.

**Parameters**

- `default_dtype` (str or None) — Default ``torch.dtype`` for model construction (e.g. ``"float32"``,
``"bfloat16"``, ``"float16"``). ``None`` leaves PyTorch's global default
unchanged. Default is ``None``.
- `max_eval_steps` (int) — Maximum number of evaluation steps per evaluation call. ``-1`` runs the
full evaluation dataset. Default is ``-1``.
- `preserve_best_model` (bool) — If ``True``, keep the checkpoint with the best value of
``best_model_metric`` protected from cleanup rotation. Default is ``False``.
- `best_model_metric` (str) — Name of the metric used to select the best checkpoint when
``preserve_best_model=True``. Default is ``"loss"``.
- `best_model_greater_is_better` (bool or None) — Whether higher values of ``best_model_metric`` are better. ``None``
auto-detects from the metric name (metrics containing ``"loss"`` or
``"perplexity"`` default to lower-is-better). Default is ``None``.
- `preserve_n_best` (int) — Number of best checkpoints to keep safe from ``save_total_limit``
cleanup. Default is ``1``.
- `eval_on_save` (bool) — Force an evaluation pass before each checkpoint save. Useful for
decoupling the save and eval schedules. Default is ``False``.
- `enable_activation_offloading` (bool) — Offload saved activation tensors to CPU during the backward pass to
reduce peak GPU memory. Best combined with activation checkpointing.
Trades GPU memory for CPU memory bandwidth. Default is ``False``.
- `detect_anomaly` (bool) — Enable ``torch.autograd`` anomaly detection for debugging NaN/Inf
gradients. Adds significant overhead — use only for debugging.
Default is ``False``.
- `sdpa_backend` (list of str, str, or None) — Scaled Dot-Product Attention backend(s). Valid string values are
``"math"``, ``"flash"``, ``"efficient"``, and ``"cudnn"``. Pass a list
to specify multiple backends; if ``sdpa_set_priority=True``, the list is
treated as a priority order. ``None`` uses PyTorch's default selection.
Default is ``None``.
- `sdpa_set_priority` (bool) — When ``sdpa_backend`` is a list, interpret it as a priority order rather
than requiring all backends to be available. Default is ``False``.
- `float32_matmul_precision` (str or None) — Float32 matrix-multiplication precision on Ampere+ GPUs. One of
``"highest"`` (full IEEE, slowest), ``"high"`` (TF32, ~10–20 % speedup),
or ``"medium"`` (more aggressive, may impact accuracy). ``None`` leaves
the PyTorch default unchanged. Default is ``None``.
- `dynamo_recompile_limit` (int or None) — Override ``torch._dynamo.config.recompile_limit``. Increase when
``torch.compile()`` produces frequent recompilation warnings. ``None``
leaves the default unchanged. Default is ``None``.
- `mixed_precision` (str or None) — Automatic Mixed Precision mode. ``None`` or ``"no"`` disables AMP.
``"bf16"`` enables bfloat16 autocast without loss scaling (recommended
for Ampere+ GPUs). ``"fp16"`` enables float16 autocast with
``GradScaler`` loss scaling. Default is ``None``.
- `fp8_recipe` (str or None) — FP8 training recipe via ``torchao``. Converts ``nn.Linear`` layers to
``Float8Linear``. One of ``"tensorwise"`` (fastest), ``"rowwise"``
(more accurate), or ``"rowwise_with_gw_hp"`` (most accurate). ``None``
disables FP8. Orthogonal to ``mixed_precision``; combine both for FP8
matmuls with bfloat16 non-linear ops. Requires CUDA SM >= 8.9.
Default is ``None``.
- `fp8_dim_alignment` (int) — Minimum alignment for FP8 ``Linear`` layer dimensions. Layers whose
``in_features`` or ``out_features`` are not divisible by this value are
skipped. Hardware requires 16. Default is ``16``.
- `qat_recipe` (str or None) — Quantization-aware training recipe via ``torchao``. Inserts
``FakeQuantizedLinear`` modules so the forward pass simulates the
target low-bit precision while backward stays in full precision.
After training, run ``forgather finalize --quantize <recipe>`` to
produce the real low-bit deployment artifact. Mutually exclusive with
``fp8_recipe``. See ``docs/trainers/qat-training.md`` for the recipe
list. Default is ``None``.

**Attributes**

- `default_dtype` (str | None)
- `max_eval_steps` (int)
- `preserve_best_model` (bool)
- `best_model_metric` (str)
- `best_model_greater_is_better` (bool | None)
- `preserve_n_best` (int)
- `eval_on_save` (bool)
- `enable_activation_offloading` (bool)
- `detect_anomaly` (bool)
- `sdpa_backend` (List[str] | str | None)
- `sdpa_set_priority` (bool)
- `float32_matmul_precision` (str | None)
- `dynamo_recompile_limit` (int | None)
- `mixed_precision` (str | None)
- `fp8_recipe` (str | None)
- `fp8_dim_alignment` (int)
- `qat_recipe` (str | None)

---

## Single-GPU Trainer

### `Trainer` {#forgather-ml-trainer-trainer-trainer}

`forgather.ml.trainer.trainer.Trainer`

```python
class Trainer(*, args: TTrainingArguments | dict, distributed_env: DistributedEnvInterface, optimizer_factory: Optional[OptimizerFactoryT] = None, optimizer_cls_and_kwargs: Optional[Tuple[Type[OptimizerT], Dict[str, Any]]] = None, lr_scheduler_factory: Optional[LRSchedulerFactoryT] = None, enable_activation_checkpoint_fn: Optional[EnableCheckpointFnT] = enable_hf_activation_checkpointing, fused_loss_factory: Optional[FusedLossFactoryT] = None, optimizer_groups: Optional[OptimGroupMap] = None, **kwargs)
```

A lightweight, single-device trainer with API close to transformers.Trainer.

This trainer provides a simplified, more comprehensible implementation of the
HuggingFace Trainer, intended as a drop-in replacement for basic use cases.

Key features:
- Compatible with HF Trainer API for basic training workflows
- Memory optimizations: fused loss, fused optimizer/backward, activation checkpointing
- Flexible model construction: default/meta/device modes for different memory/speed tradeoffs
- Full checkpoint management: saves/restores model, optimizer, scheduler, dataset state
- Best model tracking via load_best_model_at_end

For distributed training, see AccelTrainer (data parallel via Accelerate) and
PipelineTrainer (pipeline parallelism).

Basic usage:
    trainer = Trainer(
        model=model,
        args=TrainingArguments(...),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizer_factory=optimizer_factory,
        lr_scheduler_factory=lr_scheduler_factory,
    )
    trainer.train()

**Attributes**

- `args` (TTrainingArguments)
- `dist` (DistributedEnvInterface)
- `optimizer_factory` (OptimizerFactoryT | None)
- `lr_scheduler_factory` (LRSchedulerFactoryT | None)
- `enable_activation_checkpoint_fn` (EnableCheckpointFnT | None)
- `fused_loss_factory` (FusedLossFactoryT | None)
- `optimizer_groups` (OptimGroupMap | None)
- `max_steps` (int)
- `epoch_train_steps` (int)
- `do_train` (bool)
- `do_eval` (bool)
- `use_fused_loss` (bool)
- `gradient_accumulation_step` (int)
- `data_collator`

**Methods**

#### `default_callbacks` {#forgather-ml-trainer-trainer-trainer-default_callbacks}

```python
def default_callbacks(cls)
```

_No documentation._

#### `load_best_model` {#forgather-ml-trainer-trainer-trainer-load_best_model}

```python
def load_best_model()
```

Load the best model from the best checkpoint.

Called at end of training when load_best_model_at_end=True to restore
the checkpoint with the best metric value seen during training.

#### `load_checkpoint` {#forgather-ml-trainer-trainer-trainer-load_checkpoint}

```python
def load_checkpoint(*args, **kwargs)
```

_No documentation._

---

### `TrainingArguments` {#forgather-ml-trainer-trainer-trainingarguments}

`forgather.ml.trainer.trainer.TrainingArguments`

```python
class TrainingArguments(*, output_dir: str = OUTPUTDIR_NAME, logging_dir: str | None = None, per_device_eval_batch_size: int = 16, per_device_train_batch_size: int = 16, num_train_epochs: int = 1, device: Any = None, seed: int = -1, use_cpu: bool = False, epoch_train_steps: int = 100000, max_steps: int = -1, dataloader_num_workers: int = 0, dataloader_pin_memory: int = True, dataloader_persistent_workers: bool = False, dataloader_prefetch_factor: int | None = None, dataloader_drop_last: bool = False, eval_strategy: str = 'no', eval_steps: int = 100, eval_delay: int = 0, logging_strategy: str = 'steps', logging_steps: int = 50, logging_first_step: bool = False, torch_compile: bool = False, torch_compile_backend: str | None = None, torch_compile_mode: str | None = 'default', torch_compile_dynamic: bool = True, torch_compile_full_graph: bool = False, max_grad_norm: float | None = None, gradient_accumulation_steps: int = 1, save_strategy: str = 'steps', save_steps: int = 1000, save_total_limit: int = 2, save_safetensors: bool = True, save_on_each_node: bool = False, overwrite_output_dir: bool = False, resume_from_checkpoint: bool | str = True, load_best_model_at_end: bool = False, metric_for_best_model: str = 'loss', greater_is_better: bool | None = None, lr_scheduler_type: str = 'linear', lr_scheduler_kwargs: dict | None = None, warmup_steps: int = 0, learning_rate: float = 5e-05, weight_decay: float = 0.0, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, gradient_checkpointing: bool = False, default_dtype: str | None = None, max_eval_steps: int = -1, preserve_best_model: bool = False, best_model_metric: str = 'loss', best_model_greater_is_better: bool | None = None, preserve_n_best: int = 1, eval_on_save: bool = False, enable_activation_offloading: bool = False, detect_anomaly: bool = False, sdpa_backend: List[str] | str | None = None, sdpa_set_priority: bool = False, float32_matmul_precision: str | None = None, dynamo_recompile_limit: int | None = None, mixed_precision: str | None = None, fp8_recipe: str | None = None, fp8_dim_alignment: int = 16, qat_recipe: str | None = None, gc_threshold: float = 0.5, construct_model_on: str = 'default', activation_memory_budget: float | None = None, fuse_optim_with_backward: bool = False, speed_metrics_start_step: int = 1, set_dataset_epoch: bool = True, debug_optimizer_groups: bool = False)
```

Training arguments specific to the simple Trainer implementation.

Extends BaseTrainingArguments with memory optimization and model construction options.
Maintains compatibility with HuggingFace Trainer API where possible.

**Attributes**

- `gc_threshold` (float)
- `construct_model_on` (str)
- `activation_memory_budget` (float | None)
- `fuse_optim_with_backward` (bool)
- `speed_metrics_start_step` (int)
- `set_dataset_epoch` (bool)
- `debug_optimizer_groups` (bool)

---

## Distributed Data Parallel (DDP) Trainer

### `DDPTrainer` {#forgather-ml-trainer-ddp-ddp_trainer-ddptrainer}

`forgather.ml.trainer.ddp.ddp_trainer.DDPTrainer`

```python
class DDPTrainer(*, args: TDDPTrainingArguments | dict, fused_loss_factory: Optional[FusedLossFactoryT] = None, **kwargs)
```

Multi-GPU trainer using DistributedDataParallel (DDP).

Wraps the base ``Trainer`` with DDP for data-parallel training across multiple GPUs
or nodes. Each rank receives a different batch; gradients are all-reduced automatically
after each backward pass. Optionally uses PostLocalSGD for bandwidth-limited environments.

Launch with ``torchrun`` (or the ``forgather train -d ...`` shortcut)::

    torchrun --nproc_per_node=4 train.py

Key differences from single-GPU ``Trainer``:

- Model wrapped in ``torch.nn.parallel.DistributedDataParallel``
- Gradient accumulation uses DDP's ``no_sync()`` context to skip reductions on
  intermediate steps
- Dataset loading via ``DataloaderDispatcher`` (``dispatch_batches=True``, default)
  or ``SynchronizedDataLoader`` (``dispatch_batches=False``)
- Optional PostLocalSGD communication hook for reduced all-reduce frequency

**Attributes**

- `args` (TDDPTrainingArguments)
- `gradient_accumulation_step` (int)

**Methods**

#### `unwrapped_model` {#forgather-ml-trainer-ddp-ddp_trainer-ddptrainer-unwrapped_model}

```python
def unwrapped_model()
```

Get and returned the wrapped model

In the case of DDP, the original model is stored in the model's "module" attribute.

#### `get_state_components` {#forgather-ml-trainer-ddp-ddp_trainer-ddptrainer-get_state_components}

```python
def get_state_components()
```

Get state components for DDP training.

All training state is always saved to checkpoints. To skip loading a component,
delete its file from the checkpoint directory.

DDP uses data parallelism where model and optimizer state are replicated
across all ranks. DDP automatically synchronizes model weights and gradients,
so these components use REPLICATED pattern with validation enabled to catch
synchronization bugs.

Dataset pattern depends on dispatch_batches setting:
- dispatch_batches=True: GLOBAL (rank 0 loads and dispatches)
- dispatch_batches=False: PER_RANK (each rank has independent dataloader)

**Returns**

- `list of StateComponent` — State components with REPLICATED sharing patterns for DDP.

#### `get_process_groups` {#forgather-ml-trainer-ddp-ddp_trainer-ddptrainer-get_process_groups}

```python
def get_process_groups()
```

Get named process groups for checkpoint coordination.

**Returns**

- `dict` — Mapping of group names to ProcessGroup objects.
For DDP, returns the data parallel group.

---

### `DDPTrainingArguments` {#forgather-ml-trainer-ddp-ddp_trainer-ddptrainingarguments}

`forgather.ml.trainer.ddp.ddp_trainer.DDPTrainingArguments`

```python
class DDPTrainingArguments(*, output_dir: str = OUTPUTDIR_NAME, logging_dir: str | None = None, per_device_eval_batch_size: int = 16, per_device_train_batch_size: int = 16, num_train_epochs: int = 1, device: Any = None, seed: int = -1, use_cpu: bool = False, epoch_train_steps: int = 100000, max_steps: int = -1, dataloader_num_workers: int = 0, dataloader_pin_memory: int = True, dataloader_persistent_workers: bool = False, dataloader_prefetch_factor: int | None = None, dataloader_drop_last: bool = False, eval_strategy: str = 'no', eval_steps: int = 100, eval_delay: int = 0, logging_strategy: str = 'steps', logging_steps: int = 50, logging_first_step: bool = False, torch_compile: bool = False, torch_compile_backend: str | None = None, torch_compile_mode: str | None = 'default', torch_compile_dynamic: bool = True, torch_compile_full_graph: bool = False, max_grad_norm: float | None = None, gradient_accumulation_steps: int = 1, save_strategy: str = 'steps', save_steps: int = 1000, save_total_limit: int = 2, save_safetensors: bool = True, save_on_each_node: bool = False, overwrite_output_dir: bool = False, resume_from_checkpoint: bool | str = True, load_best_model_at_end: bool = False, metric_for_best_model: str = 'loss', greater_is_better: bool | None = None, lr_scheduler_type: str = 'linear', lr_scheduler_kwargs: dict | None = None, warmup_steps: int = 0, learning_rate: float = 5e-05, weight_decay: float = 0.0, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, gradient_checkpointing: bool = False, default_dtype: str | None = None, max_eval_steps: int = -1, preserve_best_model: bool = False, best_model_metric: str = 'loss', best_model_greater_is_better: bool | None = None, preserve_n_best: int = 1, eval_on_save: bool = False, enable_activation_offloading: bool = False, detect_anomaly: bool = False, sdpa_backend: List[str] | str | None = None, sdpa_set_priority: bool = False, float32_matmul_precision: str | None = None, dynamo_recompile_limit: int | None = None, mixed_precision: str | None = None, fp8_recipe: str | None = None, fp8_dim_alignment: int = 16, qat_recipe: str | None = None, gc_threshold: float = 0.5, construct_model_on: str = 'default', activation_memory_budget: float | None = None, fuse_optim_with_backward: bool = False, speed_metrics_start_step: int = 1, set_dataset_epoch: bool = True, debug_optimizer_groups: bool = False, dispatch_batches: bool = True, dispatch_eval_batches: Optional[bool] = None, ddp: DDPArguments = DDPArguments(), post_local_sgd: PostLocalSGDArguments = PostLocalSGDArguments())
```

_No documentation._

**Attributes**

- `dispatch_batches` (bool)
- `dispatch_eval_batches` (Optional[bool])
- `ddp` (DDPArguments)
- `post_local_sgd` (PostLocalSGDArguments)

## Fully Sharded Distributed Data Parallel (FSDP2) Trainer

---

### `FSDP2Trainer` {#forgather-ml-trainer-fsdp2-fsdp2_trainer-fsdp2trainer}

`forgather.ml.trainer.fsdp2.fsdp2_trainer.FSDP2Trainer`

```python
class FSDP2Trainer(*, args: TFSDP2TrainingArguments | dict, fused_loss_factory: Optional[FusedLossFactoryT] = None, **kwargs)
```

Trainer that shards model, gradients, and optimizer state via FSDP2.

Uses ``torch.distributed.fsdp.fully_shard`` (PyTorch's FSDP2 API) to distribute
parameters, gradients, and optimizer state across all ranks. Provides ZeRO-3-style
memory savings, making it suitable for models that don't fit in a single GPU's memory.

Launch with ``torchrun`` (or the ``forgather train -d ...`` shortcut)::

    torchrun --nproc_per_node=4 train.py

Key differences from DDP:

- Each rank stores only a shard of parameters, gradients, and optimizer state
- Parameters are all-gathered before each forward/backward and re-sharded after
  (controlled by ``fsdp2.reshard_after_forward``)
- Model checkpoints are saved as full HuggingFace safetensors gathered on rank 0,
  making them loadable by ``from_pretrained`` without special tooling
- Optimizer state is saved per-rank (sharded) and tied to the world size

See ``FSDP2Arguments`` for sharding configuration options (mixed precision policy,
CPU offload, transformer-layer-wise sharding).

**Attributes**

- `args` (TFSDP2TrainingArguments)

**Methods**

#### `unwrapped_model` {#forgather-ml-trainer-fsdp2-fsdp2_trainer-fsdp2trainer-unwrapped_model}

```python
def unwrapped_model()
```

_No documentation._

#### `pipeline_generate` {#forgather-ml-trainer-fsdp2-fsdp2_trainer-fsdp2trainer-pipeline_generate}

```python
def pipeline_generate(input_ids: Tensor, **kwargs)
```

All-rank generate: FSDP2 forward pass needs every rank in the
all_gather, so generation must run collectively. The
``TextgenCallback`` detects this method and uses a coordinated
broadcast-then-generate flow (same path as PipelineTrainer).

#### `get_state_components` {#forgather-ml-trainer-fsdp2-fsdp2_trainer-fsdp2trainer-get_state_components}

```python
def get_state_components()
```

State components for FSDP2.

The model is saved/loaded as HuggingFace safetensors via the model
hooks wired in ``_init_checkpoint_manager``; it is NOT registered as
a StateComponent. Optimizer state stays sharded per rank because
the DTensor layout of the optimizer moments cannot cheaply round-
trip through a gather/broadcast. Scheduler, trainer progress,
dataset and RNG mirror DDPTrainer.

#### `get_process_groups` {#forgather-ml-trainer-fsdp2-fsdp2_trainer-fsdp2trainer-get_process_groups}

```python
def get_process_groups()
```

_No documentation._

---

### `FSDP2Arguments` {#forgather-ml-trainer-fsdp2-fsdp2_trainer-fsdp2arguments}

`forgather.ml.trainer.fsdp2.fsdp2_trainer.FSDP2Arguments`

```python
class FSDP2Arguments(*, reshard_after_forward: bool = True, param_dtype: Optional[str] = None, reduce_dtype: Optional[str] = None, buffer_dtype: Optional[str] = None, cpu_offload: bool = False, shard_transformer_layers: bool = True, transformer_layers_path: str = 'causal_lm.layer_stack.layers')
```

_No documentation._

**Attributes**

- `reshard_after_forward` (bool)
- `param_dtype` (Optional[str])
- `reduce_dtype` (Optional[str])
- `buffer_dtype` (Optional[str])
- `cpu_offload` (bool)
- `shard_transformer_layers` (bool)
- `transformer_layers_path` (str)

---

## Pipeline Parallel Trainer

### `PipelineTrainer` {#forgather-ml-trainer-pipeline-pipeline_trainer-pipelinetrainer}

`forgather.ml.trainer.pipeline.pipeline_trainer.PipelineTrainer`

```python
class PipelineTrainer(*, args: TPipelineTrainingArguments | dict, model_splitter: ModelSplitter, pipe_schedule_factory: PipelineSchedulerFactorT = ScheduleGPipe, **kwargs)
```

Trainer for pipeline parallel training using PyTorch distributed pipelining.

Partitions a model across multiple GPUs — each GPU hosts one or more
sequential pipeline stages. Input batches are split into microbatches that
flow through the stages with multiple microbatches in flight simultaneously,
keeping all GPUs busy.

This trainer is designed for environments where inter-GPU bandwidth is limited
(consumer GPUs over PCIe, multi-node over Ethernet) where all-reduce–based DDP
or FSDP would be communication-bound.

Key differences from the single-device ``Trainer``:

* Model is constructed on the meta device, then each stage is materialised
  on its assigned GPU — no full model ever lives on one GPU.
* Rank 0 constructs a fully-initialised CPU model and distributes parameters
  to other ranks via point-to-point sends, avoiding N redundant initialisations.
* All ranks receive the same batch (pure model parallelism); rank 0 loads data
  via ``DataloaderDispatcher`` and broadcasts it.
* Gradient norm is all-reduced across ranks because each rank holds only a
  subset of the model's parameters.
* Effective batch size does **not** scale with ``num_processes`` (the same
  batch flows through all stages; unlike DDP, there is no data replication).

**Parameters**

- `args` (PipelineTrainingArguments or dict) — Pipeline training configuration. Dicts are converted via
``dacite.from_dict``.
- `model_splitter` (ModelSplitter) — Callable that splits the model into pipeline stages and returns
``PipelineStage`` objects. See
``src/forgather/ml/trainer/pipeline/model_splitter.py`` for the
expected signature.
- `pipe_schedule_factory` (callable) — Factory for the pipeline scheduler (e.g. ``ScheduleGPipe``,
``ScheduleZBVZeroBubble``). Default is ``ScheduleGPipe``.
- `**kwargs` — Additional arguments forwarded to the base ``Trainer``
(``model_init``, ``train_dataset``, ``optimizer_factory``, etc.).

**Raises**

- `AssertionError` — If ``model`` is provided (pipeline training requires ``model_init``).
- `AssertionError` — If ``model_init`` is not provided.
- `AssertionError` — If batch size is not divisible by ``n_microbatches``.
- `AssertionError` — If ``stages_per_rank > 1`` but ``is_multistage=False``.
- `AssertionError` — If ``mixed_precision="fp16"`` (incompatible with pipeline schedulers).
- `AssertionError` — If a zero-bubble schedule is used with ``torch_compile=True``.
- `AssertionError` — If ``world_size == 1`` (pipeline parallelism requires multiple ranks).

**Examples**

```python
>>> from torch.distributed.pipelining import ScheduleGPipe
>>> args = PipelineTrainingArguments(
...     n_microbatches=8,
...     per_device_train_batch_size=64,
...     stages_per_rank=1,
... )
>>> trainer = PipelineTrainer(
...     args=args,
...     model_init=model_factory,
...     model_splitter=my_splitter_fn,
...     pipe_schedule_factory=ScheduleGPipe,
...     train_dataset=train_dataset,
...     optimizer_factory=optimizer_factory,
... )
>>> trainer.train()
```

> **See-Also**
>
> ModelSplitter : Protocol for the model-splitting callable.

> **References**
>
> PyTorch pipeline parallelism:
> https://docs.pytorch.org/docs/stable/distributed.pipelining.html

**Attributes**

- `args` (TPipelineTrainingArguments)
- `model_splitter` (ModelSplitter)
- `pipe_schedule_factory` (PipelineSchedulerFactorT)
- `pp_group` (Any)
- `n_pipeline_stages` (int)
- `scheduler` (PipelineSchedulerT | None)
- `pipeline_modules` (List[Module] | None)
- `sharing_metadata` (SharingMetadataT | None)
- `shard_index` (ShardIndex | None)
- `stage_indices` (Tuple[int, ...] | None)
- `pp_has_last_stage` (bool)
- `pp_has_first_stage` (bool)
- `attention_mask_creator` (Callable)

**Methods**

#### `pipeline_generate` {#forgather-ml-trainer-pipeline-pipeline_trainer-pipelinetrainer-pipeline_generate}

```python
def pipeline_generate(input_ids: Tensor, max_new_tokens: int, eos_token_id: int, pad_token_id: int, do_sample: bool = True, temperature: float = 1.0, top_k: int = 0, repetition_penalty: float = 1.0)
```

Generate text autoregressively through all pipeline stages.

Bypasses the pipeline scheduler so input shapes are not constrained to
the fixed training batch dimensions. All ranks must call this method
simultaneously. The full generated sequence (prompt + new tokens) is
returned on every rank.

No KV caching is used; each decoding step reprocesses the entire
sequence. This is acceptable for infrequent, qualitative generation
checks (e.g. during a callback).

**Parameters**

- `input_ids` (Tensor) — Prompt token ids of shape ``[batch, prompt_len]``, same on all
ranks.
- `max_new_tokens` (int) — Maximum number of new tokens to generate.
- `eos_token_id` (int) — Token id that signals end of sequence. Once all sequences in the
batch have emitted this token, generation stops early.
- `pad_token_id` (int) — Token id used to pad sequences that have already finished.
- `do_sample` (bool) — If ``True``, sample from the probability distribution; if
``False``, use greedy (argmax) decoding. Default is ``True``.
- `temperature` (float) — Softmax temperature applied before top-k filtering. Values ``< 1``
sharpen the distribution; values ``> 1`` flatten it.
Default is ``1.0``.
- `top_k` (int) — When ``> 0``, restrict sampling to the top-k logits. ``0`` uses
the full vocabulary. Default is ``0``.
- `repetition_penalty` (float) — Multiplicative penalty applied to logits of tokens already present
in the sequence. ``1.0`` disables the penalty. Default is ``1.0``.

**Returns**

- `Tensor` — Generated token ids of shape ``[batch, prompt_len + n_new_tokens]``
as a ``LongTensor`` on the current device, identical on all ranks.

#### `get_state_components` {#forgather-ml-trainer-pipeline-pipeline_trainer-pipelinetrainer-get_state_components}

```python
def get_state_components()
```

Return state components for pipeline parallel training.

Because the model is split across ranks, each rank saves only its own
stage parameters. The sharing patterns reflect this:

* ``"model"`` — PER_RANK (each rank holds different stages), required.
* ``"optimizer"`` — PER_RANK (optimises different parameters), optional.
* ``"scheduler"`` — REPLICATED (same LR schedule on all ranks), optional.
* ``"trainer"`` — REPLICATED (same global step on all ranks), optional.
* ``"dataset"`` — GLOBAL (``DataloaderDispatcher`` with
  ``dp_mesh_dim=None``; rank 0 loads and broadcasts), optional.
* ``"rng"`` — PER_RANK (each stage may have different dropout), optional.

**Returns**

- `list of StateComponent` — All checkpointable state components with their sharing patterns.

#### `get_process_groups` {#forgather-ml-trainer-pipeline-pipeline_trainer-pipelinetrainer-get_process_groups}

```python
def get_process_groups()
```

Return named process groups for checkpoint coordination.

The checkpoint manager uses this mapping to implement ``PER_GROUP``
sharing patterns (e.g. saving one copy per pipeline-parallel group).

**Returns**

- `dict of str to ProcessGroup` — ``{"pp_group": self.pp_group}`` for pure pipeline parallelism.

---

### `PipelineTrainingArguments` {#forgather-ml-trainer-pipeline-pipeline_trainer-pipelinetrainingarguments}

`forgather.ml.trainer.pipeline.pipeline_trainer.PipelineTrainingArguments`

```python
class PipelineTrainingArguments(*, output_dir: str = OUTPUTDIR_NAME, logging_dir: str | None = None, per_device_eval_batch_size: int = 16, per_device_train_batch_size: int = 16, num_train_epochs: int = 1, device: Any = None, seed: int = -1, use_cpu: bool = False, epoch_train_steps: int = 100000, max_steps: int = -1, dataloader_num_workers: int = 0, dataloader_pin_memory: int = True, dataloader_persistent_workers: bool = False, dataloader_prefetch_factor: int | None = None, dataloader_drop_last: bool = False, eval_strategy: str = 'no', eval_steps: int = 100, eval_delay: int = 0, logging_strategy: str = 'steps', logging_steps: int = 50, logging_first_step: bool = False, torch_compile: bool = False, torch_compile_backend: str | None = None, torch_compile_mode: str | None = 'default', torch_compile_dynamic: bool = True, torch_compile_full_graph: bool = False, max_grad_norm: float | None = None, gradient_accumulation_steps: int = 1, save_strategy: str = 'steps', save_steps: int = 1000, save_total_limit: int = 2, save_safetensors: bool = True, save_on_each_node: bool = False, overwrite_output_dir: bool = False, resume_from_checkpoint: bool | str = True, load_best_model_at_end: bool = False, metric_for_best_model: str = 'loss', greater_is_better: bool | None = None, lr_scheduler_type: str = 'linear', lr_scheduler_kwargs: dict | None = None, warmup_steps: int = 0, learning_rate: float = 5e-05, weight_decay: float = 0.0, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, gradient_checkpointing: bool = False, default_dtype: str | None = None, max_eval_steps: int = -1, preserve_best_model: bool = False, best_model_metric: str = 'loss', best_model_greater_is_better: bool | None = None, preserve_n_best: int = 1, eval_on_save: bool = False, enable_activation_offloading: bool = False, detect_anomaly: bool = False, sdpa_backend: List[str] | str | None = None, sdpa_set_priority: bool = False, float32_matmul_precision: str | None = None, dynamo_recompile_limit: int | None = None, mixed_precision: str | None = None, fp8_recipe: str | None = None, fp8_dim_alignment: int = 16, qat_recipe: str | None = None, gc_threshold: float = 0.5, construct_model_on: str = 'default', activation_memory_budget: float | None = None, fuse_optim_with_backward: bool = False, speed_metrics_start_step: int = 1, set_dataset_epoch: bool = True, debug_optimizer_groups: bool = False, debug_pipeline: bool = False, debug_split_model: bool = False, debug_model_params: bool = False, debug_model_init: bool = False, n_microbatches: int = 4, stages_per_rank: int = 1, pp_stage_type: str = 'loop', is_multistage: bool = False)
```

Training arguments for pipeline parallel training.

Pipeline parallelism partitions a model across multiple GPUs, each handling
one or more sequential stages. Input batches are split into microbatches that
flow through the stages, allowing overlapped computation to keep all GPUs busy.

See the PyTorch pipeline parallelism documentation for background:
https://docs.pytorch.org/docs/stable/distributed.pipelining.html

**Parameters**

- `n_microbatches` (int) — Number of microbatches to split each batch into. More microbatches
improve pipeline efficiency (fewer bubbles) but increase memory usage.
The batch size must be evenly divisible by ``n_microbatches``. Typical
values are 4–16 depending on pipeline depth and memory constraints.
Default is ``4``.
- `stages_per_rank` (int) — Number of pipeline stages hosted on each GPU. Most schedulers use
``1``. Multi-stage schedulers (e.g. ``ScheduleZBVZeroBubble``) assign
multiple stages per rank to reduce pipeline bubbles. Only set ``> 1``
together with ``is_multistage=True``. Default is ``1``.
- `pp_stage_type` (str) — Stage-to-rank assignment pattern. ``"loop"`` uses round-robin (e.g. 4
stages on 2 ranks: rank0=[0,2], rank1=[1,3]). ``"v"`` uses the
V-pattern required by ZeroBubble schedulers (see
https://arxiv.org/pdf/2401.10241). Default is ``"loop"``.
- `is_multistage` (bool) — Set ``True`` when the scheduler inherits from
``PipelineScheduleMulti`` (e.g. ``ScheduleZBVZeroBubble``). Leave
``False`` for single-stage schedulers such as ``ScheduleGPipe``.
Default is ``False``.
- `debug_pipeline` (bool) — Enable debug-level logging for the pipeline scheduler. Internal
development flag. Default is ``False``.
- `debug_split_model` (bool) — Log pipeline module details after splitting. Internal development
flag. Default is ``False``.
- `debug_model_params` (bool) — Log all parameter and buffer devices/dtypes after model construction.
Internal development flag. Default is ``False``.
- `debug_model_init` (bool) — Log every send/recv during parameter distribution from rank 0.
Internal development flag. Default is ``False``.

> **Note**
>
> ``model_splitter`` is passed to ``PipelineTrainer.__init__()`` rather than
> stored here because it is a callable, not a primitive serialisable type.

**Attributes**

- `debug_pipeline` (bool)
- `debug_split_model` (bool)
- `debug_model_params` (bool)
- `debug_model_init` (bool)
- `n_microbatches` (int)
- `stages_per_rank` (int)
- `pp_stage_type` (str)
- `is_multistage` (bool)
