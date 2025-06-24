# Trainer API Reference

Complete API documentation for Forgather ML trainer classes.

## Base Classes

### AbstractBaseTrainer

**Location**: `forgather.ml.trainer_types.AbstractBaseTrainer`

Minimal abstract base class defining the core trainer interface.

#### Methods

##### `train(**kwargs) -> TrainOutput`
Main training entry point.

**Parameters:**
- `**kwargs`: Additional training parameters

**Returns:**
- `TrainOutput`: Named tuple with `(global_step, training_loss, metrics)`

##### `evaluate(eval_dataset: Optional[Dataset] = None, **kwargs) -> dict[str, float]`
Perform model evaluation.

**Parameters:**
- `eval_dataset`: Optional evaluation dataset (uses default if None)
- `**kwargs`: Additional evaluation parameters

**Returns:**
- `dict[str, float]`: Evaluation metrics

##### `save_model(output_dir: Optional[os.PathLike | str] = None) -> None`
Save model to specified directory.

**Parameters:**
- `output_dir`: Output directory (uses default if None)

### ExtensibleTrainer

**Location**: `forgather.ml.trainer_types.ExtensibleTrainer`

Extends `AbstractBaseTrainer` with callback system support.

#### Methods

##### `add_callback(callback)`
Add a callback to the trainer.

**Parameters:**
- `callback`: Callback instance or callback class

**Example:**
```python
trainer.add_callback(TBLogger(summary_writer))
trainer.add_callback(ProgressCallback)  # Class will be instantiated
```

##### `pop_callback(callback)`
Remove and return a callback.

**Parameters:**
- `callback`: Callback instance or callback class to remove

**Returns:**
- Removed callback instance or None if not found

##### `remove_callback(callback)`
Remove a callback without returning it.

**Parameters:**
- `callback`: Callback instance or callback class to remove

### BaseTrainer

**Location**: `forgather.ml.base_trainer.BaseTrainer`

Abstract base implementation providing common trainer functionality.

#### Constructor

```python
BaseTrainer(
    model: PreTrainedModel | torch.nn.Module = None,
    args: Optional[dict | TrainingArguments] = None,
    data_collator=None,
    train_dataset=None,
    eval_dataset=None,
    processing_class=None,  # Tokenizer or processor
    model_init: Optional[Callable[[], PreTrainedModel]] = None,
    callbacks: List = None,
    tokenizer=None,  # Deprecated: use processing_class
)
```

**Parameters:**
- `model`: Model instance (required if `model_init` not provided)
- `args`: Training arguments (dict or TrainingArguments instance)
- `data_collator`: Function to collate batch data
- `train_dataset`: Training dataset
- `eval_dataset`: Evaluation dataset  
- `processing_class`: Tokenizer or data processor
- `model_init`: Factory function returning model instance
- `callbacks`: List of callback instances
- `tokenizer`: **Deprecated** - use `processing_class`

#### Key Methods

##### `unwrapped_model()`
Return the base model, unwrapping any distributed wrappers.

**Returns:**
- Base model instance

##### `model_exists(output_dir) -> bool`
Check if a saved model exists in the output directory.

**Parameters:**
- `output_dir`: Directory to check

**Returns:**
- `bool`: True if model files found

##### `log(logs: Dict[str, float])`
Log metrics and dispatch to callbacks.

**Parameters:**
- `logs`: Dictionary of metric names and values

#### Checkpoint Methods

##### `_save_checkpoint() -> str`
Save checkpoint with model weights and training state.

**Returns:**
- `str`: Path to saved checkpoint

##### `_resolve_checkpoint_path() -> str | None`
Resolve checkpoint path for resuming training.

**Returns:**
- `str | None`: Path to checkpoint or None if not resuming

##### `_find_latest_checkpoint(checkpoints_dir: str = None) -> str | None`
Find most recent valid checkpoint by modification time.

**Parameters:**
- `checkpoints_dir`: Directory to search (default: `{output_dir}/checkpoints`)

**Returns:**
- `str | None`: Path to latest checkpoint

##### `_validate_checkpoint(checkpoint_path: str) -> bool`
Validate that checkpoint contains necessary files.

**Parameters:**
- `checkpoint_path`: Path to checkpoint directory

**Returns:**
- `bool`: True if checkpoint is valid

## Concrete Trainer Implementations

### Trainer

**Location**: `forgather.ml.trainer.Trainer`

Lightweight trainer for single GPU/CPU training.

#### Constructor

```python
Trainer(
    # Inherits all BaseTrainer parameters, plus:
    optimizer_factory: Optional[Callable] = None,
    optimizer_cls_and_kwargs: Optional[Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]] = None,
    lr_scheduler_factory: Optional[Callable] = None,
    **kwargs
)
```

**Additional Parameters:**
- `optimizer_factory`: Function returning optimizer instance
- `optimizer_cls_and_kwargs`: Alternative optimizer specification (HF compatibility)
- `lr_scheduler_factory`: Function returning scheduler instance

#### Key Features

- **Torch Compile Support**: Automatic model compilation when enabled
- **Periodic Operations**: Configurable logging, evaluation, and saving
- **Progress Tracking**: Built-in progress bars and info display
- **Device Management**: Automatic device selection and data movement

#### Training Arguments

Uses `TrainingArguments` class with these key options:

```python
TrainingArguments(
    output_dir: str = "tmp_trainer",
    logging_dir: str = None,  # Auto-generated if None
    
    # Training configuration
    num_train_epochs: int = 1,
    max_steps: int = -1,  # -1 means use epochs
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    
    # Logging and evaluation
    logging_steps: int = 500,
    logging_strategy: str = "steps",  # "steps" | "epoch" | "no"
    eval_steps: int = 500,
    eval_strategy: str = "no",  # "steps" | "epoch" | "no"
    
    # Saving and checkpoints
    save_steps: int = 500,
    save_strategy: str = "steps",  # "steps" | "epoch" | "no"
    save_total_limit: int = 2,
    resume_from_checkpoint: bool | str = False,
    
    # Checkpoint state management
    save_optimizer_state: bool = False,
    save_scheduler_state: bool = False,
    restore_optimizer_state: bool = True,
    restore_scheduler_state: bool = True,
    
    # Performance
    torch_compile: bool = False,
    torch_compile_backend: str = "inductor",
    torch_compile_mode: str = None,
    
    # Data loading
    dataloader_num_workers: int = 0,
    dataloader_pin_memory: bool = True,
    dataloader_drop_last: bool = False,
    
    # Optimization (HF compatibility)
    learning_rate: float = 5e-5,
    weight_decay: float = 0.0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    warmup_steps: int = 0,
    lr_scheduler_type: str = "linear",
)
```

### AccelTrainer

**Location**: `forgather.ml.accel_trainer.AccelTrainer`

Multi-GPU trainer using HuggingFace Accelerate.

#### Constructor

```python
AccelTrainer(
    # Inherits all BaseTrainer parameters, plus:
    accelerator: Optional[Accelerator] = None,
    **kwargs
)
```

**Additional Parameters:**
- `accelerator`: Accelerate instance (auto-created if None)

#### Training Arguments

Uses `AccelTrainingArguments` which extends `TrainingArguments`:

```python
AccelTrainingArguments(
    # All TrainingArguments options, plus:
    accelerator_args: dict = None,  # Accelerator configuration
)
```

#### Key Features

- **Automatic Multi-GPU**: Handles device placement and data distribution
- **Gradient Synchronization**: Automatic gradient accumulation and sync
- **Mixed Precision**: Support for FP16/BF16 training
- **Distributed Checkpoints**: Accelerate-native checkpoint format

### PipelineTrainer

**Location**: `forgather.ml.pipeline_trainer.PipelineTrainer`

Pipeline parallel trainer for very large models.

#### Constructor

```python
PipelineTrainer(
    # Inherits all BaseTrainer parameters, plus:
    distributed_env: DistributedEnvironment,
    loss_fn: Callable,
    pipe_schedule_factory: Callable = None,
    **kwargs
)
```

**Required Parameters:**
- `distributed_env`: Distributed training environment
- `loss_fn`: Loss function for pipeline training

**Optional Parameters:**
- `pipe_schedule_factory`: Pipeline scheduling strategy factory

#### Training Arguments

Uses `PipelineTrainingArguments` which extends `TrainingArguments`:

```python
PipelineTrainingArguments(
    # All TrainingArguments options, plus:
    
    # Pipeline configuration
    split_spec: dict,              # Model partitioning specification
    pipeline_chunks: int = 1,      # Number of pipeline micro-batches
    stages_per_rank: int = 1,      # Pipeline stages per rank
    
    # Scheduling
    schedule_class: str = "ScheduleGPipe",  # Pipeline schedule
    
    # Memory management
    checkpoint_sequential: bool = False,     # Sequential checkpointing
    
    # Distributed settings
    pipeline_parallel_size: int = None,     # Auto-detected if None
)
```

#### Key Features

- **Model Partitioning**: Automatic model splitting across devices/nodes
- **Pipeline Scheduling**: Multiple strategies (GPipe, 1F1B, etc.)
- **Memory Optimization**: Gradient checkpointing and activation offloading
- **Sharded Checkpoints**: Distributed checkpoint saving/loading

## TrainerState and TrainerControl

### TrainerState

Tracks training progress and metrics:

```python
TrainerState(
    epoch: float = 0.0,
    global_step: int = 0,
    max_steps: int,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    train_batch_size: int,
    num_train_epochs: int,
    log_history: List[Dict[str, float]] = [],
    is_local_process_zero: bool = True,
    is_world_process_zero: bool = True,
)
```

### TrainerControl

Controls training flow:

```python
TrainerControl(
    should_training_stop: bool = False,
    should_epoch_stop: bool = False,
    should_save: bool = False,
    should_evaluate: bool = False,
    should_log: bool = False,
)
```

## Factory Functions and Configuration

### Optimizer Factory

```python
def optimizer_factory(named_parameters):
    return torch.optim.AdamW(
        named_parameters(),
        lr=1e-3,
        weight_decay=0.01
    )

trainer = Trainer(
    optimizer_factory=optimizer_factory,
    # ...
)
```

### LR Scheduler Factory

```python
def lr_scheduler_factory(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1000
    )

trainer = Trainer(
    lr_scheduler_factory=lr_scheduler_factory,
    # ...
)
```

## Error Handling and Validation

### Common Exceptions

- `AssertionError`: Missing required parameters (model or model_init)
- `FileNotFoundError`: Invalid output directories or checkpoint paths
- `ValueError`: Invalid configuration combinations

### Validation

Trainers perform automatic validation of:
- Output directory accessibility
- Model and dataset compatibility
- Checkpoint file integrity
- Configuration consistency

## Performance Considerations

### Memory Usage

- **Trainer**: Minimal memory overhead, single device
- **AccelTrainer**: Memory distributed across devices
- **PipelineTrainer**: Model partitioned, lowest per-device memory

### Throughput

- **Trainer**: Highest single-device throughput
- **AccelTrainer**: Best data parallel scaling
- **PipelineTrainer**: Best for memory-constrained scenarios

### Checkpoint Size

- Model weights: Same across all trainers
- Training state: Varies by trainer complexity
- Sharded checkpoints: Available for PipelineTrainer