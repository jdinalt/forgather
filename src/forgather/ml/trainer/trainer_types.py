import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pprint import pformat
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    TypeAlias,
    Union,
)

from torch import Tensor, nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import Dataset

from ..utils import ConversionDescriptor, DiagnosticEnum

OUTPUTDIR_NAME = "tmp_trainer"

ArgsValueType: TypeAlias = Union[
    Dict[str, "ArgsValueType"],
    List["ArgsValueType"],
    Tuple["ArgsValueType"],
    str,
    int,
    float,
    None,
]

# The type of 'args' past to a trainer only allows for primitive types
ArgsType = Dict[str, ArgsValueType]


class BaseDataset(Protocol):
    def __len__(self):
        pass

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass

    def state_dict(self) -> Dict[str, Any]:
        pass


class IterableDatasetT(BaseDataset):
    def __iter__(self):
        pass


class DatasetT(BaseDataset):
    def __getitem__(self, key: int):
        pass

    def __iter__(self):
        pass


class TrainOutput(NamedTuple):
    global_step: int
    metrics: Dict[str, float]


class OptimizerT(Protocol):
    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass

    def state_dict(self) -> Dict[str, Any]:
        pass

    def step(self):
        pass

    def zero_grad(self):
        pass


class LRSchedulerT(Protocol):
    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass

    def state_dict(self) -> Dict[str, Any]:
        pass

    def step(self):
        pass

    def get_lr(self) -> float:
        pass

    def get_last_lr(self) -> float:
        pass


DataCollatorT: TypeAlias = Callable[[List[Dict[str, Any]]], Dict[str, Any]]

LossFunctionT: TypeAlias = Callable[[Tensor, Tensor], Tensor]

OptimizerParamsT: TypeAlias = Union[
    Iterable[Tensor], Iterable[dict[str, Any]], Iterable[tuple[str, Tensor]]
]

OptimizerFactoryT: TypeAlias = Callable[[OptimizerParamsT], OptimizerT]

LRSchedulerFactoryT: TypeAlias = Callable[[OptimizerT], LRSchedulerT]

FusedLossFactoryT: TypeAlias = Callable[[nn.Module], LossFunctionT]

PreprocessingClassT: TypeAlias = Callable

EnableCheckpointFnT: TypeAlias = Callable[[], None]


class IntervalStrategy(DiagnosticEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


@dataclass(kw_only=True)
class TrainerState:
    """
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
    """

    logging_steps: int  # How often to log metrics (in steps)
    eval_steps: int  # How often to run evaluation (in steps)
    train_batch_size: int  # Per-device training batch size
    max_steps: int  # Total optimizer updates planned
    epoch: float = 0.0  # Continuous epoch value (integer + fractional progress)
    global_step: int = 0  # Total optimizer updates completed (0-indexed)
    num_train_epochs: int  # Total epochs to train
    is_local_process_zero: bool = True  # True if rank 0 on this node
    is_world_process_zero: bool = True  # True if global rank 0
    log_history: list[Dict[str, float]] = field(
        default_factory=lambda: []
    )  # All logged metrics
    save_steps: int = 0  # How often to save checkpoints (in steps)
    best_metric: float | None = None  # Best metric value (for load_best_model_at_end)
    best_model_checkpoint: str | None = None  # Path to best checkpoint
    # HF compatibility fields (not fully implemented in all trainers)
    num_input_tokens_seen: int = 0  # Total input tokens processed
    total_flos: float = 0.0  # Total floating point operations
    is_hyper_param_search: bool = False  # Whether in hyperparameter search
    stateful_callbacks: List["TrainerCallback"] = field(default_factory=lambda: [])

    # Forgather extensions (not in HF Trainer)
    max_eval_steps: int  # Maximum eval steps to run (-1 for unlimited)
    epoch_start_step: int = 0  # Global step when current epoch started
    raw_epoch: int = 0  # Integer epoch counter (increments at end of dataset iteration)


@dataclass(slots=True)
class TrainerControl:
    """
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
    """

    should_training_stop: bool = False  # Stop training loop after current step
    should_epoch_stop: bool = False  # Stop current epoch after current step
    should_save: bool = False  # Trigger checkpoint save
    should_evaluate: bool = False  # Trigger evaluation
    should_log: bool = False  # Trigger metric logging

    # Forgather extension: abort without saving checkpoint
    should_abort_without_save: bool = False  # Abort training immediately without saving


@dataclass(kw_only=True)
class MinimalTrainingArguments:
    """
    Minimal training configuration compatible with HuggingFace Trainer.

    Provides a subset of transformers.TrainingArguments sufficient for basic training.
    This is the base configuration class - extend it for additional features rather
    than adding everything here.

    Args:
        output_dir: Directory where model predictions and checkpoints are written.
        logging_dir: TensorBoard log directory. Defaults to output_dir/runs/TIMESTAMP_HOSTNAME.
        per_device_train_batch_size: Training batch size per device. Global batch size is
            per_device_train_batch_size * num_devices * gradient_accumulation_steps.
        per_device_eval_batch_size: Evaluation batch size per device.
        num_train_epochs: Total training epochs. Can be fractional (e.g., 2.5 trains 2.5 epochs).
        max_steps: If > 0, total training steps to perform (overrides num_train_epochs).
        device: Device to use (cuda, cpu, etc.). Auto-detected if None.
        seed: Random seed for reproducibility. Use with model_init for full reproducibility.
        use_cpu: Force CPU usage even if CUDA available.

        epoch_train_steps: Fallback epoch length when dataset doesn't support len() (Forgather extension).

        dataloader_num_workers: Number of subprocesses for data loading. 0 = load in main process.
        dataloader_pin_memory: Pin memory in DataLoader for faster GPU transfer.
        dataloader_persistent_workers: Keep worker processes alive between epochs (speeds up training, uses more RAM).
        dataloader_prefetch_factor: Batches prefetched per worker. Defaults to 2 if num_workers > 0.
        dataloader_drop_last: Drop last incomplete batch if dataset size not divisible by batch size.

        eval_strategy: When to run evaluation: "no", "steps" (every eval_steps), or "epoch".
        eval_steps: Evaluation frequency in steps (if eval_strategy="steps").
        eval_delay: Number of epochs/steps to wait before first evaluation.

        logging_strategy: When to log metrics: "no", "steps" (every logging_steps), or "epoch".
        logging_steps: Logging frequency in steps (if logging_strategy="steps").
        logging_first_step: Whether to log the very first global_step.

        torch_compile: Compile model using PyTorch 2.0 torch.compile() for speedup.
        torch_compile_backend: Backend for torch.compile (e.g., "inductor", "aot_eager").
        torch_compile_mode: Compilation mode: "default", "reduce-overhead", or "max-autotune".
        torch_compile_dynamic: Allow dynamic shapes in compiled model.
        torch_compile_full_graph: Force compilation of entire model as single graph.

        max_grad_norm: Maximum gradient norm for gradient clipping (None = no clipping).
        gradient_accumulation_steps: Accumulate gradients over N steps before optimizer update.
            Effective batch size = per_device_batch * num_devices * gradient_accumulation_steps.

        save_strategy: Checkpoint save strategy: "no", "steps" (every save_steps), or "epoch".
        save_steps: Checkpoint save frequency in steps (if save_strategy="steps").
        save_total_limit: Max checkpoints to keep. Deletes oldest, but keeps best if load_best_model_at_end=True.
        save_safetensors: Use safetensors format instead of pickle (safer, compatible).
        save_on_each_node: In multi-node training, save on each node (not just main). Don't use with shared storage.
        overwrite_output_dir: Overwrite output_dir contents (use to continue from checkpoint in that dir).
        resume_from_checkpoint: Path to checkpoint to resume from, or True to auto-find latest.

        load_best_model_at_end: Load best checkpoint at end of training (requires save_strategy == eval_strategy).
        metric_for_best_model: Metric to compare models when load_best_model_at_end=True. Defaults to "loss".
        greater_is_better: True if higher metric is better. Auto-determined from metric name if None.

        lr_scheduler_type: LR scheduler type: "linear", "cosine", "polynomial", etc.
        lr_scheduler_kwargs: Additional kwargs passed to LR scheduler.
        warmup_steps: Linear warmup steps from 0 to learning_rate.
        learning_rate: Initial learning rate for AdamW optimizer.
        weight_decay: Weight decay for AdamW (applied to all layers except bias and LayerNorm).
        adam_beta1: Beta1 hyperparameter for AdamW.
        adam_beta2: Beta2 hyperparameter for AdamW.
        adam_epsilon: Epsilon hyperparameter for AdamW.

        gradient_checkpointing: Enable activation checkpointing to trade compute for memory.
            Note: Pass enable_activation_checkpoint_fn to Trainer constructor to customize behavior.

    For detailed HF Trainer documentation, see:
    ~/fg/lib/python3.12/site-packages/transformers/training_args.py:214

    Subclasses:
    - BaseTrainingArguments: Adds checkpoint control and PyTorch optimizations
    - TrainingArguments: Adds memory optimizations specific to simple Trainer
    """

    output_dir: str = OUTPUTDIR_NAME
    logging_dir: str | None = None
    per_device_eval_batch_size: int = 16
    per_device_train_batch_size: int = 16
    num_train_epochs: int = 1
    device: Any = None

    seed: int = -1
    use_cpu: bool = False

    # Not in HF trainer; number of train-batches in an epoch, when dataset does not support len()
    # This just becomes a relative value for book-keeping.
    epoch_train_steps: int = 100000
    max_steps: int = -1

    dataloader_num_workers: int = 0
    dataloader_pin_memory: int = True
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: int | None = None
    dataloader_drop_last: bool = False

    # Strategy may also be: "no" | "steps" | "epoch"
    eval_strategy: str = "no"
    eval_steps: int = 100
    eval_delay: int = 0

    logging_strategy: str = "steps"
    logging_steps: int = 50
    logging_first_step: bool = False

    torch_compile: bool = False
    torch_compile_backend: str | None = None
    torch_compile_mode: str | None = "default"
    torch_compile_dynamic: bool = True
    torch_compile_full_graph: bool = False

    max_grad_norm: float | None = None
    gradient_accumulation_steps: int = 1

    # Checkpointing options
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 2
    save_safetensors: bool = True
    save_on_each_node: bool = False
    overwrite_output_dir: bool = False
    resume_from_checkpoint: bool | str = False

    # Best model tracking and loading options
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "loss"
    greater_is_better: bool | None = None  # Auto-determined from metric name

    # Compatibility with HF Trainer -- would be better if they took a factory arg...
    lr_scheduler_type: str = "linear"
    lr_scheduler_kwargs: dict | None = None
    warmup_steps: int = 0
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8

    # Enable gradient checkpointing (a.k.a activation checkpointing) on models which support the HF API
    gradient_checkpointing: bool = False

    def __str__(self):
        return pformat(self)


class AbstractBaseTrainer(Protocol):
    """
    Minimal trainer interface based on HuggingFace Trainer API.

    Defines the core methods that any trainer must implement:
    - train(): Execute training loop
    - evaluate(): Run evaluation
    - save_model(): Save model weights
    - save_checkpoint(): Save complete training state
    - load_checkpoint(): Restore training state

    Kept minimal by design - specialized trainers add additional capabilities
    through subclassing rather than bloating this interface.

    Based on HF Trainer API for easier porting of existing code.
    """

    @abstractmethod
    def train(self, **kwargs) -> TrainOutput:
        pass

    @abstractmethod
    def evaluate(
        self, eval_dataset: Optional[BaseDataset] = None, **kwargs
    ) -> dict[str, float]:
        """
        Perform evaluation, either from the default eval dataset or from a specified dataset.

        Returns: A dictionary of metrics.
        """
        pass

    @abstractmethod
    def save_model(self, output_dir: Optional[os.PathLike | str] = None) -> None:
        """
        Save the model, either to the default location or to the specified location.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint_path=None) -> None:
        """
        Save model / trainer checkpoint
        """
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path=None) -> None:
        """
        Load model / trainer checkpoint
        """
        pass


class TrainerCallback(Protocol):
    """
    Protocol for trainer event callbacks.

    Callbacks receive notifications at key points during training and can modify
    trainer behavior by returning an updated TrainerControl.

    All methods are optional - implement only the events you need.

    Common event hooks:
    - on_init_end: After trainer initialization
    - on_train_begin/on_train_end: Training loop boundaries
    - on_epoch_begin/on_epoch_end: Epoch boundaries
    - on_step_begin/on_step_end: Training step boundaries
    - on_log: After logging metrics (use for TensorBoard, wandb, etc.)
    - on_evaluate: After evaluation (access metrics via kwargs)
    - on_save: After checkpoint save
    - on_optimizer_step: After optimizer update

    Each method receives:
    - args: Training arguments
    - state: Current trainer state (global_step, epoch, log_history, etc.)
    - control: Control flags (can modify to influence trainer)
    - **kwargs: Additional context (model, metrics, logs, etc.)

    Returns:
    - None or updated TrainerControl to modify trainer behavior

    Compatible with HuggingFace TrainerCallback for easier porting.
    See: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py
    """

    def on_init_end(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_train_begin(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_train_end(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_epoch_begin(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_epoch_end(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_step_begin(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_optimizer_step(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_substep_end(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_step_end(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_evaluate(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_predict(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_save(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_log(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_prediction_step(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass

    def on_pre_optimizer_step(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        pass


class ExtensibleTrainer(AbstractBaseTrainer):
    """
    Trainer interface extended with callback support.

    Adds callback management methods to AbstractBaseTrainer, enabling
    extensibility through the TrainerCallback system.

    Callbacks allow hooking into training events (on_step_end, on_epoch_begin, etc.)
    without modifying trainer code. Common uses:
    - Custom logging (TensorBoard, wandb, MLflow)
    - Early stopping based on metrics
    - Learning rate scheduling
    - Progress bars and notifications

    Compatible with HuggingFace TrainerCallback API.
    """

    @abstractmethod
    def add_callback(self, callback: TrainerCallback):
        """
        Add callback to the list of callbacks
        Either a type (instantiate it) or an instance
        """
        pass

    @abstractmethod
    def pop_callback(self, callback: TrainerCallback) -> TrainerCallback | None:
        """
        Callback may either be and instance or a type
        Remove the first match and return it
        """
        pass

    @abstractmethod
    def remove_callback(self, callback: TrainerCallback):
        """
        Like pop, but don't return it.
        This seems redundant, but API consistency...
        """
        pass


class CheckpointInterface(Protocol):
    """
    Protocol for checkpoint management.

    Defines interface for saving/loading complete training state (model, optimizer,
    scheduler, dataset position, RNG state, etc.) and standalone model weights.

    Implementations:
    - CheckpointManager: Standard implementation in src/forgather/ml/trainer/checkpoint_manager.py

    Key responsibilities:
    - Save complete training checkpoints with versioning and limits
    - Load checkpoints for resuming training
    - Track best checkpoint (for load_best_model_at_end)
    - Save standalone model weights (HF Trainer compatibility)
    """

    @abstractmethod
    def save_checkpoint(
        self,
        checkpoint_path: str | None = None,
        checkpoint_id: str | None = None,
    ) -> str:
        """
        Save complete training checkpoint.

        Args:
            checkpoint_path: Specific path for checkpoint, or None for auto-generated
            checkpoint_id: Identifier for checkpoint (e.g., global_step), used if path is None

        Returns:
            Path to saved checkpoint directory
        """
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str | None = None) -> None:
        """
        Load checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint, or None to load latest checkpoint
        """
        pass

    @abstractmethod
    def save_model(
        self,
        output_dir: str | os.PathLike | None = None,
        overwrite_output_dir: bool = False,
    ) -> None:
        """
        Save only model weights (not full training state).

        Args:
            output_dir: Directory to save model, or None for default
            overwrite_output_dir: Whether to overwrite existing model
        """
        pass

    @abstractmethod
    def set_best_checkpoint(self, best_checkpoint: str) -> None:
        """
        Mark a checkpoint as the best model.

        Args:
            best_checkpoint: Path to checkpoint to mark as best
        """
        pass

    @abstractmethod
    def resolve_checkpoint_path(self, checkpoint_path: str | None) -> str | None:
        """
        Resolve checkpoint path (e.g., find latest if path is None).

        Args:
            checkpoint_path: Explicit path or None for auto-resolution

        Returns:
            Resolved checkpoint path or None if not found
        """
        pass


class StatefulProvider(Protocol):
    """
    Protocol for providing stateful objects for checkpointing.

    Used by checkpoint managers to collect all components that need to be
    saved/restored during checkpointing (optimizer, scheduler, dataset, etc.).

    The protocol uses StateComponents which declare explicit sharing patterns
    (GLOBAL, PER_RANK, REPLICATED, etc.) to enable automatic distributed
    checkpoint coordination for hybrid parallelism strategies.

    All implementations must provide:
    - get_state_components(): Returns list of StateComponents with sharing patterns
    - get_process_groups(): Returns named process groups (only if using PER_GROUP pattern)
    """

    @abstractmethod
    def get_state_components(self) -> List["StateComponent"]:  # type: ignore
        """
        Get state components with explicit sharing patterns for distributed checkpointing.

        This is the new preferred API for checkpoint coordination. Each StateComponent
        declares its sharing pattern (GLOBAL, PER_RANK, REPLICATED, etc.), enabling
        automatic distributed checkpoint coordination without manual rank checks.

        Returns:
            List of StateComponent objects describing all checkpointable state

        Example implementation for single-GPU trainer:
            def get_state_components(self):
                from forgather.ml.trainer.checkpoint_types import StateComponent, SharingPattern

                return [
                    StateComponent(
                        key="model",
                        stateful=self.model,
                        sharing_pattern=SharingPattern.GLOBAL,
                    ),
                    StateComponent(
                        key="optimizer",
                        stateful=self.optimizer,
                        sharing_pattern=SharingPattern.GLOBAL,
                    ),
                    StateComponent(
                        key="scheduler",
                        stateful=self.lr_scheduler,
                        sharing_pattern=SharingPattern.GLOBAL,
                    ),
                    StateComponent(
                        key="dataset",
                        stateful=self.train_dataloader,
                        sharing_pattern=self._get_dataset_sharing_pattern(),
                    ),
                    StateComponent(
                        key="rng",
                        stateful=RNGState(),
                        sharing_pattern=SharingPattern.PER_RANK,
                    ),
                ]

        Example for DDP trainer:
            def get_state_components(self):
                return [
                    StateComponent(
                        key="model",
                        stateful=self.unwrapped_model(),
                        sharing_pattern=SharingPattern.REPLICATED,
                        validate_replication=True,  # Verify DDP synchronization
                    ),
                    # ... other components
                ]

        See: docs/checkpointing/migration_guide.md for full migration guide
        """
        pass

    def get_process_groups(self) -> Dict[str, Any]:
        """
        Get named process groups for PER_GROUP sharing pattern.

        Returns dictionary mapping group names to ProcessGroup objects.
        Only needed if using PER_GROUP sharing pattern in state components.

        Returns:
            Dictionary mapping process group names to ProcessGroup objects
            (e.g., {"dp_group": dp_pg, "pp_group": pp_pg})

        Example:
            def get_process_groups(self):
                return {
                    "dp_group": self.dp_process_group,
                    "pp_group": self.pp_process_group,
                }
        """
        return {}
