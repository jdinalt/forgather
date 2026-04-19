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
    def __len__(self): ...

    def load_state_dict(self, state_dict: Dict[str, Any]): ...

    def state_dict(self) -> Dict[str, Any]: ...


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
    def load_state_dict(self, state_dict: Dict[str, Any]): ...

    def state_dict(self) -> Dict[str, Any]: ...

    def step(self): ...

    def zero_grad(self): ...


class LRSchedulerT(Protocol):
    def load_state_dict(self, state_dict: Dict[str, Any]): ...

    def state_dict(self) -> Dict[str, Any]: ...

    def step(self): ...

    def get_lr(self) -> List[float]: ...

    def get_last_lr(self) -> List[float]: ...


DataCollatorT: TypeAlias = Callable[[List[Dict[str, Any]]], Dict[str, Any]]

LossFunctionT: TypeAlias = Callable[[Tensor, Tensor], Tensor]

OptimizerParamsT: TypeAlias = Union[
    Iterable[Tensor], Iterable[dict[str, Any]], Iterable[tuple[str, Tensor]]
]

OptimizerFactoryT: TypeAlias = Callable[[OptimizerParamsT], OptimizerT]

LRSchedulerFactoryT: TypeAlias = Callable[[OptimizerT], LRSchedulerT]

FusedLossFactoryT: TypeAlias = Callable[[nn.Module], LossFunctionT]

PreprocessingClassT: TypeAlias = Callable

EnableCheckpointFnT: TypeAlias = Callable[[int, nn.Module], None]


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
    """Minimal training configuration compatible with HuggingFace Trainer.

    Provides a subset of ``transformers.TrainingArguments`` sufficient for basic
    training. This is the base configuration class; extend it for additional
    features rather than adding fields here.

    Direct subclasses: ``BaseTrainingArguments`` (checkpoint control, PyTorch
    optimisations) and ``TrainingArguments`` (simple single-GPU memory options).

    Parameters
    ----------
    output_dir : str, optional
        Directory where model predictions and checkpoints are written.
    logging_dir : str or None, optional
        TensorBoard log directory. Defaults to ``output_dir/runs/TIMESTAMP_HOSTNAME``.
    per_device_train_batch_size : int, optional
        Training batch size per device. Effective global batch size is
        ``per_device_train_batch_size * num_devices * gradient_accumulation_steps``.
    per_device_eval_batch_size : int, optional
        Evaluation batch size per device.
    num_train_epochs : int, optional
        Total training epochs (may be fractional, e.g. ``2.5``).
    max_steps : int, optional
        If > 0, total optimiser steps to perform (overrides ``num_train_epochs``).
    device : Any, optional
        Device to use (``"cuda"``, ``"cpu"``, etc.). Auto-detected if ``None``.
    seed : int, optional
        Random seed for reproducibility. Default ``-1`` disables seeding.
    use_cpu : bool, optional
        Force CPU usage even when CUDA is available.
    epoch_train_steps : int, optional
        Fallback epoch length when the dataset does not support ``len()``.
        Used only for progress bookkeeping. Forgather extension.
    dataloader_num_workers : int, optional
        Subprocesses for data loading. ``0`` loads in the main process.
    dataloader_pin_memory : bool, optional
        Pin memory in DataLoader for faster GPU transfer.
    dataloader_persistent_workers : bool, optional
        Keep worker processes alive between epochs (faster, uses more RAM).
    dataloader_prefetch_factor : int or None, optional
        Batches prefetched per worker. Defaults to ``2`` when ``num_workers > 0``.
    dataloader_drop_last : bool, optional
        Drop the last incomplete batch when the dataset is not evenly divisible.
    eval_strategy : str, optional
        When to run evaluation: ``"no"``, ``"steps"``, or ``"epoch"``.
    eval_steps : int, optional
        Evaluation frequency in steps (when ``eval_strategy="steps"``).
    eval_delay : int, optional
        Epochs or steps to wait before the first evaluation.
    logging_strategy : str, optional
        When to log metrics: ``"no"``, ``"steps"``, or ``"epoch"``.
    logging_steps : int, optional
        Logging frequency in steps (when ``logging_strategy="steps"``).
    logging_first_step : bool, optional
        Log metrics at the very first global step.
    torch_compile : bool, optional
        Compile the model with ``torch.compile()`` for speedup.
    torch_compile_backend : str or None, optional
        Backend for ``torch.compile`` (e.g. ``"inductor"``, ``"aot_eager"``).
    torch_compile_mode : str or None, optional
        Compilation mode: ``"default"``, ``"reduce-overhead"``, or ``"max-autotune"``.
    torch_compile_dynamic : bool, optional
        Allow dynamic shapes in the compiled model.
    torch_compile_full_graph : bool, optional
        Force compilation of the entire model as a single graph.
    max_grad_norm : float or None, optional
        Maximum gradient norm for clipping. ``None`` disables clipping.
    gradient_accumulation_steps : int, optional
        Accumulate gradients over this many steps before an optimiser update.
    save_strategy : str, optional
        Checkpoint save strategy: ``"no"``, ``"steps"``, or ``"epoch"``.
    save_steps : int, optional
        Checkpoint save frequency in steps (when ``save_strategy="steps"``).
    save_total_limit : int, optional
        Maximum number of checkpoints to keep; oldest are deleted first.
    save_safetensors : bool, optional
        Write weights as Safetensors (safer and HF-compatible) rather than pickle.
    save_on_each_node : bool, optional
        In multi-node training, save on every node rather than only rank 0.
        Do not enable when using shared storage.
    overwrite_output_dir : bool, optional
        Overwrite ``output_dir`` contents on a fresh run.
    resume_from_checkpoint : bool or str, optional
        ``True`` (default) automatically finds and loads the latest checkpoint,
        falling back to fresh initialisation if none exists. A path string loads
        that specific checkpoint. ``False`` forces fresh initialisation.
    load_best_model_at_end : bool, optional
        Restore the best checkpoint at the end of training. Requires
        ``save_strategy == eval_strategy``.
    metric_for_best_model : str, optional
        Metric used to compare checkpoints when ``load_best_model_at_end=True``.
    greater_is_better : bool or None, optional
        Whether a higher metric value is better. Auto-determined from the metric
        name when ``None``.
    lr_scheduler_type : str, optional
        LR scheduler type for the built-in AdamW path (``"linear"``, ``"cosine"``, etc.).
    lr_scheduler_kwargs : dict or None, optional
        Additional keyword arguments forwarded to the LR scheduler constructor.
    warmup_steps : int, optional
        Linear warmup steps from 0 to ``learning_rate``.
    learning_rate : float, optional
        Initial learning rate for the built-in AdamW optimiser.
    weight_decay : float, optional
        Weight decay applied to all parameters except bias and LayerNorm weights.
    adam_beta1 : float, optional
        Beta1 for the built-in AdamW optimiser.
    adam_beta2 : float, optional
        Beta2 for the built-in AdamW optimiser.
    adam_epsilon : float, optional
        Epsilon for the built-in AdamW optimiser.
    gradient_checkpointing : bool, optional
        Enable activation checkpointing on models that support the HF API.
        Pass ``enable_activation_checkpoint_fn`` to the Trainer constructor to
        customise the checkpointing behaviour.
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
    resume_from_checkpoint: bool | str = True

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


class TrainerCallback:
    """
    Base class for trainer event callbacks.

    Subclasses implement only the event methods they need. Any method not
    defined is simply never called for that callback. The trainer maintains
    a lazy index mapping event names to the callbacks that define handlers,
    so only relevant callbacks are invoked per event.

    Available events (each receives args, state, control, **kwargs and
    may return None or an updated TrainerControl):

        on_init_end          - After trainer initialization
        on_train_begin       - Before training loop starts
        on_train_end         - After training loop ends
        on_epoch_begin       - Before each epoch
        on_epoch_end         - After each epoch
        on_step_begin        - Before each training step
        on_step_end          - After each training step
        on_substep_end       - After each gradient-accumulation sub-step
        on_forward_backward_begin - Before each forward+backward micro-step
                               (inside gradient accumulation loop, after data
                               loading; fires once per micro-batch)
        on_forward_backward_end   - After each forward+backward micro-step
                               (before optimizer, grad clipping, LR scheduler;
                               fires once per micro-batch)
        on_optimizer_step    - After optimizer.step()
        on_pre_optimizer_step - Before optimizer.step()
        on_evaluate          - After evaluation
        on_predict           - After prediction (also receives metrics)
        on_prediction_step   - After each prediction batch
        on_save              - After checkpoint save
        on_log               - After metric logging (receives logs kwarg)

    Forgather extensions (not in HF Trainer):

        on_log_step          - Called before on_log; receives the mutable logs dict
                               so callbacks can inject custom metrics before logging
        on_train_metrics     - Called each training step with per-step metrics
                               (loss, grad_norm, tokens, etc.) for fine-grained
                               monitoring or adaptive control

    kwargs always include:
        model, processing_class, optimizer, lr_scheduler,
        train_dataloader, eval_dataloader, trainer

    Compatible with HuggingFace TrainerCallback for easier porting.
    See: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py

    Identification:
        Each callback has a `name` property used for logging (e.g. when
        the trainer reports which callback requested an early stop).
        Subclasses inherit the default, which returns the class name, or
        can override `name` to provide a more descriptive label (useful
        when multiple instances of the same class are registered with
        different configurations).
    """

    @property
    def name(self) -> str:
        """Human-readable identifier for this callback, used in log messages."""
        return type(self).__name__


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
