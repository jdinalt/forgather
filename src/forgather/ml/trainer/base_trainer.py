import logging
import os
import platform
import time
from abc import abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    override,
)

import torch
from dacite import from_dict
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..distributed import prefix_logger_rank
from .checkpoint_manager import RNGState
from .checkpoint_types import SharingPattern, StateComponent
from .trainer_types import (
    BaseDataset,
    CheckpointInterface,
    DataCollatorT,
    ExtensibleTrainer,
    IntervalStrategy,
    IterableDatasetT,
    LossFunctionT,
    LRSchedulerT,
    MinimalTrainingArguments,
    OptimizerT,
    PreprocessingClassT,
    StatefulProvider,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainOutput,
)

logger = logging.getLogger(__name__)
prefix_logger_rank(logger)

ModelConstructor = Callable[[], torch.nn.Module]


@dataclass(kw_only=True)
class BaseTrainingArguments(MinimalTrainingArguments):
    """
    Extended training arguments with checkpoint management and PyTorch optimizations.

    Extends MinimalTrainingArguments with checkpoint preservation features and
    PyTorch runtime optimizations.

    All training state (model, optimizer, scheduler, dataset position, RNG state)
    is automatically saved in checkpoints. To skip loading specific components,
    manually delete their files from the checkpoint directory before resuming.

    NOTE: These checkpoint options are NOT compatible with HF Trainer.
    Use HF-compatible MinimalTrainingArguments for basic HF compatibility.
    """

    # Default torch dtype for model construction (e.g., "float32", "bfloat16", "float16")
    default_dtype: str | None = None

    # Limit maximum validation/eval steps (-1 for unlimited)
    max_eval_steps: int = -1

    # Checkpoint preservation
    preserve_best_model: bool = False
    best_model_metric: str = "loss"
    best_model_greater_is_better: bool | None = None
    preserve_n_best: int = 1  # Keep N best checkpoints safe from cleanup

    # Force evaluation before save (decouples save/eval scheduling)
    eval_on_save: bool = False

    # Offload activation tensors to CPU memory during backward pass to reduce GPU memory usage.
    # Best combined with activation checkpointing. Trade GPU memory for CPU memory and bandwidth.
    # https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html#saving-tensors-to-cpu
    enable_activation_offloading: bool = False

    # Enable PyTorch anomaly detection for debugging NaN/Inf in gradients.
    # Adds overhead - only use for debugging.
    # https://docs.pytorch.org/docs/stable/autograd.html#debugging-and-anomaly-detection
    detect_anomaly: bool = False

    # Set Scaled Dot-Product Attention (SDPA) backend implementation.
    # Options: "math" (reference), "flash" (Flash Attention), "efficient" (memory-efficient), "cudnn"
    # Can be a single backend or list for priority order (if sdpa_set_priority=True)
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html#torch.nn.attention.sdpa_kernel
    sdpa_backend: List[str] | str | None = (
        None  # "math" | "flash" | "efficient" | "cudnn"
    )
    sdpa_set_priority: bool = (
        False  # If True and sdpa_backend is list, interpret as priority order
    )

    # Set matmul precision for float32 operations on Ampere+ GPUs for speedup.
    # "highest": Full IEEE precision (slowest)
    # "high": TF32 precision (~10-20% speedup, minimal accuracy loss)
    # "medium": More aggressive optimization (faster but may impact accuracy)
    # https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    float32_matmul_precision: str | None = None  # "highest" | "high" | "medium"

    # Override PyTorch dynamo recompilation limit (default is quite low).
    # Increase if seeing frequent recompilations with torch.compile().
    dynamo_recompile_limit: int | None = None

    def __post_init__(self):
        if self.logging_dir is None:
            self.logging_dir = os.path.join(
                self.output_dir, "runs", f"{time.time_ns()}_{platform.node()}"
            )

        # As per https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        if self.dataloader_prefetch_factor is None and self.dataloader_num_workers > 0:
            self.dataloader_prefetch_factor = 2
        if self.torch_compile_backend is None:
            self.torch_compile_backend = "inductor"
        if self.torch_compile_mode is None:
            self.torch_compile_backend = "default"

        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {}


TBaseTrainingArguments = TypeVar("TBaseTrainingArguments", bound=BaseTrainingArguments)


class BaseTrainer(
    ExtensibleTrainer, Stateful, StatefulProvider, Generic[TBaseTrainingArguments]
):
    """
    Abstract base class implementing common trainer functionality.

    Provides shared implementation for trainer infrastructure while leaving
    the core training and evaluation loops abstract for subclasses to implement.

    Key responsibilities:
    - Callback management and event dispatching
    - Model and checkpoint save/load coordination
    - Training state management (global_step, epoch, etc.)
    - Stateful interface for checkpointing trainer components
    - Common initialization and configuration handling

    Concrete implementations must define:
    - _prepare(): Setup dataloaders, model, optimizer before training
    - _train_loop(): The actual training iteration loop
    - _eval_loop(): The evaluation loop

    HuggingFace API Compatibility:
    This class maintains API compatibility with transformers.Trainer where possible,
    making it easier to port existing training code. See MinimalTrainingArguments
    and BaseTrainingArguments for supported configuration options.
    """

    model: torch.nn.Module | None
    data_collator: DataCollatorT | None
    train_dataset: IterableDatasetT | None
    eval_dataset: IterableDatasetT | None
    processing_class: PreprocessingClassT | None
    model_init: ModelConstructor | None
    callbacks: List[TrainerCallback]
    loss_fn: LossFunctionT | None
    train_dataloader: Iterable | None
    eval_dataloader: Iterable | None
    optimizer: OptimizerT | None
    lr_scheduler: LRSchedulerT | None
    is_local_process_zero: bool
    is_world_process_zero: bool
    num_processes: int
    checkpoint_manager: CheckpointInterface | None
    state: TrainerState
    control: TrainerControl

    @classmethod
    def default_callbacks(cls):
        """
        Return list of default callbacks for this trainer class.

        Subclasses override to provide default callbacks (progress bars, logging, etc.).
        For example, Trainer adds ProgressCallback and InfoCallback by default.

        Returns:
            List of callback instances to install by default
        """
        return []

    def __init__(
        self,
        args: TBaseTrainingArguments | dict,
        model: torch.nn.Module | None = None,
        *,
        data_collator: Optional[DataCollatorT] = None,
        train_dataset: Optional[IterableDatasetT] = None,
        eval_dataset: Optional[IterableDatasetT] = None,
        processing_class: Optional[PreprocessingClassT] = None,
        model_init: Optional[ModelConstructor] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        compute_loss_func: Optional[LossFunctionT] = None,
    ):
        if isinstance(args, dict):
            args: TBaseTrainingArguments = from_dict(BaseTrainingArguments, args)

        assert (
            model or model_init
        ), "Either a model or a model constructor must be specified"

        assert (
            args.gradient_accumulation_steps > 0
        ), "gradient_accumulation_steps must be > 0"

        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class
        self.model_init = model_init
        if callbacks is None:
            self.callbacks = self.default_callbacks()
        else:
            self.callbacks = callbacks
        self.loss_fn = compute_loss_func

        # Init attributes
        self.train_dataloader = None
        self.eval_dataloader = None
        self.optimizer = None
        self.lr_scheduler = None
        self.is_local_process_zero = True
        self.is_world_process_zero = True
        self.num_processes = 1
        self.checkpoint_manager: CheckpointInterface | None = None

        self.state = TrainerState(
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            train_batch_size=args.per_device_train_batch_size,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            max_eval_steps=args.max_eval_steps,
        )
        self.control = TrainerControl()

        # Silence annoying Huggingface FastTokenizer warnings
        # If knows if it is safe or not, and does the right thing, why
        # do I need to hear about it and create a janky workaround for
        # a non-issue!?
        if self.args.dataloader_num_workers > 0:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if self.args.dynamo_recompile_limit:
            logger.info(
                f"Setting torch._dynamo.config.recompile_limit = {self.args.dynamo_recompile_limit}"
            )
            torch._dynamo.config.recompile_limit = self.args.dynamo_recompile_limit

        if self.args.detect_anomaly:
            logger.warning(
                "Enabling autograd detect anomaly; expect performance degradation"
            )
            torch.autograd.set_detect_anomaly(True)

        if self.args.float32_matmul_precision is not None:
            logger.info(
                f'Setting float32_matmul_precision to "{self.args.float32_matmul_precision}"'
            )
            torch.set_float32_matmul_precision(self.args.float32_matmul_precision)

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"model={self.model},"
            f"args={self.args},"
            f"data_collator={self.data_collator},"
            f"train_dataset={self.train_dataset},"
            f"eval_dataset={self.eval_dataset},"
            f"processing_class={self.processing_class},"
            f"model_init={self.model_init},"
            f"callbacks={self.callbacks},"
            ")"
        )

    # AbstractBaseTrainer
    @override
    def train(self, **kwargs) -> TrainOutput:
        """
        The main entry point to start training the model.
        """
        with ExitStack() as exit_stack:
            backends = self._get_sdpa_backends(self.args.sdpa_backend)
            if backends:
                logger.info(
                    f"sdpa_backends={backends}, set_priority={self.args.sdpa_set_priority}"
                )
                exit_stack.enter_context(
                    sdpa_kernel(backends, set_priority=self.args.sdpa_set_priority)
                )
            if self.args.enable_activation_offloading:
                exit_stack.enter_context(
                    torch.autograd.graph.save_on_cpu(pin_memory=True)
                )
            self._prepare(
                train_dataset=self.train_dataset, eval_dataset=self.eval_dataset
            )
            return self._train_loop()

    # AbstractBaseTrainer
    @override
    def evaluate(
        self, eval_dataset: Optional[BaseDataset] = None, **kwargs
    ) -> dict[str, float]:
        """
        The main entry point to evaluate the model.
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        with ExitStack() as exit_stack:
            backends = self._get_sdpa_backends(self.args.sdpa_backend)
            if backends:
                exit_stack.enter_context(
                    sdpa_kernel(backends, set_priority=self.args.sdpa_set_priority)
                )
            self._prepare(train_dataset=None, eval_dataset=eval_dataset)
            return self._eval_loop()

    # AbstractBaseTrainer
    @override
    def add_callback(self, callback: TrainerCallback):
        if isinstance(callback, type):
            callback = callback()
        self.callbacks.append(callback)

    # AbstractBaseTrainer
    @override
    def pop_callback(self, callback: TrainerCallback) -> TrainerCallback | None:
        if isinstance(callback, type):
            compare = lambda a, b: type(a) == b
        else:
            compare = lambda a, b: id(a) == id(b)
        for i, cb in enumerate(self.callbacks):
            if compare(cb, callback):
                return self.callbacks.pop(i)
        return None

    # AbstractBaseTrainer
    @override
    def remove_callback(self, callback: TrainerCallback):
        self.pop_callback(callback)

    def log(self, logs: Dict[str, float]):
        """
        Log metrics and dispatch to callbacks.

        Appends metrics to state.log_history and dispatches on_log event to
        all callbacks. Callbacks can use this to write to TensorBoard, wandb, etc.

        Args:
            logs: Dictionary of metric name to value (e.g., {"loss": 0.5, "lr": 1e-4})

        Returns:
            TrainerControl from callbacks (may set flags like should_save, should_evaluate)
        """
        self.state.log_history.append(logs)

        return self._dispatch_event(
            "on_log",
            logs=logs,
        )

    @staticmethod
    def _get_sdpa_backends(
        backend: List[str | SDPBackend] | str | SDPBackend | None,  # type: ignore[valid-type]
    ) -> List[SDPBackend] | SDPBackend | None:  # type: ignore[valid-type]
        """
        Normalize SDPA backend specifications to SDPBackend enum values.

        Converts string backend names ("math", "flash", "efficient", "cudnn") to
        PyTorch SDPBackend enum values. Handles both single backends and lists.

        Args:
            backend: Backend specification as string(s) or SDPBackend enum(s), or None

        Returns:
            SDPBackend enum value(s) or None if backend is None

        Raises:
            ValueError: If backend is not a valid type (str/list/SDPBackend/None)
        """
        if backend is None:
            return None

        sdpa_mapping = {
            "math": SDPBackend.MATH,
            "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "cudnn": SDPBackend.CUDNN_ATTENTION,
        }

        def get_backend(b):
            if isinstance(b, SDPBackend):
                return b
            return sdpa_mapping[b]

        if isinstance(backend, str):
            return get_backend(backend)
        elif isinstance(backend, list):
            return [get_backend(i) for i in backend]
        else:
            raise ValueError("sdpa-backend must be a List[str] or str")

    def _dispatch_event(self, event: str, **kwargs):
        """
        Dispatch event to all registered callbacks.

        Calls the named event handler method on each callback (if defined).
        Callbacks can return a new TrainerControl to modify trainer behavior
        (e.g., trigger early stopping, force checkpoint save).

        Common events:
        - on_init_end: After trainer initialization
        - on_train_begin/on_train_end: Training loop boundaries
        - on_epoch_begin/on_epoch_end: Epoch boundaries
        - on_step_begin/on_step_end: Training step boundaries
        - on_log: After logging metrics
        - on_evaluate: After evaluation
        - on_save: After checkpoint save

        Args:
            event: Name of callback method to invoke (e.g., "on_train_begin")
            **kwargs: Additional arguments passed to callback (logs, metrics, etc.)

        Returns:
            Updated TrainerControl (last non-None return from callbacks)
        """
        # Dispatch to call callbacks in list
        unwrapped_model = self.unwrapped_model()
        for callback in self.callbacks:
            event_handler = getattr(callback, event, None)
            # If handler is undefined, skip to next.
            if event_handler is None:
                continue

            new_control = event_handler(
                self.args,
                self.state,
                self.control,
                model=unwrapped_model,
                processing_class=self.processing_class,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )

            if new_control is not None:
                self.control = new_control
        return self.control

    def unwrapped_model(self) -> torch.nn.Module:
        """
        Get the underlying model without any wrappers.

        Returns the base model, unwrapped from any distributed training wrappers
        (DDP, FSDP, Accelerate, pipeline parallelism, etc.).

        Subclasses that wrap the model override this to return the unwrapped version.
        For example:
        - AccelTrainer: Returns accelerator.unwrap_model(self.model)
        - PipelineTrainer: Returns the unwrapped pipeline stage models

        Returns:
            The unwrapped base model
        """
        assert self.model
        return self.model

    # AbstractBaseTrainer
    @override
    def save_model(self, output_dir: Optional[os.PathLike | str] = None) -> None:
        """
        Save just the model weights and preprocessing class (HF Trainer API compatibility).

        Saves model to output_dir or args.output_dir. This saves ONLY the model weights,
        not the full training state. For resumable training, use save_checkpoint() instead.

        Note: This method exists primarily for HF Trainer API compatibility. In practice,
        save_checkpoint() is more useful as it saves complete training state.

        Args:
            output_dir: Directory to save model, defaults to args.output_dir
        """
        assert self.checkpoint_manager
        self.checkpoint_manager.save_model(
            output_dir=output_dir, overwrite_output_dir=self.args.overwrite_output_dir
        )

    # AbstractBaseTrainer
    @override
    def save_checkpoint(self, checkpoint_path=None) -> None:
        """
        Save complete training checkpoint.

        Saves all training state to a timestamped checkpoint directory under args.output_dir:
        - Model weights (always required)
        - Optimizer state (momentum buffers, adaptive rates, etc.)
        - LR scheduler state (current step, etc.)
        - Training progress (global_step, epoch, etc.)
        - Dataset position (if dataloader is stateful)
        - Random number generator states (for reproducibility)

        This allows resuming training from the exact point where it left off.

        Args:
            checkpoint_path: Optional specific checkpoint path, otherwise auto-generated
        """
        assert self.checkpoint_manager
        self.checkpoint_manager.save_checkpoint(checkpoint_path)

    # AbstractBaseTrainer
    @override
    def load_checkpoint(self, checkpoint_path=None) -> None:
        """
        Load complete training checkpoint to resume training.

        Restores all training state from a saved checkpoint:
        - Model weights (always loaded)
        - Optimizer state (if file exists)
        - LR scheduler state (if file exists)
        - Training progress (if file exists)
        - Dataset position (if file exists)
        - Random number generator states (if file exists)

        If checkpoint_path is None, automatically finds the latest checkpoint in args.output_dir.

        To skip loading specific components, delete their files from the checkpoint directory
        before calling this method. For example:
            rm checkpoint-1000/optimizer_state.pt  # Use fresh optimizer

        The checkpoint system will log warnings for missing components but continue loading.

        Args:
            checkpoint_path: Path to checkpoint directory, or None for latest checkpoint
        """
        assert self.checkpoint_manager
        self.checkpoint_manager.load_checkpoint(checkpoint_path)

    # StatefulProvider
    @override
    def get_state_components(self) -> List[StateComponent]:
        """
        Get state components for distributed checkpointing.

        All training state is always saved to checkpoints:
        - Model weights (required - cannot be skipped)
        - Optimizer state (momentum, adaptive rates, etc.)
        - LR scheduler state (current step, etc.)
        - Training progress (global_step, epoch, etc.)
        - Dataset state (iteration position, if dataloader is stateful)
        - RNG state (for reproducibility)

        To skip loading a component, delete its file from the checkpoint directory.
        For example, to change datasets between runs:
            rm checkpoint-1000/dataset_state.pt
            rm checkpoint-1000/trainer_state.pt

        The checkpoint system will log warnings for missing components but continue loading.

        For single-GPU trainers, all state is GLOBAL except RNG which is PER_RANK.

        Returns:
            List of StateComponent objects describing all checkpointable state
        """
        components = []
        assert self.model is not None
        # Model - REQUIRED (always must be present)
        components.append(
            StateComponent(
                key="model",
                stateful=self.model,
                sharing_pattern=SharingPattern.GLOBAL,
                required=True,  # Model is always required
            )
        )

        # Optimizer - optional (allows changing optimizer type)
        components.append(
            StateComponent(
                key="optimizer",
                stateful=self.optimizer,
                sharing_pattern=SharingPattern.GLOBAL,
                required=False,
            )
        )

        # LR Scheduler - optional (allows changing scheduler type)
        components.append(
            StateComponent(
                key="scheduler",
                stateful=self.lr_scheduler,
                sharing_pattern=SharingPattern.GLOBAL,
                required=False,
            )
        )

        # Trainer state - optional (allows fresh training progress)
        components.append(
            StateComponent(
                key="trainer",
                stateful=self,
                sharing_pattern=SharingPattern.GLOBAL,
                required=False,
            )
        )

        # Dataset state - optional, only if dataloader is stateful
        if hasattr(self.train_dataloader, "state_dict"):
            components.append(
                StateComponent(
                    key="dataset",
                    stateful=self.train_dataloader,
                    sharing_pattern=self._get_dataset_sharing_pattern(),
                    required=False,
                )
            )

        # RNG state - optional (allows fresh randomization)
        components.append(
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
                required=False,
            )
        )

        return components

    def _get_dataset_sharing_pattern(self) -> SharingPattern:
        """
        Determine dataset state sharing pattern based on dataloader type.

        The pattern depends on how data is loaded:
        - DataloaderDispatcher: Centralized loading by rank 0 → GLOBAL
        - Regular DataLoader: Independent loading per rank → PER_RANK

        For single-GPU trainers, dataset is always GLOBAL.

        Returns:
            SharingPattern for dataset state
        """
        # For single-GPU trainer, dataset is GLOBAL
        # Subclasses with distributed training should override this method
        return SharingPattern.GLOBAL

    def get_process_groups(self) -> Dict[str, Any]:
        """
        Get named process groups for PER_GROUP sharing pattern.

        Single-GPU trainers don't use process groups.
        Subclasses with distributed training should override this method.

        Returns:
            Empty dictionary (no process groups in single-GPU training)
        """
        return {}

    # Stateful
    @override
    def load_state_dict(self, state_dict):
        """
        Restore trainer training progress state from checkpoint.

        Implements PyTorch Stateful interface. Restores training step counters
        and progress tracking to resume training from the correct point.

        Args:
            state_dict: Dictionary with training state (global_step, epoch, etc.)
        """
        self.state.global_step = state_dict["global_step"]
        self.state.epoch_start_step = state_dict["epoch_start_step"]
        self.state.raw_epoch = state_dict["raw_epoch"]
        self.state.num_input_tokens_seen = state_dict["num_input_tokens_seen"]
        self.state.total_flos = state_dict["total_flos"]

    # Stateful
    @override
    def state_dict(self):
        """
        Get trainer training progress state for checkpointing.

        Implements PyTorch Stateful interface. Returns training step counters
        and progress tracking needed to resume training.

        Returns:
            Dictionary with training state:
            - global_step: Total optimizer updates since training start
            - epoch_start_step: Global step when current epoch started
            - raw_epoch: Integer epoch counter
            - num_input_tokens_seen: Total tokens processed (for logging)
            - total_flos: Total floating point operations (for efficiency metrics)
        """
        return {
            "global_step": self.state.global_step,
            "epoch_start_step": self.state.epoch_start_step,
            "raw_epoch": self.state.raw_epoch,
            "num_input_tokens_seen": self.state.num_input_tokens_seen,
            "total_flos": self.state.total_flos,
        }

    @abstractmethod
    def _prepare(
        self,
        train_dataset: Optional[BaseDataset],
        eval_dataset: Optional[BaseDataset],
    ) -> None:
        """
        Prepare all components for training and/or evaluation.

        Called at the start of train() or evaluate() to set up everything needed
        for the training/eval loop. Both datasets may be None (eval-only or train-only).

        Typical preparation sequence:
        1. Create dataloaders from datasets
        2. Initialize/move model to device(s)
        3. Set up optimizer and LR scheduler (if training)
        4. Wrap components for distributed training (DDP, Accelerate, etc.)
        5. Initialize trainer state
        6. Create checkpoint manager
        7. Load checkpoint if resuming

        Subclasses must implement this to handle their specific setup requirements.

        Args:
            train_dataset: Training dataset or None for eval-only
            eval_dataset: Evaluation dataset or None for train-only
        """
        pass

    @abstractmethod
    def _train_loop(self) -> TrainOutput:
        """
        Execute the main training loop.

        Implements the core training iteration logic:
        - Iterate over epochs and batches
        - Forward/backward passes
        - Optimizer steps and gradient accumulation
        - Periodic logging, evaluation, and checkpointing
        - Callback event dispatching
        - Early stopping and control flow

        Must be implemented by concrete trainer classes.

        Returns:
            TrainOutput with final global_step and training metrics
        """
        pass

    @abstractmethod
    def _eval_loop(self) -> dict[str, float]:
        """
        Execute the evaluation loop.

        Implements evaluation logic:
        - Iterate over evaluation dataset
        - Forward-only passes (no gradients)
        - Compute and aggregate metrics
        - Callback event dispatching

        Must be implemented by concrete trainer classes.

        Returns:
            Dictionary of evaluation metrics (e.g., {"eval_loss": 0.5})
        """
        pass


def logits_from_outputs(outputs) -> Tensor:
    """
    Extract logits from model outputs (handles multiple output formats).

    Models may return outputs in different formats:
    - Tensor: Logits directly
    - Object with .logits attribute: HF-style ModelOutput
    - Tuple: (loss, logits) or similar

    Args:
        outputs: Model forward pass outputs

    Returns:
        Logits tensor

    Raises:
        AssertionError: If outputs don't contain logits in expected format
    """
    if not isinstance(outputs, Tensor):
        assert hasattr(outputs, "logits"), f"Type is {type(outputs)}"
        return outputs.logits
    return outputs


def loss_from_outputs(outputs) -> Tensor:
    """
    Extract loss from model outputs (handles multiple output formats).

    Models may return outputs in different formats:
    - Tuple: (loss, ...) - loss is first element
    - Object with .loss attribute: HF-style ModelOutput

    Args:
        outputs: Model forward pass outputs

    Returns:
        Loss tensor

    Raises:
        AssertionError: If outputs don't contain loss in expected format
    """
    if isinstance(outputs, tuple):
        loss = outputs[0]
        assert isinstance(loss, Tensor)
        return loss
    assert hasattr(outputs, "loss")
    return outputs.loss


def loss_and_logits_from_outputs(outputs) -> Tuple[Tensor, Tensor]:
    """
    Extract both loss and logits from model outputs (handles multiple formats).

    Models may return outputs in different formats:
    - Tuple: (loss, logits)
    - Object with .loss and .logits attributes: HF-style ModelOutput

    Args:
        outputs: Model forward pass outputs

    Returns:
        Tuple of (loss, logits) tensors

    Raises:
        AssertionError: If outputs don't contain loss and logits in expected format
    """
    if isinstance(outputs, tuple):
        loss, logits = outputs
        assert isinstance(loss, Tensor) and isinstance(logits, Tensor)
        return loss, logits
    assert hasattr(outputs, "loss") and hasattr(outputs, "logits")
    return outputs.loss, outputs.logits
