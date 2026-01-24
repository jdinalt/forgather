import logging
import os
import platform
import time
from abc import abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, override

import torch
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.nn.attention import SDPBackend, sdpa_kernel

from .checkpoint_manager import RNGState
from .checkpoint_types import SharingPattern, StateComponent
from .trainer_types import (
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

ModelConstructor = Callable[[], torch.nn.Module]


@dataclass(kw_only=True)
class BaseTrainingArguments(MinimalTrainingArguments):
    """
    Extended training arguments with checkpoint control and PyTorch optimizations.

    Extends MinimalTrainingArguments with fine-grained control over what gets saved/restored
    in checkpoints, plus PyTorch runtime optimizations.

    NOTE: These checkpoint save/restore options are NOT compatible with HF Trainer.
    Use HF-compatible MinimalTrainingArguments for basic HF compatibility.
    """

    # Checkpointing options - control what gets saved in checkpoints
    # These allow fine-grained control over checkpoint size vs resumability tradeoffs
    save_optimizer_state: bool = True  # Save optimizer state (momentum, etc.)
    save_scheduler_state: bool = True  # Save LR scheduler state
    save_dataset_state: bool = True  # Save dataset iteration position
    save_rng_state: bool = True  # Save random number generator states

    # Control what gets restored from checkpoints
    restore_optimizer_state: bool = True  # Restore optimizer state
    restore_scheduler_state: bool = True  # Restore LR scheduler state
    restore_dataset_state: bool = True  # Restore dataset iteration position
    restore_rng_state: bool = True  # Restore RNG states for reproducibility

    # Default torch dtype for model construction (e.g., "float32", "bfloat16", "float16")
    default_dtype: str | None = None

    # Limit maximum validation/eval steps (-1 for unlimited)
    max_eval_steps: int = -1

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

        # Auto-determine greater_is_better from metric name if not set
        if self.greater_is_better is None:
            # Common metrics where higher is better
            higher_is_better_metrics = {
                "accuracy",
                "f1",
                "precision",
                "recall",
                "auc",
                "roc_auc",
                "ap",
                "map",
                "bleu",
                "rouge",
                "meteor",
                "bertscore",
                "exact_match",
                "squad_f1",
            }
            # Check if metric name contains any higher-is-better patterns
            metric_lower = self.metric_for_best_model.lower()
            self.greater_is_better = any(
                pattern in metric_lower for pattern in higher_is_better_metrics
            )

        # Validate alignment requirements for load_best_model_at_end
        if self.load_best_model_at_end:
            if self.save_strategy != self.eval_strategy:
                raise ValueError(
                    "load_best_model_at_end requires save_strategy and eval_strategy to be the same. "
                    f"Got save_strategy={self.save_strategy}, eval_strategy={self.eval_strategy}"
                )

            if (
                self.save_strategy == IntervalStrategy.STEPS
                and self.eval_strategy == IntervalStrategy.STEPS
            ):
                if self.save_steps % self.eval_steps != 0:
                    raise ValueError(
                        "load_best_model_at_end requires save_steps to be a multiple of eval_steps when using step-based strategies. "
                        f"Got save_steps={self.save_steps}, eval_steps={self.eval_steps}"
                    )


class BaseTrainer(ExtensibleTrainer, Stateful, StatefulProvider):
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
    - _post_init(): Post-initialization hook (device setup, wrapping, etc.)
    - _prepare(): Setup dataloaders, model, optimizer before training
    - _train_loop(): The actual training iteration loop
    - _eval_loop(): The evaluation loop

    HuggingFace API Compatibility:
    This class maintains API compatibility with transformers.Trainer where possible,
    making it easier to port existing training code. See MinimalTrainingArguments
    and BaseTrainingArguments for supported configuration options.
    """

    model: torch.nn.Module | None
    args: BaseTrainingArguments
    data_collator: DataCollatorT
    train_dataset: IterableDatasetT | None
    eval_dataset: IterableDatasetT | None
    processing_class: PreprocessingClassT | None
    model_init: ModelConstructor | None
    callbacks: List[TrainerCallback]
    loss_fn: LossFunctionT
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
        args: BaseTrainingArguments,
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

        self._post_init()

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
        self, eval_dataset: Optional[IterableDatasetT] = None, **kwargs
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
    def add_callback(self, callback):
        if isinstance(callback, type):
            callback = callback()
        self.callbacks.append(callback)

    # AbstractBaseTrainer
    @override
    def pop_callback(self, callback):
        if isinstance(callback, type):
            compare = lambda a, b: type(a) == b
        else:
            compare = lambda a, b: id(a) == id(b)
        for i, cb in enumerate(self.callbacks):
            if compare(cb, callback):
                return self.callbacks.pop(i)

    # AbstractBaseTrainer
    @override
    def remove_callback(self, callback):
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

        Saves model, optimizer, scheduler, dataset position, RNG state, and trainer state
        to a timestamped checkpoint directory under args.output_dir. This allows resuming
        training from the exact point where it left off.

        Checkpoint contents controlled by args.save_*_state flags:
        - save_optimizer_state: Optimizer state (momentum buffers, etc.)
        - save_scheduler_state: LR scheduler state
        - save_dataset_state: Dataset iteration position
        - save_rng_state: Random number generator states

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

        Restores model, optimizer, scheduler, dataset position, RNG state, and trainer state
        from a saved checkpoint. If checkpoint_path is None, automatically finds the latest
        checkpoint in args.output_dir.

        What gets restored is controlled by args.restore_*_state flags:
        - restore_optimizer_state: Restore optimizer state
        - restore_scheduler_state: Restore LR scheduler state
        - restore_dataset_state: Restore dataset iteration position
        - restore_rng_state: Restore RNG states for reproducibility

        Args:
            checkpoint_path: Path to checkpoint directory, or None for latest checkpoint
        """
        assert self.checkpoint_manager
        self.checkpoint_manager.load_checkpoint(checkpoint_path)

    # StatefulProvider
    @override
    def get_statefuls_for_save(self):
        """
        Get dictionary of stateful objects to save in checkpoint.

        Collects trainer components that need to be saved for resumable training.
        What gets included is controlled by args.save_*_state flags.

        Returns:
            Dictionary mapping component names to Stateful objects:
            - "optimizer": Optimizer state (momentum, adaptive rates, etc.)
            - "scheduler": LR scheduler state (current step, etc.)
            - "trainer": Trainer state (global_step, epoch, etc.)
            - "dataset": Dataset/dataloader iteration position
            - "rng": RNG states (torch, numpy, random module)
        """
        statefuls = {}
        save_dataset_state = False

        # Not all dataloaders are stateful and not all dataloader with
        # a state_dict() method are an instance of Stateful.
        if self.args.save_dataset_state:
            if hasattr(self.train_dataloader, "state_dict"):
                save_dataset_state = True
            else:
                logger.warning("train_dataloader doesn't have state_dict method")

        for key, obj, save in (
            ("optimizer", self.optimizer, self.args.save_optimizer_state),
            ("scheduler", self.lr_scheduler, self.args.save_scheduler_state),
            ("trainer", self, self.args.save_dataset_state),
            ("dataset", self.train_dataloader, save_dataset_state),
            ("rng", RNGState(), self.args.save_rng_state),
        ):
            if not save:
                continue
            assert obj, f"{key} is not initialized"
            statefuls[key] = obj

        return statefuls

    # StatefulProvider
    @override
    def get_statefuls_for_load(self):
        """
        Get dictionary of stateful objects to restore from checkpoint.

        Collects trainer components that should be loaded from checkpoint for
        resuming training. What gets included is controlled by args.restore_*_state flags.

        Returns:
            Dictionary mapping component names to Stateful objects:
            - "optimizer": Optimizer to restore state into
            - "scheduler": LR scheduler to restore state into
            - "trainer": Trainer to restore state into (global_step, epoch, etc.)
            - "dataset": Dataset/dataloader to restore iteration position
            - "rng": RNG states to restore for reproducibility
        """
        statefuls = {}
        restore_dataset_state = False
        if self.args.restore_dataset_state:
            if hasattr(self.train_dataloader, "load_state_dict"):
                restore_dataset_state = True
            else:
                logger.warning(
                    "Could not restored Dataloader state, as it does not have a load method"
                )

        for key, obj, load in (
            ("optimizer", self.optimizer, self.args.restore_optimizer_state),
            ("scheduler", self.lr_scheduler, self.args.restore_scheduler_state),
            ("trainer", self, self.args.restore_dataset_state),
            ("dataset", self.train_dataloader, restore_dataset_state),
            ("rng", RNGState(), self.args.restore_rng_state),
        ):
            if not load:
                continue
            assert obj, f"{key} is not initialized"
            statefuls[key] = obj

        return statefuls

    # StatefulProvider - New Checkpoint API
    def get_state_components(self) -> List[StateComponent]:
        """
        Get state components with explicit sharing patterns for distributed checkpointing.

        This is the new preferred API for checkpoint coordination. Each StateComponent
        declares its sharing pattern (GLOBAL, PER_RANK, REPLICATED, etc.), enabling
        automatic distributed checkpoint coordination without manual rank checks.

        For single-GPU trainers, all state is GLOBAL except RNG which is PER_RANK.

        Returns:
            List of StateComponent objects describing all checkpointable state

        Example:
            Components for simple trainer:
            - model: GLOBAL (single GPU, one copy)
            - optimizer: GLOBAL
            - scheduler: GLOBAL
            - trainer: GLOBAL (training progress)
            - dataset: GLOBAL or PER_RANK (depends on dataloader type)
            - rng: PER_RANK (each rank needs unique random numbers)
        """
        components = []

        # Model - always saved
        components.append(
            StateComponent(
                key="model",
                stateful=self.model,
                sharing_pattern=SharingPattern.GLOBAL,
            )
        )

        # Optimizer
        if self.args.save_optimizer_state:
            components.append(
                StateComponent(
                    key="optimizer",
                    stateful=self.optimizer,
                    sharing_pattern=SharingPattern.GLOBAL,
                    required=self.args.save_optimizer_state,
                )
            )

        # LR Scheduler
        if self.args.save_scheduler_state:
            components.append(
                StateComponent(
                    key="scheduler",
                    stateful=self.lr_scheduler,
                    sharing_pattern=SharingPattern.GLOBAL,
                    required=self.args.save_scheduler_state,
                )
            )

        # Trainer state (training progress)
        if self.args.save_dataset_state:  # Note: trainer state saved with dataset flag
            components.append(
                StateComponent(
                    key="trainer",
                    stateful=self,
                    sharing_pattern=SharingPattern.GLOBAL,
                    required=self.args.save_dataset_state,
                )
            )

        # Dataset state - pattern depends on dataloader type
        if self.args.save_dataset_state and hasattr(self.train_dataloader, "state_dict"):
            components.append(
                StateComponent(
                    key="dataset",
                    stateful=self.train_dataloader,
                    sharing_pattern=self._get_dataset_sharing_pattern(),
                    required=False,  # Optional - not all dataloaders are stateful
                )
            )

        # RNG state - always PER_RANK
        if self.args.save_rng_state:
            components.append(
                StateComponent(
                    key="rng",
                    stateful=RNGState(),
                    sharing_pattern=SharingPattern.PER_RANK,
                    required=self.args.save_rng_state,
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

    def get_process_groups(self) -> Dict[str, any]:
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
    def _post_init(self) -> None:
        """
        Post-initialization hook called at end of __init__.

        Used by concrete trainer implementations to perform additional setup after
        basic initialization. Common uses:
        - Set device if not specified
        - Validate configuration combinations
        - Set up data collators

        Distributed trainers may defer some setup to _prepare() where datasets are available.

        Examples:
        - Trainer: Sets default device, validates args, sets default data_collator
        - AccelTrainer: Initializes Accelerator instance
        - PipelineTrainer: Validates pipeline parallelism configuration
        """
        pass

    @abstractmethod
    def _prepare(
        self,
        train_dataset: IterableDatasetT | None,
        eval_dataset: IterableDatasetT | None,
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
