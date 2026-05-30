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
    cast,
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
    """Extended training arguments with checkpoint management and PyTorch optimizations.

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

    Parameters
    ----------
    default_dtype : str or None, optional
        Default ``torch.dtype`` for model construction (e.g. ``"float32"``,
        ``"bfloat16"``, ``"float16"``). ``None`` leaves PyTorch's global default
        unchanged. Default is ``None``.
    max_eval_steps : int, optional
        Maximum number of evaluation steps per evaluation call. ``-1`` runs the
        full evaluation dataset. Default is ``-1``.
    preserve_best_model : bool, optional
        If ``True``, keep the checkpoint with the best value of
        ``best_model_metric`` protected from cleanup rotation. Default is ``False``.
    best_model_metric : str, optional
        Name of the metric used to select the best checkpoint when
        ``preserve_best_model=True``. Default is ``"loss"``.
    best_model_greater_is_better : bool or None, optional
        Whether higher values of ``best_model_metric`` are better. ``None``
        auto-detects from the metric name (metrics containing ``"loss"`` or
        ``"perplexity"`` default to lower-is-better). Default is ``None``.
    preserve_n_best : int, optional
        Number of best checkpoints to keep safe from ``save_total_limit``
        cleanup. Default is ``1``.
    eval_on_save : bool, optional
        Force an evaluation pass before each checkpoint save. Useful for
        decoupling the save and eval schedules. Default is ``False``.
    enable_activation_offloading : bool, optional
        Offload saved activation tensors to CPU during the backward pass to
        reduce peak GPU memory. Best combined with activation checkpointing.
        Trades GPU memory for CPU memory bandwidth. Default is ``False``.
    detect_anomaly : bool, optional
        Enable ``torch.autograd`` anomaly detection for debugging NaN/Inf
        gradients. Adds significant overhead — use only for debugging.
        Default is ``False``.
    sdpa_backend : list of str, str, or None, optional
        Scaled Dot-Product Attention backend(s). Valid string values are
        ``"math"``, ``"flash"``, ``"efficient"``, and ``"cudnn"``. Pass a list
        to specify multiple backends; if ``sdpa_set_priority=True``, the list is
        treated as a priority order. ``None`` uses PyTorch's default selection.
        Default is ``None``.
    sdpa_set_priority : bool, optional
        When ``sdpa_backend`` is a list, interpret it as a priority order rather
        than requiring all backends to be available. Default is ``False``.
    float32_matmul_precision : str or None, optional
        Float32 matrix-multiplication precision on Ampere+ GPUs. One of
        ``"highest"`` (full IEEE, slowest), ``"high"`` (TF32, ~10–20 % speedup),
        or ``"medium"`` (more aggressive, may impact accuracy). ``None`` leaves
        the PyTorch default unchanged. Default is ``None``.
    dynamo_recompile_limit : int or None, optional
        Override ``torch._dynamo.config.recompile_limit``. Increase when
        ``torch.compile()`` produces frequent recompilation warnings. ``None``
        leaves the default unchanged. Default is ``None``.
    mixed_precision : str or None, optional
        Automatic Mixed Precision mode. ``None`` or ``"no"`` disables AMP.
        ``"bf16"`` enables bfloat16 autocast without loss scaling (recommended
        for Ampere+ GPUs). ``"fp16"`` enables float16 autocast with
        ``GradScaler`` loss scaling. Default is ``None``.
    fp8_recipe : str or None, optional
        FP8 training recipe via ``torchao``. Converts ``nn.Linear`` layers to
        ``Float8Linear``. One of ``"tensorwise"`` (fastest), ``"rowwise"``
        (more accurate), or ``"rowwise_with_gw_hp"`` (most accurate). ``None``
        disables FP8. Orthogonal to ``mixed_precision``; combine both for FP8
        matmuls with bfloat16 non-linear ops. Requires CUDA SM >= 8.9.
        Default is ``None``.
    fp8_dim_alignment : int, optional
        Minimum alignment for FP8 ``Linear`` layer dimensions. Layers whose
        ``in_features`` or ``out_features`` are not divisible by this value are
        skipped. Hardware requires 16. Default is ``16``.
    qat_recipe : str or None, optional
        Quantization-aware training recipe via ``torchao``. Inserts
        ``FakeQuantizedLinear`` modules so the forward pass simulates the
        target low-bit precision while backward stays in full precision.
        After training, run ``forgather finalize --quantize <recipe>`` to
        produce the real low-bit deployment artifact. Mutually exclusive with
        ``fp8_recipe``. See ``docs/trainers/qat-training.md`` for the recipe
        list. Default is ``None``.
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

    # Automatic Mixed Precision mode.
    # None or "no": disabled (default)
    # "bf16": bfloat16 autocast, no loss scaling (recommended for Ampere+ GPUs)
    # "fp16": float16 autocast with GradScaler loss scaling
    mixed_precision: str | None = None

    # FP8 training via torchao. Converts nn.Linear to Float8Linear for FP8 matmuls.
    # Recipes: "tensorwise" (fastest), "rowwise" (more accurate), "rowwise_with_gw_hp" (most accurate)
    # None = disabled. Orthogonal to mixed_precision (use both for FP8 matmuls + bf16 non-linear ops).
    # Requires CUDA SM >= 8.9 (RTX 4090, H100, etc.) and matmul dims divisible by 16.
    fp8_recipe: str | None = None

    # Minimum alignment for FP8 Linear layer dimensions. Layers with in_features or
    # out_features not divisible by this value are skipped. Hardware requires 16.
    fp8_dim_alignment: int = 16

    # Quantization-aware training (QAT) via torchao. Inserts FakeQuantizedLinear
    # modules in the prepare phase; convert is done post-training via
    # `forgather finalize --quantize <recipe>`. Mutually exclusive with fp8_recipe.
    # See src/forgather/ml/qat_recipes.py for the recipe table.
    qat_recipe: str | None = None

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

        # Validate mixed_precision
        if self.mixed_precision is not None:
            if self.mixed_precision == "no":
                self.mixed_precision = None
            elif self.mixed_precision not in ("bf16", "fp16"):
                raise ValueError(
                    f"mixed_precision must be None, 'no', 'bf16', or 'fp16', "
                    f"got '{self.mixed_precision}'"
                )

        # Validate fp8_recipe
        _FP8_RECIPES = ("tensorwise", "rowwise", "rowwise_with_gw_hp")
        if self.fp8_recipe is not None:
            if self.fp8_recipe not in _FP8_RECIPES:
                raise ValueError(
                    f"fp8_recipe must be one of {_FP8_RECIPES}, got '{self.fp8_recipe}'"
                )

        # Validate qat_recipe
        if self.qat_recipe is not None:
            from forgather.ml.qat_recipes import QAT_RECIPES

            if self.qat_recipe not in QAT_RECIPES:
                raise ValueError(
                    f"qat_recipe must be one of {QAT_RECIPES}, got '{self.qat_recipe}'"
                )

        # Linear-swap recipes are mutually exclusive (each replaces nn.Linear
        # with a different specialised class). Add new recipes here to keep
        # the check single-source.
        _LINEAR_SWAP_RECIPES = ("fp8_recipe", "qat_recipe")
        _active = [name for name in _LINEAR_SWAP_RECIPES if getattr(self, name)]
        if len(_active) > 1:
            raise ValueError(
                f"Linear-swap recipes are mutually exclusive "
                f"(each replaces nn.Linear); set at most one. Got: {_active}"
            )


TBaseTrainingArguments = TypeVar("TBaseTrainingArguments", bound=BaseTrainingArguments)


class BaseTrainer(
    ExtensibleTrainer, Stateful, StatefulProvider, Generic[TBaseTrainingArguments]
):
    """Abstract base class implementing common trainer infrastructure.

    Provides callback management, checkpoint coordination, training-state tracking,
    and the ``PyTorch Stateful`` interface. The actual training and evaluation loops
    are left abstract so that concrete subclasses (``Trainer``, ``AccelTrainer``,
    ``PipelineTrainer``) can implement them with their own parallelism strategy.

    This class intentionally mirrors the public surface of ``transformers.Trainer``
    to make porting existing training scripts straightforward.

    Parameters
    ----------
    args : TBaseTrainingArguments
        Training configuration dataclass. Must be an instance of
        ``BaseTrainingArguments`` or one of its subclasses.
    model : torch.nn.Module or None, optional
        Pre-constructed model. Either ``model`` or ``model_init`` must be
        provided. Default is ``None``.
    data_collator : DataCollatorT or None, optional
        Callable that collates a list of dataset examples into a batch dict.
        Default is ``None``.
    train_dataset : IterableDatasetT or None, optional
        Training dataset (``torch.utils.data.Dataset`` or any iterable).
        Default is ``None``.
    eval_dataset : IterableDatasetT or None, optional
        Evaluation dataset. Default is ``None``.
    processing_class : PreprocessingClassT or None, optional
        Tokenizer or feature extractor saved alongside model weights.
        Default is ``None``.
    model_init : Callable[[], torch.nn.Module] or None, optional
        Zero-argument factory that constructs the model. Required when ``model``
        is ``None`` (e.g. for pipeline training where construction must happen
        inside the trainer). Default is ``None``.
    callbacks : list of TrainerCallback or None, optional
        Callbacks to install. When ``None``, ``default_callbacks()`` is used.
        Default is ``None``.
    compute_loss_func : LossFunctionT or None, optional
        Custom loss function. When ``None``, the trainer computes cross-entropy
        from model logits. Default is ``None``.

    Attributes
    ----------
    state : TrainerState
        Mutable training progress state (global step, epoch, log history, etc.).
    control : TrainerControl
        Mutable flags set by callbacks to signal save/eval/stop actions.
    checkpoint_manager : CheckpointInterface or None
        Set by ``_prepare()`` before the training loop starts.

    Raises
    ------
    AssertionError
        If neither ``model`` nor ``model_init`` is provided.
    AssertionError
        If ``args.gradient_accumulation_steps`` is not greater than 0.

    Notes
    -----
    Concrete subclasses must implement three abstract methods:

    * ``_prepare(train_dataset, eval_dataset)`` — set up dataloaders, model,
      optimizer, and checkpoint manager.
    * ``_train_loop()`` — the main training iteration loop, returning
      ``TrainOutput``.
    * ``_eval_loop()`` — the evaluation loop, returning a metrics dict.
    """

    args: TBaseTrainingArguments
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
        """Return the default callbacks for this trainer class.

        Subclasses override this to provide callbacks that are always installed
        (e.g. ``ProgressCallback``, ``InfoCallback``). The base implementation
        returns an empty list.

        Returns
        -------
        list of TrainerCallback
            Default callback instances.
        """
        return []

    def __init__(
        self,
        args: TBaseTrainingArguments,
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
        # When a callback flips control.should_training_stop (or
        # should_abort_without_save), `_dispatch_event` records
        # (callback_name, event, reason) here so the trainer can log a
        # clear "stopped by X" message on the next iteration check.
        self._stop_requested_by: tuple[str, str, str] | None = None

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

        # Lazy index: event name -> list of callbacks that define that handler.
        # Built on first dispatch of each event, invalidated on callback add/remove.
        self._event_index: dict[str, list[TrainerCallback]] = {}

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
        """Run the full training loop.

        Applies any configured PyTorch context managers (SDPA backend, activation
        offloading), calls ``_prepare()`` to set up all components, then delegates
        to ``_train_loop()``.

        Returns
        -------
        TrainOutput
            Named tuple with ``global_step``, ``training_loss``, and ``metrics``.
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
        """Run evaluation on the given dataset.

        Applies the configured SDPA backend context, calls ``_prepare()`` with
        ``train_dataset=None``, then delegates to ``_eval_loop()``.

        Parameters
        ----------
        eval_dataset : BaseDataset or None, optional
            Dataset to evaluate on. Falls back to ``self.eval_dataset`` when
            ``None``. Default is ``None``.

        Returns
        -------
        dict of str to float
            Evaluation metrics, e.g. ``{"eval_loss": 1.23}``.
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
        self._event_index.clear()

    # AbstractBaseTrainer
    @override
    def pop_callback(self, callback: TrainerCallback) -> TrainerCallback | None:
        if isinstance(callback, type):
            compare = lambda a, b: type(a) == b
        else:
            compare = lambda a, b: id(a) == id(b)
        for i, cb in enumerate(self.callbacks):
            if compare(cb, callback):
                self._event_index.clear()
                return self.callbacks.pop(i)
        return None

    # AbstractBaseTrainer
    @override
    def remove_callback(self, callback: TrainerCallback):
        self.pop_callback(callback)

    def log(self, logs: Dict[str, float]):
        """Log metrics and dispatch the ``on_log`` event to all callbacks.

        Appends the metrics dict to ``state.log_history``, then fires the
        ``on_log`` callback event. Callbacks use this to write to TensorBoard,
        wandb, or other logging backends.

        Parameters
        ----------
        logs : dict of str to float
            Metrics to record, e.g. ``{"loss": 0.5, "lr": 1e-4}``.

        Returns
        -------
        TrainerControl
            Updated control object (callbacks may have set ``should_save``,
            ``should_evaluate``, etc.).
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
        """Normalize SDPA backend specifications to ``SDPBackend`` enum values.

        Converts string names (``"math"``, ``"flash"``, ``"efficient"``,
        ``"cudnn"``) or already-resolved ``SDPBackend`` enums to the form
        expected by ``torch.nn.attention.sdpa_kernel``.

        Parameters
        ----------
        backend : list of str or SDPBackend, str, SDPBackend, or None
            Backend specification. ``None`` is passed through unchanged.

        Returns
        -------
        list of SDPBackend, SDPBackend, or None
            Resolved enum value(s), or ``None`` when input is ``None``.

        Raises
        ------
        ValueError
            If ``backend`` is not a ``str``, ``list``, ``SDPBackend``, or
            ``None``.
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
        """Dispatch a trainer event to all registered callbacks.

        Uses a lazy per-event index (built on first dispatch, invalidated on
        callback add/remove) to skip callbacks that do not implement the
        requested handler method.

        Parameters
        ----------
        event : str
            Name of the callback method to invoke, e.g. ``"on_train_begin"``.
        **kwargs
            Extra arguments forwarded to each callback handler (``logs``,
            ``metrics``, etc.).

        Returns
        -------
        TrainerControl
            The ``self.control`` object after all callbacks have been invoked.
            Callbacks may mutate it in place or return a new one.
        """
        handlers = self._event_index.get(event)
        if handlers is None:
            handlers = [
                cb for cb in self.callbacks if callable(getattr(cb, event, None))
            ]
            self._event_index[event] = handlers

        if not handlers:
            return self.control

        unwrapped_model = self.unwrapped_model()
        for callback in handlers:
            # Snapshot stop-related flags before the callback runs so we
            # can attribute any state change to a specific callback. This
            # makes "training stopped by callback X" logging possible
            # without every callback having to shout about it individually.
            old_control = self.control
            prev_stop = bool(old_control.should_training_stop)
            prev_abort = bool(getattr(old_control, "should_abort_without_save", False))

            new_control = getattr(callback, event)(
                args=self.args,
                state=self.state,
                control=old_control,
                model=unwrapped_model,
                processing_class=self.processing_class,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                trainer=self,
                **kwargs,
            )

            if new_control is not None and new_control is not old_control:
                # Callbacks may either mutate the passed control in place
                # or return a new one. A small number of older callbacks
                # do both -- mutating flags on the received object and
                # then returning a fresh one. If we blindly replace
                # `self.control` with the returned object, those in-place
                # mutations would be lost. Propagate any stop flags that
                # were set on the old object forward to the new one so
                # the trainer's main stop check still sees them.
                if old_control.should_training_stop:
                    new_control.should_training_stop = True
                if getattr(old_control, "should_abort_without_save", False):
                    new_control.should_abort_without_save = True
                self.control = new_control

            # Record the first callback that flips should_training_stop or
            # should_abort_without_save. `_stop_requested_by` is read by
            # the training loop when it logs "Training stopped ..." so the
            # cause is visible even when the callback itself is silent
            # (DivergenceDetector does log, but trainer_control callbacks
            # set the flag via RPC without a clear local log line).
            # Check both the old control (in case the callback mutated it
            # in place) and the new control (in case it was replaced) so
            # we catch the transition regardless of the callback pattern.
            now_stop = bool(self.control.should_training_stop) or bool(
                old_control.should_training_stop
            )
            now_abort = bool(
                getattr(self.control, "should_abort_without_save", False)
            ) or bool(getattr(old_control, "should_abort_without_save", False))
            if (now_stop and not prev_stop) or (now_abort and not prev_abort):
                # Prefer "abort" when both transitions happen in the same
                # call; the trainer's abort path is more restrictive and
                # that's the user-facing distinction that matters.
                reason = "abort" if (now_abort and not prev_abort) else "stop"
                if getattr(self, "_stop_requested_by", None) is None:
                    self._stop_requested_by = (callback.name, event, reason)

        return self.control

    def unwrapped_model(self) -> torch.nn.Module:
        """Return the underlying model, free of any distributed wrappers.

        Subclasses that wrap ``self.model`` in DDP, FSDP, Accelerate, or
        pipeline-parallel containers override this method to strip those wrappers
        before the model is passed to callbacks.

        Returns
        -------
        torch.nn.Module
            The bare model without any framework wrapper.
        """
        assert self.model
        return self.model

    # AbstractBaseTrainer
    @override
    def save_model(self, output_dir: Optional[os.PathLike | str] = None) -> None:
        """Save model weights and the preprocessing class (HF Trainer API compatibility).

        Writes only the model weights to ``output_dir`` (or ``args.output_dir``
        when ``output_dir`` is ``None``). The full training state (optimizer,
        scheduler, RNG, etc.) is **not** saved. For resumable training, prefer
        ``save_checkpoint()``.

        Parameters
        ----------
        output_dir : path-like or str, optional
            Destination directory. Defaults to ``args.output_dir``.
        """
        assert self.checkpoint_manager
        self.checkpoint_manager.save_model(
            output_dir=output_dir, overwrite_output_dir=self.args.overwrite_output_dir
        )

    # AbstractBaseTrainer
    @override
    def save_checkpoint(self, checkpoint_path=None) -> None:
        """Save a complete training checkpoint.

        Writes all training state to a timestamped directory under
        ``args.output_dir``. The following components are always saved:

        * Model weights (required for resuming)
        * Optimizer state (momentum buffers, adaptive learning rates, etc.)
        * LR scheduler state (current step position)
        * Training progress (``global_step``, epoch counter, etc.)
        * Dataset position (when the dataloader is stateful)
        * Random number generator states (for bit-exact reproducibility)

        Parameters
        ----------
        checkpoint_path : path-like or str, optional
            Explicit checkpoint directory path. When ``None``, a path is
            auto-generated under ``args.output_dir`` based on the current
            step count.
        """
        assert self.checkpoint_manager
        self.checkpoint_manager.save_checkpoint(checkpoint_path)

    # AbstractBaseTrainer
    @override
    def load_checkpoint(self, checkpoint_path=None) -> None:
        """Load a training checkpoint to resume training.

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

        Parameters
        ----------
        checkpoint_path : path-like or str, optional
            Path to the checkpoint directory. ``None`` auto-selects the latest
            checkpoint under ``args.output_dir``.
        """
        assert self.checkpoint_manager
        self.checkpoint_manager.load_checkpoint(checkpoint_path)

    # StatefulProvider
    @override
    def get_state_components(self) -> List[StateComponent]:
        """Return state components for checkpoint save/load.

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

        Returns
        -------
        list of StateComponent
            All checkpointable state components with their sharing patterns.
        """
        components = []
        assert self.model is not None
        # Model - REQUIRED (always must be present).
        # cast: nn.Module.load_state_dict returns _IncompatibleKeys, not None,
        # so it doesn't satisfy the Stateful protocol strictly. This is a PyTorch
        # library limitation; at runtime it works correctly.
        components.append(
            StateComponent(
                key="model",
                stateful=cast(Stateful, self.model),
                sharing_pattern=SharingPattern.GLOBAL,
                required=True,  # Model is always required
            )
        )

        # Optimizer - optional (allows changing optimizer type)
        if self.optimizer is not None:
            components.append(
                StateComponent(
                    key="optimizer",
                    stateful=cast(Stateful, self.optimizer),
                    sharing_pattern=SharingPattern.GLOBAL,
                    required=False,
                )
            )

        # LR Scheduler - optional (allows changing scheduler type)
        if self.lr_scheduler is not None:
            components.append(
                StateComponent(
                    key="scheduler",
                    stateful=cast(Stateful, self.lr_scheduler),
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
        if self.train_dataloader is not None and hasattr(
            self.train_dataloader, "state_dict"
        ):
            components.append(
                StateComponent(
                    key="dataset",
                    stateful=cast(Stateful, self.train_dataloader),
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

    def get_active_state_components(self) -> List[StateComponent]:
        """The checkpoint components actually saved/loaded for this run.

        Calls the subclass's :meth:`get_state_components` (which declares the
        FULL set the trainer is capable of) and filters it by
        ``args.checkpoint_components``. ``None`` (default) returns the full
        set — today's behavior. A list restricts the checkpoint to those keys.

        This is the single chokepoint the ``CheckpointManager`` consults, so
        every trainer's ``get_state_components()`` implementation stays intact
        regardless of the per-run selection. Dropping ``"model"`` makes the
        manager skip model-weight save/load (the model component is what
        populates ``model_state_component``); see ``CheckpointManager``.
        """
        components = self.get_state_components()
        selected = getattr(self.args, "checkpoint_components", None)
        if selected is None:
            return components
        selected_set = set(selected)
        produced = {c.key for c in components}
        unknown = selected_set - produced
        if unknown:
            logger.warning(
                "checkpoint_components lists unknown key(s) %s; this trainer "
                "produces %s — they will have no effect.",
                sorted(unknown),
                sorted(produced),
            )
        dropped = produced - selected_set
        if dropped:
            logger.info(
                "checkpoint_components excludes %s from checkpoints.",
                sorted(dropped),
            )
        return [c for c in components if c.key in selected_set]

    def _get_dataset_sharing_pattern(self) -> SharingPattern:
        """Return the dataset state sharing pattern for this trainer.

        The pattern depends on the dataloader strategy:

        * ``GLOBAL`` — rank 0 loads and dispatches data (``DataloaderDispatcher``).
        * ``PER_RANK`` — each rank loads its own shard independently.

        The base implementation always returns ``GLOBAL`` (single-GPU training).
        Subclasses with distributed training should override this method.

        Returns
        -------
        SharingPattern
            ``SharingPattern.GLOBAL`` for the single-GPU base trainer.
        """
        # For single-GPU trainer, dataset is GLOBAL
        # Subclasses with distributed training should override this method
        return SharingPattern.GLOBAL

    def get_process_groups(self) -> Dict[str, Any]:
        """Return named process groups for ``PER_GROUP`` sharing pattern.

        The checkpoint manager uses this to coordinate group-level saves (e.g.
        tensor-parallel replicas). Single-GPU trainers have no process groups.
        Subclasses implementing hybrid parallelism should override this method.

        Returns
        -------
        dict of str to Any
            Empty dict for the single-GPU base trainer.
        """
        return {}

    # Stateful
    @override
    def load_state_dict(self, state_dict):
        """Restore trainer progress state from a checkpoint state dict.

        Implements the ``torch.distributed.checkpoint.stateful.Stateful``
        interface. Restores step counters and progress tracking so training
        resumes at the exact point where it was saved. Also restores the
        ``GradScaler`` state when fp16 AMP is active.

        Parameters
        ----------
        state_dict : dict
            State dictionary previously returned by ``state_dict()``. Expected
            keys: ``global_step``, ``epoch_start_step``, ``raw_epoch``,
            ``num_input_tokens_seen``, ``total_flos``, and optionally
            ``grad_scaler``.
        """
        self.state.global_step = state_dict["global_step"]
        self.state.epoch_start_step = state_dict["epoch_start_step"]
        self.state.raw_epoch = state_dict["raw_epoch"]
        self.state.num_input_tokens_seen = state_dict["num_input_tokens_seen"]
        self.state.total_flos = state_dict["total_flos"]

        # Restore GradScaler state if present and AMP is active
        if "grad_scaler" in state_dict and hasattr(self, "amp_context"):
            self.amp_context.load_state_dict(state_dict["grad_scaler"])

    # Stateful
    @override
    def state_dict(self):
        """Return trainer progress state for checkpointing.

        Implements the ``torch.distributed.checkpoint.stateful.Stateful``
        interface. The returned dict is consumed by ``load_state_dict()`` to
        restore training from the exact saved point.

        Returns
        -------
        dict
            Training state with the following keys:

            * ``global_step`` — total optimizer updates performed.
            * ``epoch_start_step`` — global step at the start of the current epoch.
            * ``raw_epoch`` — integer epoch counter.
            * ``num_input_tokens_seen`` — total tokens processed (for throughput logging).
            * ``total_flos`` — total floating-point operations (for efficiency metrics).
            * ``grad_scaler`` — ``GradScaler`` state dict (only when fp16 AMP is active).
        """
        state = {
            "global_step": self.state.global_step,
            "epoch_start_step": self.state.epoch_start_step,
            "raw_epoch": self.state.raw_epoch,
            "num_input_tokens_seen": self.state.num_input_tokens_seen,
            "total_flos": self.state.total_flos,
        }

        # Save GradScaler state if AMP fp16 is active
        if hasattr(self, "amp_context"):
            scaler_state = self.amp_context.state_dict()
            if scaler_state:
                state["grad_scaler"] = scaler_state

        return state

    @abstractmethod
    def _prepare(
        self,
        train_dataset: Optional[BaseDataset],
        eval_dataset: Optional[BaseDataset],
    ) -> None:
        """Prepare all components for training and/or evaluation.

        Called at the start of ``train()`` or ``evaluate()``. Either dataset
        argument may be ``None`` (eval-only or train-only runs).

        Subclasses implement a typical sequence of:

        1. Create dataloaders from datasets.
        2. Initialize and move the model to the target device(s).
        3. Set up the optimizer and LR scheduler (training only).
        4. Wrap components for distributed training (DDP, Accelerate, etc.).
        5. Initialize the ``TrainerState``.
        6. Create the ``CheckpointManager``.
        7. Load from checkpoint when ``args.resume_from_checkpoint`` is set.

        Parameters
        ----------
        train_dataset : BaseDataset or None
            Training dataset. ``None`` when only evaluation is requested.
        eval_dataset : BaseDataset or None
            Evaluation dataset. ``None`` when only training is requested.
        """
        pass

    @abstractmethod
    def _train_loop(self) -> TrainOutput:
        """Execute the main training loop.

        Concrete implementations iterate over epochs and batches, perform
        forward/backward passes with gradient accumulation, step the optimizer
        and LR scheduler, dispatch callback events, and handle periodic logging,
        evaluation, and checkpointing.

        Returns
        -------
        TrainOutput
            Named tuple with ``global_step``, ``training_loss``, and a metrics
            dict aggregated over the full training run.
        """
        pass

    @abstractmethod
    def _eval_loop(self) -> dict[str, float]:
        """Execute the evaluation loop.

        Concrete implementations iterate over the evaluation dataset, perform
        forward-only passes (no gradient computation), aggregate metrics, and
        dispatch callback events.

        Returns
        -------
        dict of str to float
            Computed evaluation metrics, e.g. ``{"eval_loss": 0.5}``.
        """
        pass


def logits_from_outputs(outputs) -> Tensor:
    """Extract logits from model forward-pass outputs.

    Handles the three output conventions used across Forgather models:

    * ``Tensor`` — outputs **are** the logits.
    * Object with a ``.logits`` attribute — HuggingFace ``ModelOutput`` style.

    Parameters
    ----------
    outputs : Tensor or object with .logits
        Return value of the model's ``forward()`` call.

    Returns
    -------
    Tensor
        Logits tensor.

    Raises
    ------
    AssertionError
        If ``outputs`` is not a ``Tensor`` and has no ``.logits`` attribute.
    """
    if not isinstance(outputs, Tensor):
        assert hasattr(outputs, "logits"), f"Type is {type(outputs)}"
        return outputs.logits
    return outputs


def loss_from_outputs(outputs) -> Tensor:
    """Extract the loss scalar from model forward-pass outputs.

    Handles two output conventions:

    * ``tuple`` — the loss is the first element, ``outputs[0]``.
    * Object with a ``.loss`` attribute — HuggingFace ``ModelOutput`` style.

    Parameters
    ----------
    outputs : tuple or object with .loss
        Return value of the model's ``forward()`` call.

    Returns
    -------
    Tensor
        Loss scalar tensor.

    Raises
    ------
    AssertionError
        If ``outputs`` is a tuple whose first element is not a ``Tensor``, or
        if it has no ``.loss`` attribute.
    """
    if isinstance(outputs, tuple):
        loss = outputs[0]
        assert isinstance(loss, Tensor)
        return loss
    assert hasattr(outputs, "loss")
    return outputs.loss


def loss_and_logits_from_outputs(outputs) -> Tuple[Tensor, Tensor]:
    """Extract both loss and logits from model forward-pass outputs.

    Handles two output conventions:

    * ``tuple`` — ``(loss, logits)`` in that order.
    * Object with ``.loss`` and ``.logits`` attributes — HuggingFace
      ``ModelOutput`` style.

    Parameters
    ----------
    outputs : tuple or object with .loss and .logits
        Return value of the model's ``forward()`` call.

    Returns
    -------
    tuple of (Tensor, Tensor)
        ``(loss, logits)`` where both are ``Tensor`` instances.

    Raises
    ------
    AssertionError
        If ``outputs`` is a tuple whose elements are not both ``Tensor``, or if
        the object lacks ``.loss`` or ``.logits`` attributes.
    """
    if isinstance(outputs, tuple):
        loss, logits = outputs
        assert isinstance(loss, Tensor) and isinstance(logits, Tensor)
        return loss, logits
    assert hasattr(outputs, "loss") and hasattr(outputs, "logits")
    return outputs.loss, outputs.logits
