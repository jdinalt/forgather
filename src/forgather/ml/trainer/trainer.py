# A light-weight Trainer with an API close enough to "transformers.Trainer"
# to act as a stand-in for basic use-cases.
import gc
import logging
import os
import time
from collections.abc import Sized
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeGuard,
    TypeVar,
    cast,
    override,
)

import torch
import torchdata.nodes as tn
from dacite import from_dict
from torch import Tensor
from torch import distributed as dist
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

from forgather.ml.construct import torch_dtype
from forgather.ml.datasets import sync_dataset_state_from_dataloader
from forgather.ml.utils import default_dtype

from ..distributed import DistributedEnvInterface, prefix_logger_rank
from ..loss import RescaleLoss
from ..no_init_weights import no_init_weights
from ..sharded_checkpoint import (
    create_sharing_metadata,
    next_checkpoint_path,
    retie_parameters,
    save_checkpoint_metrics,
)
from .base_trainer import BaseTrainer, BaseTrainingArguments, logits_from_outputs
from .callbacks.default_callbacks import InfoCallback, ProgressCallback
from .checkpoint_manager import CheckpointConfig, CheckpointManager
from .periodic_function import PeriodicFunction
from .trainer_types import (
    BaseDataset,
    CheckpointInterface,
    EnableCheckpointFnT,
    FusedLossFactoryT,
    IntervalStrategy,
    LossFunctionT,
    LRSchedulerFactoryT,
    OptimizerFactoryT,
    OptimizerT,
)
from .trainer_types import TrainerState as BaseTrainerState
from .trainer_types import (
    TrainOutput,
)

logger = logging.getLogger(__name__)
prefix_logger_rank(logger)


# Type checking protocols
class ModelWithCheckpointing(Protocol):
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Any):
        pass


class HasBatchSize(Protocol):
    batch_size: int


class HasMainInputName(Protocol):
    main_input_name: str


def has_gradient_checkpointing_enable(obj: object) -> TypeGuard[ModelWithCheckpointing]:
    return hasattr(obj, "gradient_checkpointing_enable")


def has_batch_size(obj: object) -> TypeGuard[HasBatchSize]:
    return hasattr(obj, "batch_size")


def has_main_input_name(obj: object) -> TypeGuard[HasMainInputName]:
    return hasattr(obj, "main_input_name")


def enable_hf_activation_checkpointing(
    rank, module, gradient_checkpointing_kwargs=None
):
    """
    Enable activation checkpointing via Huggingface protocol
    """

    if has_gradient_checkpointing_enable(module):
        if rank == 0:
            logger.info("rank0: Enabling HF gradient checkpointing")

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = dict(
                use_reentrant=False,
            )
        module.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    else:
        logger.warning(
            "rank{rank}: Gradient HF checkpointing requested, but model does not support it!"
        )


@dataclass(kw_only=True)
class TrainerState(BaseTrainerState):
    # --- Not in HF TrainerState ---
    # The total number of processes used for training
    num_processes: int = 1  # Non-standard
    # The number of batches in an epoch
    epoch_train_steps: int = 0


@dataclass(kw_only=True)
class TrainingArguments(BaseTrainingArguments):
    """
    Training arguments specific to the simple Trainer implementation.

    Extends BaseTrainingArguments with memory optimization and model construction options.
    Maintains compatibility with HuggingFace Trainer API where possible.
    """

    # Ratio of reserved to total GPU memory to trigger GC
    # If OOM from fragmentation, lower ratio
    gc_threshold: float = 0.5

    # Construct model on meta-device and materialize directly on device
    # default: Construct model on CPU and move to device. Safest option, works in all cases.
    #          Uses no_init_weights() context when loading checkpoint to skip initialization.
    # device:  Construct model directly on device with initialization. Faster than default
    #          but may fail when model needs sharding across devices. Use when checkpoint
    #          doesn't save all buffers (e.g., RoPE).
    # meta:    Construct on meta device (no memory backing) and materialize as empty tensors
    #          on target device. Fastest option but requires loading checkpoint since model
    #          is uninitialized. May have issues with buffers not saved in checkpoint.
    construct_model_on: str = "default"  # "default" | "meta" | "device"

    # https://pytorch.org/blog/activation-checkpointing-techniques/
    # Requires "torch_compile = True" option
    activation_memory_budget: float | None = None

    # Combine gradient calculation with optimizer step, to save memory.
    # As each gradient is computed during backward(), it's immediately applied by the
    # optimizer and freed. Incompatible with max_grad_norm and gradient_accumulation_steps > 1.
    # Greatest memory savings when combined with gradient checkpointing.
    # https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html
    fuse_optim_with_backward: bool = False

    # The step at which to start collecting speed metrics
    # We default to 1, to remove the effects from torch.compile().
    # Set this to 0 to include all steps or > 0 for compile warmup time.
    speed_metrics_start_step: int = 1

    # If the train dataset has a `set_epoch(epoch: int)` method, call it at the start of each epoch.
    set_dataset_epoch: bool = True


@contextmanager
def set_train(model: torch.nn.Module, mode: bool):
    """
    Context manager which saves the mode (train/eval) on entry,
    then sets the specified mode (train = True), and finally
    restores the original mode on exit.
    """
    previous_mode = model.training
    try:
        model.train(mode)
        yield
    finally:
        model.train(previous_mode)


def maybe_cleanup_memory(alloc_threshold):
    """
    Trigger garbage collection and CUDA cache cleanup if memory usage exceeds threshold.

    Helps prevent OOM errors from memory fragmentation during long training runs.

    Args:
        alloc_threshold: Ratio of reserved to total GPU memory (0.0 to 1.0).
                        If current usage exceeds this, cleanup is triggered.
    """
    if torch.cuda.is_available():
        reserved = torch.cuda.memory_reserved()
        max_memory = torch.cuda.get_device_properties(0).total_memory
        usage_ratio = reserved / max_memory

        if usage_ratio > alloc_threshold:
            gc.collect()
            torch.cuda.empty_cache()


def optimizer_hook(optimizer, total_grad_squared, name, parameter):
    """
    Hook for fusing optimizer step with backward pass.

    This hook is registered via register_post_accumulate_grad_hook() when
    fuse_optim_with_backward=True. As each gradient is computed during backward(),
    it's immediately applied by the optimizer and freed, reducing peak memory usage.

    Greatest memory savings when combined with gradient checkpointing.

    Args:
        optimizer: The optimizer instance to apply the gradient update
        total_grad_squared: Accumulator for computing total gradient norm across all parameters
        name: Parameter name (for debugging)
        parameter: The parameter whose gradient was just computed
    """
    if total_grad_squared is not None:
        total_grad_squared += parameter.grad.square().sum().to(dtype=torch.float32)
        # norm = parameter.grad.square().sum().sqrt()
        # logger.info(f"{name} {norm}")
    optimizer.step()
    optimizer.zero_grad()


# Help static-type-checking correctly infer the type of "args"
TTrainingArguments = TypeVar("TTrainingArguments", bound=TrainingArguments)


class Trainer(BaseTrainer[TTrainingArguments], Generic[TTrainingArguments]):
    """
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
    """

    args: TTrainingArguments
    dist: DistributedEnvInterface
    optimizer_factory: OptimizerFactoryT | None
    lr_scheduler_factory: LRSchedulerFactoryT | None
    enable_activation_checkpoint_fn: EnableCheckpointFnT | None
    fused_loss_factory: FusedLossFactoryT | None

    max_steps: int
    epoch_train_steps: int
    do_train: bool
    do_eval: bool
    use_fused_loss: bool
    gradient_accumulation_step: int

    @classmethod
    def default_callbacks(cls):
        return [ProgressCallback(), InfoCallback()]

    def __init__(
        self,
        *,
        args: TTrainingArguments | dict,
        distributed_env: DistributedEnvInterface,
        optimizer_factory: Optional[OptimizerFactoryT] = None,
        # Alternative, for compatibility with transformers.Trainer
        optimizer_cls_and_kwargs: Optional[
            Tuple[Type[OptimizerT], Dict[str, Any]]
        ] = None,
        lr_scheduler_factory: Optional[LRSchedulerFactoryT] = None,
        enable_activation_checkpoint_fn: Optional[
            EnableCheckpointFnT
        ] = enable_hf_activation_checkpointing,
        fused_loss_factory: Optional[FusedLossFactoryT] = None,
        **kwargs,
    ):
        if isinstance(args, dict):
            args = cast(TTrainingArguments, from_dict(TrainingArguments, args))
        super().__init__(args=args, **kwargs)

        # HF Trainer compatibility.
        if not optimizer_factory:
            if not optimizer_cls_and_kwargs:
                optimizer_factory = partial(  # type: ignore[assignment]
                    torch.optim.AdamW,
                    lr=args.learning_rate,
                    betas=(args.adam_beta1, args.adam_beta2),
                    weight_decay=args.weight_decay,
                    eps=args.adam_epsilon,
                )
            else:
                optimizer_factory = partial(
                    optimizer_cls_and_kwargs[0], **optimizer_cls_and_kwargs[1]
                )
        self.dist = distributed_env
        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory
        self.enable_activation_checkpoint_fn = enable_activation_checkpoint_fn
        self.fused_loss_factory = fused_loss_factory
        self.use_fused_loss = False
        assert (self.model is not None) or (
            self.model_init is not None
        ), "Either a model or a model constructor must be specified."

        assert (
            self.args.max_grad_norm is None or not self.args.fuse_optim_with_backward
        ), "max_grad_norm is incompatible with fuse_optim_with_backward"

        assert (
            self.args.gradient_accumulation_steps == 1
            or not self.args.fuse_optim_with_backward
        ), "gradient_accumulation_steps={self.args.gradient_accumulation_steps} is incompatible with fuse_optim_with_backward"

        assert (
            self.loss_fn or self.args.gradient_accumulation_steps == 1
        ), f"gradient_accumulation_steps [{self.args.gradient_accumulation_steps}] > 1 requires loss_fn"

        if self.data_collator is None:
            self.data_collator = torch.utils.data.default_collate

        # Compute FLOPs per token for tracking
        # Note: model not initialized yet, will be set in _prepare()

    def _compute_flops_per_token(self) -> float:
        """
        Estimate FLOPs per token for forward + backward pass.

        Uses standard transformer formula:
        - Forward pass: 6 * num_params FLOPs per token
        - Backward pass: 2 * forward (approximately)
        - Total: 6 * num_params * 3 = 18 * num_params per token

        Returns:
            Estimated FLOPs per token for forward + backward pass
        """
        assert self.model is not None
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # 6N per forward token, 2x for backward (12N), total ~18N
        # More precise: 6N forward + 12N backward = 18N
        return 18.0 * num_params

    def _count_batch_tokens(self, input_dict: dict[str, Tensor], labels: Tensor) -> int:
        """
        Count non-padding tokens using labels tensor.

        Uses labels as the primary source since the cross-entropy ignore_index (-100)
        marks padding and special tokens that should not be counted.

        Args:
            input_dict: Batch input dictionary (unused in base, available for overrides)
            labels: Target labels with -100 (ignore_index) for positions to ignore

        Returns:
            Number of non-ignored tokens in the batch
        """
        # Labels have -100 for padding/special tokens; count only real target tokens
        return int((labels != -100).sum().item())

    def _distributed_tokens(self, tokens: Tensor) -> Tensor:
        """
        Aggregate token counts across processes.

        Base implementation for single-device training.
        Distributed trainers override to sum across ranks.

        Args:
            tokens: Token count tensor from current process

        Returns:
            Token count tensor (single-device) or aggregated across ranks (distributed)
        """
        return tokens

    def _init_distributed(self):
        """
        Subclasses are expected to override, if they support distributed training.
        If distributed training is supported, set the following variables:
          self.is_local_process_zero
          self.is_world_process_zero
          self.num_processes
        """
        assert (
            self.dist.world_size == 1
        ), "'Trainer' does not support distributed training. See subclasses for implementations which do support it."

    def _init_device(self):
        """Update / init trainer's device"""
        # If unspecified, set a default device
        if self.args.device is None:
            self.args.device = self.dist.device
        # Override for debug.
        if self.args.use_cpu:
            self.args.device = "cpu"

    def _get_dataloader(self, dataset, batch_size):
        if not isinstance(dataset, tn.BaseNode | DataLoader):
            dataloader_kwargs = {
                "batch_size": batch_size,
                "collate_fn": self.data_collator,
                "drop_last": self.args.dataloader_drop_last,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "prefetch_factor": self.args.dataloader_prefetch_factor,
                "persistent_workers": self.args.dataloader_persistent_workers,
            }

            # Use StatefulDataLoader for datasets with state if available and requested
            return StatefulDataLoader(dataset, **dataloader_kwargs)
        else:
            return dataset

    @override
    def _prepare(
        self, train_dataset: Optional[BaseDataset], eval_dataset: Optional[BaseDataset]
    ) -> None:
        """
        Prepare for training and/or evaluation
        """
        self._init_distributed()
        self._init_device()

        # Set the random seed
        if self.args.seed != -1:
            import random

            torch.manual_seed(self.args.seed)
            random.seed(self.args.seed)

        if self.args.activation_memory_budget:
            logger.info(
                f"Setting memory budget to {self.args.activation_memory_budget}"
            )
            try:
                torch._functorch.config.activation_memory_budget = (  # type: ignore[attr-defined]
                    self.args.activation_memory_budget
                )
            except AttributeError:
                logger.warning(
                    "PyTorch does not appear to support the experimental activation_memory_budget API"
                )

        self._init_dataloaders(train_dataset, eval_dataset)
        self._prepare_model()
        if self.args.torch_compile:
            logger.info(
                f"Compiling model: backend={self.args.torch_compile_backend}, mode={self.args.torch_compile_mode},"
                f"dynamic={self.args.torch_compile_dynamic}, fullgraph={self.args.torch_compile_full_graph},"
                f"activation_memory_budget={self.args.activation_memory_budget}"
            )

            if os.environ.get("PARTITIONER_MEMORY_BUDGET_PARETO", 0):
                logger.warning(
                    "Computing paritioner Pareto memory budget -- be patient, this takes time..."
                )
            self._compile_model()

        if self.do_train:
            self._init_optimizer()

        self._wrap()
        self.state = self._init_state()
        self.checkpoint_manager = self._init_checkpoint_manager()

        # Restore from checkpoint if specified (after state is initialized)
        if self.args.resume_from_checkpoint:
            checkpoint_path = self.args.resume_from_checkpoint
            if isinstance(checkpoint_path, bool) or checkpoint_path == "":
                checkpoint_path = None
            self.load_checkpoint(checkpoint_path)

        self._dispatch_event("on_init_end")

    def _wrap(self) -> None:
        """
        Hook for wrapping objects after construction in _prepare().

        Called after dataloaders, model, optimizer, and scheduler are initialized
        but before training begins. Subclasses use this to wrap objects for
        distributed training or other runtime modifications.

        Examples:
        - AccelTrainer: Wraps model/optimizer/dataloaders with Accelerate
        - PipelineTrainer: Sets up pipeline parallel stage wrapping

        See src/forgather/ml/trainer/accelerate/accel_trainer.py and
        src/forgather/ml/trainer/pipeline/pipeline_trainer.py for examples.
        """
        pass

    def _compile_model(self):
        """
        Compile model using torch.compile().

        Hook for model compilation. Subclasses may override to customize compilation
        behavior or apply compilation to additional wrapped objects.
        """
        assert self.model is not None
        self.model.compile(
            backend=self.args.torch_compile_backend,
            mode=self.args.torch_compile_mode,
            dynamic=self.args.torch_compile_dynamic,
            fullgraph=self.args.torch_compile_full_graph,
        )

    def _init_checkpoint_manager(self) -> CheckpointManager:  # type: ignore[override]
        """
        Initialize checkpoint manager hook.

        Creates the CheckpointManager responsible for saving/loading complete training
        state including model, optimizer, scheduler, dataset state, and RNG state.

        Subclasses may override to provide custom checkpoint management behavior.

        Returns:
            CheckpointInterface: Initialized checkpoint manager
        """
        cp_config = CheckpointConfig(
            output_dir=self.args.output_dir,
            save_total_limit=self.args.save_total_limit,
            save_on_each_node=self.args.save_on_each_node,
            save_safetensors=self.args.save_safetensors,
        )

        checkpoint_manager = CheckpointManager(
            config=cp_config,
            dist=self.dist,
            model=self.unwrapped_model(),
            model_preprocessor=self.processing_class,
            stateful_provider=self,
        )
        # Set trainer reference for callback state save/load
        checkpoint_manager.trainer = self
        # Set preserve_n_best from training args
        if hasattr(self.args, "preserve_n_best"):
            checkpoint_manager.preserve_n_best = self.args.preserve_n_best
        return checkpoint_manager

    def _init_dataloaders(self, train_dataset, eval_dataset) -> None:
        """
        Initialize train and evaluation dataloaders (_prepare() sub-step 1).

        Creates StatefulDataLoader instances that support checkpointing dataset iteration state.
        Also computes training step counts for scheduling logging/evaluation/checkpointing.

        Args:
            train_dataset: Training dataset or None for eval-only
            eval_dataset: Evaluation dataset or None for train-only
        """
        # _prepare() sub-step 1
        self.max_steps = 0
        self.epoch_train_steps = self.args.epoch_train_steps

        self.do_train = train_dataset is not None
        self.do_eval = eval_dataset is not None

        if self.do_train:
            if (
                self.args.set_dataset_epoch
                and self.args.num_train_epochs > 1.0
                and not hasattr(train_dataset, "set_epoch")
            ):
                logger.warning(
                    "Train dataset does not support `set_epoch` and training for > 1 epoch. Dataset will not be reshuffled after each epoch"
                )
                self.args.set_dataset_epoch = False
            self.train_dataloader = self._get_dataloader(
                train_dataset, self.args.per_device_train_batch_size
            )

            self._update_training_steps()

        if self.do_eval:
            self.eval_dataloader = self._get_dataloader(
                eval_dataset, self.args.per_device_eval_batch_size
            )

    def _prepare_model(self) -> None:
        """
        Construct/initialize model and move to device (_prepare() sub-step 2).

        Handles three model construction strategies based on construct_model_on:
        - default: Safe, works everywhere. Constructs on CPU (with no_init_weights if loading
                  checkpoint), then moves to device.
        - meta: Fastest. Constructs on meta device (no memory), materializes empty on device.
                Requires loading checkpoint. May have issues with non-persistent buffers.
        - device: Middle ground. Constructs directly on device with initialization. Faster
                 than default but may fail with model sharding. Good for models with buffers
                 not saved in checkpoint (e.g., RoPE).

        Also sets up gradient checkpointing if enabled and initializes fused loss if available.
        """
        # _prepare() sub-step 2
        match self.args.construct_model_on:
            case "default":
                if self.model_init:
                    logger.info(
                        f"Constructing model on default device and moving to {self.args.device}"
                    )
                    with ExitStack() as exit_stack:
                        if self.args.default_dtype:
                            exit_stack.enter_context(
                                default_dtype(torch_dtype(self.args.default_dtype))
                            )
                        if self.args.resume_from_checkpoint:
                            exit_stack.enter_context(no_init_weights())
                        self.model = self.model_init()
                else:
                    logger.info(f"Moving model to {self.args.device}")
                    assert self.model is not None
                self.model = self.model.to(self.args.device)
            case "meta":
                assert (
                    self.model_init
                ), "Constructing the model on meta device requires model_init"
                assert (
                    self.args.resume_from_checkpoint
                ), "Constructing model on meta-device requires loading parameters from checkpoint"

                logger.info(
                    f"Constructing model on meta device and materializing on {self.args.device}"
                )
                # TODO: Identify if the model has buffers with "persist=False" and warn loudly!
                if not self.args.resume_from_checkpoint:
                    logger.warning(
                        f"Uninitialized model constructed on meta-device and not loading from checkpoint!"
                    )
                with ExitStack() as exit_stack:
                    if self.args.default_dtype:
                        exit_stack.enter_context(
                            default_dtype(torch_dtype(self.args.default_dtype))
                        )
                    exit_stack.enter_context(torch.device("meta"))
                    self.model = self.model_init()
                sharing_metadata = create_sharing_metadata(self.model)
                self.model.to_empty(device=self.args.device)
                # to_empty() will break tied parameters. Fix them!
                retie_parameters(self.model, sharing_metadata)
            case "device":
                assert (
                    self.model_init
                ), "Constructing the model on device requires model_init"
                logger.info(
                    f"Constructing and initializing model directly on {self.args.device}"
                )
                with ExitStack() as exit_stack:
                    if self.args.default_dtype:
                        exit_stack.enter_context(
                            default_dtype(torch_dtype(self.args.default_dtype))
                        )
                    exit_stack.enter_context(torch.device(self.args.device))
                    if self.args.resume_from_checkpoint:
                        exit_stack.enter_context(no_init_weights())
                    self.model = self.model_init()
            case _:
                raise ValueError("Requires one of: default|meta|device")
        assert self.model is not None
        if self.args.gradient_checkpointing:
            if self.enable_activation_checkpoint_fn is None:
                if self.dist.rank == 0:
                    logger.warning(
                        f"Activation checkpointing requested, but no function defined!"
                    )
            else:
                # Enable activation checkpointing for all modules in the pipeline.
                self.enable_activation_checkpoint_fn(self.dist.rank, self.model)
        self.loss_fn = self._maybe_get_fused_loss_fn(self.model, self.loss_fn)
        self._wrap_loss_fn()

        # Compute FLOPs per token for performance tracking
        self._flops_per_token = self._compute_flops_per_token()
        if self.dist.rank == 0:
            logger.info(f"Estimated FLOPs per token: {self._flops_per_token:.2e}")

    def _wrap_loss_fn(self):
        # Rescale loss by gradient accumulation steps.
        self.loss_fn = RescaleLoss(
            self.loss_fn, 1 / self.args.gradient_accumulation_steps
        )

    def _maybe_get_fused_loss_fn(
        self, module: torch.nn.Module, default_loss_fn: Optional[LossFunctionT]
    ):
        """
        Attempt to enable fused loss-logits computation for memory optimization.

        Fused loss combines the final linear layer (computing logits from hidden states)
        with the cross-entropy loss computation. This avoids materializing the full logits
        tensor in memory, which is critical for models with large vocabulary sizes where
        logits can be gigabytes in size.

        For example, with vocab_size=50k, batch_size=8, seq_len=2048:
        - Unfused: logits tensor is 8 * 2048 * 50000 * 4 bytes = ~3.2 GB
        - Fused: Only computes logits for one token at a time, dramatically less memory

        Requires:
        - fused_loss_factory provided to Trainer constructor
        - Model supports get_output_embeddings() (returns final linear layer)
        - Model supports return_hidden_states=True (returns hidden states instead of logits)

        See src/forgather/ml/loss.py and docs/fused_loss/ for implementations and details.

        If model supports fused-loss, and fused loss function is returned, sets:
            self.use_fused_loss = True

        Args:
            module: The prospective module to enable fused-loss on
            default_loss_fn: The default loss function, if unsupported

        Returns:
            The fused loss function or default_loss_fn, if unsupported
        """
        if self.fused_loss_factory:
            if not hasattr(module, "get_output_embeddings"):
                logger.warning(
                    "Model does not support get_output_embeddings() for fused_loss_factory()"
                )
                return default_loss_fn
            if not getattr(module, "can_return_hidden_states", False):
                logger.warning(
                    f"Model does not support 'return_hidden_states' API; fused loss will not be used."
                )
                return default_loss_fn
            logger.info("Enabled fused loss-logits function")
            self.use_fused_loss = True
            return self.fused_loss_factory(module.get_output_embeddings())  # type: ignore[operator]
        return default_loss_fn

    def _init_optimizer(self) -> None:
        """
        Initialize optimizer and learning rate scheduler (_prepare() sub-step 3).

        Creates optimizer from factory and optionally sets up:
        - Learning rate scheduler (from factory or HF get_scheduler)
        - Fused backward/optimizer hooks if fuse_optim_with_backward=True
        """
        # _prepare() sub-step 3
        assert self.model is not None
        if self.optimizer is None:
            assert self.optimizer_factory is not None
            self.optimizer = self.optimizer_factory(self.model.named_parameters())

            # Combine backward with optimizer step?
            if self.args.fuse_optim_with_backward:
                self._total_grad_squared = torch.zeros(
                    1, device=self.args.device, dtype=torch.float32
                )

                for name, p in self.model.named_parameters():
                    if p.requires_grad:
                        hook = partial(
                            optimizer_hook,
                            self.optimizer,
                            self._total_grad_squared,
                            name,
                        )
                        p.register_post_accumulate_grad_hook(hook)

        if self.lr_scheduler is None:
            if self.lr_scheduler_factory is not None:
                assert self.optimizer is not None
                self.lr_scheduler = self.lr_scheduler_factory(self.optimizer)
            elif self.args.lr_scheduler_type:
                from transformers import get_scheduler

                self.lr_scheduler = get_scheduler(
                    name=self.args.lr_scheduler_type,
                    optimizer=cast(Any, self.optimizer),  # type: ignore[arg-type]
                    num_warmup_steps=self.args.warmup_steps,
                    num_training_steps=self.max_steps,
                    scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
                )

    def _maybe_log_save_evaluate(
        self,
        loss_log,
        total_norm_log,
        tokens_log,
        periodic_log,
        periodic_eval,
        periodic_save,
    ):
        # The logic diverges slighlty from HF, in that this in an 'and'
        # It's not clear if this a bug? It's also not clear if any callbacks depend on this?
        # Until proven otherwise, try to do the right thing here.
        if periodic_log.step() or self.control.should_log:
            log_steps = periodic_log.reset()
            self.control.should_log = False

            self._log_step(loss_log, total_norm_log, tokens_log)

        # Handle evaluation (normal schedule or control-triggered)
        eval_metrics = None
        should_eval = periodic_eval.step() or self.control.should_evaluate

        # Force eval if saving and eval_on_save enabled
        should_save = periodic_save.step() or self.control.should_save
        if should_save and self.args.eval_on_save and self.eval_dataset is not None:
            should_eval = True

        if should_eval:
            periodic_eval.reset()
            self.control.should_evaluate = False

            # Do eval
            maybe_cleanup_memory(self.args.gc_threshold)
            eval_metrics = self._eval_loop()

        # Handle checkpointing - normal schedule or control-triggered
        if should_save:
            periodic_save.reset()
            self.control.should_save = False
            assert self.checkpoint_manager

            # Determine checkpoint path BEFORE saving
            checkpoint_path = next_checkpoint_path(
                self.args.output_dir, str(self.state.global_step)
            )

            # Update best checkpoints list BEFORE saving (so preserved list is correct)
            if self.args.preserve_best_model and eval_metrics:
                checkpoint_manager = cast(CheckpointManager, self.checkpoint_manager)
                checkpoint_manager.update_best_checkpoints(
                    checkpoint_path=checkpoint_path,
                    metrics=eval_metrics,
                    metric_key=self.args.best_model_metric,
                    greater_is_better=self.args.best_model_greater_is_better,
                    preserve_n_best=self.args.preserve_n_best,
                )
                # Update state for compatibility
                if checkpoint_manager.best_checkpoints:
                    self.state.best_metric = checkpoint_manager.best_checkpoints[0][1]
                    self.state.best_model_checkpoint = (
                        checkpoint_manager.best_checkpoints[0][0]
                    )

            # Now save checkpoint (deletion will use updated preserved list)
            saved_path = self.checkpoint_manager.save_checkpoint(
                checkpoint_id=str(self.state.global_step)
            )
            self._dispatch_event("on_save")

            # Save metrics file
            if eval_metrics:
                save_checkpoint_metrics(saved_path, eval_metrics)

    def load_best_model(self) -> None:
        """
        Load the best model from the best checkpoint.

        Called at end of training when load_best_model_at_end=True to restore
        the checkpoint with the best metric value seen during training.
        """
        if not self.state.best_model_checkpoint:
            logger.warning("No best model checkpoint available to load")
            return

        if not os.path.exists(self.state.best_model_checkpoint):
            logger.warning(
                f"Best model checkpoint path does not exist: {self.state.best_model_checkpoint}"
            )
            return

        logger.info(f"Loading best model from {self.state.best_model_checkpoint}")
        self.load_checkpoint(self.state.best_model_checkpoint)

    @override
    def _train_loop(self) -> TrainOutput:
        periodic_log = PeriodicFunction(
            global_step=self.state.global_step,
            strategy=self.args.logging_strategy,
            period=self.args.logging_steps,
            epoch_period=self.epoch_train_steps,
            first_step=self.args.logging_first_step,
        )
        periodic_eval = PeriodicFunction(
            global_step=self.state.global_step,
            strategy=self.args.eval_strategy,
            period=self.args.eval_steps,
            epoch_period=self.epoch_train_steps,
            first_step=self.args.eval_delay,
        )
        periodic_save = PeriodicFunction(
            global_step=self.state.global_step,
            strategy=self.args.save_strategy,
            period=self.args.save_steps,
            epoch_period=self.epoch_train_steps,
            first_step=0,
        )

        assert self.optimizer is not None
        assert self.model is not None
        assert self.train_dataloader is not None

        # Just to be sure...
        self.optimizer.zero_grad()

        start_time = None
        train_steps = 0
        self._dispatch_event("on_train_begin")

        # Holds loss, grad-norm, and token samples between log steps
        accumulated_grad_norm = []
        accumulated_loss = []
        accumulated_tokens = []

        # Context manager for setting model.train()/eval()
        with set_train(self.model, True):
            # Epoch loop
            while True:
                self.control.should_epoch_stop = False
                if self.args.set_dataset_epoch and self.state.raw_epoch > 0:
                    # If supported, reshuffle dataset at the start of each epoch
                    logger.debug(f"Setting dataset epoch {self.state.raw_epoch}")
                    self.train_dataloader.dataset.set_epoch(self.state.raw_epoch)  # type: ignore[union-attr]
                data_iterator = iter(self.train_dataloader)
                self._dispatch_event("on_epoch_begin")

                while True:
                    self._dispatch_event("on_step_begin")

                    try:
                        loss, total_norm, tokens = self._train_step(data_iterator)
                    except StopIteration:
                        self.state.raw_epoch += 1
                        self.state.epoch_start_step = self.state.global_step
                        if self.state.raw_epoch >= self.args.num_train_epochs:
                            self.control.should_epoch_stop = True
                        self._dispatch_event("on_step_end")
                        break

                    accumulated_grad_norm.append(total_norm)
                    accumulated_loss.append(loss)
                    accumulated_tokens.append(tokens)

                    # Increment global step
                    self.state.global_step += 1

                    # Compute epoch as continuous value from global steps
                    self.state.epoch = float(self.state.raw_epoch) + (
                        float(self.state.global_step - self.state.epoch_start_step)
                        / float(self.epoch_train_steps)
                    )

                    self._dispatch_event("on_step_end")
                    self._maybe_log_save_evaluate(
                        accumulated_loss,
                        accumulated_grad_norm,
                        accumulated_tokens,
                        periodic_log,
                        periodic_eval,
                        periodic_save,
                    )

                    train_steps += 1

                    # Check for early stop condition
                    if (
                        self.control.should_training_stop
                        or self.control.should_abort_without_save
                        or self.state.global_step >= self.max_steps
                    ):
                        self.control.should_epoch_stop = True
                        break

                    # Periodic GC
                    maybe_cleanup_memory(self.args.gc_threshold)

                    # maybe delay start of metrics recording for torch.compile()
                    if train_steps == self.args.speed_metrics_start_step:
                        start_time = time.time()

                self._dispatch_event("on_epoch_end")
                if self.control.should_epoch_stop:
                    break

        # Final log-eval-save step; skip on abort condition
        if not self.control.should_abort_without_save:
            # Force save, if we have not already saved on this step and save enabled.
            if periodic_save.rel_step != 0 and self.args.save_strategy != "no":
                logger.info(f"Saving final checkpoint at step {self.state.global_step}")
                # If load best model, we need to evaluate it too.
                if self.args.load_best_model_at_end:
                    self.control.should_evaluate = True
                self.control.should_save = True
            self._maybe_log_save_evaluate(
                accumulated_loss,
                accumulated_grad_norm,
                accumulated_tokens,
                periodic_log,
                periodic_eval,
                periodic_save,
            )

        # Load best model at end if requested
        if self.args.load_best_model_at_end:
            logger.info(f'Loading best model "{self.state.best_model_checkpoint}"')
            self.load_best_model()

        metrics = self._end_train_loop(
            start_time, train_steps - self.args.speed_metrics_start_step
        )
        self.log(metrics)

        # Log best checkpoints summary at end of training
        if (
            self.args.preserve_best_model
            and self.checkpoint_manager
            and self.state.is_world_process_zero
        ):
            summary = cast(
                CheckpointManager, self.checkpoint_manager
            ).get_best_checkpoints_summary(metric_key=self.args.best_model_metric)
            logger.info(f"\n{'='*60}\nTraining complete!\n{summary}\n{'='*60}")

        self._dispatch_event("on_train_end")
        return TrainOutput(self.state.global_step, metrics)

    @override
    @torch.no_grad()
    def _eval_loop(self) -> Dict[str, float]:
        """
        The inner evaluation loop
        """
        assert self.model is not None
        assert self.eval_dataloader is not None
        with set_train(self.model, False):
            total_loss = torch.zeros(1, device=self.args.device)
            step = -1
            for step, batch in enumerate(self.eval_dataloader):
                if self.args.max_eval_steps > 0 and step >= self.args.max_eval_steps:
                    break
                input_dict, labels = self._prepare_batch(batch)
                outputs = self._prediction_step(input_dict, labels)
                loss = outputs["loss"]
                assert loss is not None
                total_loss += loss
                self._dispatch_event("on_prediction_step")
            assert step >= 0, "The eval dataloader did not yield any examples"

            metrics = {"eval_loss": (total_loss / (step + 1)).item()}
            if isinstance(self.eval_dataloader, StatefulDataLoader):
                sync_dataset_state_from_dataloader(self.eval_dataloader)
            self._dispatch_event("on_evaluate", metrics=metrics)
            return metrics

    def _clip_grad_norm(
        self, max_grad_norm: float | None, norm_type: float = 2.0
    ) -> Optional[Tensor]:
        """
        Clip gradients by norm.

        Returns:
            The total norm of the parameters (as per PyTorch's API).

        Raises:
            RuntimeError: If parameters are not of supported types for foreach=True.
        """
        # In the case of fused backward / optimizer step, we accumulate squared norm
        # in the optimizer hook. Compute norm via sqrt() and reset accumulator
        if self.args.fuse_optim_with_backward:
            total_norm = self._total_grad_squared.sqrt()
            self._total_grad_squared -= self._total_grad_squared
            return total_norm

        assert self.model is not None
        # If not clipping, just compute and return it
        if max_grad_norm is None or max_grad_norm == 0:
            grads = [p.grad for p in self.model.parameters() if p.grad is not None]

            total_norm = torch.nn.utils.get_total_norm(
                grads, norm_type=norm_type, foreach=True
            )
            return total_norm

        # Otherwise, use fused clip_grad_norm_
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_grad_norm,
            norm_type=norm_type,
            foreach=False if self.args.device == "cpu" else True,
        )

        return total_norm

    def _train_step(
        self, data_iterator: Iterator[dict[str, Tensor]]
    ) -> Tuple[Tensor, Tensor | None, Tensor]:
        """
        Perform a single training step, with optional gradient accumulation.

        Args:
            data_iterator: Iterator over training batches

        Returns:
            Tuple of (loss, grad_norm, tokens). Loss is unscaled for logging consistency.
            grad_norm is None if not computed on this step.
            tokens is the total non-padding tokens processed in this step.
        """
        accumulated_losses = []
        accumulated_tokens = 0

        for gradient_step in range(self.args.gradient_accumulation_steps):
            self.gradient_accumulation_step = gradient_step + 1
            input_dict, labels = self._prepare_batch(next(data_iterator))
            loss = self._forward_backward_step(input_dict, labels)
            accumulated_losses.append(loss)
            # Count tokens in this micro-batch (local, not yet synchronized across ranks)
            accumulated_tokens += self._count_batch_tokens(input_dict, labels)

        assert self._should_sync_gradients()
        assert self.optimizer is not None

        total_norm = self._clip_grad_norm(self.args.max_grad_norm)
        if not self.args.fuse_optim_with_backward:
            self.optimizer.step()
            self.optimizer.zero_grad()
        self._dispatch_event("on_optimizer_step")
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        loss = torch.sum(torch.stack(accumulated_losses))
        loss = self._distributed_loss(loss)

        # Return local token count as tensor; synchronization deferred to _log_step()
        # to amortize the cost of all_reduce across many steps
        tokens_tensor = torch.tensor(
            accumulated_tokens, device=self.args.device, dtype=torch.int64
        )
        return loss, total_norm, tokens_tensor

    def _forward_backward_step(
        self, input_dict: dict[str, Tensor], labels: Tensor
    ) -> Tensor:
        """
        Execute forward pass followed by backward pass.

        Handles both standard and fused loss computation. When fused loss is enabled,
        requests hidden states instead of logits to avoid materializing the large
        logits tensor.

        Args:
            input_dict: Model inputs (input_ids, attention_mask, etc.)
            labels: Target labels for loss computation

        Returns:
            Detached loss tensor for logging
        """
        assert self.model is not None
        assert self.loss_fn is not None
        if self.use_fused_loss:
            input_dict["return_hidden_states"] = True  # type: ignore[assignment]
        outputs = self.model(**input_dict)
        logits = logits_from_outputs(outputs)
        loss = self.loss_fn(logits, labels)
        self._backward(loss)
        return loss.detach()

    def _backward(self, loss: Tensor) -> None:
        """
        Execute backward pass to compute gradients.

        Subclasses may override to customize backward behavior (e.g., for pipeline parallelism).

        Args:
            loss: Loss tensor to backpropagate
        """
        loss.backward()

    def _should_sync_gradients(self) -> bool:
        """
        Determine if gradients should be synchronized across processes.

        Normally, this just compares the gradient accumulation step against the total
        accumulation steps.

        If overridden, as is the case for the Accelerate trainer, an assert verifies that the logic
        agrees with when gradient synchronization is expected.

        Returns:
            True if gradients should be synchronized on the current forward-backward step
        """
        return self.gradient_accumulation_step == self.args.gradient_accumulation_steps

    def _update_training_steps(self) -> None:
        """
        Compute training step counts from dataloader length.

        Calculates:
        - epoch_train_steps: Number of batches per epoch
        - max_steps: Total optimizer updates across all epochs

        With gradient accumulation: global_steps = batch_count // gradient_accumulation_steps

        This may need to be called multiple times if dataloader is wrapped (e.g., by Accelerate)
        and its length changes. Also synchronizes dataset state to ensure accurate length.
        """
        if isinstance(self.train_dataloader, StatefulDataLoader):
            sync_dataset_state_from_dataloader(self.train_dataloader)

        # The number of training steps in a single epoch (batch processing steps)
        if isinstance(self.train_dataloader, Sized):
            self.epoch_train_steps = len(self.train_dataloader)

        # Convert to effective global steps (optimizer updates) with gradient accumulation
        effective_epoch_steps = (
            self.epoch_train_steps // self.args.gradient_accumulation_steps
        )

        # The total number of global training steps (optimizer updates) in all epochs
        self.max_steps = self.args.num_train_epochs * effective_epoch_steps

        # If limit is specified, constrain to limit.
        if self.args.max_steps >= 0:
            self.max_steps = min(self.args.max_steps, self.max_steps)

        if self.state is not None:
            self.state.max_steps = self.max_steps

    def _init_state(self) -> TrainerState:
        """
        Initialize trainer state for tracking training progress.

        Creates TrainerState with training step counts, batch sizes, and process info.
        State is saved/restored with checkpoints to resume training accurately.

        Key state fields:
        - global_step: Total optimizer updates since training start (0-indexed)
        - raw_epoch: Integer epoch counter, increments at end of each dataset iteration
        - epoch_start_step: Global step when raw_epoch was last incremented
        - epoch: Continuous value = raw_epoch + fractional progress through current epoch
        - best_metric/best_model_checkpoint: Best model tracking for load_best_model_at_end

        Returns:
            TrainerState: Initialized state object
        """

        if self.do_train:
            assert has_batch_size(self.train_dataloader)
            return TrainerState(
                max_steps=self.max_steps,
                logging_steps=self.args.logging_steps,
                eval_steps=self.args.eval_steps,
                num_train_epochs=int(self.args.num_train_epochs),
                train_batch_size=self.train_dataloader.batch_size,
                epoch_train_steps=self.epoch_train_steps,
                is_local_process_zero=self.is_local_process_zero,
                is_world_process_zero=self.is_world_process_zero,
                num_processes=self.num_processes,
                save_steps=self.args.save_steps,
                # Initialize best model tracking
                best_metric=None,
                best_model_checkpoint=None,
                max_eval_steps=self.args.max_eval_steps,
            )
        else:
            return TrainerState(
                max_steps=0,
                logging_steps=0,
                eval_steps=0,
                num_train_epochs=0,
                train_batch_size=0,
                epoch_train_steps=0,
                is_local_process_zero=self.is_local_process_zero,
                is_world_process_zero=self.is_world_process_zero,
                num_processes=self.num_processes,
                save_steps=0,
                # Initialize best model tracking
                best_metric=None,
                best_model_checkpoint=None,
                max_eval_steps=self.args.max_eval_steps,
            )

    def _end_train_loop(
        self, start_time: float | None, train_steps: int
    ) -> dict[str, int | float]:
        if start_time:
            runtime = time.time() - start_time
        else:
            runtime = None
        # Calculate effective batch size including gradient accumulation
        effective_batch_size = self._calculate_effective_batch_size()
        total_train_samples = effective_batch_size * train_steps
        metrics = self._speed_metrics(
            "train", runtime, total_train_samples, train_steps
        )
        metrics["epoch"] = self.state.epoch
        metrics["effective_batch_size"] = effective_batch_size

        # Add token and FLOP metrics
        metrics["total_tokens"] = self.state.num_input_tokens_seen
        if runtime and runtime > 0:
            metrics["tokens_per_second"] = round(
                self.state.num_input_tokens_seen / runtime, 2
            )

        if self.state.total_flos > 0:
            metrics["total_flops"] = self.state.total_flos
            if runtime and runtime > 0:
                metrics["flops_per_second"] = round(self.state.total_flos / runtime, 2)

        return metrics

    def _calculate_effective_batch_size(self) -> int:
        """
        Calculate effective batch size accounting for gradient accumulation and parallelism.

        The effective batch size is the total number of examples processed per optimizer update.

        Calculation depends on parallelism strategy:
        - Data parallelism (DDP): Multiply by num_processes (each process sees different batch)
        - Pipeline parallelism: Don't multiply (same batch flows through pipeline stages)
        - Model parallelism: Don't multiply (same batch processed by different model shards)
        - Combined: Don't multiply (same batch)

        Formula: base_batch = per_device_batch * gradient_accumulation_steps
                 effective = base_batch * num_processes (only for data parallel)

        Returns:
            Total number of examples per optimizer update
        """
        base_batch_size = (
            self.state.train_batch_size * self.args.gradient_accumulation_steps
        )

        # Check if any form of model parallelism is being used
        # If so, don't multiply by num_processes since the same batch is processed
        # across different parts of the model (stages and/or shards)
        is_pipeline_parallel = (
            hasattr(self, "_is_pipeline_parallel") and self._is_pipeline_parallel()
        )
        is_model_parallel = (
            hasattr(self, "_is_model_parallel") and self._is_model_parallel()
        )

        if is_pipeline_parallel or is_model_parallel:
            # Any form of model parallelism: don't multiply by num_processes
            return base_batch_size
        else:
            # Data parallel (DDP) or single process: multiply by num_processes
            return self.num_processes * base_batch_size

    def _is_pipeline_parallel(self) -> bool:
        """
        Indicate if trainer uses pipeline parallelism.

        Used for effective batch size calculation. Pipeline parallel trainers process
        the same batch across different pipeline stages, so don't multiply by num_processes.

        Returns:
            False for single-device Trainer, True in PipelineTrainer
        """
        return False

    def _is_model_parallel(self) -> bool:
        """
        Indicate if trainer uses model parallelism (tensor/expert parallelism).

        Used for effective batch size calculation. Model parallel trainers process
        the same batch across different model shards, so don't multiply by num_processes.

        Returns:
            False for single-device Trainer, True in model parallel trainers
        """
        return False

    def _prediction_step(
        self, input_dict: dict[str, Tensor], labels: Tensor
    ) -> Dict[str, Tensor | None]:
        """
        Perform a single evaluation batch forward pass.

        Computes loss without gradient computation (wrapped in @torch.no_grad()).
        Uses unscaled loss (via loss_fn.no_rescale()) for accurate eval metrics.

        Args:
            input_dict: Model inputs (input_ids, attention_mask, etc.)
            labels: Target labels for loss computation

        Returns:
            Dictionary with 'loss', 'logits', and 'labels' tensors
        """
        assert self.model is not None
        assert isinstance(self.loss_fn, RescaleLoss)
        if self.use_fused_loss:
            input_dict["return_hidden_states"] = True  # type: ignore[assignment]
        with self.loss_fn.no_rescale():
            outputs = self.model(**input_dict)
            logits = logits_from_outputs(outputs)
            loss = self.loss_fn(logits, labels)

        loss = self._distributed_loss(loss.detach())
        return {
            "loss": loss,
            "logits": logits.detach(),
            "labels": labels,
        }

    def _speed_metrics(
        self, prefix: str, runtime: float | None, samples: int, steps: int
    ) -> dict[str, int | float]:
        if runtime is not None and steps > 0:
            samples_per_second = round(samples / runtime, 3)
            steps_per_second = round(steps / runtime, 3)
        else:
            samples_per_second = float("nan")
            steps_per_second = float("nan")
        metrics = {
            f"{prefix}_runtime": runtime,
            f"{prefix}_samples": samples,
            "step": steps,
            f"{prefix}_samples_per_second": samples_per_second,
            f"{prefix}_steps_per_second": steps_per_second,
        }
        return metrics

    def _log_step(
        self,
        loss_log: list[Tensor],
        total_norm_log: list[Tensor],
        tokens_log: list[Tensor],
    ):
        self._update_training_steps()

        logs = {
            "epoch": self.state.epoch,
        }

        if not len(loss_log):
            # No losses to log - this can happen with gradient accumulation
            # when logging is called multiple times in the same step
            return

        # Reduce loss and total_norm
        mean_loss = torch.stack(loss_log).mean().item()
        logs["loss"] = mean_loss
        loss_log.clear()

        if len(total_norm_log):
            total_norm = torch.stack(total_norm_log)
            logs["grad_norm"] = total_norm.square().mean().sqrt().item()
            logs["max_grad_norm"] = total_norm.max().item()
            total_norm_log.clear()
        else:
            raise ValueError("No grad norm!")

        # Synchronize token counts across ranks at log step (amortizes all_reduce cost)
        if len(tokens_log):
            local_tokens = torch.stack(tokens_log).sum()
            synced_tokens = self._distributed_tokens(local_tokens)
            tokens_count = int(synced_tokens.item())
            self.state.num_input_tokens_seen += tokens_count
            self.state.total_flos += self._flops_per_token * tokens_count
            logs["tokens"] = tokens_count
            logs["total_tokens"] = self.state.num_input_tokens_seen
            logs["total_flos"] = self.state.total_flos
            tokens_log.clear()

        if self.lr_scheduler is not None:
            last_lr = self.lr_scheduler.get_last_lr()[0]
            if last_lr is not None:
                if torch.is_tensor(last_lr):
                    last_lr = last_lr.item()
                logs["learning_rate"] = last_lr

        # Capture peak CUDA memory for this rank and reset stats so each interval
        # reflects only the high-water mark since the previous log step.  Doing this
        # here (rather than in individual callbacks) ensures the reset happens exactly
        # once regardless of how many callbacks query memory statistics.
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            logs["peak_mem_allocated"] = torch.cuda.max_memory_allocated(device=device)
            torch.cuda.reset_peak_memory_stats(device=device)

        # Capture control object from log callbacks for trainer control
        return self.log(logs)

    def _prepare_batch(
        self, batch: Dict[str, Tensor]
    ) -> tuple[Dict[str, Tensor], Tensor]:
        """
        Move batch tensors to target device and extract labels.

        Args:
            batch: Dictionary of tensors from dataloader, must include 'labels' key

        Returns:
            Tuple of (input_dict, labels) where labels are separated for loss computation
        """
        batch = {k: v.to(self.args.device) for k, v in batch.items()}
        labels = batch.pop("labels")

        return (batch, labels)

    def _distributed_loss(self, loss: Tensor) -> Tensor:
        """
        Reduce loss across all processes for accurate logging.

        Single-device trainer just returns the input. Distributed trainers (AccelTrainer,
        PipelineTrainer) override this to all-reduce loss values so logging reflects
        the average loss across all processes/devices.

        See src/forgather/ml/trainer/accelerate/accel_trainer.py for distributed implementation.

        Args:
            loss: Loss tensor from current process

        Returns:
            Loss tensor (single-device) or all-reduced loss (distributed)
        """
        return loss

    @override
    def load_checkpoint(self, *args, **kwargs) -> None:
        super().load_checkpoint(*args, **kwargs)
        self._update_training_steps()
