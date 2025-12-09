# A light-weight Trainer with an API close enough to "transformers.Trainer"
# to act as a stand-in for basic use-cases.
from typing import (
    TypeGuard,
    Any,
    Dict,
    Callable,
    Iterable,
    Tuple,
    Optional,
    Type,
    Protocol,
    Iterator,
    cast,
    override,
)
from collections.abc import Sized
from functools import partial
from dataclasses import dataclass
from contextlib import contextmanager
import time
import os
import logging
import gc
from contextlib import ExitStack

import torch
from torch import Tensor

import torchdata.nodes as tn
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

from .trainer_types import (
    TrainOutput,
    IntervalStrategy,
    CheckpointInterface,
    OptimizerT,
    LRSchedulerT,
    OptimizerFactoryT,
    LRSchedulerFactoryT,
    FusedLossFactoryT,
    EnableCheckpointFnT,
    IterableDatasetT,
)
from .base_trainer import (
    BaseTrainer,
    BaseTrainingArguments,
    logits_from_outputs,
)
from .trainer_types import TrainerState as BaseTrainerState
from .callbacks.default_callbacks import (
    ProgressCallback,
    InfoCallback,
)
from .periodic_function import PeriodicFunction
from ..sharded_checkpoint import (
    create_sharing_metadata,
    retie_parameters,
    save_checkpoint_metrics,
)

from .checkpoint_manager import CheckpointManager, CheckpointConfig
from ..distributed import DistributedEnvInterface
from ..no_init_weights import no_init_weights
from forgather.ml.utils import default_dtype
from forgather.ml.construct import torch_dtype
from ..loss import RescaleLoss

logger = logging.getLogger(__name__)


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
    # Ratio of reserved to total GPU memory to trigger GC
    # If OOM from fragmentation, lower ratio
    gc_threshold: float = 0.5

    # Construct model on meta-device and materialize directly on device
    # default: Construct model on CPU on move to device; slow, but reliable.
    # device: Construct model directly on device. This is can faster, but may result in OOM
    # meta: Construct on meta device and materialize on target device. The resulting model
    #   is uninitialized and will need to be loaded with a checkpoint.
    construct_model_on: str = "default"  # "default" | "meta" | "device"

    # https://pytorch.org/blog/activation-checkpointing-techniques/
    # Requires "torch_compile = True" option
    activation_memory_budget: float | None = None

    # Combine gradient calculation with optimizer step, to save memory.
    # https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html
    fuse_optim_with_backward: bool = False

    # The step at which to start collecting speed metrics
    # We default to 1, to remove the effects from torch.compile().
    # Set this to 0 to include all steps or > 0 for compile warmup time.
    speed_metrics_start_step: int = 1


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
    """Check if memory usage exceeds threshold"""
    if torch.cuda.is_available():
        reserved = torch.cuda.memory_reserved()
        max_memory = torch.cuda.get_device_properties(0).total_memory
        usage_ratio = reserved / max_memory

        if usage_ratio > alloc_threshold:
            gc.collect()
            torch.cuda.empty_cache()


def optimizer_hook(optimizer, total_grad_squared, name, parameter):
    if total_grad_squared is not None:
        total_grad_squared += parameter.grad.square().sum().to(dtype=torch.float32)
        # norm = parameter.grad.square().sum().sqrt()
        # logger.info(f"{name} {norm}")
    optimizer.step()
    optimizer.zero_grad()


class Trainer(BaseTrainer):
    """
    This transformer trainer is a simplified version of the HF Trainer class
    The intent is to hopefully make the workings of such a class more comprehensible and
    easier to customize.
    """

    args: TrainingArguments
    dist: DistributedEnvInterface
    optimizer_factory: OptimizerFactoryT | None
    lr_scheduler_factory: LRSchedulerFactoryT | None
    enable_activation_checkpoint_fn: EnableCheckpointFnT
    fused_loss_factory: FusedLossFactoryT | None

    max_steps: int
    epoch_train_steps: int
    do_train: bool
    do_eval: bool
    use_fused_loss: bool

    @classmethod
    def default_callbacks(cls):
        return [ProgressCallback(), InfoCallback()]

    def __init__(
        self,
        *,
        args: TrainingArguments,
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
        assert isinstance(args, TrainingArguments)
        self.args = args  # For type checking hint
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
        super().__init__(args=args, **kwargs)

    @override
    def _post_init(self) -> None:
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

        # If unspecified, set a default device
        if self.args.device is None:
            self.args.device = (
                torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            )
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
        self, train_dataset: IterableDatasetT, eval_dataset: IterableDatasetT
    ) -> None:
        """
        Prepare for training and/or evaluation
        """
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
        """Hook for wrapping object, after they have all be constructed in _prepare()"""
        pass

    def _compile_model(self):
        """Compile model hook"""
        self.model.compile(
            backend=self.args.torch_compile_backend,
            mode=self.args.torch_compile_mode,
            dynamic=self.args.torch_compile_dynamic,
            fullgraph=self.args.torch_compile_full_graph,
        )

    def _init_checkpoint_manager(self) -> CheckpointInterface:
        """Init checkpoint manager hook"""
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
        return checkpoint_manager

    def _init_dataloaders(self, train_dataset, eval_dataset) -> None:
        # _prepare() sub-step 1
        self.max_steps = 0
        self.epoch_train_steps = self.args.epoch_train_steps

        self.do_train = train_dataset is not None
        self.do_eval = eval_dataset is not None

        if self.do_train:
            self.train_dataloader = self._get_dataloader(
                train_dataset, self.args.per_device_train_batch_size
            )

            self._update_training_steps()

        if self.do_eval:
            self.eval_dataloader = self._get_dataloader(
                eval_dataset, self.args.per_device_eval_batch_size
            )

    def _prepare_model(self) -> None:
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

        # Rescale loss by gradient accumulation steps.
        self.loss_fn = RescaleLoss(
            self.loss_fn, 1 / self.args.gradient_accumulation_steps
        )

    def _maybe_get_fused_loss_fn(
        self, module: torch.nn.Module, default_loss_fn: Callable
    ):
        """Experimental API for handling fused classifier-loss-function

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
            return self.fused_loss_factory(module.get_output_embeddings())
        logger.warning(
            f"Fused loss factory not provided. Provide a fused-loss factory for enhanced performance."
        )
        return default_loss_fn

    def _init_optimizer(self) -> None:
        # _prepare() sub-step 3
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
                self.lr_scheduler = self.lr_scheduler_factory(
                    optimizer=self.optimizer,
                )
            elif self.args.lr_scheduler_type:
                from transformers import get_scheduler

                self.lr_scheduler = get_scheduler(
                    name=self.args.lr_scheduler_type,
                    optimizer=cast(Any, self.optimizer),  # type: ignore[arg-type]
                    num_warmup_steps=self.args.warmup_steps,
                    num_training_steps=self.max_steps,
                    scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
                )

    def _dataloader_iter(
        self, dataloader: Iterable[Dict[str, Tensor]]
    ) -> Iterable[Dict[str, Tensor]]:
        """
        Get the next batch from the dataloader.
        This is a generator that yields batches from the dataloader.
        """
        for batch in dataloader:
            yield batch

    def _maybe_log_save_evaluate(
        self,
        loss_log,
        total_norm_log,
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

            self._log_step(loss_log, total_norm_log)

        # Handle evaluation (normal schedule or control-triggered)
        eval_metrics = None

        if periodic_eval.step() or self.control.should_evaluate:
            periodic_eval.reset()
            self.control.should_evaluate = False

            # Do eval
            maybe_cleanup_memory(self.args.gc_threshold)
            eval_metrics = self._eval_loop()

        # Handle checkpointing - normal schedule or control-triggered
        if periodic_save.step() or self.control.should_save:
            periodic_save.reset()
            self.control.should_save = False
            assert self.checkpoint_manager
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                checkpoint_id=str(self.state.global_step)
            )
            self._dispatch_event("on_save")

            # For load_best_model_at_end, we need metrics from the most recent evaluation
            if self.args.load_best_model_at_end:
                if not eval_metrics:
                    logger.error(
                        "load_best_model_at_end requires that save and eval occur on the same step\n"
                        f"periodic_eval_step={str(periodic_eval)}\nperiodic_save_step={periodic_save}\n"
                        "Skipping update of best model and continuing..."
                    )
                else:
                    # Update best model tracking with current evaluation metrics
                    logger.info(
                        f"Updating best model to {checkpoint_path} with metrics {eval_metrics}"
                    )
                    self._update_best_model(checkpoint_path, eval_metrics)
                    save_checkpoint_metrics(checkpoint_path, eval_metrics)

    def _update_best_model(
        self, checkpoint_path: str, metrics: Dict[str, float]
    ) -> None:
        """Update best model tracking based on current metrics."""
        if not self.args.load_best_model_at_end:
            return

        # Try exact match first, then with eval_ prefix
        metric_value = metrics.get(self.args.metric_for_best_model)
        if metric_value is None:
            metric_value = metrics.get(f"eval_{self.args.metric_for_best_model}")

        if metric_value is None:
            logger.warning(
                f"Metric '{self.args.metric_for_best_model}' not found in evaluation metrics. "
                f"Available metrics: {list(metrics.keys())}"
            )
            return

        # Initialize best metric on first evaluation
        if self.state.best_metric is None:
            self.state.best_metric = metric_value
            self.state.best_model_checkpoint = checkpoint_path
            logger.info(
                f"First evaluation - setting best {self.args.metric_for_best_model}: {metric_value}"
            )
            return

        # Check if current metric is better
        is_better = (
            self.args.greater_is_better and metric_value > self.state.best_metric
        ) or (not self.args.greater_is_better and metric_value < self.state.best_metric)

        if is_better:
            previous_best = self.state.best_metric
            self.state.best_metric = metric_value
            self.state.best_model_checkpoint = checkpoint_path
            assert self.checkpoint_manager
            self.checkpoint_manager.set_best_checkpoint(checkpoint_path)
            logger.info(
                f"New best {self.args.metric_for_best_model}: {metric_value} (previous: {previous_best})"
            )
            logger.info(f"Saving new best model checkpoint to {checkpoint_path}")

    def load_best_model(self) -> None:
        """Load the best model from the best checkpoint."""
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
        assert isinstance(self.args.logging_strategy, IntervalStrategy)
        assert isinstance(self.args.eval_strategy, IntervalStrategy)
        assert isinstance(self.args.save_strategy, IntervalStrategy)
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

        # Just to be sure...
        self.optimizer.zero_grad()

        start_time = None
        train_steps = 0
        self._dispatch_event("on_train_begin")

        # Holds loss and grad-norm samples between logs steps
        accumulated_grad_norm = []
        accumulated_loss = []

        # Context manager for setting model.train()/eval()
        with set_train(self.model, True):
            # Epoch loop
            while True:
                self.control.should_epoch_stop = False
                self._dispatch_event("on_epoch_begin")
                data_iterator = iter(self._dataloader_iter(self.train_dataloader))
                while True:
                    self._dispatch_event("on_step_begin")

                    try:
                        loss, total_norm = self._train_step(data_iterator)
                    except StopIteration:
                        self._dispatch_event("on_step_end")
                        break

                    accumulated_grad_norm.append(total_norm)
                    accumulated_loss.append(loss)

                    # Increment global step
                    self.state.global_step += 1

                    # Compute epoch as continuous value from global steps
                    self.state.epoch = float(self.state.global_step) / float(
                        self.epoch_train_steps
                    )

                    self._dispatch_event("on_step_end")
                    self._maybe_log_save_evaluate(
                        accumulated_loss,
                        accumulated_grad_norm,
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
            if (
                periodic_save.rel_step != 0
                and self.args.save_strategy != IntervalStrategy.NO
            ):
                logger.info(f"Saving final checkpoint at step {self.state.global_step}")
                # If load best model, we need to evaluate it too.
                if self.args.load_best_model_at_end:
                    self.control.should_evaluate = True
                self.control.should_save = True
            self._maybe_log_save_evaluate(
                accumulated_loss,
                accumulated_grad_norm,
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
        self._dispatch_event("on_train_end")
        return TrainOutput(self.state.global_step, metrics)

    @override
    @torch.no_grad()
    def _eval_loop(self) -> Dict[str, float]:
        """
        The inner evaluation loop
        """
        with set_train(self.model, False):
            total_loss = torch.zeros(1, device=self.args.device)
            for step, batch in enumerate(self._dataloader_iter(self.eval_dataloader)):
                if self.args.max_eval_steps > 0 and step >= self.args.max_eval_steps:
                    break
                input_dict, labels = self._prepare_batch(batch)
                outputs = self._prediction_step(input_dict, labels)
                loss = outputs["loss"]
                assert loss is not None
                total_loss += loss
                self._dispatch_event("on_prediction_step")
            metrics = {"eval_loss": (total_loss / (step + 1)).item()}
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
    ) -> Tuple[Tensor, Tensor | None]:
        """
        Perform a single training step, with optional gradient accumulation.

        Args:


        Returns:
            Tuple of (loss, grad_norm). Loss is unscaled for logging consistency.
            grad_norm is None if not computed on this step.
        """
        accumulated_losses = []

        for _ in range(self.args.gradient_accumulation_steps):
            input_dict, labels = self._prepare_batch(next(data_iterator))
            loss = self._forward_backward_step(input_dict, labels)
            accumulated_losses.append(loss)

        assert self._should_sync_gradients()

        total_norm = self._clip_grad_norm(self.args.max_grad_norm)
        if not self.args.fuse_optim_with_backward:
            self.optimizer.step()
            self.optimizer.zero_grad()
        self._dispatch_event("on_optimizer_step")
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        loss = torch.sum(torch.stack(accumulated_losses))
        loss = self._distributed_loss(loss)
        return loss, total_norm

    def _forward_backward_step(
        self, input_dict: dict[str, Tensor], labels: Tensor
    ) -> Tensor:
        if self.use_fused_loss:
            input_dict["return_hidden_states"] = True
        outputs = self.model(**input_dict)
        logits = logits_from_outputs(outputs)
        loss = self.loss_fn(logits, labels)
        self._backward(loss)
        return loss.detach()

    def _backward(self, loss: Tensor) -> None:
        loss.backward()

    def _should_sync_gradients(self) -> bool:
        return True

    def _update_training_steps(self) -> None:
        """
        Estimate the training steps from the train data-loader
        This value can potentially change when wrapping the dataloder, e.g., with Accelerate
        Should this occur, call this to updatte the step-count and epoch info.

        With gradient accumulation, global steps = batch_count // gradient_accumulation_steps
        """
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

    def _init_state(self) -> TrainerState:
        """
        Init public training state
        This should be retained when saving a checkpoint
        """

        max_eval_steps = min(
            self.args.max_eval_steps,
            len(self.eval_dataloader),
        )

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
                max_eval_steps=max_eval_steps,
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
                max_eval_steps=max_eval_steps,
            )

    def _end_train_loop(
        self, start_time: float, train_steps: int
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
        return metrics

    def _calculate_effective_batch_size(self) -> int:
        """
        Calculate the effective batch size accounting for gradient accumulation and parallelism.

        For different parallelism strategies:
        - Data parallelism (DDP): Multiply by num_processes (different batches per process)
        - Pipeline parallelism: Don't multiply by num_processes (same batch across stages)
        - Model parallelism: Don't multiply by num_processes (same batch across shards)
        - Combined (pipeline + model): Don't multiply by num_processes (same batch)
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
            return self.state.num_processes * base_batch_size

    def _is_pipeline_parallel(self) -> bool:
        """Override in PipelineTrainer to return True"""
        return False

    def _is_model_parallel(self) -> bool:
        """Override in model parallel trainers to return True"""
        return False

    def _prediction_step(
        self, input_dict: dict[str, Tensor], labels: Tensor
    ) -> Dict[str, Tensor | None]:
        """
        Perform a single batch of predictions
        """
        if self.use_fused_loss:
            input_dict["return_hidden_states"] = True
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
        self, prefix: str, runtime: float, samples: int, steps: int
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

    def _log_step(self, loss_log: list[Tensor], total_norm_log: list[Tensor]):
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

        if self.lr_scheduler is not None:
            last_lr = self.lr_scheduler.get_last_lr()[0]
            if last_lr is not None:
                if torch.is_tensor(last_lr):
                    last_lr = last_lr.item()
                logs["learning_rate"] = last_lr

        # Capture control object from log callbacks for trainer control
        return self.log(logs)

    def _prepare_batch(
        self, batch: Dict[str, Tensor]
    ) -> tuple[Dict[str, Tensor], Tensor]:
        """
        Move the batch to the device
        """
        batch = {k: v.to(self.args.device) for k, v in batch.items()}
        labels = batch.pop("labels")

        return (batch, labels)

    def _distributed_loss(self, loss: Tensor) -> Tensor:
        """
        Gather / reduce loss across all processes
        This implementaiton only supports a single process, so we just return the input.
        """
        return loss
