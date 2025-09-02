# A light-weight Trainer with an API close enough to "transformers.Trainer"
# to act as a stand-in for basic use-cases.
from typing import (
    Any,
    Dict,
    Callable,
    Iterable,
    Tuple,
    Optional,
    Type,
    # override, # PEP-698, introduced in Python 3.12
)
from collections.abc import Sequence
from functools import partial
from dataclasses import dataclass, field
import time
from contextlib import contextmanager, ExitStack
import os
import logging
import gc


import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import torchdata.nodes as tn
from torchdata.stateful_dataloader import StatefulDataLoader

from .base_trainer import BaseTrainer
from .trainer_types import TrainerState as BaseTrainerState
from .trainer_types import (
    ExtensibleTrainer,
    TrainingArguments,
    TrainOutput,
    TrainerControl,
    TrainerCallback,
    IntervalStrategy,
)
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


@dataclass(kw_only=True)
class TrainerState(BaseTrainerState):
    # --- Not in HF TrainerState ---
    # The total number of processes used for training
    num_processes: int = 1  # Non-standard
    # The number of batches in an epoch
    epoch_train_steps: int = 0


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


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def maybe_cleanup_memory(alloc_threshold):
    """Check if memory usage exceeds threshold"""
    if torch.cuda.is_available():
        reserved = torch.cuda.memory_reserved()
        max_memory = torch.cuda.get_device_properties(0).total_memory
        usage_ratio = reserved / max_memory

        if usage_ratio > alloc_threshold:
            gc.collect()
            torch.cuda.empty_cache()

def optimzer_hook(optimizer, total_grad_squared, parameter):
    if total_grad_squared is not None:
        total_grad_squared += parameter.grad.square().sum().to(dtype=torch.float32)
    optimizer.step()
    optimizer.zero_grad()


class Trainer(BaseTrainer):
    """
    This transformer trainer is a simplified version of the HF Trainer class
    The intent is to hopefully make the workings of such a class more comprehensible and
    easier to customize.
    """

    @classmethod
    def default_callbacks(cls):
        return [ProgressCallback(), InfoCallback()]

    def __init__(
        self,
        *,
        args: TrainingArguments,
        optimizer_factory: Optional[Callable] = None,
        # Alernative, for compatibility with transformers.Trainer
        optimizer_cls_and_kwargs: Optional[
            Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]
        ] = None,
        lr_scheduler_factory: Optional[Callable] = None,
        **kwargs,
    ):
        assert isinstance(args, TrainingArguments)
        # HF Trainer compatibility.
        if not optimizer_factory:
            if not optimizer_cls_and_kwargs:
                optimizer_factory = partial(
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
        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory
        super().__init__(args=args, **kwargs)

    # @override
    def _post_init(self) -> None:
        assert (self.model is not None) or (
            self.model_init is not None
        ), "Either a model or a model constructor must be specified."

        assert (
            self.args.max_grad_norm is None or not self.args.fuse_optim_with_backward
        ), "max_grad_norm is incompatible with fuse_optim_with_backward"

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

    # @override
    def _prepare(self, train_dataset, eval_dataset) -> None:
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
            torch._functorch.config.activation_memory_budget = (
                self.args.activation_memory_budget
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

        self.state = self._init_state()

        if self.do_train:
            # Restore from checkpoint if specified (after state is initialized)
            if self.args.resume_from_checkpoint:
                self.load_checkpoint()

        self._dispatch_event("on_init_end")

    def _compile_model(self):
        self.model.compile(
            backend=self.args.torch_compile_backend,
            mode=self.args.torch_compile_mode,
            dynamic=self.args.torch_compile_dynamic,
            fullgraph=self.args.torch_compile_full_graph,
        )

    def _init_dataloaders(self, train_dataset, eval_dataset) -> None:
        # _prepare() sub-step 1
        self.max_steps = 0
        self.epoch_train_steps = self.args.epoch_train_steps
        self.train_ds_has_length = False

        self.do_train = train_dataset is not None
        self.do_eval = eval_dataset is not None

        if self.do_train:
            self.train_dataloader = self._get_dataloader(
                train_dataset, self.args.per_device_train_batch_size
            )

            self.train_ds_has_length = hasattr(train_dataset, "__len__")
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
                    f"Consturcting model on meta device and materializing on {self.args.device}"
                )
                # TODO: Identify if the model has buffers with "persist=False" and warn loudly!
                if not self.args.resume_from_checkpoint:
                    logger.warning(
                        f"Uninitialized model constructed on meta-device and not loading from checkpoint!"
                    )
                with torch.device("meta"):
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
                    f"Constructig and initializing model directly on {self.args.device}"
                )
                with torch.device(self.args.device):
                    self.model = self.model_init()
            case _:
                raise ValueError("Requires one of: default|meta|device")
        if self.args.gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                logger.info("Enabling gradient checkpointing")
                self.model.gradient_checkpointing_enable(
                    **self.args.gradient_checkpointing_kwargs
                )
            else:
                logger.warning(
                    "Gradient checkpointing requested, but model does not support it!"
                )

    def _init_optimizer(self) -> None:
        # _prepare() sub-step 3
        if self.optimizer is None:
            self.optimizer = self.optimizer_factory(self.model.named_parameters())

            # Combine backward with optimizer step?
            if self.args.fuse_optim_with_backward:
                self._total_grad_squared = torch.zeros(
                    1, device=self.args.device, dtype=torch.float32
                )
                hook = partial(optimzer_hook, self.optimizer, self._total_grad_squared)
                for p in self.model.parameters():
                    if p.requires_grad:
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
                    optimizer=self.optimizer,
                    num_warmup_steps=self.args.warmup_steps,
                    num_training_steps=self.max_steps,
                    **self.args.lr_scheduler_kwargs,
                )

    def _dataloader_iter(self, dataloader: DataLoader) -> Iterable:
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
            checkpoint_path = self.save_checkpoint()

            # For load_best_model_at_end, we need metrics from the most recent evaluation
            if self.args.load_best_model_at_end:
                assert eval_metrics, (
                    "BUG: load_best_model_at_end requires that save and eval occur on the same step\m"
                    f"periodic_eval_step={str(periodic_eval)}\nperiodic_save_step={periodic_save}"
                )

                # Update best model tracking with current evaluation metrics
                self._update_best_model(checkpoint_path, eval_metrics)

                save_checkpoint_metrics(checkpoint_path, eval_metrics)

    # @override
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

        # Just to be sure...
        self.optimizer.zero_grad()

        # This keeps track of the sum of all log losses and the number of log steps
        # We need this to compute the mean loss at the end.
        self._total_loss = 0.0
        self._total_log_steps = 0

        is_abort = False  # Track if training was aborted without save

        start_time = time.time()
        self._dispatch_event("on_train_begin")

        # Holds loss and grad-norm samples between logs steps
        total_norm_log = []
        loss_log = []

        # Context manager for setting model.train()/eval()
        with set_train(self.model, True):
            # Epoch loop
            while True:
                self._dispatch_event("on_epoch_begin")
                # Batch within epoch loop
                for batch in self._dataloader_iter(self.train_dataloader):
                    self._dispatch_event("on_step_begin")
                    loss, total_norm = self._train_step(batch)

                    # Update counters and stats
                    if total_norm is not None:
                        total_norm_log.append(total_norm)
                    loss_log.append(loss)
                    self.state.global_step += 1
                    # Compute epoch as continous value from steps
                    self.state.epoch = float(self.state.global_step) / float(
                        self.epoch_train_steps
                    )

                    self._dispatch_event("on_step_end")
                    self._maybe_log_save_evaluate(
                        loss_log,
                        total_norm_log,
                        periodic_log,
                        periodic_eval,
                        periodic_save,
                    )

                    # Check for stop request from log callbacks
                    if self.control.should_training_stop:
                        # Check if this is an abort (no save) vs graceful stop
                        is_abort = (
                            hasattr(self.control, "should_abort_without_save")
                            and self.control.should_abort_without_save
                        )
                        if is_abort:
                            logger.info(
                                "Training aborted by control command - skipping final checkpoint"
                            )
                        else:
                            logger.info("Training stopped by control command")
                        break

                    # Break both loops when we reach the target global steps
                    if self.state.global_step >= self.max_steps:
                        break

                    # Periodic GC
                    maybe_cleanup_memory(self.args.gc_threshold)
                else:  # Continue, if loop exits normally
                    self._dispatch_event("on_epoch_end")
                    if self.control.should_epoch_stop:
                        break
                    continue
                break  # Break, if inner-loop breaks

        # Save final checkpoint if checkpointing is enabled
        # Skip if we just saved a checkpoint in the final step, or if training was aborted
        if (
            not is_abort
            and self.args.save_strategy != IntervalStrategy.NO
            and periodic_save.rel_step != 0  # Step resets to zero on save
        ):
            logger.info(f"Saving final checkpoint at step {self.state.global_step}")
            self.save_checkpoint()

        # Load best model at end if requested
        if self.args.load_best_model_at_end:
            logger.info(f'Loading best model "{self.state.best_model_checkpoint}"')
            self.load_best_model()

        metrics = self._end_train_loop(start_time)
        # Save final training metrics
        self.log_metrics("train", metrics)
        self.save_metrics("train", metrics)

        self.log(metrics)
        self._dispatch_event("on_train_end")
        if self._total_log_steps:
            mean_loss = self._total_loss / self._total_log_steps
        else:
            mean_loss = float("nan")
        return TrainOutput(self.state.global_step, mean_loss, metrics)

    # @override
    @torch.no_grad()
    def _eval_loop(self) -> Dict[str, float]:
        """
        The inner evaluation loop
        """
        with set_train(self.model, False):
            total_loss = torch.zeros(1, device=self.args.device)
            for step, batch in enumerate(self._dataloader_iter(self.eval_dataloader)):
                outputs = self._prediction_step(batch)
                total_loss += outputs["loss"]
                self._dispatch_event("on_prediction_step")
            metrics = {"eval_loss": (total_loss / (step + 1)).item()}
            self._dispatch_event("on_evaluate", metrics=metrics)
            return metrics

    def _clip_grad_norm(self, max_grad_norm, norm_type=2.0) -> Optional[Tensor]:
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

    def _train_step(self, batch: dict | tuple) -> Tensor:
        """
        Perform a single training step

        Returns: mean loss (detached from graph)
        """
        args, kwargs = self._prepare_batch(batch)

        with ExitStack() as stack:
            if self.args.enable_activation_offloading:
                stack.enter_context(torch.autograd.graph.save_on_cpu(pin_memory=True))
            if self.loss_fn:
                # TODO: We are making a guess as to how to interpret the args. Can we do better?
                if len(args):
                    main_input = args[0]
                    labels = args[1]
                else:
                    main_input = kwargs[self.model.main_input_name]
                    labels = kwargs["labels"]
                outputs = self.model(main_input)
                loss = self.loss_fn(outputs, labels)
            else:
                loss = self.model(*args, **kwargs)[0]

        self._backward(loss)
        total_norm = self._clip_grad_norm(self.args.max_grad_norm)
        if not self.args.fuse_optim_with_backward:
            self.optimizer.step()
            self.optimizer.zero_grad()
        self._dispatch_event("on_optimizer_step")
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        loss = self._distributed_loss(loss.detach().mean())
        return loss, total_norm

    def _backward(self, loss):
        loss.backward()

    def _save_dataloader_state(self):
        """Save StatefulDataLoader state if available."""
        logger.debug(
            f"_save_dataloader_state called, train_dataloader type: {type(getattr(self, 'train_dataloader', None))}"
        )

        if hasattr(self, "train_dataloader") and hasattr(
            self.train_dataloader, "state_dict"
        ):
            try:
                state = self.train_dataloader.state_dict()
                logger.debug(
                    f"Successfully saved dataloader state with keys: {list(state.keys())}"
                )
                return state
            except Exception as e:
                logger.warning(f"Failed to save dataloader state: {e}")
        else:
            logger.debug("train_dataloader doesn't have state_dict method")
        return None

    def _load_dataloader_state(self, dataloader_state):
        """Load StatefulDataLoader state if available."""
        if hasattr(self, "train_dataloader"):
            # Depending upon the class of the dataloader, it may have a method named either load_state_dict() or reset()
            # which can load a state dictionary, with the latter being part of the torchdata.nodes API
            load_method = getattr(
                self.train_dataloader,
                "load_state_dict",
                getattr(self.train_dataloader, "reset", None),
            )
            if load_method:
                try:
                    logger.info(
                        f"Loading dataloader state: {dataloader_state.keys()} via {load_method.__name__}()"
                    )
                    load_method(dataloader_state)
                except Exception as e:
                    logger.warning(f"Failed to load dataloader state: {e}")
            else:
                logger.warning(
                    "Could not restored Dataloader state, as it does not have a load method"
                )

    def _update_training_steps(self) -> None:
        """
        Estimate the training steps from the train data-loader
        This value can potentially change when using parallel training.
        Should this occur, update the value be calling this again.
        """
        # The number of training steps in a single epoch

        if self.train_ds_has_length:
            self.epoch_train_steps = len(self.train_dataloader)

        # The total number of training steps in all epochs
        self.max_steps = self.args.num_train_epochs * self.epoch_train_steps

        # If limit is specified, constrain to limit.
        if self.args.max_steps >= 0:
            self.max_steps = min(self.args.max_steps, self.max_steps)

    def _init_state(self) -> TrainerState:
        """
        Init public training state
        This should be retained when saving a checkpoint
        """
        if self.do_train:
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
            )

    def _end_train_loop(self, start_time):
        runtime = time.time() - start_time
        total_train_batch_size = self.state.num_processes * self.state.train_batch_size
        total_train_samples = total_train_batch_size * self.max_steps
        metrics = self._speed_metrics(
            "train", runtime, total_train_samples, self.max_steps
        )
        metrics["epoch"] = self.state.epoch
        return metrics

    def _prediction_step(self, batch: dict | tuple) -> Tensor:
        """
        Perform a single batch of predictions
        """
        args, kwargs = self._prepare_batch(batch)
        # Note that some models may outpus more items than (loss, logits)
        outputs = self.model(*args, **kwargs)
        loss, logits = outputs[0], outputs[1]
        labels = kwargs.get("labels", None)
        loss = self._distributed_loss(loss.detach().mean())
        return {
            "loss": loss,
            "logits": logits.detach(),
            "labels": labels,
        }

    def _speed_metrics(self, prefix, runtime, samples, steps):
        metrics = {
            f"{prefix}_runtime": runtime,
            f"{prefix}_samples": samples,
            "step": steps,
            f"{prefix}_samples_per_second": round(samples / runtime, 3),
            f"{prefix}_steps_per_second": round(steps / runtime, 3),
        }
        return metrics

    def _log_step(self, loss_log, total_norm_log):
        logs = {
            "epoch": self.state.epoch,
        }

        assert len(loss_log)

        # Reduce loss and total_norm
        mean_loss = torch.stack(loss_log).mean().item()
        self._total_loss += mean_loss
        self._total_log_steps += 1
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

    def _prepare_batch(self, batch):
        """
        Move the batch to the device and returns (args, kwargs) in batch
        """
        if isinstance(batch, Sequence):
            return (tuple(x.to(self.args.device) for x in batch), {})
        else:
            return (tuple(), {k: v.to(self.args.device) for k, v in batch.items()})

    def _distributed_loss(self, loss: Tensor) -> Tensor:
        """
        Gather / reduce loss across all processes
        This implementaiton only supports a single process, so we just return the input.
        """
        return loss
