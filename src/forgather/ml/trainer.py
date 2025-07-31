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
from contextlib import contextmanager
import os
import logging
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import set_seed

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
from .default_callbacks import (
    ProgressCallback,
    InfoCallback,
)
from .periodic_function import PeriodicFunction


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
        args,
        optimizer_factory: Optional[Callable] = None,
        # Alernative, for compatibility with transformers.Trainer
        optimizer_cls_and_kwargs: Optional[
            Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]
        ] = None,
        lr_scheduler_factory: Optional[Callable] = None,
        **kwargs,
    ):
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
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            persistent_workers=self.args.dataloader_persistent_workers,
        )

    # @override
    def _prepare(self, train_dataset, eval_dataset) -> None:
        """
        Prepare for training and/or evaluation
        """
        # Set the random seed
        if self.args.seed != -1:
            set_seed(self.args.seed)

        self._init_dataloaders(train_dataset, eval_dataset)
        self._prepare_model()
        if self.args.torch_compile:
            print(
                "Compiling model",
                self.args.torch_compile_backend,
                self.args.torch_compile_mode,
            )
            self.model.compile(
                backend=self.args.torch_compile_backend,
                mode=self.args.torch_compile_mode,
                dynamic=True,
            )

        if self.do_train:
            self._init_optimizer()

        self.state = self._init_state()

        if self.do_train:
            # Restore from checkpoint if specified (after state is initialized)
            checkpoint_path = self._resolve_checkpoint_path()
            if checkpoint_path:
                logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
                self._load_model_from_checkpoint(checkpoint_path)
                self._load_training_state(checkpoint_path)

        self._dispatch_event("on_init_end")

    def _init_dataloaders(self, train_dataset, eval_dataset) -> None:
        # _prepare() sub-step 1
        self.max_steps = 0
        self.epoch_train_steps = self.args.epoch_train_steps
        self.train_ds_has_length = False

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
        if self.model_init:
            self.optimizer = None
            self.lr_scheduler = None
            self.model = self.model_init()
        self.model = self.model.to(self.args.device)

    def _init_optimizer(self) -> None:
        # _prepare() sub-step 3
        if self.optimizer is None:
            self.optimizer = self.optimizer_factory(self.model.named_parameters())
        if self.lr_scheduler is None:
            if self.lr_scheduler_factory is not None:
                self.lr_scheduler = self.lr_scheduler_factory(
                    optimizer=self.optimizer,
                )
            elif self.args.lr_scheduler_type:
                self.lr_scheduler = transformers.get_scheduler(
                    name=self.args.lr_scheduler_type,
                    optimizer=self.optimizer,
                    num_warmup_steps=self.args.warmup_steps,
                    num_training_steps=self.max_steps,
                    **self.args.lr_scheduler_kwargs,
                )

    # @override
    def _train_loop(self) -> TrainOutput:
        """
        The inner training loop
        """
        self._dispatch_event("on_train_begin")

        start_time = time.time()

        periodic_log = PeriodicFunction(
            strategy=self.args.logging_strategy,
            period=self.args.logging_steps,
            epoch_period=self.epoch_train_steps,
            phase=0 if not self.args.logging_first_step else -1,
        )
        periodic_eval = PeriodicFunction(
            strategy=self.args.eval_strategy,
            period=self.args.eval_steps,
            epoch_period=self.epoch_train_steps,
            phase=self.args.eval_delay,
        )

        periodic_save = PeriodicFunction(
            strategy=self.args.save_strategy,
            period=self.args.save_steps,
            epoch_period=self.epoch_train_steps,
            phase=0,
        )

        # Just to be sure...
        self.optimizer.zero_grad()

        # Tracks mean loss for each log-step
        total_loss = torch.zeros(1, device=self.args.device)
        loss_steps = 0
        last_save_step = 0
        mean_loss = float("nan")

        # Context manager for setting model.train()/eval()
        with set_train(self.model, True):
            # Epoch loop
            while True:
                self._dispatch_event("on_epoch_begin")
                # Batch within epoch loop
                for batch in self.train_dataloader:
                    self._dispatch_event("on_step_begin")
                    loss = self._train_step(batch)
                    loss = self._gather_reduce_loss(loss)
                    total_loss += loss
                    self.state.global_step += 1
                    loss_steps += 1

                    # Compute epoch as continous value from steps
                    self.state.epoch = float(self.state.global_step) / float(
                        self.epoch_train_steps
                    )

                    if periodic_log.step():
                        mean_loss = total_loss.item() / loss_steps
                        total_loss -= total_loss  # zero total
                        loss_steps = 0
                        self._log_step(mean_loss)

                    if periodic_eval.step():
                        self._eval_loop()

                    if periodic_save.step():
                        last_save_step = self.state.global_step
                        self._save_checkpoint()

                    # Stop, if requested by callback.
                    control = self._dispatch_event("on_step_end")
                    if control is not None and control.should_training_stop:
                        break

                    # Break both loops when we reach the target global steps
                    if self.state.global_step >= self.max_steps:
                        break
                else:  # Continue, if loop exits normally
                    control = self._dispatch_event("on_epoch_end")
                    if control is not None and control.should_epoch_stop:
                        break
                    continue
                break  # Break, if inner-loop breaks

        # Save final checkpoint if checkpointing is enabled
        # Skip if we just saved a checkpoint in the final step
        if (
            self.args.save_strategy != IntervalStrategy.NO
            and self.state.global_step != last_save_step
        ):
            logger.info(f"Saving final checkpoint at step {self.state.global_step}")
            self._save_checkpoint()

        metrics = self._end_train_loop(start_time)
        self.log(metrics)
        self._dispatch_event("on_train_end")
        return TrainOutput(self.state.global_step, mean_loss, metrics)

    # @override
    @torch.no_grad()
    def _eval_loop(self) -> Dict[str, float]:
        """
        The inner evaluation loop
        """
        with set_train(self.model, False):
            total_loss = torch.zeros(1, device=self.args.device)
            for step, batch in enumerate(self.eval_dataloader):
                outputs = self._prediction_step(batch)
                loss = self._gather_reduce_loss(outputs["loss"])
                total_loss += loss
                self._dispatch_event("on_prediction_step")
            metrics = {"eval_loss": (total_loss / (step + 1)).item()}
            self._dispatch_event("on_evaluate", metrics=metrics)
            return metrics

    def _train_step(self, batch: dict | tuple) -> Tensor:
        """
        Perform a single training step

        Returns: mean loss (detached from graph)
        """
        args, kwargs = self._prepare_batch(batch)
        loss = self.model(*args, **kwargs)[0]

        self._backward(loss)
        self.optimizer.step()
        self._dispatch_event("on_optimizer_step")
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return loss.detach().mean()

    def _backward(self, loss):
        loss.backward()

    def _update_training_steps(self) -> None:
        """
        Estimate the training steps from the train data-loader
        This value can potentially change when using parallel training.
        Should this occur, update the value be calling this again.
        """
        # The number of training steps in a single epoch

        if self.train_ds_has_length:
            self.epoch_train_steps = len(self.train_dataloader)

        # If limit is specified, constrain to limit.
        if self.args.max_steps >= 0:
            self.max_steps = min(self.args.max_steps, self.epoch_train_steps)
        else:
            # The total number of training steps in all epochs
            self.max_steps = self.args.num_train_epochs * self.epoch_train_steps

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
        return {
            "loss": loss.mean().detach(),
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

    def _log_step(self, mean_loss):
        logs = {
            "epoch": self.state.epoch,
            "loss": mean_loss,
        }

        if self.lr_scheduler is not None:
            last_lr = self.lr_scheduler.get_last_lr()[0]
            if last_lr is not None:
                if torch.is_tensor(last_lr):
                    last_lr = last_lr.item()
                logs["learning_rate"] = last_lr

        self.log(logs)

    def _prepare_batch(self, batch):
        """
        Move the batch to the device and returns (args, kwargs) in batch
        """
        if isinstance(batch, Sequence):
            return (tuple(x.to(self.args.device) for x in batch), {})
        else:
            return (tuple(), {k: v.to(self.args.device) for k, v in batch.items()})

    def _gather_reduce_loss(self, loss: Tensor) -> Tensor:
        """
        Gather / reduce loss across all processes
        This implementaiton only supports a single process, so we just return the input.
        """
        return loss
