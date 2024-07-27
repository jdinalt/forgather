# A light-weight Trainer with an API close enough to "transformers.Trainer"
# to act as a stand-in for basic use-cases.
from typing import (
    Any,
    Dict,
)
from dataclasses import dataclass, field
import time
from contextlib import contextmanager
import os
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import set_seed, DataCollatorForLanguageModeling

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


# For compatible with return-type of HF Trainer.
def default_optimizer_factory(model, training_args):
    """
    Construct the default optimizer
    """
    return torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)


def default_lr_scheduler_factory(optimizer, num_training_steps, training_args):
    """
    Construct the default learning-rate scheduler
    """
    return transformers.get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_training_steps,
    )


@dataclass(kw_only=True, slots=True)
class PrivateTrainerState:
    """
    Data to save with checkpoint, which is not in TrainerState
    """

    start_time: float
    periodic_log: PeriodicFunction
    periodic_eval: PeriodicFunction
    periodic_save: PeriodicFunction
    log_step_loss: Tensor
    total_loss: Tensor


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
        optimizer_factory=default_optimizer_factory,
        lr_scheduler_factory=default_lr_scheduler_factory,
        *args,
        **kwargs,
    ):
        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory
        super().__init__(*args, **kwargs)

    def _post_init(self) -> None:
        assert (self.model is not None) or (
            self.model_init is not None
        ), "Either a model or a model constructor must be specified."
        assert (self.data_collator is not None) or (
            self.tokenizer is not None
        ), "Either a tokenizer of data_collator must be specified."
        if self.data_collator is None:
            self.data_collator = DataCollatorForLanguageModeling(
                self.tokenizer,
                mlm=False,
                return_tensors="pt",
            )
        # Holds the mean loss for the most recent train log-step
        self.mean_train_loss = float("NaN")
        # If unspecified, set a default device
        if self.args.device is None:
            self.args.device = (
                torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            )
        # Override for debug.
        if self.args.use_cpu:
            self.args.device = "cpu"

    def _get_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            persistent_workers=self.args.dataloader_persistent_workers,
        )

    def _prepare(self, train_dataset, eval_dataset) -> None:
        """
        Prepare for training and/or evaluation
        """
        # Set the random seed
        if self.args.seed != -1:
            set_seed(self.args.seed)

        if self.model_init is not None:
            self.model = self.model_init()
            self.optimizer = None
            self.lr_scheduler = None

        if self.args.torch_compile:
            print("Compiling model", self.args.torch_compile_backend, self.args.torch_compile_mode)
            self.model.compile(
                backend=self.args.torch_compile_backend,
                mode=self.args.torch_compile_mode,
                dynamic=True,
            )
        else:
            print("not compiling model")
        
        if train_dataset is not None:
            assert train_dataset is not None, "Training requires a train_dataset"
            self.train_dataloader = self._get_dataloader(train_dataset)
            self._update_training_steps()
            if self.optimizer is None:
                self.optimizer = self.optimizer_factory(self.model, self.args)
            if self.lr_scheduler is None:
                self.lr_scheduler = self.lr_scheduler_factory(
                    self.optimizer, self.max_steps, self.args
                )
            self.mean_train_loss = float("NaN")

        if eval_dataset is not None:
            assert self.train_dataset is not None, "Evaluation requires an eval_dataset"
            if self.eval_dataset is not None:
                self.eval_dataloader = self._get_dataloader(eval_dataset)

    def _train_loop(self) -> TrainOutput:
        """
        The inner training loop
        """
        state = self._private_state = self._init_private_state()
        self.state = self._init_state()
        self._dispatch_event("on_train_begin")

        # Context manager for setting model.train()/eval()
        with set_train(self.model, True):
            # Epoch loop
            while True:
                self._dispatch_event("on_epoch_begin")
                # Batch within epoch loop
                for batch in self.train_dataloader:
                    self._dispatch_event("on_step_begin")
                    batch = self._prepare_batch(batch)
                    loss = self._train_step(batch)
                    loss = self._reduce_loss(loss)
                    state.total_loss += loss
                    state.log_step_loss += loss
                    self.state.global_step += 1

                    # Compute epoch as continous value from steps
                    self.state.epoch = float(self.state.global_step) / float(
                        self.epoch_train_steps
                    )
                    state.periodic_log.step(
                        state.log_step_loss, state.periodic_log.count()
                    )
                    state.periodic_eval.step()
                    state.periodic_save.step()

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
        # Flush the last (poentially) partial log-step.
        self._log_step(state.log_step_loss, state.periodic_log.count())
        metrics = self._end_train_loop()
        self.log(metrics)
        self._dispatch_event("on_train_end")
        return TrainOutput(self.state.global_step, self.mean_train_loss, metrics)

    @torch.no_grad()
    def _eval_loop(self) -> Dict[str, float]:
        """
        The inner evaluation loop
        """
        with set_train(self.model, False):
            total_loss = torch.zeros(1, device=self.args.device)
            step = 0
            for step, batch in enumerate(self.eval_dataloader):
                loss, _, _ = self._prediction_step(self._prepare_batch(batch))
                loss = self._reduce_loss(loss)
                total_loss += loss
                self._dispatch_event("on_prediction_step")
            metrics = {"eval_loss": (total_loss / step).item()}
            self._dispatch_event("on_evaluate", metrics=metrics)
            return metrics

    def _train_step(self, batch: Tensor) -> Tensor:
        """
        Perform a single training step

        Returns: mean loss (detached from graph)
        """
        outputs = self.model(**batch)
        loss = outputs[0]
        self._backward(loss)
        self.optimizer.step()
        self._dispatch_event("on_optimizer_step")
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
            save_steps=self._private_state.periodic_save.period,
        )

    def _init_private_state(self) -> PrivateTrainerState:
        """
        Init private training state
        This should be retained when saving a checkpoint
        """
        start_time = time.time()
        self.model.zero_grad()
        periodic_log = PeriodicFunction(
            strategy=self.args.logging_strategy,
            period=self.args.logging_steps,
            epoch_period=self.epoch_train_steps,
            f=self._log_step,
            first_step=0 if not self.args.logging_first_step else -1,
        )
        periodic_eval = PeriodicFunction(
            strategy=self.args.eval_strategy,
            period=self.args.eval_steps,
            epoch_period=self.epoch_train_steps,
            f=self._eval_loop,
            first_step=self.args.eval_delay,
        )

        periodic_save = PeriodicFunction(
            strategy=self.args.save_strategy,
            period=self.args.save_steps,
            epoch_period=self.epoch_train_steps,
            f=self._save_checkpoint,
            first_step=self.args.eval_delay,
        )
        # Tracks mean loss for each log-step
        log_step_loss = torch.zeros(1, device=self.args.device)
        # Tracks mean loss for whole session.
        total_loss = torch.zeros(1, device=self.args.device)

        return PrivateTrainerState(
            start_time=start_time,
            periodic_log=periodic_log,
            periodic_eval=periodic_eval,
            periodic_save=periodic_save,
            log_step_loss=log_step_loss,
            total_loss=total_loss,
        )

    def _end_train_loop(self):
        state = self._private_state
        runtime = time.time() - state.start_time
        total_train_batch_size = self.state.num_processes * self.state.train_batch_size
        total_train_samples = total_train_batch_size * self.max_steps
        metrics = self._speed_metrics(
            "train", runtime, total_train_samples, self.max_steps
        )
        metrics["train_loss"] = (state.total_loss / self.state.global_step).item()
        metrics["epoch"] = self.state.epoch
        return metrics

    def _prediction_step(self, batch: Tensor) -> Tensor:
        """
        Perform a single batch of predictions
        """
        outputs = self.model(**batch)
        logits = outputs[1]
        loss = outputs[0]
        return loss.mean().detach(), logits.detach(), batch["labels"]

    def _speed_metrics(self, prefix, runtime, samples, steps):
        metrics = {
            f"{prefix}_runtime": runtime,
            f"{prefix}_samples": samples,
            "step": steps,
            f"{prefix}_samples_per_second": round(samples / runtime, 3),
            f"{prefix}_steps_per_second": round(steps / runtime, 3),
        }
        return metrics

    def _update_train_loss(self, total_loss: Tensor, log_steps: int):
        if log_steps == 0:
            return self.mean_train_loss
        self.mean_train_loss = (total_loss / log_steps).item()
        total_loss -= total_loss

    def _log_step(self, total_loss: Tensor, log_steps: int):
        """
        Log training progress; called every 'log_steps' and once more at the end of training.
        Note: mean loss must be gathered if multi-process
        """
        if log_steps == 0:
            return
        self._update_train_loss(total_loss, log_steps)

        last_lr = self.lr_scheduler.get_last_lr()[0]
        if torch.is_tensor(last_lr):
            last_lr = last_lr.item()

        logs = {
            "epoch": self.state.epoch,
            "loss": self.mean_train_loss,
            "learning_rate": last_lr,
        }

        self.log(logs)

    def _prepare_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Performs any required steps to ready the data for the model to process.
        For example, normal torch training requires moving the batch to the target device.
        """
        return {k: v.to(self.args.device) for k, v in batch.items()}

    def _reduce_loss(self, loss: Tensor) -> Tensor:
        return loss
