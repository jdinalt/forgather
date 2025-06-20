# A subclass of Trainer, which adds support for the Acclerate library.
from dataclasses import dataclass, field
from collections.abc import Sequence

import torch
from torch import Tensor
from accelerate import Accelerator

from .trainer_types import TrainingArguments, TrainerState
from .trainer import Trainer


@dataclass(kw_only=True)
class AccelTrainingArguments(TrainingArguments):
    accelerator_args: dict = None


class AccelTrainer(Trainer):
    """
    Modify the base Trainer to use the Accelerate library.
    """

    def _post_init(self) -> None:
        super()._post_init()
        self.accelerator = Accelerator(**self.args.accelerator_args)

        # Accel uses a special device target
        self.args.device = self.accelerator.device
        # Update process info
        self.is_local_process_zero = self.accelerator.is_local_main_process
        self.is_world_process_zero = self.accelerator.is_main_process
        self.num_processes = self.accelerator.num_processes

    def _prepare(self, train_dataset, eval_dataset) -> None:
        super()._prepare(train_dataset, eval_dataset)

        # Wrap relevant componens with accelerator
        (
            self.train_dataloader,
            self.eval_dataloader,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.train_dataloader,
            self.eval_dataloader,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        )
        # Accelerate modifies the dataloaders, which can change both the length and the batch size.
        if train_dataset is not None:
            self._update_training_steps()
        self.accelerator.wait_for_everyone()

    def _backward(self, loss):
        self.accelerator.backward(loss)

    def _reduce_loss(self, loss: Tensor):
        """
        Reduces loss accross processes
        """
        return self.accelerator.reduce(loss, "mean")

    def _prepare_batch(self, batch):
        # The accelerate will have already moved the batch to the right device
        # We just need to split it into positional/kw-args
        if isinstance(batch, Sequence):
            return (batch, {})
        else:
            return (tuple(), batch)

    def _init_state(self) -> TrainerState:
        """
        Modifies parent state by setting process rank info
        """
        state = super()._init_state()
        # Split-batches option divides the requested batch size by the number of GPUs
        if self.args.accelerator_args["dataloader_config"].split_batches:
            state.train_batch_size = (
                self.args.per_device_train_batch_size // state.num_processes
            )
        else:
            state.train_batch_size = self.args.per_device_train_batch_size
        return state

    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    def _save(self, output_dir):
        self.accelerator.wait_for_everyone()
        super()._save(output_dir)
