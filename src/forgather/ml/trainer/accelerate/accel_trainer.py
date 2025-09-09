# A subclass of Trainer, which adds support for the Acclerate library.
from typing import Any, Optional
from dataclasses import dataclass, field
from collections.abc import Sequence
import os
import logging
from functools import partial

import torch
from torch import Tensor
from accelerate import Accelerator

from ..trainer_types import TrainingArguments, TrainerState
from ..trainer import Trainer

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class AccelTrainingArguments(TrainingArguments):
    pass


class AccelTrainer(Trainer):
    """
    Modify the base Trainer to use the Accelerate library.
    """

    def __init__(
        self,
        *,
        args,
        accelerator: Accelerator,
        **kwargs,
    ):
        assert isinstance(args, AccelTrainingArguments)
        assert isinstance(accelerator, Accelerator)
        self.accelerator = accelerator
        super().__init__(args=args, **kwargs)

    def _post_init(self) -> None:
        super()._post_init()
        assert (
            not self.args.fuse_optim_with_backward
        ), "AccelTrainer does not support option fuse_optim_with_backward"

        # Accel uses a special device target
        self.args.device = self.accelerator.device
        # Update process info
        self.is_local_process_zero = self.accelerator.is_local_main_process
        self.is_world_process_zero = self.accelerator.is_main_process
        self.num_processes = self.accelerator.num_processes

    # @override
    def _wrap(
        self,
    ) -> None:
        super()._wrap()

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
        # TODO: Need to enable stateful dataloader correctly
        # Using the following seems to cause issues, but only some of the time.
        # self.train_dataloader = accelerate.data_loader.prepare_data_loader(
        #    self.train_dataloader,
        #    use_stateful_dataloader=True
        # )
        # Accelerate modifies the dataloaders, which can change both the length and the batch size.
        if self.train_dataloader is not None:
            self._update_training_steps()

    # @override
    def _backward(self, loss):
        self.accelerator.backward(loss)

    # @override
    def _distributed_loss(self, loss: Tensor):
        """
        Reduces loss accross processes
        """
        return self.accelerator.reduce(loss, "mean")

    # @override
    def _prepare_batch(self, batch):
        # The accelerate will have already moved the batch to the right device
        # We just need to split it into positional/kw-args
        if isinstance(batch, Sequence):
            return (batch, {})
        else:
            return (tuple(), batch)

    # @override
    def _init_state(self) -> TrainerState:
        """
        Modifies parent state by setting process rank info
        """
        state = super()._init_state()
        # Split-batches option divides the requested batch size by the number of GPUs
        if self.accelerator.dataloader_config.split_batches:
            state.train_batch_size = (
                self.args.per_device_train_batch_size // state.num_processes
            )
        else:
            state.train_batch_size = self.args.per_device_train_batch_size
        return state

    # @override
    def unwrapped_model(self) -> torch.nn.Module:
        assert self.model
        return self.accelerator.unwrap_model(self.model)

    # @override
    def _dispatch_event(self, event: str, **kwargs):
        match event:
            case "on_train_end":
                self.accelerator.end_training()
        super()._dispatch_event(event, **kwargs)

    # @override
    def _clip_grad_norm(self, max_grad_norm, norm_type=2.0) -> Optional[Tensor]:
        # If not clipping, just compute and return it
        # It's unclear if this will work right with Accelerate?
        total_norm = None
        if self.accelerator.sync_gradients:
            if max_grad_norm is None or max_grad_norm == 0:
                grads = [p.grad for p in self.model.parameters() if p.grad is not None]

                total_norm = torch.nn.utils.get_total_norm(
                    grads, norm_type=norm_type, foreach=True
                )
                return total_norm

            # Otherwise, use fused clip_grad_norm_
            total_norm = self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                max_grad_norm,
                norm_type=int(norm_type),
            )

        return total_norm
