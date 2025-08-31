# A subclass of Trainer, which adds support for the Acclerate library.
from dataclasses import dataclass, field
from collections.abc import Sequence
import os
import logging

import torch
from torch import Tensor
from accelerate import Accelerator

from ..trainer_types import TrainingArguments, TrainerState
from ..trainer import Trainer

# Import StatefulDataLoader if available
try:
    from torchdata.stateful_dataloader import StatefulDataLoader

    STATEFUL_DATALOADER_AVAILABLE = True
except ImportError:
    StatefulDataLoader = None
    STATEFUL_DATALOADER_AVAILABLE = False

logger = logging.getLogger(__name__)


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

    # @override
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
        self._barrier()

    # @override
    def _backward(self, loss):
        self.accelerator.backward(loss)

    # @override
    def _distributed_loss(self, loss: Tensor):
        """
        Reduces loss accross processes
        """
        return self.accelerator.reduce(loss, "mean")

    def _save_dataloader_state(self):
        """Save StatefulDataLoader state if available, accounting for Accelerate wrapping."""
        # After accelerate.prepare(), the dataloader may be wrapped
        dataloader = getattr(self, "train_dataloader", None)
        if dataloader is None:
            return None

        # Try to get state from wrapped dataloader
        if hasattr(dataloader, "state_dict"):
            try:
                return dataloader.state_dict()
            except Exception as e:
                logger.warning(f"Failed to save dataloader state: {e}")

        return super()._save_dataloader_state()

    def _load_dataloader_state(self, dataloader_state):
        """Load StatefulDataLoader state if available, accounting for Accelerate wrapping."""
        dataloader = getattr(self, "train_dataloader", None)
        if dataloader is None:
            return

        # Try to load state into wrapped dataloader
        if hasattr(dataloader, "load_state_dict"):
            try:
                dataloader.load_state_dict(dataloader_state)
                return
            except Exception as e:
                logger.warning(f"Failed to load dataloader state: {e}")

        super()._load_dataloader_state(dataloader_state)

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
        if self.args.accelerator_args["dataloader_config"].split_batches:
            state.train_batch_size = (
                self.args.per_device_train_batch_size // state.num_processes
            )
        else:
            state.train_batch_size = self.args.per_device_train_batch_size
        return state

    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    def _should_use_accelerate_checkpoint(self) -> bool:
        """Determine if we should use Accelerate's native checkpointing."""
        return (
            self.args.save_optimizer_state
            and self.args.save_scheduler_state
            and hasattr(self.accelerator, "save_state")
        )

    # @override
    def _save_training_state(self, output_dir: str) -> None:
        """Override to use Accelerate's checkpoint saving when possible."""
        if self._should_save_unique():
            if (
                hasattr(self.accelerator, "save_state")
                and self._should_use_accelerate_checkpoint()
            ):
                accelerate_checkpoint_path = os.path.join(
                    output_dir, "accelerate_state"
                )
                self.accelerator.save_state(accelerate_checkpoint_path)
                logger.info(f"Saved accelerate state to {accelerate_checkpoint_path}")
            else:
                super()._save_training_state(output_dir)

    # @override
    def _save_model(self, output_dir):
        if self._should_save_unique():
            super()._save_model()

    # @override
    def _load_training_state(self, checkpoint_path: str) -> None:
        """Override to use Accelerate's checkpoint loading when possible."""
        accelerate_checkpoint_path = os.path.join(checkpoint_path, "accelerate_state")

        if os.path.exists(accelerate_checkpoint_path) and hasattr(
            self.accelerator, "load_state"
        ):
            self.accelerator.load_state(accelerate_checkpoint_path)
            logger.info(f"Loaded accelerate state from {accelerate_checkpoint_path}")
        else:
            super()._load_training_state(checkpoint_path)
        self._barrier()

    # @override
    def _barrier(self):
        self.accelerator.wait_for_everyone()

    # @override
    def _should_save_unique(self):
        """
        Should this process save a unique file?
        """
        return self.accelerator.is_main_process
