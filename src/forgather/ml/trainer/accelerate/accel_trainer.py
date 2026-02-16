# A subclass of Trainer, which adds support for the Acclerate library.
import logging
from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, TypeVar, override

import torch
from accelerate import Accelerator
from torch import Tensor

from ..checkpoint_manager import RNGState
from ..checkpoint_types import SharingPattern, StateComponent
from ..trainer import Trainer, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class AccelTrainingArguments(TrainingArguments):
    pass


TAccelTrainingArguments = TypeVar(
    "TAccelTrainingArguments", bound=AccelTrainingArguments
)


class AccelTrainer(Trainer[TAccelTrainingArguments], Generic[TAccelTrainingArguments]):
    """
    Modify the base Trainer to use the Accelerate library.
    """

    def __init__(
        self,
        *,
        args: TAccelTrainingArguments,
        accelerator: Accelerator,
        **kwargs,
    ):
        assert isinstance(args, AccelTrainingArguments)
        assert isinstance(accelerator, Accelerator)
        super().__init__(args=args, **kwargs)

        self.accelerator = accelerator

        # Ensure Accelerator and TrainingArguments gradient accumulation settings are consistent
        if hasattr(accelerator, "gradient_accumulation_steps"):
            if (
                accelerator.gradient_accumulation_steps
                != args.gradient_accumulation_steps
            ):
                logger.warning(
                    f"Accelerator gradient_accumulation_steps ({accelerator.gradient_accumulation_steps}) "
                    f"differs from TrainingArguments ({args.gradient_accumulation_steps}). "
                    f"Using Accelerator's setting: {accelerator.gradient_accumulation_steps}"
                )
                args.gradient_accumulation_steps = (
                    accelerator.gradient_accumulation_steps
                )

        assert (
            not self.args.fuse_optim_with_backward
        ), "AccelTrainer does not support option fuse_optim_with_backward"

    @override
    def _init_distributed(self):
        self.is_local_process_zero = self.accelerator.is_local_main_process
        self.is_world_process_zero = self.accelerator.is_main_process
        self.num_processes = self.accelerator.num_processes

    @override
    def _init_device(self):
        # Accel uses a special device target
        self.args.device = self.accelerator.device

    @override
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

    @override
    def _distributed_loss(self, loss: Tensor) -> Tensor:
        """
        Reduces loss accross processes
        """
        reduced_loss = self.accelerator.reduce(loss, "mean")
        assert isinstance(reduced_loss, Tensor)
        return reduced_loss

    @override
    def _prepare_batch(
        self, batch: Dict[str, Tensor]
    ) -> tuple[Dict[str, Tensor], Tensor]:
        # The accelerate will have already moved the batch to the right device
        labels = batch.pop("labels")
        return (batch, labels)

    @override
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

    @override
    def unwrapped_model(self) -> torch.nn.Module:
        assert self.model
        return self.accelerator.unwrap_model(self.model)

    @override
    def get_state_components(self) -> List[StateComponent]:
        """
        Get state components for Accelerate-based distributed training.

        All training state is always saved to checkpoints. To skip loading a component,
        delete its file from the checkpoint directory.

        Accelerate uses DDP for multi-GPU training, which synchronizes model
        and optimizer state across all ranks. Therefore, we use REPLICATED
        pattern for these components with validation enabled to catch sync bugs.

        Returns:
            List of StateComponent objects with REPLICATED patterns for DDP state
        """
        components = []

        # Model - REQUIRED, REPLICATED in DDP
        # Accelerate synchronizes model weights across all ranks
        components.append(
            StateComponent(
                key="model",
                stateful=self.unwrapped_model(),
                sharing_pattern=SharingPattern.REPLICATED,
                validate_replication=True,  # Verify DDP synchronization
                validation_level="tensor",  # Good balance of speed vs accuracy
                required=True,  # Model is always required
            )
        )

        # Optimizer - optional, REPLICATED in DDP
        # Accelerate synchronizes optimizer state across all ranks
        # Note: Validation disabled - AcceleratedOptimizer wrapper may have rank-specific state
        components.append(
            StateComponent(
                key="optimizer",
                stateful=self.optimizer,
                sharing_pattern=SharingPattern.REPLICATED,
                validate_replication=False,  # Disabled: AcceleratedOptimizer has rank-specific state
                validation_level="quick",
                required=False,
            )
        )

        # LR Scheduler - optional, REPLICATED
        # Same schedule across all ranks
        components.append(
            StateComponent(
                key="scheduler",
                stateful=self.lr_scheduler,
                sharing_pattern=SharingPattern.REPLICATED,
                required=False,
            )
        )

        # Trainer state - optional, REPLICATED
        # Training progress is synchronized across all ranks
        components.append(
            StateComponent(
                key="trainer",
                stateful=self,
                sharing_pattern=SharingPattern.REPLICATED,
                required=False,
            )
        )

        # Dataset state - optional, depends on dataloader configuration
        # Accelerate can use different data loading strategies
        if hasattr(self.train_dataloader, "state_dict"):
            components.append(
                StateComponent(
                    key="dataset",
                    stateful=self.train_dataloader,
                    sharing_pattern=self._get_dataset_sharing_pattern(),
                    required=False,
                )
            )

        # RNG state - optional, PER_RANK
        # Each rank needs different random numbers for data augmentation, dropout, etc.
        components.append(
            StateComponent(
                key="rng",
                stateful=RNGState(),
                sharing_pattern=SharingPattern.PER_RANK,
                required=False,
            )
        )

        return components

    @override
    def _get_dataset_sharing_pattern(self) -> SharingPattern:
        """
        Determine dataset sharing pattern for Accelerate training.

        Accelerate can handle data loading in different ways:
        - split_batches=True: Batches split across GPUs (needs coordination)
        - split_batches=False: Each GPU gets full batch (independent)

        For now, conservatively use PER_RANK for independent iteration.
        In future, could detect DataloaderDispatcher and use GLOBAL.

        Returns:
            SharingPattern for dataset state
        """
        # TODO: Could check for DataloaderDispatcher and return GLOBAL
        # For now, assume independent dataloaders per rank
        return SharingPattern.PER_RANK

    @override
    def _end_train_loop(
        self, start_time: float, train_steps: int
    ) -> dict[str, int | float]:
        self.accelerator.end_training()
        return super()._end_train_loop(start_time, train_steps)

    @override
    def _clip_grad_norm(
        self, max_grad_norm: float | None, norm_type: float = 2.0
    ) -> Optional[Tensor]:
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

    @override
    def _backward(self, loss: Tensor) -> None:
        """
        Use Accelerate's backward method which handles gradient scaling and accumulation.
        """
        # Note: This method is kept for compatibility with the base trainer's _train_step
        # The _train_step_with_accumulation method uses accelerator.backward directly
        self.accelerator.backward(loss)

    @override
    def _should_sync_gradients(self) -> bool:
        return self.accelerator.sync_gradients
