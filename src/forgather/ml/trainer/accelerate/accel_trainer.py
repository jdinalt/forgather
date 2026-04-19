# A subclass of Trainer, which adds support for the Acclerate library.
import logging
from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, TypeVar, cast, override

import torch
from accelerate import Accelerator
from dacite import from_dict
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful

from ..checkpoint_manager import RNGState
from ..checkpoint_types import SharingPattern, StateComponent
from ..trainer import Trainer, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class AccelTrainingArguments(TrainingArguments):
    """Training arguments for Accelerate-based multi-GPU training.

    Extends ``TrainingArguments`` with no additional fields. Mixed precision
    and gradient accumulation are configured through the ``Accelerator``
    object passed to ``AccelTrainer.__init__()``, not through these arguments.

    .. note::
        ``mixed_precision`` from the parent class is **ignored** by
        ``AccelTrainer``; set it on the ``Accelerator`` instead.
        ``fuse_optim_with_backward`` is not supported and will raise if set.
    """

    pass


TAccelTrainingArguments = TypeVar(
    "TAccelTrainingArguments", bound=AccelTrainingArguments
)


class AccelTrainer(Trainer[TAccelTrainingArguments], Generic[TAccelTrainingArguments]):
    """Trainer that uses HuggingFace Accelerate for multi-GPU / distributed training.

    Wraps ``Trainer`` with an ``Accelerator`` instance so that model, optimizer,
    scheduler, and dataloaders are automatically prepared for the configured
    distributed backend (DDP, FSDP, DeepSpeed, etc.).

    Key behavioural differences from the base ``Trainer``:

    * Model and optimizer weights are synchronised across ranks by DDP
      (``REPLICATED`` checkpoint sharing pattern).
    * Mixed precision is configured on the ``Accelerator``; ``args.mixed_precision``
      is explicitly ignored and reset to ``None`` with a warning.
    * Gradient accumulation step count is taken from the ``Accelerator`` when
      it conflicts with ``args.gradient_accumulation_steps``.
    * ``args.fuse_optim_with_backward`` is not supported and raises on init.
    * Gradient clipping delegates to ``accelerator.clip_grad_norm_()``.

    Parameters
    ----------
    args : AccelTrainingArguments or dict
        Training configuration. Dicts are converted via ``dacite.from_dict``.
    accelerator : accelerate.Accelerator
        Pre-configured Accelerator instance. Controls device placement, mixed
        precision, and gradient synchronisation.
    **kwargs
        Additional keyword arguments forwarded to the base ``Trainer``
        (``model``, ``train_dataset``, ``eval_dataset``, ``callbacks``, etc.).

    Raises
    ------
    AssertionError
        If ``accelerator`` is not an ``Accelerator`` instance.
    AssertionError
        If ``args.fuse_optim_with_backward`` is ``True``.
    """

    args: TAccelTrainingArguments

    def __init__(
        self,
        *,
        args: TAccelTrainingArguments | dict,
        accelerator: Accelerator,
        **kwargs,
    ):
        if isinstance(args, dict):
            args = cast(
                TAccelTrainingArguments, from_dict(AccelTrainingArguments, args)
            )
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

        if self.args.mixed_precision is not None:
            logger.warning(
                "AccelTrainer ignores args.mixed_precision. "
                "Configure mixed precision via Accelerator(mixed_precision=...) instead. "
                f"Ignoring mixed_precision='{self.args.mixed_precision}'"
            )
            self.args.mixed_precision = None

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
        """Reduce the per-rank loss to a mean across all processes.

        Parameters
        ----------
        loss : Tensor
            Local loss scalar on this rank.

        Returns
        -------
        Tensor
            Mean loss averaged across all participating ranks.
        """
        reduced_loss = self.accelerator.reduce(loss, "mean")
        assert isinstance(reduced_loss, Tensor)
        return reduced_loss

    @override
    def _distributed_peak_mem(self, local_peak: int) -> list[int]:
        """All-gather per-rank peak CUDA memory usage via Accelerate.

        Falls back to the single-rank implementation when ``world_size == 1``.

        Parameters
        ----------
        local_peak : int
            Peak CUDA memory allocated on this rank, in bytes.

        Returns
        -------
        list of int
            Peak memory in bytes for each rank, indexed by rank.
        """
        if self.dist.world_size == 1:
            return super()._distributed_peak_mem(local_peak)

        value = torch.tensor(
            [int(local_peak)], dtype=torch.long, device=self.args.device
        )
        gathered = self.accelerator.gather(value)
        assert isinstance(gathered, Tensor)
        return [int(v) for v in gathered.tolist()]

    @override
    def _prepare_batch(
        self, batch: Dict[str, Tensor]
    ) -> tuple[Dict[str, Tensor], Tensor]:
        # The accelerate will have already moved the batch to the right device
        labels = batch.pop("labels")
        return (batch, labels)

    @override
    def _init_state(self) -> TrainerState:
        """Initialise trainer state, adjusting batch size for Accelerate's split-batch mode.

        When ``split_batches=True`` on the ``Accelerator``, the requested per-device
        batch size is divided across GPUs rather than replicated, so
        ``state.train_batch_size`` must reflect that.

        Returns
        -------
        TrainerState
            Initialised state with correct ``train_batch_size``.
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
        """Return state components for Accelerate-based distributed training.

        Accelerate uses DDP, which keeps model and optimizer state synchronised
        across all ranks. This is reflected in the ``REPLICATED`` sharing pattern
        for most components, enabling the checkpoint manager to save from rank 0
        only and validate synchronisation.

        Returned components (in order):

        * ``"model"`` — REPLICATED, required, replication validation enabled.
        * ``"optimizer"`` — REPLICATED, optional; validation disabled due to
          ``AcceleratedOptimizer`` wrapper holding rank-specific state.
        * ``"scheduler"`` — REPLICATED, optional.
        * ``"trainer"`` — REPLICATED, optional.
        * ``"dataset"`` — sharing pattern from ``_get_dataset_sharing_pattern()``,
          optional; only added when the dataloader exposes a ``state_dict``.
        * ``"rng"`` — PER_RANK, optional.

        Returns
        -------
        list of StateComponent
            All checkpointable state components with their sharing patterns.
        """
        components = []

        # Model - REQUIRED, REPLICATED in DDP
        # Accelerate synchronizes model weights across all ranks
        # cast: nn.Module doesn't structurally satisfy Stateful (state_dict returns None, not dict)
        components.append(
            StateComponent(
                key="model",
                stateful=cast(Stateful, self.unwrapped_model()),
                sharing_pattern=SharingPattern.REPLICATED,
                validate_replication=True,  # Verify DDP synchronization
                validation_level="tensor",  # Good balance of speed vs accuracy
                required=True,  # Model is always required
            )
        )

        # Optimizer - optional, REPLICATED in DDP
        # Accelerate synchronizes optimizer state across all ranks
        # Note: Validation disabled - AcceleratedOptimizer wrapper may have rank-specific state
        if self.optimizer is not None:
            components.append(
                StateComponent(
                    key="optimizer",
                    stateful=cast(Stateful, self.optimizer),
                    sharing_pattern=SharingPattern.REPLICATED,
                    validate_replication=False,  # Disabled: AcceleratedOptimizer has rank-specific state
                    validation_level="quick",
                    required=False,
                )
            )

        # LR Scheduler - optional, REPLICATED
        # Same schedule across all ranks
        if self.lr_scheduler is not None:
            components.append(
                StateComponent(
                    key="scheduler",
                    stateful=cast(Stateful, self.lr_scheduler),
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
        # cast: DataLoader doesn't structurally satisfy Stateful without stateful dataloader support
        if hasattr(self.train_dataloader, "state_dict"):
            components.append(
                StateComponent(
                    key="dataset",
                    stateful=cast(Stateful, self.train_dataloader),
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
        """Return the dataset sharing pattern for Accelerate training.

        Accelerate supports two data-loading strategies:

        * ``split_batches=True`` — a single batch is split across GPUs; a
          ``DataloaderDispatcher`` would be used and ``GLOBAL`` would be ideal.
        * ``split_batches=False`` — each GPU iterates its own shard
          independently.

        Currently returns ``PER_RANK`` as a conservative default for both
        strategies. Future work may detect ``DataloaderDispatcher`` and return
        ``GLOBAL`` for the split-batches case.

        Returns
        -------
        SharingPattern
            ``SharingPattern.PER_RANK``.
        """
        # TODO: Could check for DataloaderDispatcher and return GLOBAL
        # For now, assume independent dataloaders per rank
        return SharingPattern.PER_RANK

    @override
    def _end_train_loop(
        self, start_time: float | None, train_steps: int
    ) -> dict[str, int | float]:
        self.accelerator.end_training()
        return super()._end_train_loop(start_time, train_steps)

    @override
    def _clip_grad_norm(
        self, max_grad_norm: float | None, norm_type: float = 2.0
    ) -> Optional[Tensor]:
        assert self.model is not None
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
        """Run the backward pass using Accelerate, which handles gradient scaling.

        Delegates to ``accelerator.backward(loss)`` so that mixed-precision loss
        scaling and gradient-accumulation context management are handled by the
        Accelerator rather than by the base trainer.

        Parameters
        ----------
        loss : Tensor
            Scalar loss tensor to differentiate.
        """
        # Note: This method is kept for compatibility with the base trainer's _train_step
        # The _train_step_with_accumulation method uses accelerator.backward directly
        self.accelerator.backward(loss)

    @override
    def _should_sync_gradients(self) -> bool:
        return self.accelerator.sync_gradients
