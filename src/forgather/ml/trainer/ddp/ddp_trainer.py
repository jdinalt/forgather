import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, override

import torch
from torch import Tensor
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.parallel import DistributedDataParallel as DDP

from ..checkpoint_manager import RNGState
from ..checkpoint_types import SharingPattern, StateComponent
from ..dataloader_dispatcher import DataloaderDispatcher
from ..trainer import Trainer, TrainingArguments

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DDPTrainingArguments(TrainingArguments):
    # Load and preprocess all batches on rank-0, then dispatch to other ranks
    # All ranks are sent full batches, as specified by `per_device_train_batch_size`, where
    # the total effective batch size is per_device_train_batch_size * world_size
    #
    # This avoid the need to manually specify how to shard the dataset, at the expense of
    # adding some non-zero amount of processing latency. This also greatly simplifies dataset
    # checkpointing, as there is only one global state to keep track of.
    #
    # When set to False, care must be taken to ensure that each rank receives different examples.
    dispatch_batches: bool = True

    # These are the same as the arguments to DDP
    # See: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
    ddp_broadcast_buffers: bool = True
    ddp_init_sync: bool = True
    ddp_bucket_cap_mb: Optional[int] = None
    ddp_find_unused_parameters: bool = False
    ddp_gradient_as_bucket_view: bool = True
    ddp_static_graph: bool = False
    ddp_skip_all_reduce_unused_params: bool = False


class DDPTrainer(Trainer):
    """
    Modify the base Trainer to use the Accelerate library.
    """

    gradient_accumulation_step: int

    def __init__(
        self,
        *,
        args: DDPTrainingArguments,
        **kwargs,
    ):
        self.args = args  # For type checking hint
        super().__init__(args=args, **kwargs)

    def _post_init(self) -> None:
        super()._post_init()

    @override
    def _init_distributed(self):
        assert (
            dist.is_initialized()
        ), "DDP trainer requires that torch.distributed has been initialized."
        self.is_local_process_zero = self.dist.local_rank == 0
        self.is_world_process_zero = self.dist.rank == 0
        self.num_processes = self.dist.world_size

        self.mesh = init_device_mesh(
            self.dist.device_type,
            (self.dist.world_size,),
            mesh_dim_names=("data_parallel",),
        )
        self.ddp_group = self.mesh.get_group(0)  # data-parallel group

    @override
    def _init_device(self):
        self.args.device = self.dist.device

    @override
    def _wrap(
        self,
    ) -> None:
        """
        Wrap assets for DDP
        """

        self.model = DDP(
            self.model,
            device_ids=[self.args.device] if self.dist.device_type != "cpu" else None,
            process_group=self.ddp_group,
            broadcast_buffers=self.args.ddp_broadcast_buffers,
            init_sync=self.args.ddp_init_sync,
            bucket_cap_mb=self.args.ddp_bucket_cap_mb,
            find_unused_parameters=self.args.ddp_find_unused_parameters,
            gradient_as_bucket_view=self.args.ddp_gradient_as_bucket_view,
            static_graph=self.args.ddp_static_graph,
            skip_all_reduce_unused_params=self.args.ddp_skip_all_reduce_unused_params,
        )

        # TODO:
        # - Fused loss function will probably not work.

        if self.args.dispatch_batches:
            if self.train_dataloader:
                self.train_dataloader = DataloaderDispatcher(
                    self.train_dataloader,
                    self.mesh,
                    self.args.device,
                )

            if self.eval_dataloader:
                self.eval_dataloader = DataloaderDispatcher(
                    self.eval_dataloader,
                    self.mesh,
                    self.args.device,
                )

    @override
    def unwrapped_model(self) -> torch.nn.Module:
        """
        Get and returned the wrapped model

        In the case of DDP, the original model is stored in the model's "module" attribute.
        """
        assert self.model
        return self.model.module

    @override
    def _distributed_loss(self, loss: Tensor) -> Tensor:
        """
        Reduces loss across processes
        """
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        return loss

    @override
    def _forward_backward_step(
        self,
        *args,
        **kwargs,
    ) -> Tensor:
        """
        Skip gradient reduction when not a gradient sync step.

        This is achieved with DDP's "no_sync" context manager.
        """
        with nullcontext() if self._should_sync_gradients() else self.model.no_sync():
            return super()._forward_backward_step(*args, **kwargs)

    @override
    def get_state_components(self) -> List[StateComponent]:
        """
        Get state components for DDP training.

        DDP uses data parallelism where model and optimizer state are replicated
        across all ranks. DDP automatically synchronizes model weights and gradients,
        so these components use REPLICATED pattern with validation enabled to catch
        synchronization bugs.

        Dataset pattern depends on dispatch_batches setting:
        - dispatch_batches=True: GLOBAL (rank 0 loads and dispatches)
        - dispatch_batches=False: PER_RANK (each rank has independent dataloader)

        Returns:
            List of StateComponent objects with REPLICATED patterns for DDP state
        """
        components = []

        # Model - REPLICATED in DDP
        # DDP synchronizes model weights across all ranks
        components.append(
            StateComponent(
                key="model",
                stateful=self.unwrapped_model(),
                sharing_pattern=SharingPattern.REPLICATED,
                validate_replication=True,  # Verify DDP synchronization
                validation_level="tensor",  # Good balance of speed vs accuracy
            )
        )

        # Optimizer - REPLICATED in DDP
        # DDP synchronizes gradients, so optimizer state should be identical
        if self.args.save_optimizer_state:
            components.append(
                StateComponent(
                    key="optimizer",
                    stateful=self.optimizer,
                    sharing_pattern=SharingPattern.REPLICATED,
                    validate_replication=True,
                    validation_level="quick",  # Faster for large optimizers
                    required=self.args.save_optimizer_state,
                )
            )

        # LR Scheduler - REPLICATED
        # Same schedule across all ranks
        if self.args.save_scheduler_state:
            components.append(
                StateComponent(
                    key="scheduler",
                    stateful=self.lr_scheduler,
                    sharing_pattern=SharingPattern.REPLICATED,
                    required=self.args.save_scheduler_state,
                )
            )

        # Trainer state - REPLICATED
        # Training progress is synchronized across all ranks
        if self.args.save_dataset_state:
            components.append(
                StateComponent(
                    key="trainer",
                    stateful=self,
                    sharing_pattern=SharingPattern.REPLICATED,
                    required=self.args.save_dataset_state,
                )
            )

        # Dataset state - depends on dispatch_batches setting
        if self.args.save_dataset_state and hasattr(self.train_dataloader, "state_dict"):
            components.append(
                StateComponent(
                    key="dataset",
                    stateful=self.train_dataloader,
                    sharing_pattern=self._get_dataset_sharing_pattern(),
                    required=False,
                )
            )

        # RNG state - PER_RANK
        # Each rank needs different random numbers for data augmentation, dropout, etc.
        if self.args.save_rng_state:
            components.append(
                StateComponent(
                    key="rng",
                    stateful=RNGState(),
                    sharing_pattern=SharingPattern.PER_RANK,
                    required=self.args.save_rng_state,
                )
            )

        return components

    @override
    def _get_dataset_sharing_pattern(self) -> SharingPattern:
        """
        Determine dataset sharing pattern for DDP training.

        The pattern depends on the dispatch_batches setting:
        - dispatch_batches=True: Uses DataloaderDispatcher where rank 0 loads
          data and broadcasts to all ranks (GLOBAL pattern)
        - dispatch_batches=False: Each rank has independent dataloader iteration
          (PER_RANK pattern)

        Returns:
            SharingPattern for dataset state (GLOBAL or PER_RANK)
        """
        if self.args.dispatch_batches:
            # DataloaderDispatcher: rank 0 loads and broadcasts
            return SharingPattern.GLOBAL
        else:
            # Independent dataloaders per rank
            return SharingPattern.PER_RANK

    @override
    def get_process_groups(self) -> Dict[str, any]:
        """
        Get named process groups for checkpoint coordination.

        Returns:
            Dictionary mapping group names to ProcessGroup objects.
            For DDP, returns the data parallel group.
        """
        return {
            "ddp_group": self.ddp_group,
        }
