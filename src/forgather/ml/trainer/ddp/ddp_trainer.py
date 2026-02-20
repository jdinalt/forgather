import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar, cast, override

import torch
import torch.distributed.algorithms.model_averaging.averagers as averagers
from dacite import from_dict
from torch import Tensor
from torch import distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import (
    PostLocalSGDState,
    post_localSGD_hook,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.optim import PostLocalSGDOptimizer
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

from forgather.ml.datasets import sync_dataset_state_from_dataloader
from forgather.ml.distributed import prefix_logger_rank
from forgather.ml.loss import RescaleLoss
from forgather.ml.trainer import DataloaderDispatcher
from forgather.ml.trainer.base_trainer import logits_from_outputs
from forgather.ml.trainer.checkpoint_manager import RNGState
from forgather.ml.trainer.checkpoint_types import SharingPattern, StateComponent
from forgather.ml.trainer.synchronized_dataloader import SynchronizedDataLoader
from forgather.ml.trainer.trainer import Trainer, TrainingArguments, set_train
from forgather.ml.trainer.trainer_types import FusedLossFactoryT

logger = logging.getLogger(__name__)
prefix_logger_rank(logger, lambda rank: True)


@dataclass(kw_only=True)
class DDPArguments:
    # These are the same as the arguments to DDP
    # See: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
    broadcast_buffers: bool = True
    init_sync: bool = True
    bucket_cap_mb: Optional[int] = None
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    skip_all_reduce_unused_params: bool = False


@dataclass(kw_only=True)
class PostLocalSGDArguments:
    enabled: bool = False
    start_step: int = 500
    period: int = 4
    post_local_gradient_allreduce: bool = False


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

    ddp: DDPArguments = field(default_factory=DDPArguments)
    post_local_sgd: PostLocalSGDArguments = field(default_factory=PostLocalSGDArguments)


TDDPTrainingArguments = TypeVar("TDDPTrainingArguments", bound=DDPTrainingArguments)


class DDPTrainer(Trainer[TDDPTrainingArguments], Generic[TDDPTrainingArguments]):
    """
    Modify the base Trainer to use the Accelerate library.
    """

    args: TDDPTrainingArguments
    gradient_accumulation_step: int

    def __init__(
        self,
        *,
        args: TDDPTrainingArguments | dict,
        fused_loss_factory: Optional[FusedLossFactoryT] = None,
        **kwargs,
    ):
        if isinstance(args, dict):
            args = cast(TDDPTrainingArguments, from_dict(DDPTrainingArguments, args))

        super().__init__(args=args, fused_loss_factory=fused_loss_factory, **kwargs)

        assert (
            not self.args.fuse_optim_with_backward
        ), "DDPTrainer does not support option fuse_optim_with_backward"

    @override
    def _init_distributed(self):
        assert (
            dist.is_available and dist.is_initialized()
        ) or self.dist.world_size == 1, (
            "DDP trainer requires that torch.distributed has been initialized."
        )

        if self.dist.world_size == 1:
            return super()._init_distributed()

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
    def _wrap(
        self,
    ) -> None:
        """
        Wrap assets for DDP
        """
        if self.dist.world_size == 1:
            return super()._wrap()

        self.model = DDP(
            self.model,
            device_ids=[self.args.device] if self.dist.device_type != "cpu" else None,
            process_group=self.ddp_group,
            broadcast_buffers=self.args.ddp.broadcast_buffers,
            init_sync=self.args.ddp.init_sync,
            bucket_cap_mb=self.args.ddp.bucket_cap_mb,
            find_unused_parameters=self.args.ddp.find_unused_parameters,
            gradient_as_bucket_view=self.args.ddp.gradient_as_bucket_view,
            static_graph=self.args.ddp.static_graph,
            skip_all_reduce_unused_params=self.args.ddp.skip_all_reduce_unused_params,
        )

        if self.args.dispatch_batches:
            # Use DataloaderDispatcher for centralized batch loading
            if self.train_dataloader:
                self.train_dataloader = DataloaderDispatcher(
                    cast(DataLoader, self.train_dataloader),
                    self.mesh,
                    self.args.device,
                )

            if self.eval_dataloader:
                self.eval_dataloader = DataloaderDispatcher(
                    cast(DataLoader, self.eval_dataloader),
                    self.mesh,
                    self.args.device,
                )
        else:
            # Use SynchronizedDataLoader for sharded datasets
            # Ensures all ranks agree on when to stop iterating
            if self.train_dataloader:
                self.train_dataloader = SynchronizedDataLoader(
                    self.train_dataloader,
                    device=self.args.device,
                    process_group=self.ddp_group,
                )

            if self.eval_dataloader:
                self.eval_dataloader = SynchronizedDataLoader(
                    self.eval_dataloader,
                    device=self.args.device,
                    process_group=self.ddp_group,
                )

        assert self.optimizer is not None
        if self.args.post_local_sgd.enabled:
            logger.info(f"Enabling post-local-SGD: {self.args.post_local_sgd}")
            self.post_local_sgd_state = PostLocalSGDState(
                process_group=self.ddp_group,
                subgroup=None,
                start_localSGD_iter=self.args.post_local_sgd.start_step,
                post_local_gradient_allreduce=self.args.post_local_sgd.post_local_gradient_allreduce,
            )
            self.model.register_comm_hook(self.post_local_sgd_state, post_localSGD_hook)
            self.optimizer = PostLocalSGDOptimizer(
                optim=cast(Optimizer, self.optimizer),
                averager=averagers.PeriodicModelAverager(
                    period=self.args.post_local_sgd.period,
                    warmup_steps=self.args.post_local_sgd.start_step,
                ),
            )

    @override
    def unwrapped_model(self) -> torch.nn.Module:
        """
        Get and returned the wrapped model

        In the case of DDP, the original model is stored in the model's "module" attribute.
        """
        if self.dist.world_size == 1:
            return super().unwrapped_model()

        assert self.model
        return cast(Module, self.model.module)

    @override
    @torch.no_grad()
    def _eval_loop(self) -> Dict[str, float]:
        """
        Evaluation loop for DDP training.

        For dispatch_batches=True or single-GPU, delegates to the base Trainer._eval_loop().

        For dispatch_batches=False, bypasses SynchronizedDataLoader's MIN-based
        synchronization to let each rank process ALL its local validation data
        independently. This prevents data loss when token packing creates highly
        uneven shard lengths across ranks (e.g., shard lengths [85, 167, 239, 247]
        would otherwise be truncated to the shortest rank's count).
        """
        if self.dist.world_size == 1 or self.args.dispatch_batches:
            return super()._eval_loop()
        return self._eval_loop_all_shards()

    @torch.no_grad()
    def _eval_loop_all_shards(self) -> Dict[str, float]:
        """
        Eval loop that lets each rank process all its local validation data.

        Instead of using SynchronizedDataLoader (which stops all ranks when ANY
        rank runs out of data via MIN reduction), this synchronizes step-by-step
        using MAX reduction: the loop continues while ANY rank still has data.
        Ranks that exhaust their data early continue participating in the sync
        loop without processing, keeping all ranks in lockstep so rank 0's
        progress bar updates smoothly throughout.

        After all ranks are exhausted, total_loss is aggregated via all_reduce.
        The step count is implicit since each rank tracked its own local count.
        """
        assert self.model is not None
        assert self.eval_dataloader is not None
        assert isinstance(self.loss_fn, RescaleLoss)

        # Access the underlying dataloader, bypassing SynchronizedDataLoader
        assert isinstance(self.eval_dataloader, SynchronizedDataLoader)
        raw_dataloader = self.eval_dataloader._dataloader

        with set_train(self.model, False):
            total_loss = torch.zeros(1, device=self.args.device)
            local_steps = 0
            has_data = True
            iterator = iter(raw_dataloader)

            # Reuse a single tensor for the per-step synchronization
            has_data_tensor = torch.zeros(1, dtype=torch.int32, device=self.args.device)

            while True:
                # Try to get next batch from this rank's data
                batch = None
                if has_data:
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        has_data = False

                    # Respect max_eval_steps on this rank's own data
                    if (
                        has_data
                        and self.args.max_eval_steps > 0
                        and local_steps >= self.args.max_eval_steps
                    ):
                        has_data = False
                        batch = None

                # Synchronize: continue while ANY rank still has data (MAX as OR)
                has_data_tensor.fill_(1 if has_data else 0)
                dist.all_reduce(has_data_tensor, op=dist.ReduceOp.MAX)

                if has_data_tensor.item() == 0:
                    # All ranks are done
                    break

                # Process batch if this rank has one
                if batch is not None:
                    input_dict, labels = self._prepare_batch(batch)

                    # Inline the forward pass from _prediction_step, but without
                    # _distributed_loss (which would require an additional
                    # per-step all_reduce).
                    if self.use_fused_loss:
                        input_dict["return_hidden_states"] = True  # type: ignore[assignment]
                    with self.loss_fn.no_rescale(), self.amp_context.autocast():
                        outputs = self.model(**input_dict)
                        logits = logits_from_outputs(outputs)
                        loss = self.loss_fn(logits, labels)

                    total_loss += loss.detach()
                    local_steps += 1

                # Dispatch on every synchronized step so rank 0's progress bar
                # keeps updating even after rank 0 exhausts its own data.
                self._dispatch_event("on_prediction_step")

            # Aggregate loss across all ranks
            step_count = torch.tensor(
                local_steps, device=self.args.device, dtype=torch.int64
            )
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(step_count, op=dist.ReduceOp.SUM)

            total_steps = step_count.item()
            assert total_steps > 0, "No eval examples were processed across any rank"
            eval_loss = (total_loss / total_steps).item()

            # Sync dataset state on the underlying StatefulDataLoader
            if isinstance(raw_dataloader, StatefulDataLoader):
                sync_dataset_state_from_dataloader(raw_dataloader)

            metrics = {"eval_loss": eval_loss}
            self._dispatch_event("on_evaluate", metrics=metrics)
            return metrics

    @override
    def _distributed_loss(self, loss: Tensor) -> Tensor:
        """
        Reduces loss across processes
        """
        if self.dist.world_size == 1:
            return super()._distributed_loss(loss)

        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        return loss

    @override
    def _distributed_tokens(self, tokens: Tensor) -> Tensor:
        """
        Sum token counts across all DDP ranks.

        In DDP, each rank processes different batches, so token counts must be
        summed across all ranks to get the total tokens processed per step.

        Args:
            tokens: Token count from current rank

        Returns:
            Total token count across all DDP ranks
        """
        if self.dist.world_size == 1:
            return super()._distributed_tokens(tokens)

        dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
        return tokens

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
        if self.dist.world_size == 1:
            return super()._forward_backward_step(*args, **kwargs)

        with (
            nullcontext()
            if self._should_sync_gradients()
            else cast(DDP, self.model).no_sync()
        ):
            return super()._forward_backward_step(*args, **kwargs)

    @override
    def get_state_components(self) -> List[StateComponent]:
        """
        Get state components for DDP training.

        All training state is always saved to checkpoints. To skip loading a component,
        delete its file from the checkpoint directory.

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
        if self.dist.world_size == 1:
            return super().get_state_components()

        components = []

        # Model - REQUIRED, REPLICATED in DDP
        # DDP synchronizes model weights across all ranks
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
        # DDP synchronizes gradients, so optimizer state should be identical
        if self.optimizer is not None:
            components.append(
                StateComponent(
                    key="optimizer",
                    stateful=cast(Stateful, self.optimizer),
                    sharing_pattern=SharingPattern.REPLICATED,
                    validate_replication=True,
                    validation_level="tensor",  # Per-tensor checksums for accurate validation
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

        # Dataset state - optional, depends on dispatch_batches setting
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
        if self.dist.world_size == 1:
            return super()._get_dataset_sharing_pattern()

        if self.args.dispatch_batches:
            # DataloaderDispatcher: rank 0 loads and broadcasts
            return SharingPattern.GLOBAL
        else:
            # Independent dataloaders per rank
            return SharingPattern.PER_RANK

    @override
    def get_process_groups(self) -> Dict[str, Any]:
        """
        Get named process groups for checkpoint coordination.

        Returns:
            Dictionary mapping group names to ProcessGroup objects.
            For DDP, returns the data parallel group.
        """
        if self.dist.world_size == 1:
            return super().get_process_groups()

        return {
            "ddp_group": self.ddp_group,
        }
