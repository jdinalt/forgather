import itertools
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
prefix_logger_rank(logger, show_all_ranks=True)


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

    # Optional eval-only override for `dispatch_batches`. ``None`` (default) means
    # eval follows whatever `dispatch_batches` is set to. Set to ``True`` to make
    # eval centralise on rank 0 even when training shards across ranks; this is
    # useful when the eval split is small enough that per-rank shards may not
    # contain enough examples to fill a single batch (especially with
    # ``dataloader_drop_last=True``). See
    # ``docs/trainers/distributed-eval-zero-batches.md`` for the failure mode
    # this guards against.
    dispatch_eval_batches: Optional[bool] = None

    ddp: DDPArguments = field(default_factory=DDPArguments)
    post_local_sgd: PostLocalSGDArguments = field(default_factory=PostLocalSGDArguments)


TDDPTrainingArguments = TypeVar("TDDPTrainingArguments", bound=DDPTrainingArguments)


class DDPTrainer(Trainer[TDDPTrainingArguments], Generic[TDDPTrainingArguments]):
    """
    Multi-GPU trainer using DistributedDataParallel (DDP).

    Wraps the base ``Trainer`` with DDP for data-parallel training across multiple GPUs
    or nodes. Each rank receives a different batch; gradients are all-reduced automatically
    after each backward pass. Optionally uses PostLocalSGD for bandwidth-limited environments.

    Launch with ``torchrun`` (or the ``forgather train -d ...`` shortcut)::

        torchrun --nproc_per_node=4 train.py

    Key differences from single-GPU ``Trainer``:

    - Model wrapped in ``torch.nn.parallel.DistributedDataParallel``
    - Gradient accumulation uses DDP's ``no_sync()`` context to skip reductions on
      intermediate steps
    - Dataset loading via ``DataloaderDispatcher`` (``dispatch_batches=True``, default)
      or ``SynchronizedDataLoader`` (``dispatch_batches=False``)
    - Optional PostLocalSGD communication hook for reduced all-reduce frequency
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
        """
        Parameters
        ----------
        args : DDPTrainingArguments or dict
            Training configuration including DDP-specific options (``ddp``,
            ``post_local_sgd``, ``dispatch_batches``). Accepts a dict for
            programmatic construction.
        fused_loss_factory : callable, optional
            Factory for fused logits-loss computation. See ``Trainer`` for details.
        **kwargs
            Forwarded to ``Trainer.__init__``: ``distributed_env``, ``model``,
            ``model_init``, ``train_dataset``, ``eval_dataset``, ``optimizer_factory``,
            ``lr_scheduler_factory``, ``callbacks``, etc.
        """
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

        dispatch_eval = self._dispatch_eval_batches()

        if self.train_dataloader:
            if self.args.dispatch_batches:
                # Use DataloaderDispatcher for centralized batch loading
                self.train_dataloader = DataloaderDispatcher(
                    cast(DataLoader, self.train_dataloader),
                    self.mesh,
                    self.args.device,
                )
            else:
                # Use SynchronizedDataLoader for sharded datasets
                # Ensures all ranks agree on when to stop iterating
                self.train_dataloader = SynchronizedDataLoader(
                    self.train_dataloader,
                    device=self.args.device,
                    process_group=self.ddp_group,
                )

        if self.eval_dataloader:
            if dispatch_eval:
                self.eval_dataloader = DataloaderDispatcher(
                    cast(DataLoader, self.eval_dataloader),
                    self.mesh,
                    self.args.device,
                )
            else:
                self.eval_dataloader = SynchronizedDataLoader(
                    self.eval_dataloader,
                    device=self.args.device,
                    process_group=self.ddp_group,
                )

        # Optimizer is only required for training. When running under
        # ``trainer.evaluate()`` alone, ``do_train`` is False and no optimizer
        # is constructed, which is fine — we just need the DDP wrapping above.
        if self.do_train:
            assert self.optimizer is not None
        if self.do_train and self.args.post_local_sgd.enabled:
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

    def _dispatch_eval_batches(self) -> bool:
        """Resolve the effective eval-time dispatch_batches setting.

        ``dispatch_eval_batches`` overrides ``dispatch_batches`` for eval only.
        ``None`` (the default) means "follow ``dispatch_batches``".
        """
        if self.args.dispatch_eval_batches is not None:
            return self.args.dispatch_eval_batches
        return self.args.dispatch_batches

    @override
    @torch.no_grad()
    def _eval_loop(self) -> Dict[str, float]:
        """
        Evaluation loop for DDP training.

        For dispatch_eval_batches=True or single-GPU, delegates to the base
        Trainer._eval_loop().

        For dispatch_eval_batches=False, bypasses SynchronizedDataLoader's
        MIN-based synchronization to let each rank process ALL its local
        validation data independently. This prevents data loss when token
        packing creates highly uneven shard lengths across ranks (e.g., shard
        lengths [85, 167, 239, 247] would otherwise be truncated to the
        shortest rank's count).
        """
        if self.dist.world_size == 1 or self._dispatch_eval_batches():
            return super()._eval_loop()
        return self._eval_loop_all_shards()

    def _check_eval_shards_nonempty(self, iterator) -> Any:
        """Confirm every rank's eval iterator yields at least one batch.

        Fetches one batch from ``iterator`` on each rank, then runs an
        ``all_reduce(SUM)`` over per-rank "has-first-batch" flags. If any
        rank failed to fetch, raises ``RuntimeError`` *on every rank*
        with the empty-rank set listed in the diagnostic.

        Returns the first batch on success so the caller can prepend it
        back into the eval loop's iterator.

        This is the contract that lets ``_eval_loop_all_shards`` rely on
        every rank having a valid "last seen" batch to reuse as a dummy
        once its real shard exhausts.
        """
        try:
            first_batch = next(iterator)
            local_has_first = 1
        except StopIteration:
            first_batch = None
            local_has_first = 0
        per_rank_has_first = torch.zeros(
            self.dist.world_size,
            dtype=torch.int32,
            device=self.args.device,
        )
        per_rank_has_first[self.dist.rank] = local_has_first
        dist.all_reduce(per_rank_has_first, op=dist.ReduceOp.SUM)
        empty_ranks = [
            r for r, has in enumerate(per_rank_has_first.tolist()) if not has
        ]
        if empty_ranks:
            raise RuntimeError(self._zero_eval_batches_message(empty_ranks=empty_ranks))
        return first_batch

    @torch.no_grad()
    def _eval_loop_all_shards(self) -> Dict[str, float]:
        """
        Eval loop that lets each rank process all its local validation data.

        Two design constraints shape this loop:

        1. **No data loss across uneven shards.** ``SynchronizedDataLoader``'s
           per-step MIN sync would stop every rank as soon as the *shortest*
           shard exhausted, truncating the longer shards (e.g. shard lengths
           ``[85, 167, 239, 247]`` would all drop to 85). For an eval pass we
           want every example evaluated exactly once, not just the first 85.

        2. **Symmetric DDP collectives.** The wrapped DDP module participates
           in collectives (buffer broadcast, parameter-sync hooks) on every
           ``self.model(...)`` call, even under ``torch.no_grad()``. If one
           rank skips a forward call while peers run it, the process group
           deadlocks. So every rank must call ``self.model(...)`` the same
           number of times across the entire loop.

        Strategy:

        - **Pre-flight** (`_check_eval_shards_nonempty`): every rank fetches
          its first batch and gathers per-rank "has-first-batch" flags via
          ``all_reduce(SUM)``. If any rank is empty, every rank raises with
          the empty-rank set and a remediation pointer. This guarantees the
          rest of the loop has a usable per-rank "last-seen" batch shape.
        - **Symmetric loop**: every iteration, every rank calls
          ``self.model(...)``. Ranks with a real next batch use it and
          accumulate loss; ranks that have exhausted their shard reuse
          their last-seen real batch as a *dummy* (same shape, no
          recompile, ignored result) so the DDP collectives stay
          balanced. A per-step ``all_reduce(MAX)`` over a "rank still has
          real data" flag terminates the loop once every shard is
          exhausted.
        - **Aggregation**: ``total_loss`` and ``step_count`` are
          ``all_reduce(SUM)`` across ranks, so only real batches
          contribute to ``eval_loss``.
        """
        assert self.model is not None
        assert self.eval_dataloader is not None
        assert isinstance(self.loss_fn, RescaleLoss)

        # Access the underlying dataloader, bypassing SynchronizedDataLoader
        assert isinstance(self.eval_dataloader, SynchronizedDataLoader)
        raw_dataloader = self.eval_dataloader._dataloader

        with set_train(self.model, False):
            total_loss = torch.zeros(1, device=self.args.device)
            local_real_steps = 0
            iterator = iter(raw_dataloader)

            # Pre-flight: detect empty shards before any model forward call.
            first_batch = self._check_eval_shards_nonempty(iterator)
            assert first_batch is not None  # post-pre-flight invariant

            # Re-prepend the first batch so the loop can yield it on iter 0.
            iterator = itertools.chain([first_batch], iterator)
            # Last real batch on this rank — reused as a shape-matched dummy
            # once the local iterator exhausts. Always populated post-pre-flight.
            last_real_batch = first_batch

            # Reuse a single tensor for the per-step "any rank has real data"
            # synchronization (MAX as logical OR over int32 flags).
            any_real_tensor = torch.zeros(1, dtype=torch.int32, device=self.args.device)

            while True:
                # Try to get the next real batch on this rank. If none, fall
                # back to the dummy (last_real_batch) so the model.forward()
                # call below still runs with a valid shape; mark this iteration
                # as having no real data for this rank so its loss is discarded.
                try:
                    batch = next(iterator)
                    last_real_batch = batch
                    local_has_real = 1
                except StopIteration:
                    batch = last_real_batch
                    local_has_real = 0

                # Respect max_eval_steps on this rank's own real data only.
                if (
                    local_has_real
                    and self.args.max_eval_steps > 0
                    and local_real_steps >= self.args.max_eval_steps
                ):
                    local_has_real = 0
                    batch = last_real_batch

                # Continue while ANY rank still has real data.
                any_real_tensor.fill_(local_has_real)
                dist.all_reduce(any_real_tensor, op=dist.ReduceOp.MAX)
                if any_real_tensor.item() == 0:
                    break

                # Symmetric forward on every rank. Inline the work from
                # _prediction_step but skip _distributed_loss — that would
                # add an extra per-step all_reduce that we don't need here.
                input_dict, labels = self._prepare_batch(batch)
                if self.use_fused_loss:
                    input_dict["return_hidden_states"] = True  # type: ignore[assignment]
                with self.loss_fn.no_rescale(), self.amp_context.autocast():
                    outputs = self.model(**input_dict)
                    logits = logits_from_outputs(outputs)
                    loss = self.loss_fn(logits, labels)

                # Only real batches contribute to the eval metric.
                if local_has_real:
                    total_loss += loss.detach()
                    local_real_steps += 1

                # Dispatch on every synchronized step so rank 0's progress
                # indicator keeps advancing even after its own shard is done.
                self._dispatch_event("on_prediction_step")

            # Aggregate loss across all ranks. Only real-batch contributions
            # are counted; dummy iterations on exhausted ranks were skipped
            # in the accumulator above.
            step_count = torch.tensor(
                local_real_steps, device=self.args.device, dtype=torch.int64
            )
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(step_count, op=dist.ReduceOp.SUM)

            total_steps = step_count.item()
            if total_steps == 0:
                # Defence-in-depth: the pre-flight should have caught this,
                # but if max_eval_steps=0 or some other edge case leaves
                # everyone with zero real steps, surface the diagnostic.
                raise RuntimeError(self._zero_eval_batches_message())
            eval_loss = (total_loss / total_steps).item()

            # Sync dataset state on the underlying StatefulDataLoader
            if isinstance(raw_dataloader, StatefulDataLoader):
                sync_dataset_state_from_dataloader(raw_dataloader)

            metrics = {"eval_loss": eval_loss}
            self._dispatch_event("on_evaluate", metrics=metrics)
            return metrics

    @override
    def _zero_eval_batches_message(
        self, empty_ranks: Optional[List[int]] = None
    ) -> str:
        """Build a diagnostic for the zero-eval-batches failure mode.

        Fires from three places:

        - ``_eval_loop_all_shards`` pre-flight when *some* ranks have an
          empty shard (``empty_ranks`` is the list of those ranks); this
          would otherwise deadlock on asymmetric ``self.model(...)`` calls.
        - ``_eval_loop_all_shards`` post-loop when *every* rank produced
          zero steps (``empty_ranks`` is None or covers every rank).
        - The inherited base ``_eval_loop`` when ``dispatch_eval_batches``
          is effectively True and the rank-0 dispatcher terminated before
          yielding a batch (e.g. it could not assemble one full batch per
          rank from the available eval data); ``empty_ranks`` is None.
        """
        effective_dispatch_eval = self._dispatch_eval_batches()
        partial = (
            empty_ranks is not None and 0 < len(empty_ranks) < self.dist.world_size
        )
        if effective_dispatch_eval:
            header = (
                f"Distributed evaluation produced zero batches across all "
                f"{self.dist.world_size} ranks."
            )
            explanation = (
                "With dispatch_eval_batches=True (effective) rank 0 loads\n"
                "the eval dataloader and broadcasts batches to the other\n"
                "ranks. The dispatcher needs to assemble at least one batch\n"
                "per rank before any rank can step; if rank 0 exhausts the\n"
                "dataloader before that point (small eval split, or\n"
                "dataloader_drop_last=True dropping the only partial batch),\n"
                "every rank reports zero steps."
            )
        elif partial:
            assert empty_ranks is not None
            ranks_str = ", ".join(str(r) for r in empty_ranks)
            plural = "s" if len(empty_ranks) > 1 else ""
            header = (
                f"Distributed evaluation produced zero batches on "
                f"{len(empty_ranks)} of {self.dist.world_size} ranks "
                f"(empty rank{plural}: {ranks_str}).\n"
                "Refusing to enter the eval loop because asymmetric "
                "self.model(...) calls across the DDP process group "
                "would deadlock."
            )
            explanation = (
                "With dispatch_eval_batches=False (effective) each rank\n"
                "evaluates its own shard of the eval dataset; if a shard\n"
                "contains fewer than per_device_eval_batch_size examples\n"
                "and dataloader_drop_last is True, that rank yields no\n"
                "batches. The other ranks would still try to step, and\n"
                "their model.forward() calls participate in DDP collectives\n"
                "that the empty ranks must mirror — so the eval loop would\n"
                "hang. This pre-flight check fails fast on every rank."
            )
        else:
            header = (
                f"Distributed evaluation produced zero batches across all "
                f"{self.dist.world_size} ranks."
            )
            explanation = (
                "With dispatch_eval_batches=False (effective) each rank\n"
                "evaluates its own shard of the eval dataset; if every\n"
                "shard contains fewer than per_device_eval_batch_size\n"
                "examples and dataloader_drop_last is True, every shard\n"
                "is dropped and total_steps collapses to zero."
            )
        return self._format_zero_eval_batches_diagnostic(
            header=header,
            settings=[
                ("dispatch_batches", self.args.dispatch_batches),
                (
                    "dispatch_eval_batches",
                    f"{self.args.dispatch_eval_batches}"
                    f" (effective: {effective_dispatch_eval})",
                ),
                ("dataloader_drop_last", self.args.dataloader_drop_last),
                ("per_device_eval_batch_size", self.args.per_device_eval_batch_size),
                ("max_eval_steps", self.args.max_eval_steps),
                ("world_size", self.dist.world_size),
            ],
            explanation=explanation,
        )

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

        Parameters
        ----------
        tokens : Tensor
            Token count from the current rank.

        Returns
        -------
        Tensor
            Total token count summed across all DDP ranks.
        """
        if self.dist.world_size == 1:
            return super()._distributed_tokens(tokens)

        dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
        return tokens

    @override
    def _distributed_peak_mem(self, local_peak: int) -> list[int]:
        """
        All-gather per-rank peak CUDA memory across DDP ranks.
        """
        if self.dist.world_size == 1:
            return super()._distributed_peak_mem(local_peak)

        value = torch.tensor(
            [int(local_peak)], dtype=torch.long, device=self.args.device
        )
        gathered = [torch.zeros_like(value) for _ in range(self.dist.world_size)]
        dist.all_gather(gathered, value)
        return [int(t.item()) for t in gathered]

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

        Returns
        -------
        list of StateComponent
            State components with REPLICATED sharing patterns for DDP.
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

        Returns
        -------
        SharingPattern
            GLOBAL when ``dispatch_batches=True``, PER_RANK otherwise.
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

        Returns
        -------
        dict
            Mapping of group names to ProcessGroup objects.
            For DDP, returns the data parallel group.
        """
        if self.dist.world_size == 1:
            return super().get_process_groups()

        return {
            "ddp_group": self.ddp_group,
        }
