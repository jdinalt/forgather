import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, override

import torch
from torch import Tensor
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.parallel import DistributedDataParallel as DDP

from ..dataloader_dispatcher import DataloaderDispatcher
from ..trainer import Trainer, TrainingArguments

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DDPTrainingArguments(TrainingArguments):
    dispatch_batches: bool = True
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
            "cuda",
            (self.dist.world_size, 1),
            mesh_dim_names=("data_parallel", "model_parallel"),
        )
        self.ddp_group = self.mesh.get_group(0)  # data-parallel group

    @override
    def _init_device(self):
        self.args.device = self.dist.device

    @override
    def _wrap(
        self,
    ) -> None:
        self.model = DDP(
            self.model,
            device_ids=[self.args.device],
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
        with nullcontext() if self._should_sync_gradients() else self.model.no_sync():
            return super()._forward_backward_step(*args, **kwargs)
