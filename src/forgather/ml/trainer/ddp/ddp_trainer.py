# A subclass of Trainer, which adds support for the Acclerate library.
import itertools
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, override

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
    split_batches: bool = True


class DDPTrainer(Trainer):
    """
    Modify the base Trainer to use the Accelerate library.
    """

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
            # broadcast_buffers=True,
            # init_sync=True,
            # bucket_cap_mb=None,
            # find_unused_parameters=False,
            # gradient_as_bucket_view=True,
            # static_graph=True,
            # skip_all_reduce_unused_params=True,
        )

        # TODO:
        # - Fused loss function will probably not work.

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
