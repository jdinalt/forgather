# A subclass of Trainer, which adds support for the Acclerate library.
import itertools
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, override

import torch
from torch import Tensor
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
        self.ddp_group = dist.new_group()

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
    def _dataloader_iter(
        self, dataloader: Iterable[Dict[str, Tensor]]
    ) -> Iterable[Dict[str, Tensor]]:
        """
        Broadcast batches from rank 0 to all other ranks
        """
        if not self.args.split_batches:
            for batch in dataloader:
                yield batch
            return

        world_size = self.dist.world_size
        if self.dist.rank == 0:
            dataloader_iter = iter(dataloader)
            for step in itertools.count(start=0, step=1):
                batches = []
                try:
                    for _ in range(world_size):
                        batch = next(dataloader_iter)
                        batch = {
                            k: v.to(self.args.device, non_blocking=True)
                            for k, v in batch.items()
                        }
                        batches.append(batch)
                except StopIteration:
                    if step == 0:
                        # Send empty meta data to indicate we are done
                        dist.broadcast_object_list(
                            [{}],
                            group=self.ddp_group,
                            group_src=0,
                            device=self.dist.device,
                        )
                    else:
                        dist.broadcast(
                            torch.zeros_like(batch_lengths),
                            group=self.ddp_group,
                            group_src=0,
                        )
                    return

                if step == 0:
                    # Send meta-data on the first step

                    meta_data = {
                        "keys": [k for k in batches[0].keys()],
                        "dtypes": [v.dtype for v in batches[0].values()],
                    }
                    objects = [meta_data]

                    dist.broadcast_object_list(
                        objects,
                        group=self.ddp_group,
                        group_src=0,
                        device=self.dist.device,
                    )
                    batch_lengths = torch.empty(
                        world_size - 1,
                        len(meta_data["keys"]),
                        2,
                        device=self.dist.device,
                        dtype=torch.int,
                    )

                # Populate lengths table with tensor lengths
                for i, batch in enumerate(batches[1:]):
                    for j, v in enumerate(batch.values()):
                        assert (
                            len(v.shape) == 2
                        ), "Expected two dimensional tensors, but found {meta_data['keys'][j]} shape = {v.shape}"
                        batch_lengths[i, j, 0] = v.shape[0]
                        batch_lengths[i, j, 1] = v.shape[1]

                # Distribute batch lengths
                dist.broadcast(batch_lengths, group=self.ddp_group, group_src=0)

                # Send the correct batch to each other rank
                requests = []
                for i, batch in enumerate(batches[1:]):
                    for data in batch.values():
                        req = dist.isend(data, group=self.ddp_group, group_dst=i + 1)
                        requests.append(req)

                # Wait for all data to be sent
                for req in requests:
                    req.wait()

                # Finally, return our own data
                yield batches[0]

        else:
            for step in itertools.count(start=0, step=1):
                if step == 0:
                    objects = [None]
                    dist.broadcast_object_list(
                        objects,
                        group=self.ddp_group,
                        group_src=0,
                        device=self.dist.device,
                    )

                    meta_data = objects[0]
                    # Early abort?
                    if len(meta_data.keys()) == 0:
                        return
                    batch_lengths = torch.empty(
                        world_size - 1,
                        len(meta_data["keys"]),
                        2,
                        device=self.dist.device,
                        dtype=torch.int,
                    )

                # Get lengths
                dist.broadcast(batch_lengths, group=self.ddp_group, group_src=0)

                # Out of data?
                if batch_lengths[0][0][0] == 0:
                    return

                # Get the lengths for this rank
                rank_batch_lengths = batch_lengths[self.dist.rank - 1]

                requests = []
                batch = {}
                for key, dtype, shape in zip(
                    meta_data["keys"], meta_data["dtypes"], rank_batch_lengths
                ):
                    data = torch.empty(
                        shape[0], shape[1], dtype=dtype, device=self.dist.device
                    )
                    req = dist.irecv(data, group=self.ddp_group, group_src=0)
                    requests.append(req)
                    batch[key] = data

                for req in requests:
                    req.wait()

                yield batch
