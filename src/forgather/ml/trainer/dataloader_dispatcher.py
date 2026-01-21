import itertools
from typing import Dict, Iterable, Optional

import torch
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import DataLoader


class DataloaderDispatcher:

    def __init__(
        self,
        dataloader: DataLoader,
        mesh: DeviceMesh,  # (data_parallel, model_parallel)
        device: torch.device,
    ):
        self._dataloader = dataloader
        self._mesh = mesh
        self._device = device

    def __iter__(self):
        dp_group = self._mesh.get_group(0)  # data-parallel group
        dp_rank = self._mesh.get_local_rank(0)
        dp_group_size = len(dist.get_process_group_ranks(dp_group))

        mp_group = self._mesh.get_group(1)  # model-parallel group
        mp_rank = self._mesh.get_local_rank(1)
        mp_group_size = len(dist.get_process_group_ranks(mp_group))

        global_rank = self._mesh.get_rank()

        if global_rank == 0:
            dataloader_iter = iter(self._dataloader)
            for step in itertools.count(start=0, step=1):
                batches = []
                try:
                    for _ in range(dp_group_size):
                        batch = next(dataloader_iter)
                        batch = {
                            k: v.to(self._device, non_blocking=True)
                            for k, v in batch.items()
                        }
                        batches.append(batch)
                except StopIteration:
                    if step == 0:
                        # Send empty meta data to indicate we are done
                        dist.broadcast_object_list(
                            [{}],
                            group=dp_group,
                            group_src=0,
                            device=self._device,
                        )
                    else:
                        dist.broadcast(
                            torch.zeros_like(batch_lengths),
                            group=dp_group,
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
                        group=dp_group,
                        group_src=0,
                        device=self._device,
                    )
                    batch_lengths = torch.empty(
                        dp_group_size - 1,
                        len(meta_data["keys"]),
                        2,
                        device=self._device,
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
                dist.broadcast(batch_lengths, group=dp_group, group_src=0)

                # Send the correct batch to each other rank
                requests = []
                for i, batch in enumerate(batches[1:]):
                    for data in batch.values():
                        req = dist.isend(data, group=dp_group, group_dst=i + 1)
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
                        group=dp_group,
                        group_src=0,
                        device=self._device,
                    )

                    meta_data = objects[0]
                    # Early abort?
                    if len(meta_data.keys()) == 0:
                        return
                    batch_lengths = torch.empty(
                        dp_group_size - 1,
                        len(meta_data["keys"]),
                        2,
                        device=self._device,
                        dtype=torch.int,
                    )

                # Get lengths
                dist.broadcast(batch_lengths, group=dp_group, group_src=0)

                # Out of data?
                if batch_lengths[0][0][0] == 0:
                    return

                # Get the lengths for this rank
                rank_batch_lengths = batch_lengths[dp_rank - 1]

                requests = []
                batch = {}
                for key, dtype, shape in zip(
                    meta_data["keys"], meta_data["dtypes"], rank_batch_lengths
                ):
                    data = torch.empty(
                        shape[0], shape[1], dtype=dtype, device=self._device
                    )
                    req = dist.irecv(data, group=dp_group, group_src=0)
                    requests.append(req)
                    batch[key] = data

                for req in requests:
                    req.wait()

                yield batch

    def __len__(self):
        dp_group = self._mesh.get_group(0)  # data-parallel group
        dp_group_size = len(dist.get_process_group_ranks(dp_group))
        return len(self._dataloader) // dp_group_size

    def __getattr__(self, name):
        """Forward all unknown attributes/methods to the wrapped dataloader."""
        return getattr(self._dataloader, name)
