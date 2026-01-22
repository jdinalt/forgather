import itertools
from typing import Dict, Iterator, Optional

import torch
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import DataLoader


class DataloaderDispatcher:
    """
    Dispatches batches from a dataloader to distributed ranks.

    Supports three parallelism modes based on mesh dimensions:
    - Pure DP (mp_size=1): Each rank gets a different batch
    - Pure MP (dp_size=1): All ranks get the same batch via broadcast
    - Hybrid (dp_size>1, mp_size>1): Different batches across DP groups,
      same batch within each MP group

    Communication pattern for hybrid mode:
    - Global coordinator (rank 0) loads and dispatches to DP leaders
    - DP leaders broadcast to their MP group followers
    """

    def __init__(
        self,
        dataloader: DataLoader,
        mesh: DeviceMesh,  # (data_parallel, model_parallel)
        device: torch.device,
    ):
        self._dataloader = dataloader
        self._mesh = mesh
        self._device = device

        # Topology properties
        self._dp_group = mesh.get_group(0)
        self._mp_group = mesh.get_group(1)
        self._dp_rank = mesh.get_local_rank(0)
        self._mp_rank = mesh.get_local_rank(1)
        self._dp_size = len(dist.get_process_group_ranks(self._dp_group))
        self._mp_size = len(dist.get_process_group_ranks(self._mp_group))
        self._global_rank = mesh.get_rank()

        # Role identification
        self._is_dp_leader = self._mp_rank == 0
        self._is_global_coordinator = self._global_rank == 0

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self._mp_size == 1:
            yield from self._iter_pure_dp()
        elif self._dp_size == 1:
            yield from self._iter_pure_mp()
        else:
            yield from self._iter_hybrid()

    # -------------------------------------------------------------------------
    # Pure DP Mode (original behavior)
    # -------------------------------------------------------------------------

    def _iter_pure_dp(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Original DP-only iteration. Each rank gets a different batch."""
        if self._is_global_coordinator:
            yield from self._dp_coordinator_iter()
        else:
            yield from self._dp_receiver_iter()

    def _dp_coordinator_iter(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Coordinator loads batches and dispatches to other DP ranks."""
        dataloader_iter = iter(self._dataloader)
        batch_lengths: Optional[torch.Tensor] = None
        meta_data: Optional[Dict] = None

        for step in itertools.count(start=0, step=1):
            batches = []
            try:
                for _ in range(self._dp_size):
                    batch = next(dataloader_iter)
                    batch = {
                        k: v.to(self._device, non_blocking=True)
                        for k, v in batch.items()
                    }
                    batches.append(batch)
            except StopIteration:
                self._dp_signal_end(step, batch_lengths)
                return

            if step == 0:
                meta_data = self._broadcast_metadata(batches[0])
                batch_lengths = torch.empty(
                    self._dp_size - 1,
                    len(meta_data["keys"]),
                    2,
                    device=self._device,
                    dtype=torch.int,
                )

            # Populate lengths table
            for i, batch in enumerate(batches[1:]):
                for j, v in enumerate(batch.values()):
                    assert (
                        len(v.shape) == 2
                    ), f"Expected 2D tensors, found {meta_data['keys'][j]} shape={v.shape}"
                    batch_lengths[i, j, 0] = v.shape[0]
                    batch_lengths[i, j, 1] = v.shape[1]

            # Distribute batch lengths
            dist.broadcast(batch_lengths, group=self._dp_group, group_src=0)

            # Send batches to other ranks via isend
            requests = []
            for i, batch in enumerate(batches[1:]):
                for data in batch.values():
                    req = dist.isend(data, group=self._dp_group, group_dst=i + 1)
                    requests.append(req)

            for req in requests:
                req.wait()

            yield batches[0]

    def _dp_receiver_iter(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Non-coordinator DP rank receives batches from coordinator."""
        meta_data: Optional[Dict] = None
        batch_lengths: Optional[torch.Tensor] = None

        for step in itertools.count(start=0, step=1):
            if step == 0:
                meta_data = self._receive_metadata()
                if not meta_data.get("keys"):
                    return
                batch_lengths = torch.empty(
                    self._dp_size - 1,
                    len(meta_data["keys"]),
                    2,
                    device=self._device,
                    dtype=torch.int,
                )

            dist.broadcast(batch_lengths, group=self._dp_group, group_src=0)

            if batch_lengths[0][0][0] == 0:
                return

            rank_batch_lengths = batch_lengths[self._dp_rank - 1]

            requests = []
            batch = {}
            for key, dtype, shape in zip(
                meta_data["keys"], meta_data["dtypes"], rank_batch_lengths
            ):
                data = torch.empty(
                    shape[0], shape[1], dtype=dtype, device=self._device
                )
                req = dist.irecv(data, group=self._dp_group, group_src=0)
                requests.append(req)
                batch[key] = data

            for req in requests:
                req.wait()

            yield batch

    def _dp_signal_end(
        self, step: int, batch_lengths: Optional[torch.Tensor]
    ) -> None:
        """Signal end of data to other DP ranks."""
        if step == 0:
            dist.broadcast_object_list(
                [{}],
                group=self._dp_group,
                group_src=0,
                device=self._device,
            )
        else:
            dist.broadcast(
                torch.zeros_like(batch_lengths),
                group=self._dp_group,
                group_src=0,
            )

    # -------------------------------------------------------------------------
    # Pure MP Mode
    # -------------------------------------------------------------------------

    def _iter_pure_mp(self) -> Iterator[Dict[str, torch.Tensor]]:
        """All ranks get the same batch. Only DP leader loads data."""
        if self._is_dp_leader:
            yield from self._mp_leader_iter()
        else:
            yield from self._mp_follower_iter()

    def _mp_leader_iter(self) -> Iterator[Dict[str, torch.Tensor]]:
        """DP leader loads data and broadcasts to MP group."""
        last_step = -1
        num_keys = 0
        for step, batch in enumerate(self._dataloader):
            batch = {k: v.to(self._device, non_blocking=True) for k, v in batch.items()}

            if step == 0:
                self._mp_broadcast_metadata(batch)
                num_keys = len(batch)

            self._mp_broadcast_shapes(batch)
            self._mp_broadcast_batch(batch)
            last_step = step
            yield batch

        self._mp_signal_end(last_step, num_keys)

    def _mp_follower_iter(self) -> Iterator[Dict[str, torch.Tensor]]:
        """MP follower receives batches via broadcast from DP leader."""
        meta_data: Optional[Dict] = None

        for step in itertools.count(start=0, step=1):
            if step == 0:
                meta_data = self._mp_receive_metadata()
                if not meta_data.get("keys"):
                    return

            shapes = self._mp_receive_shapes(len(meta_data["keys"]))

            if shapes[0][0] == 0:
                return

            batch = self._mp_receive_batch(meta_data, shapes)
            yield batch

    def _mp_broadcast_metadata(self, batch: Dict[str, torch.Tensor]) -> Dict:
        """Broadcast metadata to MP group."""
        meta_data = {
            "keys": list(batch.keys()),
            "dtypes": [v.dtype for v in batch.values()],
        }
        dist.broadcast_object_list(
            [meta_data],
            group=self._mp_group,
            group_src=0,
            device=self._device,
        )
        return meta_data

    def _mp_receive_metadata(self) -> Dict:
        """Receive metadata from MP leader."""
        objects = [None]
        dist.broadcast_object_list(
            objects,
            group=self._mp_group,
            group_src=0,
            device=self._device,
        )
        return objects[0]

    def _mp_broadcast_shapes(self, batch: Dict[str, torch.Tensor]) -> None:
        """Broadcast tensor shapes to MP group."""
        shapes = torch.tensor(
            [[v.shape[0], v.shape[1]] for v in batch.values()],
            device=self._device,
            dtype=torch.int,
        )
        dist.broadcast(shapes, group=self._mp_group, group_src=0)

    def _mp_receive_shapes(self, num_keys: int) -> torch.Tensor:
        """Receive tensor shapes from MP leader."""
        shapes = torch.empty(num_keys, 2, device=self._device, dtype=torch.int)
        dist.broadcast(shapes, group=self._mp_group, group_src=0)
        return shapes

    def _mp_broadcast_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Broadcast batch tensors to MP group."""
        for data in batch.values():
            dist.broadcast(data, group=self._mp_group, group_src=0)

    def _mp_receive_batch(
        self, meta_data: Dict, shapes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Receive batch tensors via broadcast from MP leader."""
        batch = {}
        for key, dtype, shape in zip(meta_data["keys"], meta_data["dtypes"], shapes):
            data = torch.empty(shape[0], shape[1], dtype=dtype, device=self._device)
            dist.broadcast(data, group=self._mp_group, group_src=0)
            batch[key] = data
        return batch

    def _mp_signal_end(self, step: int, num_keys: int = 1) -> None:
        """Signal end of data to MP followers."""
        if step < 0:
            # No data was ever sent, send empty metadata
            dist.broadcast_object_list(
                [{}],
                group=self._mp_group,
                group_src=0,
                device=self._device,
            )
        else:
            # Send zero shapes to signal end (must match expected num_keys)
            shapes = torch.zeros(num_keys, 2, device=self._device, dtype=torch.int)
            dist.broadcast(shapes, group=self._mp_group, group_src=0)

    # -------------------------------------------------------------------------
    # Hybrid Mode (DP + MP)
    # -------------------------------------------------------------------------

    def _iter_hybrid(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Hybrid parallelism: Different batches across DP groups,
        same batch within each MP group.

        Roles:
        - Global coordinator (dp_rank=0, mp_rank=0): Load and dispatch to DP leaders
        - DP leader (mp_rank=0): Receive from coordinator, broadcast to MP followers
        - MP follower (mp_rank>0): Receive via broadcast from DP leader
        """
        if self._is_global_coordinator:
            yield from self._hybrid_coordinator_iter()
        elif self._is_dp_leader:
            yield from self._hybrid_dp_leader_iter()
        else:
            yield from self._hybrid_mp_follower_iter()

    def _hybrid_coordinator_iter(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Global coordinator: Load batches, send to other DP leaders,
        broadcast own batch to local MP group.
        """
        dataloader_iter = iter(self._dataloader)
        batch_lengths: Optional[torch.Tensor] = None
        meta_data: Optional[Dict] = None

        for step in itertools.count(start=0, step=1):
            batches = []
            try:
                for _ in range(self._dp_size):
                    batch = next(dataloader_iter)
                    batch = {
                        k: v.to(self._device, non_blocking=True)
                        for k, v in batch.items()
                    }
                    batches.append(batch)
            except StopIteration:
                self._hybrid_signal_end_from_coordinator(step, batch_lengths, meta_data)
                return

            if step == 0:
                # Broadcast metadata to both DP and MP groups
                meta_data = self._broadcast_metadata(batches[0])
                self._mp_broadcast_metadata(batches[0])
                batch_lengths = torch.empty(
                    self._dp_size - 1,
                    len(meta_data["keys"]),
                    2,
                    device=self._device,
                    dtype=torch.int,
                )

            # Populate lengths table for DP distribution
            for i, batch in enumerate(batches[1:]):
                for j, v in enumerate(batch.values()):
                    assert (
                        len(v.shape) == 2
                    ), f"Expected 2D tensors, found {meta_data['keys'][j]} shape={v.shape}"
                    batch_lengths[i, j, 0] = v.shape[0]
                    batch_lengths[i, j, 1] = v.shape[1]

            # Distribute batch lengths to other DP leaders
            dist.broadcast(batch_lengths, group=self._dp_group, group_src=0)

            # Send batches to other DP leaders via isend
            requests = []
            for i, batch in enumerate(batches[1:]):
                for data in batch.values():
                    req = dist.isend(data, group=self._dp_group, group_dst=i + 1)
                    requests.append(req)

            for req in requests:
                req.wait()

            # Broadcast own batch to local MP group
            self._mp_broadcast_shapes(batches[0])
            self._mp_broadcast_batch(batches[0])

            yield batches[0]

    def _hybrid_dp_leader_iter(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Non-coordinator DP leader: Receive from coordinator,
        broadcast to local MP group.
        """
        meta_data: Optional[Dict] = None
        batch_lengths: Optional[torch.Tensor] = None

        for step in itertools.count(start=0, step=1):
            if step == 0:
                # Receive DP metadata from coordinator
                meta_data = self._receive_metadata()
                if not meta_data.get("keys"):
                    # Signal end to local MP group
                    self._mp_signal_end(-1)
                    return
                # Broadcast metadata to local MP group
                self._mp_broadcast_metadata_from_dict(meta_data)
                batch_lengths = torch.empty(
                    self._dp_size - 1,
                    len(meta_data["keys"]),
                    2,
                    device=self._device,
                    dtype=torch.int,
                )

            dist.broadcast(batch_lengths, group=self._dp_group, group_src=0)

            if batch_lengths[0][0][0] == 0:
                self._mp_signal_end(step, len(meta_data["keys"]))
                return

            rank_batch_lengths = batch_lengths[self._dp_rank - 1]

            # Receive batch from coordinator
            requests = []
            batch = {}
            for key, dtype, shape in zip(
                meta_data["keys"], meta_data["dtypes"], rank_batch_lengths
            ):
                data = torch.empty(
                    shape[0], shape[1], dtype=dtype, device=self._device
                )
                req = dist.irecv(data, group=self._dp_group, group_src=0)
                requests.append(req)
                batch[key] = data

            for req in requests:
                req.wait()

            # Broadcast to local MP group
            self._mp_broadcast_shapes(batch)
            self._mp_broadcast_batch(batch)

            yield batch

    def _hybrid_mp_follower_iter(self) -> Iterator[Dict[str, torch.Tensor]]:
        """MP follower: Receive batches via broadcast from local DP leader."""
        meta_data: Optional[Dict] = None

        for step in itertools.count(start=0, step=1):
            if step == 0:
                meta_data = self._mp_receive_metadata()
                if not meta_data.get("keys"):
                    return

            shapes = self._mp_receive_shapes(len(meta_data["keys"]))

            if shapes[0][0] == 0:
                return

            batch = self._mp_receive_batch(meta_data, shapes)
            yield batch

    def _mp_broadcast_metadata_from_dict(self, meta_data: Dict) -> None:
        """Broadcast pre-existing metadata dict to MP group."""
        dist.broadcast_object_list(
            [meta_data],
            group=self._mp_group,
            group_src=0,
            device=self._device,
        )

    def _hybrid_signal_end_from_coordinator(
        self, step: int, batch_lengths: Optional[torch.Tensor], meta_data: Optional[Dict]
    ) -> None:
        """Signal end of data from coordinator to both DP and MP groups."""
        # Signal to other DP leaders
        self._dp_signal_end(step, batch_lengths)
        # Signal to local MP group
        num_keys = len(meta_data["keys"]) if meta_data else 1
        self._mp_signal_end(step - 1 if step > 0 else -1, num_keys)

    # -------------------------------------------------------------------------
    # Shared Helpers
    # -------------------------------------------------------------------------

    def _broadcast_metadata(self, batch: Dict[str, torch.Tensor]) -> Dict:
        """Broadcast metadata to DP group."""
        meta_data = {
            "keys": list(batch.keys()),
            "dtypes": [v.dtype for v in batch.values()],
        }
        dist.broadcast_object_list(
            [meta_data],
            group=self._dp_group,
            group_src=0,
            device=self._device,
        )
        return meta_data

    def _receive_metadata(self) -> Dict:
        """Receive metadata from coordinator via DP group."""
        objects = [None]
        dist.broadcast_object_list(
            objects,
            group=self._dp_group,
            group_src=0,
            device=self._device,
        )
        return objects[0]

    def __len__(self):
        return len(self._dataloader) // self._dp_size

    def __getattr__(self, name):
        """Forward all unknown attributes/methods to the wrapped dataloader."""
        return getattr(self._dataloader, name)
