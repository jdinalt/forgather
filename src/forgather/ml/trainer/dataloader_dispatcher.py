import itertools
from typing import Dict, Iterator, List, Optional, cast

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

    Args:
        dataloader: Source dataloader (only coordinator loads from this)
        mesh: Device mesh (1D or 2D)
        device: Target device for tensors
        dp_mesh_dim: Which mesh dimension is data-parallel.
            - 0: dimension 0 is DP (default)
            - 1: dimension 1 is DP (for 2D mesh only)
            - None: pure MP mode, all ranks get same batch

            For 1D mesh: dp_mesh_dim=0 means pure DP, dp_mesh_dim=None means pure MP
            For 2D mesh: dp_mesh_dim specifies DP dimension, other is MP
    """

    def __init__(
        self,
        dataloader: DataLoader,
        mesh: DeviceMesh,
        device: torch.device,
        dp_mesh_dim: Optional[int] = 0,
    ):
        self._dataloader = dataloader
        self._mesh = mesh
        self._device = device
        self._global_rank = mesh.get_rank()

        # Configure groups based on mesh dimensionality and dp_mesh_dim
        ndim = mesh.ndim
        if ndim == 1:
            self._init_1d_mesh(mesh, dp_mesh_dim)
        elif ndim == 2:
            self._init_2d_mesh(mesh, dp_mesh_dim)
        else:
            raise ValueError(f"Only 1D and 2D meshes supported, got {ndim}D")

        # Role identification
        self._is_dp_leader = self._mp_rank == 0
        self._is_global_coordinator = self._global_rank == 0

    def _init_1d_mesh(self, mesh: DeviceMesh, dp_mesh_dim: Optional[int]) -> None:
        """Initialize from 1D mesh."""
        if dp_mesh_dim == 0:
            # Pure DP: all ranks in DP group, no MP
            self._dp_group = mesh.get_group(0)
            self._mp_group = None
            self._dp_size = mesh.size(0)
            self._mp_size = 1
            self._dp_rank = mesh.get_local_rank(0)
            self._mp_rank = 0
        elif dp_mesh_dim is None:
            # Pure MP: no DP, all ranks in MP group
            self._dp_group = None
            self._mp_group = mesh.get_group(0)
            self._dp_size = 1
            self._mp_size = mesh.size(0)
            self._dp_rank = 0
            self._mp_rank = mesh.get_local_rank(0)
        else:
            raise ValueError(
                f"dp_mesh_dim must be 0 or None for 1D mesh, got {dp_mesh_dim}"
            )

    def _init_2d_mesh(self, mesh: DeviceMesh, dp_mesh_dim: Optional[int]) -> None:
        """Initialize from 2D mesh."""
        if dp_mesh_dim == 0:
            # Dimension 0 is DP, dimension 1 is MP
            self._dp_group = mesh.get_group(0)
            self._mp_group = mesh.get_group(1)
            self._dp_size = mesh.size(0)
            self._mp_size = mesh.size(1)
            self._dp_rank = mesh.get_local_rank(0)
            self._mp_rank = mesh.get_local_rank(1)
        elif dp_mesh_dim == 1:
            # Dimension 1 is DP, dimension 0 is MP
            self._dp_group = mesh.get_group(1)
            self._mp_group = mesh.get_group(0)
            self._dp_size = mesh.size(1)
            self._mp_size = mesh.size(0)
            self._dp_rank = mesh.get_local_rank(1)
            self._mp_rank = mesh.get_local_rank(0)
        elif dp_mesh_dim is None:
            # Pure MP over entire 2D mesh - not yet supported
            # Would need a flattened process group spanning all ranks
            raise NotImplementedError(
                "Pure MP over 2D mesh (dp_mesh_dim=None) not yet supported. "
                "Use a 1D mesh with dp_mesh_dim=None instead."
            )
        else:
            raise ValueError(
                f"dp_mesh_dim must be 0, 1, or None for 2D mesh, got {dp_mesh_dim}"
            )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self._mp_size == 1:
            yield from self._iter_pure_dp()
        elif self._dp_size == 1:
            yield from self._iter_pure_mp()
        else:
            yield from self._iter_hybrid()

    # -------------------------------------------------------------------------
    # Group-Parameterized Communication Helpers
    # -------------------------------------------------------------------------

    def _broadcast_metadata(
        self, batch: Dict[str, torch.Tensor], group: dist.ProcessGroup
    ) -> Dict:
        """Broadcast metadata (keys and dtypes) to a process group."""
        meta_data = {
            "keys": list(batch.keys()),
            "dtypes": [v.dtype for v in batch.values()],
        }
        dist.broadcast_object_list(
            [meta_data], group=group, group_src=0, device=self._device
        )
        return meta_data

    def _receive_metadata(self, group: dist.ProcessGroup) -> Dict:
        """Receive metadata from group source."""
        objects: List[Optional[object]] = [None]
        dist.broadcast_object_list(
            objects, group=group, group_src=0, device=self._device
        )
        result = objects[0]
        assert result is not None
        return cast(Dict, result)

    def _broadcast_metadata_from_dict(
        self, meta_data: Dict, group: dist.ProcessGroup
    ) -> None:
        """Broadcast pre-existing metadata dict to a group."""
        dist.broadcast_object_list(
            [meta_data], group=group, group_src=0, device=self._device
        )

    def _broadcast_shapes(
        self, batch: Dict[str, torch.Tensor], group: dist.ProcessGroup
    ) -> None:
        """Broadcast tensor shapes to a group."""
        shapes = torch.tensor(
            [[v.shape[0], v.shape[1]] for v in batch.values()],
            device=self._device,
            dtype=torch.int,
        )
        dist.broadcast(shapes, group=group, group_src=0)

    def _receive_shapes(self, num_keys: int, group: dist.ProcessGroup) -> torch.Tensor:
        """Receive tensor shapes from group source."""
        shapes = torch.empty(num_keys, 2, device=self._device, dtype=torch.int)
        dist.broadcast(shapes, group=group, group_src=0)
        return shapes

    def _broadcast_batch(
        self, batch: Dict[str, torch.Tensor], group: dist.ProcessGroup
    ) -> None:
        """Broadcast batch tensors to a group."""
        for data in batch.values():
            dist.broadcast(data, group=group, group_src=0)

    def _receive_batch(
        self, meta_data: Dict, shapes: torch.Tensor, group: dist.ProcessGroup
    ) -> Dict[str, torch.Tensor]:
        """Receive batch tensors via broadcast from group source."""
        batch = {}
        for key, dtype, shape in zip(meta_data["keys"], meta_data["dtypes"], shapes):
            data = torch.empty(
                int(shape[0].item()),
                int(shape[1].item()),
                dtype=dtype,
                device=self._device,
            )
            dist.broadcast(data, group=group, group_src=0)
            batch[key] = data
        return batch

    def _signal_end_metadata(self, group: dist.ProcessGroup) -> None:
        """Signal end by sending empty metadata."""
        dist.broadcast_object_list([{}], group=group, group_src=0, device=self._device)

    def _signal_end_shapes(self, num_keys: int, group: dist.ProcessGroup) -> None:
        """Signal end by sending zero shapes."""
        shapes = torch.zeros(num_keys, 2, device=self._device, dtype=torch.int)
        dist.broadcast(shapes, group=group, group_src=0)

    # -------------------------------------------------------------------------
    # DP Point-to-Point Communication Helpers
    # -------------------------------------------------------------------------

    def _create_batch_lengths_buffer(self, num_keys: int) -> torch.Tensor:
        """Create buffer for batch lengths table."""
        return torch.empty(
            self._dp_size - 1, num_keys, 2, device=self._device, dtype=torch.int
        )

    def _populate_batch_lengths(
        self, batches: List[Dict[str, torch.Tensor]], batch_lengths: torch.Tensor
    ) -> None:
        """Populate lengths table from batches (excluding first batch)."""
        for i, batch in enumerate(batches[1:]):
            for j, v in enumerate(batch.values()):
                assert len(v.shape) == 2, f"Expected 2D tensors, found shape={v.shape}"
                batch_lengths[i, j, 0] = v.shape[0]
                batch_lengths[i, j, 1] = v.shape[1]

    def _send_batches_to_dp_ranks(self, batches: List[Dict[str, torch.Tensor]]) -> None:
        """Send batches to other DP ranks via batched isend (excluding first batch)."""
        p2p_ops = []
        for i, batch in enumerate(batches[1:]):
            for data in batch.values():
                op = dist.P2POp(dist.isend, data, i + 1, group=self._dp_group)
                p2p_ops.append(op)
        if p2p_ops:
            reqs = dist.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

    def _receive_batch_from_coordinator(
        self, meta_data: Dict, shapes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Receive batch from coordinator via batched irecv."""
        p2p_ops = []
        batch = {}
        for key, dtype, shape in zip(meta_data["keys"], meta_data["dtypes"], shapes):
            data = torch.empty(
                int(shape[0].item()),
                int(shape[1].item()),
                dtype=dtype,
                device=self._device,
            )
            op = dist.P2POp(dist.irecv, data, 0, group=self._dp_group)
            p2p_ops.append(op)
            batch[key] = data
        if p2p_ops:
            reqs = dist.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()
        return batch

    # -------------------------------------------------------------------------
    # Pure DP Mode
    # -------------------------------------------------------------------------

    def _iter_pure_dp(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Original DP-only iteration. Each rank gets a different batch."""
        if self._is_global_coordinator:
            yield from self._dp_coordinator_iter(broadcast_to_mp=False)
        else:
            yield from self._dp_receiver_iter(broadcast_to_mp=False)

    def _dp_coordinator_iter(
        self, broadcast_to_mp: bool
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Coordinator loads batches and dispatches to other DP ranks."""
        dp_group = self._dp_group
        assert dp_group is not None
        mp_group = self._mp_group  # May be None when broadcast_to_mp=False
        dataloader_iter = iter(self._dataloader)
        batch_lengths: Optional[torch.Tensor] = None
        meta_data: Optional[Dict] = None

        for step in itertools.count(start=0, step=1):
            batches = self._load_dp_batches(dataloader_iter)
            if batches is None:
                self._signal_dp_end(step, batch_lengths)
                if broadcast_to_mp:
                    self._signal_mp_end(step - 1, meta_data)
                return

            if step == 0:
                meta_data = self._broadcast_metadata(batches[0], dp_group)
                if broadcast_to_mp:
                    assert mp_group is not None
                    self._broadcast_metadata(batches[0], mp_group)
                batch_lengths = self._create_batch_lengths_buffer(
                    len(meta_data["keys"])
                )

            assert batch_lengths is not None
            self._populate_batch_lengths(batches, batch_lengths)
            dist.broadcast(batch_lengths, group=dp_group, group_src=0)
            self._send_batches_to_dp_ranks(batches)

            if broadcast_to_mp:
                assert mp_group is not None
                self._broadcast_shapes(batches[0], mp_group)
                self._broadcast_batch(batches[0], mp_group)

            yield batches[0]

    def _dp_receiver_iter(
        self, broadcast_to_mp: bool
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Non-coordinator DP rank receives batches from coordinator."""
        dp_group = self._dp_group
        assert dp_group is not None
        mp_group = self._mp_group  # May be None when broadcast_to_mp=False
        meta_data: Optional[Dict] = None
        batch_lengths: Optional[torch.Tensor] = None

        for step in itertools.count(start=0, step=1):
            if step == 0:
                meta_data = self._receive_metadata(dp_group)
                if not meta_data.get("keys"):
                    if broadcast_to_mp:
                        assert mp_group is not None
                        self._signal_end_metadata(mp_group)
                    return
                if broadcast_to_mp:
                    assert mp_group is not None
                    self._broadcast_metadata_from_dict(meta_data, mp_group)
                batch_lengths = self._create_batch_lengths_buffer(
                    len(meta_data["keys"])
                )

            assert batch_lengths is not None
            dist.broadcast(batch_lengths, group=dp_group, group_src=0)

            assert meta_data is not None
            if batch_lengths[0][0][0] == 0:
                if broadcast_to_mp:
                    assert mp_group is not None
                    self._signal_end_shapes(len(meta_data["keys"]), mp_group)
                return

            rank_shapes = batch_lengths[self._dp_rank - 1]
            batch = self._receive_batch_from_coordinator(meta_data, rank_shapes)

            if broadcast_to_mp:
                assert mp_group is not None
                self._broadcast_shapes(batch, mp_group)
                self._broadcast_batch(batch, mp_group)

            yield batch

    def _load_dp_batches(
        self, dataloader_iter: Iterator
    ) -> Optional[List[Dict[str, torch.Tensor]]]:
        """Load dp_size batches from dataloader. Returns None on exhaustion."""
        batches = []
        try:
            for _ in range(self._dp_size):
                batch = next(dataloader_iter)
                batch = {
                    k: v.to(self._device, non_blocking=True) for k, v in batch.items()
                }
                batches.append(batch)
            return batches
        except StopIteration:
            return None

    def _signal_dp_end(self, step: int, batch_lengths: Optional[torch.Tensor]) -> None:
        """Signal end of data to other DP ranks."""
        dp_group = self._dp_group
        assert dp_group is not None
        if step == 0:
            self._signal_end_metadata(dp_group)
        else:
            assert batch_lengths is not None
            dist.broadcast(torch.zeros_like(batch_lengths), group=dp_group, group_src=0)

    def _signal_mp_end(self, step: int, meta_data: Optional[Dict]) -> None:
        """Signal end of data to MP followers."""
        mp_group = self._mp_group
        assert mp_group is not None
        if step < 0 or meta_data is None:
            self._signal_end_metadata(mp_group)
        else:
            self._signal_end_shapes(len(meta_data["keys"]), mp_group)

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
        mp_group = self._mp_group
        assert mp_group is not None
        meta_data: Optional[Dict] = None
        step = -1
        for step, batch in enumerate(self._dataloader):
            batch = {k: v.to(self._device, non_blocking=True) for k, v in batch.items()}

            if step == 0:
                meta_data = self._broadcast_metadata(batch, mp_group)

            self._broadcast_shapes(batch, mp_group)
            self._broadcast_batch(batch, mp_group)
            yield batch

        self._signal_mp_end(step if meta_data else -1, meta_data)

    def _mp_follower_iter(self) -> Iterator[Dict[str, torch.Tensor]]:
        """MP follower receives batches via broadcast from DP leader."""
        mp_group = self._mp_group
        assert mp_group is not None
        meta_data: Optional[Dict] = None

        for step in itertools.count(start=0, step=1):
            if step == 0:
                meta_data = self._receive_metadata(mp_group)
                if not meta_data.get("keys"):
                    return

            assert meta_data is not None
            shapes = self._receive_shapes(len(meta_data["keys"]), mp_group)
            if shapes[0][0] == 0:
                return

            batch = self._receive_batch(meta_data, shapes, mp_group)
            yield batch

    # -------------------------------------------------------------------------
    # Hybrid Mode (DP + MP)
    # -------------------------------------------------------------------------

    def _iter_hybrid(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Hybrid parallelism: Different batches across DP groups,
        same batch within each MP group.
        """
        if self._is_global_coordinator:
            yield from self._dp_coordinator_iter(broadcast_to_mp=True)
        elif self._is_dp_leader:
            yield from self._dp_receiver_iter(broadcast_to_mp=True)
        else:
            yield from self._mp_follower_iter()

    # -------------------------------------------------------------------------
    # DataLoader Interface
    # -------------------------------------------------------------------------

    def __len__(self):
        return len(self._dataloader) // self._dp_size

    def __getattr__(self, name):
        """Forward all unknown attributes/methods to the wrapped dataloader."""
        return getattr(self._dataloader, name)
