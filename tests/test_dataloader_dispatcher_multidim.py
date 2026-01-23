#!/usr/bin/env python3
"""
Multi-dimensional parallelism test for DataloaderDispatcher.

Tests three modes:
- Pure DP (mp_size=1): Each rank gets a different batch
- Pure MP (dp_size=1): All ranks get the same batch via broadcast
- Hybrid (dp_size>1, mp_size>1): Different batches across DP groups,
  same batch within each MP group

Supports both 1D and 2D mesh configurations.

Usage:
    # 1D Pure DP - 4 ranks
    torchrun --nproc_per_node 4 test_dataloader_dispatcher_multidim.py --dp 4

    # 1D Pure MP - 4 ranks
    torchrun --nproc_per_node 4 test_dataloader_dispatcher_multidim.py --mp 4

    # 2D Hybrid - 6 ranks: 2 DP groups x 3 MP each
    torchrun --nproc_per_node 6 test_dataloader_dispatcher_multidim.py --dp 2 --mp 3

    # 2D with swapped dimensions (dp_mesh_dim=1)
    torchrun --nproc_per_node 6 test_dataloader_dispatcher_multidim.py --dp 2 --mp 3 --dp-dim 1
"""
import argparse
import os
import sys
from argparse import RawTextHelpFormatter
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader, Dataset

from forgather.ml.trainer.dataloader_dispatcher import DataloaderDispatcher


class DummyDataset(Dataset):
    """Dataset that produces unique batches identifiable by batch index."""

    def __init__(self, num_batches: int, seq_len: int = 8, batch_size: int = 2):
        self.num_batches = num_batches
        self.seq_len = seq_len
        self.batch_size = batch_size

    def __len__(self):
        return self.num_batches * self.batch_size

    def __getitem__(self, idx):
        batch_idx = idx // self.batch_size
        sample_in_batch = idx % self.batch_size
        return {
            "input_ids": torch.full(
                (self.seq_len,), batch_idx * 1000 + sample_in_batch, dtype=torch.long
            ),
            "labels": torch.full(
                (self.seq_len,),
                batch_idx * 1000 + sample_in_batch + 1,
                dtype=torch.long,
            ),
        }


def collate_fn(batch):
    """Stack samples into a batch tensor."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]).reshape(
            len(batch), -1
        ),
        "labels": torch.stack([b["labels"] for b in batch]).reshape(len(batch), -1),
    }


def init_distributed(backend: str):
    """Initialize distributed process group."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend=backend)

    if backend == "gloo":
        device = torch.device("cpu")
        device_type = "cpu"
    else:
        assert torch.accelerator.is_available()

        torch.accelerator.set_device_index(local_rank)
        acc = torch.accelerator.current_accelerator()
        device_type = acc.type
        idx = torch.accelerator.current_device_index()
        device = torch.device(f"{device_type}:{idx}")

    return rank, local_rank, world_size, device, device_type


def create_mesh_and_dispatcher(
    dataloader: DataLoader,
    device: torch.device,
    device_type: str,
    dp_size: int,
    mp_size: int,
    dp_mesh_dim: Optional[int],
):
    """Create mesh and dispatcher based on configuration."""
    if mp_size == 1 and dp_size > 1:
        # 1D Pure DP
        mesh = init_device_mesh(
            device_type,
            (dp_size,),
            mesh_dim_names=("data_parallel",),
        )
        dispatcher = DataloaderDispatcher(dataloader, mesh, device, dp_mesh_dim=0)
        mode = "1D Pure DP"
    elif dp_size == 1 and mp_size > 1:
        # 1D Pure MP
        mesh = init_device_mesh(
            device_type,
            (mp_size,),
            mesh_dim_names=("model_parallel",),
        )
        dispatcher = DataloaderDispatcher(dataloader, mesh, device, dp_mesh_dim=None)
        mode = "1D Pure MP"
    elif dp_size == 1 and mp_size == 1:
        # Single rank - use 1D DP mesh
        mesh = init_device_mesh(
            device_type,
            (1,),
            mesh_dim_names=("data_parallel",),
        )
        dispatcher = DataloaderDispatcher(dataloader, mesh, device, dp_mesh_dim=0)
        mode = "Single rank"
    else:
        # 2D Hybrid
        if dp_mesh_dim is None or dp_mesh_dim == 0:
            # Default: dim 0 is DP, dim 1 is MP
            mesh = init_device_mesh(
                device_type,
                (dp_size, mp_size),
                mesh_dim_names=("data_parallel", "model_parallel"),
            )
            dispatcher = DataloaderDispatcher(dataloader, mesh, device, dp_mesh_dim=0)
            mode = "2D Hybrid (dp_dim=0)"
        else:
            # Swapped: dim 0 is MP, dim 1 is DP
            mesh = init_device_mesh(
                device_type,
                (mp_size, dp_size),
                mesh_dim_names=("model_parallel", "data_parallel"),
            )
            dispatcher = DataloaderDispatcher(dataloader, mesh, device, dp_mesh_dim=1)
            mode = "2D Hybrid (dp_dim=1)"

    return mesh, dispatcher, mode


def verify_pure_dp(dispatcher, dp_size, rank, device):
    """Verify pure DP mode: each rank gets different batches."""
    print(f"[Rank {rank}] Testing Pure DP mode (dp_size={dp_size})")

    batches_received = []
    for batch in dispatcher:
        batches_received.append(batch)

    # Each rank should receive different batches
    local_batch_ids = torch.tensor(
        [b["input_ids"][0, 0].item() for b in batches_received],
        device=device,
    )

    # Gather all batch IDs to rank 0
    all_batch_ids = [torch.zeros_like(local_batch_ids) for _ in range(dp_size)]
    dist.all_gather(all_batch_ids, local_batch_ids)

    if rank == 0:
        # Verify all batches are unique across ranks
        all_ids = torch.cat(all_batch_ids).tolist()
        unique_ids = set(all_ids)
        assert len(unique_ids) == len(
            all_ids
        ), f"DP mode: Expected unique batches, got duplicates. IDs: {all_ids}"
        print(
            f"[Rank {rank}] Pure DP verified: {len(batches_received)} unique batches per rank"
        )

    return True


def verify_pure_mp(dispatcher, mp_size, rank, device):
    """Verify pure MP mode: all ranks get the same batches."""
    print(f"[Rank {rank}] Testing Pure MP mode (mp_size={mp_size})")

    batches_received = []
    for batch in dispatcher:
        batches_received.append(batch)

    # All ranks should receive identical batches
    local_batch_ids = torch.tensor(
        [b["input_ids"][0, 0].item() for b in batches_received],
        device=device,
    )

    # Gather all batch IDs to rank 0
    all_batch_ids = [torch.zeros_like(local_batch_ids) for _ in range(mp_size)]
    dist.all_gather(all_batch_ids, local_batch_ids)

    if rank == 0:
        # Verify all ranks got the same batches
        reference = all_batch_ids[0].tolist()
        for r, batch_ids in enumerate(all_batch_ids[1:], start=1):
            assert batch_ids.tolist() == reference, (
                f"MP mode: Rank {r} got different batches than rank 0. "
                f"Rank 0: {reference}, Rank {r}: {batch_ids.tolist()}"
            )
        print(
            f"[Rank {rank}] Pure MP verified: all {mp_size} ranks received identical batches"
        )

    return True


def verify_hybrid(dispatcher, mesh, dp_size, mp_size, rank, device, dp_mesh_dim):
    """Verify hybrid mode: same batch within MP group, different across DP groups."""
    print(f"[Rank {rank}] Testing Hybrid mode (dp_size={dp_size}, mp_size={mp_size})")

    # Get ranks based on dp_mesh_dim
    if dp_mesh_dim == 0:
        dp_rank = mesh.get_local_rank(0)
        mp_rank = mesh.get_local_rank(1)
        dp_group = mesh.get_group(0)
        mp_group = mesh.get_group(1)
    else:
        dp_rank = mesh.get_local_rank(1)
        mp_rank = mesh.get_local_rank(0)
        dp_group = mesh.get_group(1)
        mp_group = mesh.get_group(0)

    batches_received = []
    for batch in dispatcher:
        batches_received.append(batch)

    # Collect batch IDs
    local_batch_ids = torch.tensor(
        [b["input_ids"][0, 0].item() for b in batches_received],
        device=device,
    )

    # Gather within MP group - should all be identical
    mp_batch_ids = [torch.zeros_like(local_batch_ids) for _ in range(mp_size)]
    dist.all_gather(mp_batch_ids, local_batch_ids, group=mp_group)

    # Verify within MP group
    reference = mp_batch_ids[0].tolist()
    for r, batch_ids in enumerate(mp_batch_ids[1:], start=1):
        assert batch_ids.tolist() == reference, (
            f"Hybrid mode: MP rank {r} got different batches than MP rank 0. "
            f"MP rank 0: {reference}, MP rank {r}: {batch_ids.tolist()}"
        )

    # Gather across DP groups (only DP leaders)
    if mp_rank == 0:
        dp_batch_ids = [torch.zeros_like(local_batch_ids) for _ in range(dp_size)]
        dist.all_gather(dp_batch_ids, local_batch_ids, group=dp_group)

        # Verify across DP groups - should be different
        all_dp_ids = [ids.tolist() for ids in dp_batch_ids]
        for i in range(len(all_dp_ids)):
            for j in range(i + 1, len(all_dp_ids)):
                # Check that no two DP groups have the same batches
                common = set(all_dp_ids[i]) & set(all_dp_ids[j])
                assert (
                    len(common) == 0
                ), f"Hybrid mode: DP groups {i} and {j} share batches: {common}"

    if rank == 0:
        print(
            f"[Rank {rank}] Hybrid verified: same batches within MP groups, "
            f"different across DP groups"
        )

    return True


@record
def main(args):
    """Main test function."""
    rank, local_rank, world_size, device, device_type = init_distributed(args.backend)

    dp_size = args.dp
    mp_size = args.mp

    assert dp_size * mp_size == world_size, (
        f"dp_size ({dp_size}) * mp_size ({mp_size}) must equal "
        f"world_size ({world_size})"
    )

    # Create dataset and dataloader
    num_batches_per_rank = args.num_batches
    dataset = DummyDataset(
        num_batches=dp_size * num_batches_per_rank,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # Create mesh and dispatcher
    mesh, dispatcher, mode = create_mesh_and_dispatcher(
        dataloader, device, device_type, dp_size, mp_size, args.dp_dim
    )

    if rank == 0:
        print("=" * 60)
        print("DataloaderDispatcher Multi-Dimensional Parallelism Test")
        print(f"Backend: {args.backend}, Device Type: {device_type}")
        print(f"World size: {world_size}, DP size: {dp_size}, MP size: {mp_size}")
        print(f"Mode: {mode}")
        print("=" * 60)
        print(f"Dataset size: {len(dataset)}, Batches: {len(dataloader)}")
        print(f"Expected batches per rank: {num_batches_per_rank}")
        print()

    # Run appropriate verification based on mode
    try:
        if mp_size == 1:
            verify_pure_dp(dispatcher, dp_size, rank, device)
        elif dp_size == 1:
            verify_pure_mp(dispatcher, mp_size, rank, device)
        else:
            verify_hybrid(
                dispatcher,
                mesh,
                dp_size,
                mp_size,
                rank,
                device,
                args.dp_dim if args.dp_dim is not None else 0,
            )

        dist.barrier()
        if rank == 0:
            print()
            print("=" * 60)
            print("All tests passed!")
            print("=" * 60)

    except AssertionError as e:
        print(f"[Rank {rank}] Test FAILED: {e}")
        dist.barrier()
        dist.destroy_process_group()
        return 1

    dist.destroy_process_group()
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Test DataloaderDispatcher multi-dimensional parallelism",
        epilog=(
            "Examples:\n"
            "  # 1D Pure DP\n"
            "  torchrun --nproc_per_node 4 --standalone test.py --dp 4\n"
            "  # 1D Pure MP\n"
            "  torchrun --nproc_per_node 4 --standalone test.py --mp 4\n"
            "  # 2D Hybrid\n"
            "  torchrun --nproc_per_node 6 --standalone test.py --dp 2 --mp 3\n"
        ),
    )
    parser.add_argument(
        "--dp",
        type=int,
        default=1,
        help="Data parallel size",
    )
    parser.add_argument(
        "--mp",
        type=int,
        default=1,
        help="Model parallel size",
    )
    parser.add_argument(
        "--dp-dim",
        type=int,
        default=None,
        help="Which mesh dimension is DP (0 or 1, for 2D mesh). Default: 0",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="Number of batches per rank",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8,
        help="Sequence length",
    )
    parser.add_argument(
        "--backend",
        default="nccl",
        help="Torch Distributed backend. e.g. 'nccl', 'gloo'",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))
