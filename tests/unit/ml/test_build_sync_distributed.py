#!/usr/bin/env python3
"""
Distributed test for build_sync context manager.

Tests the torch.distributed barrier-based synchronization path of build_sync.
Uses gloo backend for CPU testing.

Usage:
    # Test with 4 ranks using global process group
    torchrun --nproc_per_node 4 --standalone tests/unit/ml/test_build_sync_distributed.py

    # Test with local process group (per-node sync)
    torchrun --nproc_per_node 4 --standalone tests/unit/ml/test_build_sync_distributed.py --local
"""
import argparse
import os
import sys
import tempfile
import time

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from forgather.ml.construct import build_sync


def init_distributed(backend: str = "gloo"):
    """Initialize distributed process group."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend=backend)

    return rank, local_rank, world_size


def test_build_sync_exactly_one_builder(rank, world_size, target_path, local=False):
    """
    Test that exactly one rank yields True from build_sync.

    All ranks call build_sync simultaneously. Exactly one (rank 0) should
    get should_build=True, others should get False.
    """
    print(f"[Rank {rank}] Testing build_sync (local={local})")

    # Ensure all ranks start together
    dist.barrier()
    start_time = time.time()

    with build_sync(target_path, local=local) as should_build:
        elapsed = time.time() - start_time
        print(f"[Rank {rank}] should_build={should_build}, elapsed={elapsed:.3f}s")

        if should_build:
            # Simulate building - create the target file
            time.sleep(0.2)  # Simulate work
            with open(target_path, "w") as f:
                f.write(f"Built by rank {rank}\n")

    # Verify results
    dist.barrier()

    # Gather should_build values from all ranks
    should_build_tensor = torch.tensor([1 if should_build else 0], dtype=torch.int)
    all_should_build = [torch.zeros(1, dtype=torch.int) for _ in range(world_size)]
    dist.all_gather(all_should_build, should_build_tensor)

    if rank == 0:
        builders = [i for i, t in enumerate(all_should_build) if t.item() == 1]
        print(f"[Rank {rank}] Builders: {builders}")

        # Exactly one builder
        assert (
            len(builders) == 1
        ), f"Expected exactly 1 builder, got {len(builders)}: {builders}"

        # Builder should be rank 0 (for global process group)
        if not local:
            assert (
                builders[0] == 0
            ), f"Expected rank 0 to be builder, got rank {builders[0]}"

        # Verify target was created
        assert os.path.exists(target_path), "Target file was not created"

        with open(target_path, "r") as f:
            content = f.read()
        print(f"[Rank {rank}] Target content: {content.strip()}")

        print(f"[Rank {rank}] Test PASSED: exactly one builder")

    return True


def test_build_sync_waiters_blocked(rank, world_size, target_path, local=False):
    """
    Test that non-builder ranks wait until the builder completes.

    The builder sleeps for a noticeable duration. Non-builder ranks should
    not yield until after this duration.
    """
    print(f"[Rank {rank}] Testing waiter blocking (local={local})")

    build_duration = 0.5  # Builder will sleep this long

    # Ensure all ranks start together
    dist.barrier()
    start_time = time.time()

    with build_sync(target_path, local=local) as should_build:
        yield_time = time.time() - start_time

        if should_build:
            # Builder: sleep to simulate work
            time.sleep(build_duration)
            with open(target_path, "w") as f:
                f.write(f"Built by rank {rank} at {time.time()}\n")
            print(f"[Rank {rank}] Builder finished after {build_duration}s")
        else:
            # Non-builder: should have waited for builder
            print(f"[Rank {rank}] Waiter yielded after {yield_time:.3f}s")

    end_time = time.time() - start_time

    # Gather timing info
    dist.barrier()
    timing_tensor = torch.tensor([yield_time], dtype=torch.float)
    all_timings = [torch.zeros(1, dtype=torch.float) for _ in range(world_size)]
    dist.all_gather(all_timings, timing_tensor)

    if rank == 0:
        timings = [t.item() for t in all_timings]
        print(f"[Rank {rank}] Yield timings: {[f'{t:.3f}s' for t in timings]}")

        # Non-builder ranks should have waited at least build_duration
        # (with some tolerance for timing variations)
        min_wait_time = build_duration * 0.8  # 80% of build duration
        for r, timing in enumerate(timings):
            if r != 0:  # Skip builder (rank 0)
                assert timing >= min_wait_time, (
                    f"Rank {r} yielded too early ({timing:.3f}s < {min_wait_time:.3f}s). "
                    f"Waiters should block until builder completes."
                )

        print(f"[Rank {rank}] Test PASSED: waiters properly blocked")

    return True


def test_build_sync_all_continue_together(rank, world_size, target_path, local=False):
    """
    Test that all ranks continue together after build_sync exits.

    After the context manager exits, all ranks should be synchronized
    (within reasonable timing tolerance).
    """
    print(f"[Rank {rank}] Testing synchronized exit (local={local})")

    dist.barrier()
    start_time = time.time()

    with build_sync(target_path, local=local) as should_build:
        if should_build:
            time.sleep(0.3)
            with open(target_path, "w") as f:
                f.write(f"Built by rank {rank}\n")

    exit_time = time.time() - start_time

    # Gather exit times
    dist.barrier()
    exit_tensor = torch.tensor([exit_time], dtype=torch.float)
    all_exits = [torch.zeros(1, dtype=torch.float) for _ in range(world_size)]
    dist.all_gather(all_exits, exit_tensor)

    if rank == 0:
        exit_times = [t.item() for t in all_exits]
        print(f"[Rank {rank}] Exit timings: {[f'{t:.3f}s' for t in exit_times]}")

        # All ranks should exit within a small window of each other
        min_exit = min(exit_times)
        max_exit = max(exit_times)
        spread = max_exit - min_exit

        # Allow 100ms spread for timing variations
        max_spread = 0.1
        assert spread < max_spread, (
            f"Ranks did not exit together. Spread: {spread:.3f}s > {max_spread}s. "
            f"Exit times: {exit_times}"
        )

        print(
            f"[Rank {rank}] Test PASSED: all ranks exited together (spread={spread:.3f}s)"
        )

    return True


@record
def main(args):
    """Main test function."""
    rank, local_rank, world_size = init_distributed(args.backend)

    if rank == 0:
        print("=" * 60)
        print("build_sync Distributed Test")
        print(f"Backend: {args.backend}")
        print(f"World size: {world_size}")
        print(f"Local mode: {args.local}")
        print("=" * 60)
        print()

    # Create temp directory for test targets
    if rank == 0:
        tmpdir = tempfile.mkdtemp(prefix="build_sync_test_")
    else:
        tmpdir = None

    # Broadcast tmpdir to all ranks
    tmpdir_list = [tmpdir]
    dist.broadcast_object_list(tmpdir_list, src=0)
    tmpdir = tmpdir_list[0]

    try:
        # Test 1: Exactly one builder
        target1 = os.path.join(tmpdir, "test1_target")
        test_build_sync_exactly_one_builder(rank, world_size, target1, args.local)

        dist.barrier()
        if rank == 0:
            print()

        # Test 2: Waiters blocked
        target2 = os.path.join(tmpdir, "test2_target")
        test_build_sync_waiters_blocked(rank, world_size, target2, args.local)

        dist.barrier()
        if rank == 0:
            print()

        # Test 3: Synchronized exit
        target3 = os.path.join(tmpdir, "test3_target")
        test_build_sync_all_continue_together(rank, world_size, target3, args.local)

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

    finally:
        # Cleanup
        if rank == 0:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

    dist.destroy_process_group()
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test build_sync distributed synchronization"
    )
    parser.add_argument(
        "--backend",
        default="gloo",
        help="Torch Distributed backend (default: gloo for CPU testing)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local process group (per-node sync) instead of global",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))
