"""
SynchronizedDataLoader: Wrapper for handling uneven dataset lengths in DDP.

When using sharded datasets (dispatch_batches=False) in distributed training,
different ranks may have shards of different lengths. This wrapper ensures
all ranks agree on when to stop iterating, preventing collective operation
mismatches and deadlocks.

Pattern: Similar to DataloaderDispatcher, transparently wraps any dataloader
to add cross-rank synchronization.
"""

import logging
from typing import Any, Iterator

import torch
from torch import distributed as dist

logger = logging.getLogger(__name__)


class SynchronizedDataLoader:
    """
    Wraps a dataloader to synchronize iteration across distributed ranks.

    When any rank's dataloader is exhausted, all ranks stop iterating together.
    This prevents deadlocks when dataset shards have uneven lengths.

    Synchronization strategy:
    Per-step all_reduce(MIN) on a small int32 "has_batch" flag. All ranks
    continue only while every rank successfully fetched a batch; as soon as
    any rank's underlying iterator raises StopIteration, every rank stops
    together on the same iteration. To keep the overhead small, the next
    batch is prefetched and the next all_reduce is launched *before* yielding,
    so the .item() at the top of the loop typically overlaps with the
    caller's forward/backward and doesn't stall the CUDA pipeline.

    This replaces an earlier design that did a single upfront all_reduce(MIN)
    on len(dataloader) and then iterated that many steps with zero per-step
    sync. That was fragile because len() on iterable datasets is an estimate:
    when the real length drifted below the estimate on one rank, that rank
    would StopIteration early and move on to end-of-training collectives
    while the other ranks were still doing DDP gradient all-reduces, causing
    a cross-op NCCL hang.

    Usage:
        train_dataloader = SynchronizedDataLoader(
            dataloader=raw_dataloader,
            device=torch.device("cuda:0"),
            process_group=None,  # Use default group
        )

        for batch in train_dataloader:
            # All ranks guaranteed to process same number of batches
            ...

    Args:
        dataloader: The underlying dataloader to wrap
        device: Device for synchronization tensors
        process_group: DDP process group (None = default group)
        enabled: If False, pass through without synchronization
    """

    def __init__(
        self,
        dataloader: Any,
        device: torch.device,
        process_group: Any = None,
        enabled: bool = True,
    ):
        self._dataloader = dataloader
        self._device = device
        self._process_group = process_group
        self._enabled = enabled and dist.is_initialized() and dist.get_world_size() > 1

        if self._enabled:
            logger.debug(
                f"SynchronizedDataLoader enabled: world_size={dist.get_world_size()}"
            )

    def __iter__(self) -> Iterator:
        """Return an iterator that synchronizes across ranks."""
        if not self._enabled:
            # Pass through without synchronization
            yield from self._dataloader
            return

        iterator = iter(self._dataloader)
        # Reused GPU-side int32 scalar for the per-step MIN reduction.
        has_batch = torch.zeros(1, dtype=torch.int32, device=self._device)

        def prefetch():
            """Fetch next batch from the local iterator; fill has_batch flag."""
            try:
                b = next(iterator)
                has_batch.fill_(1)
                return b
            except StopIteration:
                has_batch.fill_(0)
                return None

        # Prefetch first batch and launch the first MIN all_reduce so that
        # the .item() at the top of the loop can overlap with the caller's
        # compute on subsequent iterations.
        batch = prefetch()
        dist.all_reduce(has_batch, op=dist.ReduceOp.MIN, group=self._process_group)

        while True:
            # Block until the previously-launched all_reduce has completed.
            # For all steps after the first, this is typically cheap: the
            # all_reduce was launched before the last yield, and the caller
            # has since run forward+backward, so the result is already here.
            if has_batch.item() == 0:
                if batch is not None:
                    # This rank had a batch prefetched but at least one
                    # peer rank is exhausted. Drop it to stay in lockstep.
                    logger.warning(
                        "SynchronizedDataLoader: another rank exhausted "
                        "before this one; dropping 1 prefetched local batch "
                        "to keep all ranks aligned."
                    )
                return

            current = batch

            # Prefetch the next batch and launch the next MIN all_reduce
            # BEFORE yielding, so the collective overlaps with the caller's
            # compute. On the next loop iteration, has_batch.item() just
            # reads an already-completed value.
            batch = prefetch()
            dist.all_reduce(has_batch, op=dist.ReduceOp.MIN, group=self._process_group)

            yield current

    def __len__(self):
        """Return length of underlying dataloader (may differ across ranks!)."""
        return len(self._dataloader)

    def __getattr__(self, name):
        """Forward all unknown attributes/methods to the wrapped dataloader."""
        return getattr(self._dataloader, name)

    def state_dict(self):
        """Forward state_dict to underlying dataloader if supported."""
        if hasattr(self._dataloader, "state_dict"):
            return self._dataloader.state_dict()
        logger.warning(
            f"Wrapped Dataloader does not support state_dict(). State will not be saved."
        )
        return {}

    def load_state_dict(self, state_dict):
        """Forward load_state_dict to underlying dataloader if supported."""
        if hasattr(self._dataloader, "load_state_dict"):
            self._dataloader.load_state_dict(state_dict)
        else:
            logger.warning(
                f"Wrapped Dataloader does not support load_state_dict(). State can't be restored."
            )
