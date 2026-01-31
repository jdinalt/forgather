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

    Works by performing an all_reduce AND operation before each batch:
    - Each rank reports: 1 if batch available, 0 if StopIteration
    - MIN reduction acts as AND: result is 1 only if all ranks have data
    - If any rank exhausted (result=0), all ranks raise StopIteration

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

        self._has_batch_tensor = torch.empty(
            1,
            dtype=torch.int32,
            device=self._device,
        )

    def __iter__(self) -> Iterator:
        """Return an iterator that synchronizes across ranks."""
        if not self._enabled:
            # Pass through without synchronization
            yield from self._dataloader
            return

        # Synchronized iteration
        iterator = iter(self._dataloader)
        while True:
            # Try to get next batch
            batch = None
            has_batch = True
            try:
                batch = next(iterator)
            except StopIteration:
                has_batch = False

            # Synchronize: all ranks must agree to continue
            self._has_batch_tensor.fill_(1 if has_batch else 0)

            dist.all_reduce(
                self._has_batch_tensor,
                op=dist.ReduceOp.MIN,  # MIN acts as AND for 0/1 values
                group=self._process_group,
            )

            # If any rank exhausted, all ranks stop
            if self._has_batch_tensor.item() == 0:
                return

            yield batch

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
