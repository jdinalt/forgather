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
    At the start of each iteration, all ranks exchange their dataloader lengths
    via a single all_reduce(MIN).  Each rank then iterates for exactly that
    many batches with zero per-step synchronization.  This avoids the GPU-CPU
    sync (.item()) that would break the CUDA pipeline and force the GPU idle
    while the CPU prepares the next batch.

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

        # Determine the minimum length across all ranks with a single
        # all_reduce at the start of iteration.  This avoids a per-step
        # GPU-CPU sync (.item()) that would break the CUDA pipeline.
        local_len = len(self._dataloader)
        len_tensor = torch.tensor(local_len, dtype=torch.int64, device=self._device)
        dist.all_reduce(len_tensor, op=dist.ReduceOp.MIN, group=self._process_group)
        min_len = int(len_tensor.item())

        if min_len != local_len:
            logger.debug(
                f"SynchronizedDataLoader: local length {local_len}, "
                f"global min {min_len} (dropping {local_len - min_len} batches)"
            )

        iterator = iter(self._dataloader)
        for _ in range(min_len):
            try:
                batch = next(iterator)
            except StopIteration:
                # Underlying dataloader exhausted before reaching min_len.
                # `min_len` is computed from len(dataloader) at the start of
                # iteration; for iterable datasets with a dynamic length
                # estimator the real length can drift below the estimate,
                # in which case next() raises StopIteration here. Letting
                # that StopIteration propagate would violate PEP 479 -- this
                # method is a generator, and a StopIteration raised inside
                # a generator body is converted to `RuntimeError: generator
                # raised StopIteration` by the interpreter -- bypassing the
                # trainer's end-of-training save path. Instead, `return`
                # cleanly so the outer `for batch in loader` loop sees a
                # normal iteration end.
                logger.warning(
                    "SynchronizedDataLoader: underlying dataloader exhausted "
                    "before reaching the all-ranks minimum length "
                    f"({min_len}); returning early."
                )
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
