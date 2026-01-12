"""
Utilities for working with StatefulDataLoader and multi-worker dataset iteration.

Provides helpers for syncing dataset state between workers and the main process,
which is necessary for length estimation and other stateful features to work
correctly with num_workers > 0.
"""

from typing import Any, Dict, Optional

from torchdata.stateful_dataloader import StatefulDataLoader


def sync_dataset_state_from_dataloader(
    dataloader: StatefulDataLoader,
    dataset: Any,
    debug: bool = False,
) -> None:
    """
    Sync dataset state from DataLoader workers back to the main process.

    With multi-worker DataLoader (num_workers > 0), each worker has its own
    copy of the dataset and updates its state independently. The main process
    dataset doesn't see these updates automatically. This function extracts
    worker states from the DataLoader and aggregates them back to the main
    process dataset.

    This is particularly useful for length estimation with N->M mapped datasets,
    where workers track input/output counts that need to be aggregated.

    Args:
        dataloader: StatefulDataLoader instance (must have been iterated)
        dataset: The dataset to update (should be the same as dataloader.dataset)
        debug: If True, print debug information about state structure

    Example:
        >>> dataloader = StatefulDataLoader(dataset, num_workers=4)
        >>> for batch in dataloader:
        ...     train_step(batch)
        >>>
        >>> # Sync state to get updated length
        >>> sync_dataset_state_from_dataloader(dataloader, dataset)
        >>> print(f"Updated length: {len(dataset)}")
    """
    # Get DataLoader state which includes worker states
    dl_state = dataloader.state_dict()

    if debug:
        print("DataLoader state keys:", dl_state.keys() if dl_state else "None")

    # Check if we have any state to sync
    if not dl_state or "_snapshot" not in dl_state:
        if debug:
            print("No _snapshot in dl_state")
        return

    snapshot = dl_state["_snapshot"]

    if debug:
        print("Snapshot keys:", snapshot.keys() if snapshot else "None")

    # For single worker (num_workers=0), state is stored differently
    if "main_snapshot" in dl_state or dataloader.num_workers == 0:
        # Single process - dataset should already be up to date
        if debug:
            print("Single process mode, skipping")
        return

    # Multi-worker: aggregate state from all workers
    worker_snapshots = snapshot.get("_worker_snapshots", {})
    if not worker_snapshots:
        if debug:
            print("No worker_snapshots found")
        return

    if debug:
        print(f"Found {len(worker_snapshots)} worker snapshots")
        for key, val in worker_snapshots.items():
            print(f"  {key}: {type(val).__name__}")
            if isinstance(val, dict):
                print(f"    keys: {val.keys()}")

    _aggregate_worker_state_to_dataset(
        dataset, worker_snapshots, dataloader.num_workers, debug=debug
    )


def _aggregate_worker_state_to_dataset(
    dataset: Any,
    worker_snapshots: Dict[str, Any],
    num_workers: int,
    debug: bool = False,
) -> None:
    """
    Aggregate length estimation state from multiple workers to the main dataset.

    Each worker tracks its own input/output counts for length estimation.
    We need to sum these across workers and update the main process dataset.
    """
    from .interleaved import InterleavedDataset

    # Handle InterleavedDataset by extracting child states and syncing children
    # (before checking length_estimate_mode, since InterleavedDataset doesn't have it)
    if isinstance(dataset, InterleavedDataset):
        if debug:
            print(
                f"Recursively syncing {len(dataset.datasets)} children of InterleavedDataset"
            )

        # Extract child states from each worker's InterleavedDataset state
        # and create synthetic worker snapshots for each child
        num_children = len(dataset.datasets)

        for child_idx in range(num_children):
            if debug:
                print(f"  Syncing child {child_idx}...")

            # Collect states from all workers for this specific child
            child_worker_snapshots = {}
            for worker_key, worker_state in worker_snapshots.items():
                if worker_state is None:
                    continue

                dataset_state = worker_state.get("dataset_state")
                if dataset_state is None:
                    continue

                # InterleavedDataset state contains 'child_states' list
                child_states = dataset_state.get("child_states", [])
                if child_idx >= len(child_states):
                    continue

                # Create a synthetic worker snapshot for this child
                child_state = child_states[child_idx]
                if child_state is not None:
                    child_worker_snapshots[worker_key] = {
                        "worker_id": worker_state["worker_id"],
                        "dataset_state": child_state,
                        "fetcher_state": None,
                    }

            # Recursively sync this child with its aggregated worker states
            _aggregate_worker_state_to_dataset(
                dataset.datasets[child_idx],
                child_worker_snapshots,
                num_workers,
                debug=debug,
            )
        return

    # Check if dataset supports length estimation
    if not hasattr(dataset, "length_estimate_mode"):
        if debug:
            print(
                f"Dataset {type(dataset).__name__} does not support length estimation"
            )
        return

    # Aggregate counts from all workers
    total_input_count = 0
    total_output_count = 0
    cached_exact_lengths = []  # Collect all cached lengths
    num_workers_with_state = 0

    for worker_key, worker_state in worker_snapshots.items():
        if worker_state is None:
            if debug:
                print(f"  {worker_key}: None")
            continue

        if debug:
            print(f"  {worker_key}: {list(worker_state.keys())}")

        # Extract dataset state from worker snapshot
        # Keys are without underscores: 'dataset_state' not '_dataset_state'
        dataset_state = worker_state.get("dataset_state")
        if dataset_state is None:
            if debug:
                print(f"    No dataset_state")
            continue

        num_workers_with_state += 1

        if debug:
            print(
                f"    dataset_state keys: {list(dataset_state.keys()) if isinstance(dataset_state, dict) else type(dataset_state).__name__}"
            )

        # Aggregate length estimation state
        input_count = dataset_state.get("input_count", 0)
        output_count = dataset_state.get("output_count", 0)
        worker_cached = dataset_state.get("cached_exact_length")

        if debug:
            print(
                f"    input={input_count}, output={output_count}, cached={worker_cached}"
            )

        total_input_count += input_count
        total_output_count += output_count

        # Collect cached lengths (including None)
        cached_exact_lengths.append(worker_cached)

    # Determine final cached length: only use if ALL workers have non-None values
    final_cached_length = None
    if len(cached_exact_lengths) == num_workers_with_state:
        # Check if all workers have cached values
        if all(c is not None for c in cached_exact_lengths):
            # Sum cached lengths from all workers
            final_cached_length = sum(cached_exact_lengths)

    if debug:
        print(
            f"Total: input={total_input_count}, output={total_output_count}, cached={final_cached_length}"
        )

    # Update main process dataset with aggregated state
    # IMPORTANT: Don't set _input_count and _output_count on the main process dataset!
    # With persistent_workers=False, new workers are forked from the main process dataset,
    # and would inherit these counts, causing accumulation across iterations.
    # Only store the synced cached length which is computed from worker counts.

    # For cached length, store in a separate attribute to avoid corrupting worker state
    # We use a separate attribute (_synced_cached_length) for multi-worker scenarios
    if final_cached_length is not None:
        # Store the aggregated cached length separately
        if not hasattr(dataset, "_synced_cached_length"):
            dataset._synced_cached_length = None
        dataset._synced_cached_length = final_cached_length
        # Don't overwrite _cached_exact_length as it might be used by workers
    else:
        # Use ratio-based estimate if we don't have complete cached values
        if total_input_count > 0 and total_output_count > 0:
            ratio = total_output_count / total_input_count
            original = (
                dataset._get_original_length()
                if hasattr(dataset, "_get_original_length")
                else len(dataset)
            )
            estimated = int(original * ratio)
            if not hasattr(dataset, "_synced_cached_length"):
                dataset._synced_cached_length = None
            dataset._synced_cached_length = estimated
        else:
            # Clear synced length if we don't have data
            if hasattr(dataset, "_synced_cached_length"):
                dataset._synced_cached_length = None

    if debug:
        synced = getattr(dataset, "_synced_cached_length", None)
        print(
            f"Updated dataset: synced_cached={synced} (from worker counts: input={total_input_count}, output={total_output_count}, cached={final_cached_length})"
        )


def create_length_sync_callback(
    dataloader: StatefulDataLoader,
    dataset: Any,
    sync_every_n_steps: int = 100,
) -> "LengthSyncCallback":
    """
    Create a callback that automatically syncs dataset length during training.

    This creates a trainer callback that periodically syncs dataset state from
    DataLoader workers, ensuring that len(dataset) returns up-to-date values
    even with multi-worker DataLoader.

    Args:
        dataloader: StatefulDataLoader for training
        dataset: The dataset to sync
        sync_every_n_steps: How often to sync (default: every 100 steps)

    Returns:
        Callback instance that can be added to trainer.callbacks

    Example:
        >>> train_dataloader = StatefulDataLoader(train_dataset, num_workers=4)
        >>> sync_callback = create_length_sync_callback(
        ...     train_dataloader,
        ...     train_dataset,
        ...     sync_every_n_steps=50
        ... )
        >>> trainer = Trainer(
        ...     model=model,
        ...     args=args,
        ...     train_dataset=train_dataset,
        ...     callbacks=[sync_callback, ...],
        ... )
    """
    return LengthSyncCallback(
        dataloader=dataloader,
        dataset=dataset,
        sync_every_n_steps=sync_every_n_steps,
    )


class LengthSyncCallback:
    """
    Trainer callback that syncs dataset length from DataLoader workers.

    Automatically calls sync_dataset_state_from_dataloader() during training
    at specified intervals.
    """

    def __init__(
        self,
        dataloader: StatefulDataLoader,
        dataset: Any,
        sync_every_n_steps: int = 100,
    ):
        self.dataloader = dataloader
        self.dataset = dataset
        self.sync_every_n_steps = sync_every_n_steps
        self._last_sync_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        current_step = state.global_step

        # Sync at specified intervals
        if current_step - self._last_sync_step >= self.sync_every_n_steps:
            sync_dataset_state_from_dataloader(self.dataloader, self.dataset)
            self._last_sync_step = current_step

    def on_evaluate(self, args, state, control, **kwargs):
        """Called before evaluation - sync to get accurate length."""
        sync_dataset_state_from_dataloader(self.dataloader, self.dataset)
