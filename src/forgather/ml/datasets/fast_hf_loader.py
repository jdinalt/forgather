"""
Fast HuggingFace Dataset Loader - Simple Generator Approach

Uses a simple generator that reads Arrow files sequentially.
No expensive interleave_datasets() calls. Just a simple, fast generator.
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.utils.data
from torch.utils.data import IterableDataset as TorchIterableDataset

from datasets import Dataset
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset

from .iterable_with_length import IterableDatasetWithLength

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# Metadata version - increment when index format changes
METADATA_VERSION = 2  # v2: Added per-file example counts and version check


def _identity_func(x):
    """Identity function for map with no function specified."""
    return x


def _parse_split_notation(split: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Parse HuggingFace split notation into base split and slice bounds.

    This allows us to use only the base split for caching while applying
    slices virtually after loading.

    Examples:
        "train" → ("train", None, None)
        "train[100:]" → ("train", 100, None)
        "train[:1000]" → ("train", None, 1000)
        "train[100:1000]" → ("train", 100, 1000)
        "train[10%:20%]" → ("train", "10%", "20%")

    Args:
        split: Split string potentially with slice notation

    Returns:
        Tuple of (base_split, start_idx, end_idx) where indices can be
        int, str (for percentages like "10%"), or None
    """
    if not split:
        return split, None, None

    # Match pattern: split_name[start:end]
    # Where start and end are optional and can be numbers or percentages
    match = re.match(r"^([^[]+)(?:\[([^:]*):([^\]]*)\])?$", split)
    if not match:
        # Invalid format, return as-is
        return split, None, None

    base_split = match.group(1)
    start_str = match.group(2)
    end_str = match.group(3)

    # Parse start index (can be int, percentage string, or None)
    if start_str:
        if "%" in start_str:
            start_idx = start_str
        else:
            start_idx = int(start_str)
    else:
        start_idx = None

    # Parse end index (can be int, percentage string, or None)
    if end_str:
        if "%" in end_str:
            end_idx = end_str
        else:
            end_idx = int(end_str)
    else:
        end_idx = None

    return base_split, start_idx, end_idx


class SimpleArrowIterableDataset(TorchIterableDataset):
    """
    Simple IterableDataset wrapper around Arrow files.

    Implements:
    - Sequential reading of Arrow files
    - Shard-level shuffling (shuffles file order)
    - Sharding for DDP
    - Stateful checkpoint protocol (state_dict/load_state_dict)
    - Compatible with torch.utils.data.DataLoader
    - Compatible with torchdata.stateful_dataloader.StatefulDataLoader
    """

    def __init__(
        self, arrow_files: List[str], file_lengths: Optional[List[int]] = None
    ):
        self.arrow_files = arrow_files
        self.file_lengths = file_lengths  # Per-file example counts (if available)
        self._shuffled_files = None
        self._shuffled_lengths = None  # Parallel to _shuffled_files
        self._shard_config = None  # (num_shards, shard_index, shard_mode)

        # Checkpoint state
        self._current_file_index = 0
        self._current_example_index = 0
        self._shuffle_seed = None
        self._shuffle_buffer_size = None  # Example-level shuffle buffer size
        self._total_examples = None  # Cached length

        # Epoch-based shuffle state (for multi-epoch training)
        self._base_shuffle_seed = None  # Original seed from shuffle() call
        self._epoch = 0  # Current epoch (default 0)
        self._last_iter_epoch = None  # Track which epoch was used for last iteration

        # Virtual split boundaries (global example indices, before sharding)
        self._split_start_idx = None  # Inclusive
        self._split_end_idx = None  # Exclusive

        # Example-level sharding boundaries (global example indices, after split)
        self._shard_start_idx = None  # Inclusive
        self._shard_end_idx = None  # Exclusive

        # Metadata attributes (lazy loaded from Arrow schema)
        self._column_names = None
        self._features = None

        # Map configuration (for lazy transformation during iteration)
        self._map_function = None
        self._map_batched = False
        self._map_batch_size = 1000
        self._map_drop_last_batch = False
        self._map_remove_columns = None
        self._map_with_indices = False
        self._map_input_columns = None
        self._map_fn_kwargs = None

        # Length estimation configuration and state
        self.length_estimate_mode = "dynamic"  # 'static', 'dynamic', or 'exact'
        self._reset_length_on_iter = False  # Configurable: reset counts each iteration
        self._original_length = None  # Pre-map length (set lazily)
        self._input_count = 0  # Examples consumed from source
        self._output_count = 0  # Examples yielded after map
        self._cached_exact_length = None  # Exact count after full iteration
        self._length_invalidated = False  # Flag for operations that invalidate stats
        self._current_batch_buffer_size = 0  # For batched maps: track pending batch
        self._restored_from_checkpoint = False  # Track if load_state_dict was called

    def __repr__(self):
        return (
            "SimpleArrowIterableDataset: "
            f"arrow_files={len(self.arrow_files)}, "
            f"examples={len(self)}, "
            f"current_file_index={self._current_file_index}, "
            f"current_example_index={self._current_example_index}"
        )

    @staticmethod
    def _shuffle_files_and_lengths(
        arrow_files: List[str],
        file_lengths: Optional[List[int]],
        seed: int,
    ) -> Tuple[List[str], Optional[List[int]]]:
        """
        Shuffle Arrow files and their lengths together.

        Args:
            arrow_files: List of Arrow file paths
            file_lengths: Optional list of example counts per file
            seed: Random seed for shuffling

        Returns:
            Tuple of (shuffled_files, shuffled_lengths)
        """
        import random

        if file_lengths:
            # Shuffle files and lengths together
            paired = list(zip(arrow_files, file_lengths))
            rng = random.Random(seed)
            rng.shuffle(paired)
            shuffled_files, shuffled_lengths = zip(*paired)
            return list(shuffled_files), list(shuffled_lengths)
        else:
            shuffled_files = arrow_files.copy()
            rng = random.Random(seed)
            rng.shuffle(shuffled_files)
            return shuffled_files, None

    def _copy(self, **overrides) -> "SimpleArrowIterableDataset":
        """
        Create a copy of this dataset with optional overrides.

        Args:
            **overrides: Keyword arguments to override specific instance variables.
                        Use the attribute name without the leading underscore for
                        private attributes (e.g., shuffle_seed not _shuffle_seed).

        Returns:
            New SimpleArrowIterableDataset with copied state

        Example:
            >>> new_ds = ds._copy(shuffle_seed=42, epoch=1)
        """
        # Create new instance with same arrow files
        new_dataset = SimpleArrowIterableDataset(
            self.arrow_files, self.file_lengths
        )

        # Copy all instance variables
        # Checkpoint state
        new_dataset._current_file_index = overrides.get(
            "current_file_index", self._current_file_index
        )
        new_dataset._current_example_index = overrides.get(
            "current_example_index", self._current_example_index
        )

        # Shuffle state
        new_dataset._shuffled_files = overrides.get(
            "shuffled_files", self._shuffled_files
        )
        new_dataset._shuffled_lengths = overrides.get(
            "shuffled_lengths", self._shuffled_lengths
        )
        new_dataset._shuffle_seed = overrides.get("shuffle_seed", self._shuffle_seed)
        new_dataset._base_shuffle_seed = overrides.get(
            "base_shuffle_seed", self._base_shuffle_seed
        )
        new_dataset._epoch = overrides.get("epoch", self._epoch)
        new_dataset._last_iter_epoch = overrides.get(
            "last_iter_epoch", self._last_iter_epoch
        )
        new_dataset._shuffle_buffer_size = overrides.get(
            "shuffle_buffer_size", self._shuffle_buffer_size
        )

        # Sharding and splitting
        new_dataset._shard_config = overrides.get("shard_config", self._shard_config)
        new_dataset._split_start_idx = overrides.get(
            "split_start_idx", self._split_start_idx
        )
        new_dataset._split_end_idx = overrides.get("split_end_idx", self._split_end_idx)
        new_dataset._shard_start_idx = overrides.get(
            "shard_start_idx", self._shard_start_idx
        )
        new_dataset._shard_end_idx = overrides.get("shard_end_idx", self._shard_end_idx)

        # Metadata (lazy loaded)
        new_dataset._column_names = overrides.get("column_names", self._column_names)
        new_dataset._features = overrides.get("features", self._features)
        new_dataset._total_examples = overrides.get(
            "total_examples", self._total_examples
        )

        # Map configuration
        new_dataset._map_function = overrides.get("map_function", self._map_function)
        new_dataset._map_batched = overrides.get("map_batched", self._map_batched)
        new_dataset._map_batch_size = overrides.get(
            "map_batch_size", self._map_batch_size
        )
        new_dataset._map_drop_last_batch = overrides.get(
            "map_drop_last_batch", self._map_drop_last_batch
        )
        new_dataset._map_remove_columns = overrides.get(
            "map_remove_columns", self._map_remove_columns
        )
        new_dataset._map_with_indices = overrides.get(
            "map_with_indices", self._map_with_indices
        )
        new_dataset._map_input_columns = overrides.get(
            "map_input_columns", self._map_input_columns
        )
        new_dataset._map_fn_kwargs = overrides.get("map_fn_kwargs", self._map_fn_kwargs)

        # Length estimation
        new_dataset.length_estimate_mode = overrides.get(
            "length_estimate_mode", self.length_estimate_mode
        )
        new_dataset._reset_length_on_iter = overrides.get(
            "reset_length_on_iter", self._reset_length_on_iter
        )
        new_dataset._original_length = overrides.get(
            "original_length", self._original_length
        )
        new_dataset._input_count = overrides.get("input_count", self._input_count)
        new_dataset._output_count = overrides.get("output_count", self._output_count)
        new_dataset._cached_exact_length = overrides.get(
            "cached_exact_length", self._cached_exact_length
        )
        new_dataset._length_invalidated = overrides.get(
            "length_invalidated", self._length_invalidated
        )
        new_dataset._current_batch_buffer_size = overrides.get(
            "current_batch_buffer_size", self._current_batch_buffer_size
        )

        return new_dataset

    def shuffle(self, seed: Optional[int] = None, buffer_size: int = 1000):
        """
        Shuffle at both file and example level.

        Implements two-level shuffling:
        1. Shard-level: Shuffles Arrow file order
        2. Example-level: Maintains a shuffle buffer during iteration

        Args:
            seed: Random seed for shuffling. If None, generates a random seed.
                  Storing the seed allows checkpoint/restore to reproduce the shuffle.
            buffer_size: Size of shuffle buffer for example-level shuffling.
                        Larger buffers provide better randomization but use more memory.
                        Set to 0 or None to disable example-level shuffling (shard-level only).

        Returns:
            New dataset instance with shuffling configured

        Note:
            With checkpoint resumption, the shuffle pattern after the checkpoint will
            differ from a non-interrupted run, but randomness is still maintained.
        """
        import random

        # Generate seed if not provided, so shuffle can be reproduced from checkpoint
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        # Normalize buffer_size
        if buffer_size is None or buffer_size <= 0:
            buffer_size = None  # Disable example-level shuffling

        # Shuffle Arrow file order using static helper
        shuffled_files, shuffled_lengths = self._shuffle_files_and_lengths(
            self.arrow_files, self.file_lengths, seed
        )

        # Create copy with shuffled state
        return self._copy(
            shuffle_seed=seed,  # Current effective seed
            base_shuffle_seed=seed,  # Store base seed for epoch-based re-shuffling
            epoch=0,  # Initialize epoch to 0
            last_iter_epoch=0,  # Mark that files are already shuffled for epoch 0
            shuffle_buffer_size=buffer_size,
            shuffled_files=shuffled_files,
            shuffled_lengths=shuffled_lengths,
            # Invalidate cached stats but preserve ratio estimate
            length_invalidated=True,
            # Keep input/output counts for ratio estimate (inherited via _copy)
            # Clear exact cache since order changed
            cached_exact_length=None,
        )

    def set_epoch(self, epoch: int):
        """
        Set the epoch for the dataset.

        Enables different shuffling patterns for each epoch while maintaining
        reproducibility. This should be called at the start of each training epoch:

            for epoch in range(num_epochs):
                train_dataset.set_epoch(epoch)
                for batch in dataloader:
                    # Training code
                    pass

        Args:
            epoch: The current epoch number (0-indexed)

        Behavior:
            - If shuffle() was called: effective_seed = base_seed + epoch
            - If shuffle() was not called but epoch > 0: effective_seed = epoch
            - If shuffle() was not called and epoch == 0: no shuffle occurs

        This allows set_epoch() to work even without explicitly calling shuffle(),
        enabling epoch-based shuffling with deterministic seeds derived from the
        epoch number itself.

        Note:
            This method must be called BEFORE iteration starts for the epoch.
        """
        self._epoch = epoch

    def _reshuffle_files_with_seed(self, seed: int):
        """
        Re-shuffle the Arrow file list with a new seed.

        This updates _shuffled_files and _shuffled_lengths in-place.
        Used by set_epoch() to change shuffle pattern between epochs.

        Args:
            seed: Random seed for shuffling
        """
        # Use static helper to shuffle
        self._shuffled_files, self._shuffled_lengths = self._shuffle_files_and_lengths(
            self.arrow_files, self.file_lengths, seed
        )
        # Update current working seed
        self._shuffle_seed = seed

    def _get_effective_seed(self, fallback_seed: Optional[int] = None) -> Optional[int]:
        """
        Compute the effective shuffle seed for the current epoch.

        This handles three cases:
        1. If shuffle() was called: effective_seed = base_seed + epoch
        2. If shuffle() was not called but epoch > 0: effective_seed = epoch
        3. Otherwise: use fallback_seed (or None)

        Args:
            fallback_seed: Seed to use if no base_seed and epoch == 0.
                          Defaults to None if not provided.

        Returns:
            Effective seed to use for shuffling, or None for no shuffle
        """
        if self._base_shuffle_seed is not None:
            # shuffle() was called - use base_seed + epoch
            return self._base_shuffle_seed + self._epoch
        elif self._epoch > 0:
            # shuffle() not called, but set_epoch() was called with non-zero epoch
            # Use epoch itself as seed for deterministic per-epoch shuffling
            return self._epoch
        else:
            # No shuffle() or epoch == 0 - use fallback
            return fallback_seed

    def select(self, indices):
        """
        Select examples by indices, similar to HuggingFace datasets API.

        This is an adapter that translates to the slice() method for contiguous ranges.

        Args:
            indices: Range, list, iterable, ndarray, or Series of integer indices.
                     If the indices correspond to a contiguous range, the dataset
                     is efficiently sliced. Non-contiguous indices are not yet supported.

        Returns:
            New dataset with selected examples

        Examples:
            >>> # Select first 100 examples
            >>> ds.select(range(100))
            >>> ds.select(list(range(100)))
            >>>
            >>> # Select examples 100-200
            >>> ds.select(range(100, 200))

        Raises:
            ValueError: If indices are empty
            NotImplementedError: If non-contiguous indices are provided
        """
        # Convert to list if needed (handles range, numpy arrays, pandas Series, etc.)
        if hasattr(indices, "tolist"):
            # numpy array or pandas Series
            indices_list = indices.tolist()
        elif not isinstance(indices, list):
            # range, iterator, etc.
            indices_list = list(indices)
        else:
            indices_list = indices

        if not indices_list:
            raise ValueError("Cannot select from empty indices")

        # Check if indices are contiguous
        start_idx = indices_list[0]
        end_idx = indices_list[-1] + 1  # End is exclusive in slice()

        # Verify all indices are present and contiguous
        if len(indices_list) != (end_idx - start_idx):
            raise NotImplementedError(
                "Non-contiguous indices are not yet supported. "
                "The current implementation only supports contiguous ranges. "
                "Use dataset.select(range(start, end)) for contiguous selections."
            )

        # Verify the indices are actually the expected sequence
        if indices_list != list(range(start_idx, end_idx)):
            raise NotImplementedError(
                "Indices are not in sequential order. "
                "The current implementation only supports contiguous, ordered ranges."
            )

        # Translate to slice call
        return self.slice(start_idx, end_idx)

    def slice(self, start=None, end=None):
        """
        Create a virtual split by selecting a slice of examples.

        Useful for train/val/test splits without copying data.

        Args:
            start: Start index (inclusive). Can be:
                - None: Start from beginning (index 0)
                - int: Absolute example index
                - float in (0,1): Percentage of dataset (e.g., 0.8 = 80%)
                - str: Percentage string (e.g., "80%")
            end: End index (exclusive). Can be:
                - None: Go to end of dataset
                - int: Absolute example index
                - float in (0,1): Percentage of dataset
                - str: Percentage string (e.g., "80%")

        Returns:
            New dataset with virtual split applied

        Examples:
            >>> # First 80% for training
            >>> train_ds = ds.slice(None, 0.8)
            >>> train_ds = ds.slice(None, "80%")
            >>>
            >>> # Last 20% for validation
            >>> val_ds = ds.slice(0.8, None)
            >>> val_ds = ds.slice("80%", None)
            >>>
            >>> # Specific range
            >>> subset = ds.slice(100, 200)  # Examples 100-199
            >>> subset = ds.slice(0.1, 0.2)  # 10%-20% of dataset
        """

        def parse_index(idx, total):
            """Convert index to absolute integer."""
            if idx is None:
                return None
            if isinstance(idx, str):
                # Parse percentage string like "80%"
                if idx.endswith("%"):
                    idx = float(idx[:-1]) / 100.0
                else:
                    idx = float(idx)
            if isinstance(idx, float):
                # Convert percentage to absolute index
                if not 0 <= idx <= 1:
                    raise ValueError(f"Percentage must be in range [0, 1], got {idx}")
                return int(idx * total)
            if isinstance(idx, int):
                # Absolute index
                if idx < 0:
                    # Support negative indexing
                    return total + idx
                return idx
            raise ValueError(f"Invalid index type: {type(idx)}")

        # Get current slice boundaries (these are absolute indices in original dataset)
        current_start = self._split_start_idx if self._split_start_idx is not None else 0
        current_end = self._split_end_idx

        if current_end is None:
            # Need to compute total length of original dataset
            if self.file_lengths:
                current_end = sum(self.file_lengths)
            else:
                # Must iterate to compute - create temp dataset without split
                temp_dataset = SimpleArrowIterableDataset(
                    self.arrow_files, self.file_lengths
                )
                temp_dataset._shuffled_files = self._shuffled_files
                temp_dataset._shuffled_lengths = self._shuffled_lengths
                current_end = len(temp_dataset)

        # Current slice has (current_end - current_start) examples
        current_length = current_end - current_start

        # Parse indices RELATIVE to current slice
        relative_start = parse_index(start, current_length) if start is not None else 0
        relative_end = (
            parse_index(end, current_length) if end is not None else current_length
        )

        # Validate relative indices
        if relative_start < 0 or relative_start > current_length:
            raise ValueError(
                f"Start index {relative_start} out of range [0, {current_length}]"
            )
        if relative_end < 0 or relative_end > current_length:
            raise ValueError(
                f"End index {relative_end} out of range [0, {current_length}]"
            )
        if relative_start >= relative_end:
            raise ValueError(
                f"Start index {relative_start} must be < end index {relative_end}"
            )

        # Convert to ABSOLUTE indices in original dataset space
        absolute_start = current_start + relative_start
        absolute_end = current_start + relative_end

        # Create copy with updated split boundaries (using absolute indices)
        # Note: Don't copy old shard boundaries - sharding should happen on the sliced dataset
        return self._copy(
            split_start_idx=absolute_start,
            split_end_idx=absolute_end,
            last_iter_epoch=None,  # Reset for new dataset instance
            # Don't copy counts - this is a different slice
            original_length=None,
            input_count=0,
            output_count=0,
            cached_exact_length=None,
            length_invalidated=False,
        )

    def shard(self, num_shards: int, index: int, mode: str = "auto"):
        """
        Shard the dataset (for DDP).

        Args:
            num_shards: Number of shards to split into
            index: Index of this shard (0 to num_shards-1)
            mode: Sharding mode:
                - 'auto': Use file-level if possible, otherwise example-level
                - 'file': File-level sharding (each shard gets subset of files)
                - 'example': Example-level sharding (divide examples evenly)

        File-level sharding:
            - More efficient (less overhead during iteration)
            - Requires num_shards <= num_files
            - Shard i gets files at indices i, i+num_shards, i+2*num_shards, ...

        Example-level sharding:
            - Works with any num_shards (even > num_files)
            - Divides total examples evenly across shards
            - Slightly more overhead during iteration
        """
        if index >= num_shards:
            raise ValueError(
                f"Shard index ({index}) must be less than num_shards ({num_shards})"
            )
        if index < 0:
            raise ValueError(f"Shard index must be non-negative, got {index}")

        # Determine sharding mode
        files = self._shuffled_files if self._shuffled_files else self.arrow_files
        num_files = len(files)

        if mode == "auto":
            # Automatically choose the best sharding mode
            # File-level sharding doesn't work correctly with virtual splits because
            # split boundaries apply across all files, so each shard would still see
            # all examples in the split range
            has_virtual_split = (
                self._split_start_idx is not None or self._split_end_idx is not None
            )

            if has_virtual_split:
                # Must use example-level sharding when split is active
                shard_mode = "example"
            elif num_shards <= num_files:
                # Can use efficient file-level sharding
                shard_mode = "file"
            else:
                # Too many shards for file-level, use example-level
                shard_mode = "example"
        else:
            shard_mode = mode

        # Prepare override kwargs based on sharding mode
        copy_kwargs = {
            "last_iter_epoch": None,  # Reset for new dataset instance
            # Don't copy counts - this is a different shard
            "original_length": None,
            "input_count": 0,
            "output_count": 0,
            "cached_exact_length": None,
            "length_invalidated": False,
        }

        if shard_mode == "file":
            # File-level sharding
            if num_shards > num_files:
                logger.warning(
                    f"File-level sharding with num_shards={num_shards} > num_files={num_files}. "
                    f"Some shards will be empty. Consider using mode='example'."
                )
            copy_kwargs["shard_config"] = (num_shards, index, "file")

        elif shard_mode == "example":
            # Example-level sharding - compute boundaries
            total_examples = len(self)  # Compute total examples
            examples_per_shard = total_examples // num_shards
            remainder = total_examples % num_shards

            # Distribute remainder examples to first 'remainder' shards
            if index < remainder:
                shard_start = index * (examples_per_shard + 1)
                shard_end = shard_start + examples_per_shard + 1
            else:
                shard_start = index * examples_per_shard + remainder
                shard_end = shard_start + examples_per_shard

            copy_kwargs["shard_config"] = (num_shards, index, "example")
            copy_kwargs["shard_start_idx"] = shard_start
            copy_kwargs["shard_end_idx"] = shard_end
        else:
            raise ValueError(
                f"Invalid shard mode: {mode}. Must be 'auto', 'file', or 'example'."
            )

        return self._copy(**copy_kwargs)

    def map(
        self,
        function=None,
        with_indices: bool = False,
        input_columns=None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns=None,
        fn_kwargs=None,
    ):
        """
        Apply a function to all examples (individually or in batches).

        The function is applied lazily during iteration, preserving our
        efficient checkpoint protocol.

        Args:
            function: Function to apply. If None, identity function is used.
            with_indices: Provide example indices to function
            input_columns: Specific columns to pass to function
            batched: Process examples in batches
            batch_size: Number of examples per batch if batched=True
            drop_last_batch: Drop incomplete final batch
            remove_columns: Columns to remove after mapping
            fn_kwargs: Keyword arguments for function

        Returns:
            New SimpleArrowIterableDataset with map applied
        """
        # Normalize arguments
        if isinstance(input_columns, str):
            input_columns = [input_columns]
        if isinstance(remove_columns, str):
            remove_columns = [remove_columns]
        if function is None:
            function = _identity_func
        if fn_kwargs is None:
            fn_kwargs = {}

        # Prepare copy kwargs with common settings
        copy_kwargs = {
            "last_iter_epoch": None,  # Reset for new dataset instance
            # Don't inherit counts/stats - those are specific to this dataset instance
            "original_length": None,
            "input_count": 0,
            "output_count": 0,
            "cached_exact_length": None,
            "length_invalidated": False,
        }

        # Handle function composition if there's already a map
        if self._map_function is not None:
            # Chain maps: apply existing map first, then new map
            prev_function = self._map_function
            prev_batched = self._map_batched
            prev_fn_kwargs = self._map_fn_kwargs or {}

            # For now, don't support chaining maps with different batched modes
            if prev_batched != batched:
                raise ValueError(
                    "Cannot chain maps with different batched modes. "
                    "Convert to same batched mode or use single map."
                )

            # Compose functions
            def composed_function(example, *args, **comp_kwargs):
                result = prev_function(example, **prev_fn_kwargs)
                return function(result, *args, **fn_kwargs)

            copy_kwargs["map_function"] = composed_function
            copy_kwargs["map_fn_kwargs"] = {}  # Already incorporated
        else:
            # First map operation
            copy_kwargs["map_function"] = function
            copy_kwargs["map_batched"] = batched
            copy_kwargs["map_batch_size"] = batch_size
            copy_kwargs["map_drop_last_batch"] = drop_last_batch
            copy_kwargs["map_remove_columns"] = remove_columns
            copy_kwargs["map_with_indices"] = with_indices
            copy_kwargs["map_input_columns"] = input_columns
            copy_kwargs["map_fn_kwargs"] = fn_kwargs

        return self._copy(**copy_kwargs)

    def _apply_map_to_example(self, example, example_idx):
        """
        Apply map function to a single example.

        Args:
            example: Dictionary of column values
            example_idx: Global example index (for with_indices)

        Returns:
            Transformed example dictionary
        """
        if self._map_function is None:
            return example

        # Handle input_columns: only pass specified columns to function
        if self._map_input_columns is not None:
            fn_input = {
                col: example[col] for col in self._map_input_columns if col in example
            }
        else:
            fn_input = example

        # Apply function
        fn_kwargs = self._map_fn_kwargs or {}
        if self._map_with_indices:
            result = self._map_function(fn_input, example_idx, **fn_kwargs)
        else:
            result = self._map_function(fn_input, **fn_kwargs)

        # If function returned None, filter out this example
        if result is None:
            return None
        elif not isinstance(result, dict):
            raise ValueError(
                f"Map function must return a dictionary or None, got {type(result)}"
            )

        # Merge result with original example (HF behavior)
        # Result columns overwrite original if same key exists
        final_result = example.copy()
        final_result.update(result)

        # Handle remove_columns: remove after merging
        if self._map_remove_columns is not None:
            for col in self._map_remove_columns:
                final_result.pop(col, None)  # Remove if exists

        return final_result

    def _apply_map_to_batch(self, examples, start_idx):
        """
        Apply map function to a batch of examples.

        Args:
            examples: List of example dictionaries
            start_idx: Starting global index for this batch (for with_indices)

        Returns:
            List of transformed example dictionaries (may be different length due to filtering)
        """
        if self._map_function is None or not examples:
            return examples

        # Convert list of examples to batch format (dict with list values)
        # Example: [{"text": "a"}, {"text": "b"}] -> {"text": ["a", "b"]}
        batch = {}
        if self._map_input_columns is not None:
            # Only include specified columns
            for col in self._map_input_columns:
                batch[col] = [ex.get(col) for ex in examples]
        else:
            # Include all columns
            all_columns = set()
            for ex in examples:
                all_columns.update(ex.keys())
            for col in all_columns:
                batch[col] = [ex.get(col) for ex in examples]

        # Apply function
        fn_kwargs = self._map_fn_kwargs or {}
        if self._map_with_indices:
            indices = list(range(start_idx, start_idx + len(examples)))
            result = self._map_function(batch, indices, **fn_kwargs)
        else:
            result = self._map_function(batch, **fn_kwargs)

        # Handle None result (filter all examples)
        if result is None:
            return []

        if not isinstance(result, dict):
            raise ValueError(
                f"Map function must return a dictionary or None, got {type(result)}"
            )

        # Validate batch map result structure
        if result:
            # Check that all values are lists of the same length or all scalars
            list_lengths = []
            has_scalars = False
            for col, values in result.items():
                if isinstance(values, list):
                    list_lengths.append(len(values))
                else:
                    has_scalars = True

            if list_lengths and has_scalars:
                # Mixed lists and scalars - this is allowed (scalars are broadcast)
                pass
            elif list_lengths:
                # All lists - check they have the same length
                if len(set(list_lengths)) > 1:
                    raise ValueError(
                        f"Batched map function returned lists of different lengths: "
                        f"{dict(zip(result.keys(), list_lengths))}. "
                        f"All lists must have the same length."
                    )

        # Convert batch format back to list of examples
        # Example: {"input_ids": [[1,2], [3,4]]} -> [{"input_ids": [1,2]}, {"input_ids": [3,4]}]
        result_examples = []

        # Determine number of output examples
        if result:
            first_key = next(iter(result.keys()))
            num_outputs = (
                len(result[first_key]) if isinstance(result[first_key], list) else 1
            )
        else:
            num_outputs = 0

        for i in range(num_outputs):
            result_example = {}
            for col, values in result.items():
                if isinstance(values, list):
                    result_example[col] = values[i]
                else:
                    # Scalar value, repeat for all examples
                    result_example[col] = values

            # Merge with original example if within range
            # (N->M mapping may produce more or fewer examples)
            if i < len(examples):
                final_example = examples[i].copy()
                final_example.update(result_example)
            else:
                final_example = result_example

            # Handle remove_columns
            if self._map_remove_columns is not None:
                for col in self._map_remove_columns:
                    final_example.pop(col, None)

            result_examples.append(final_example)

        return result_examples

    def _get_files_for_shard(self):
        """Get the Arrow files for this shard."""
        files = self._shuffled_files if self._shuffled_files else self.arrow_files

        if self._shard_config:
            shard_mode = (
                self._shard_config[2] if len(self._shard_config) >= 3 else "file"
            )

            if shard_mode == "file":
                num_shards, shard_index = self._shard_config[0], self._shard_config[1]
                # Distribute files: shard i gets files i, i+num_shards, i+2*num_shards, ...
                files = [
                    f for idx, f in enumerate(files) if idx % num_shards == shard_index
                ]
            elif shard_mode == "example":
                # Example-level sharding uses all files, filtering happens in __iter__
                pass

        return files

    @staticmethod
    def _get_worker_info():
        # Get worker info for multi-worker DataLoader support
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
        return worker_id, num_workers

    def _flush_batch_buffer(self, batch_buffer, batch_start_idx):
        """
        Process and yield any pending batch buffer.

        This helper is used to flush partial batches before early termination
        (e.g., when reaching split/shard boundaries).

        Args:
            batch_buffer: List of pending examples
            batch_start_idx: Starting index for the batch

        Yields:
            Processed examples from the batch
        """
        if not batch_buffer or self._map_drop_last_batch:
            return

        result_examples = self._apply_map_to_batch(batch_buffer, batch_start_idx)

        # Track counts in dynamic/exact mode
        # IMPORTANT: Count both inputs and outputs atomically before yielding
        # to avoid race conditions where counts are temporarily out of sync
        if self.length_estimate_mode in ("dynamic", "exact"):
            self._input_count += len(batch_buffer)
            self._output_count += len(result_examples)
            self._current_batch_buffer_size = 0  # Reset buffer size

        # Yield results
        for result_example in result_examples:
            yield result_example

    def _apply_shuffle_buffer(self, iterator, buffer_size, seed):
        """
        Apply shuffle buffer to an iterator of examples.

        This implements reservoir sampling-style shuffling:
        1. Fill buffer with first buffer_size examples
        2. For each new example, randomly replace an example in the buffer
        3. Yield the replaced example
        4. At the end, shuffle and yield remaining buffer contents

        Args:
            iterator: Iterator yielding examples
            buffer_size: Size of shuffle buffer
            seed: Random seed for reproducibility

        Yields:
            Shuffled examples
        """
        import random

        if buffer_size is None or buffer_size <= 0:
            # No shuffling, just pass through
            yield from iterator
            return

        # Initialize RNG with seed for reproducibility
        rng = random.Random(seed)

        # Fill initial buffer
        buffer = []
        for example in iterator:
            buffer.append(example)
            if len(buffer) >= buffer_size:
                break

        if not buffer:
            # Empty iterator
            return

        # Main shuffling loop
        for example in iterator:
            # Randomly select an index to replace
            idx = rng.randint(0, buffer_size - 1)
            # Yield the example being replaced
            yield buffer[idx]
            # Replace it with the new example
            buffer[idx] = example

        # Shuffle remaining buffer and yield all
        rng.shuffle(buffer)
        for example in buffer:
            yield example

    # ========== Helper Methods for _base_iter ==========

    def _prepare_iteration(self) -> tuple[int, int]:
        """
        Prepare iteration state: handle epoch changes and reset counts.

        Returns:
            (start_input_count, start_output_count): Starting counts for tracking this iteration
        """
        # Re-shuffle files if epoch changed
        if self._last_iter_epoch != self._epoch:
            effective_seed = self._get_effective_seed(fallback_seed=None)
            if effective_seed is not None:
                self._reshuffle_files_with_seed(effective_seed)
            self._last_iter_epoch = self._epoch

        # Reset counts if configured or if stats were invalidated
        if self._reset_length_on_iter or self._length_invalidated:
            # Keep ratio estimate as starting point if invalidated
            if not (
                self._length_invalidated
                and self._output_count > 0
                and self._input_count > 0
            ):
                # Reset completely if not preserving ratio
                self._input_count = 0
                self._output_count = 0
            self._cached_exact_length = None
            self._length_invalidated = False

        # Track starting counts for this iteration
        return self._input_count, self._output_count

    def _initialize_iteration_state(self) -> tuple[list[str], bool, int, int]:
        """
        Initialize iteration state and determine if fresh vs checkpoint restoration.

        Returns:
            (worker_files, is_fresh_iteration, worker_id, num_workers)
        """
        worker_id, num_workers = self._get_worker_info()
        files = self._get_files_for_shard()

        # Use all files - we'll shard at example level, not file level
        # This ensures consistent results regardless of num_workers
        worker_files = files

        # Determine if this is a fresh iteration vs checkpoint restoration
        # Fresh iteration: reset both position and counts
        # Checkpoint restoration: preserve both position and counts
        is_fresh_iteration = not self._restored_from_checkpoint

        if is_fresh_iteration:
            # Reset position to start for fresh iteration
            self._current_file_index = 0
            self._current_example_index = 0
            # Reset counts to track this iteration
            self._input_count = 0
            self._output_count = 0
            # Don't reset _cached_exact_length - it persists across iterations
        # else: checkpoint restoration - preserve position and counts

        # Clear the checkpoint restoration flag after first iteration starts
        self._restored_from_checkpoint = False

        return worker_files, is_fresh_iteration, worker_id, num_workers

    def _compute_iteration_boundaries(
        self, num_workers: int, worker_id: int
    ) -> tuple[int, int, bool, bool, bool]:
        """
        Compute iteration boundaries for splits, shards, and worker-level sharding.

        Args:
            num_workers: Total number of workers
            worker_id: Current worker ID

        Returns:
            (split_start, split_end, use_virtual_split, use_example_sharding, worker_sharding_enabled)
        """
        # Track global position for virtual splits and example-level sharding
        use_virtual_split = (
            self._split_start_idx is not None or self._split_end_idx is not None
        )
        use_example_sharding = self._shard_start_idx is not None

        # Get split boundaries (default to full dataset)
        split_start = self._split_start_idx if self._split_start_idx is not None else 0
        split_end = (
            self._split_end_idx if self._split_end_idx is not None else float("inf")
        )

        # Multi-worker sharding: shard at example level (not file level)
        # Following PyTorch's recommendation for IterableDataset
        # This ensures consistent results regardless of num_workers
        # Note: Only enable for num_workers >= 2 (single worker doesn't need sharding)
        worker_sharding_enabled = False
        if num_workers >= 2:
            # Compute total examples to shard (after split/shard filtering)
            if use_virtual_split:
                # Operating on a virtual split
                total_examples = split_end - split_start
                worker_split_start = split_start
            elif use_example_sharding:
                # Operating on an example-level shard
                total_examples = self._shard_end_idx - self._shard_start_idx
                worker_split_start = self._shard_start_idx
            else:
                # Operating on full dataset - need to compute total length
                total_examples = self._get_original_length()
                worker_split_start = 0

            # Compute per-worker range
            import math

            per_worker = int(math.ceil(total_examples / float(num_workers)))
            worker_start = worker_split_start + worker_id * per_worker
            worker_end = min(
                worker_start + per_worker, worker_split_start + total_examples
            )

            # Override split boundaries with worker-specific range
            split_start = worker_start
            split_end = worker_end
            use_virtual_split = True  # Enable filtering by these boundaries
            worker_sharding_enabled = True

        return split_start, split_end, use_virtual_split, use_example_sharding, worker_sharding_enabled

    def _get_file_length(self, file_idx: int, arrow_file: str) -> tuple[int, bool]:
        """
        Get length of Arrow file, using cache if available.

        Args:
            file_idx: Index of file in files list
            arrow_file: Path to Arrow file

        Returns:
            (file_len, file_len_cached): Length and whether it came from cache
        """
        if self._shuffled_lengths:
            return self._shuffled_lengths[file_idx], True
        elif self.file_lengths:
            return self.file_lengths[file_idx], True
        else:
            # No cached length - must load file to get it
            ds = Dataset.from_file(arrow_file)
            return len(ds), False

    def _skip_processed_file(
        self,
        file_idx: int,
        arrow_file: str,
        use_virtual_split: bool,
        use_example_sharding: bool
    ) -> tuple[bool, int]:
        """
        Check if file should be skipped and compute global index delta.

        Args:
            file_idx: Current file index
            arrow_file: Path to Arrow file
            use_virtual_split: Whether virtual split is enabled
            use_example_sharding: Whether example sharding is enabled

        Returns:
            (should_skip, global_idx_delta): Whether to skip and how much to advance global index
        """
        # Check checkpoint position dynamically (not captured as local var)
        # This allows load_state_dict() to work even if called after __iter__
        if file_idx < self._current_file_index:
            # Skip entire file, but track global position
            if use_virtual_split or use_example_sharding:
                # Use cached file lengths instead of loading file (FAST!)
                if self._shuffled_lengths:
                    file_len = self._shuffled_lengths[file_idx]
                elif self.file_lengths:
                    file_len = self.file_lengths[file_idx]
                else:
                    # Fallback: must load file to get length (rare case)
                    ds = Dataset.from_file(arrow_file)
                    file_len = len(ds)
                return True, file_len
            return True, 0
        return False, 0

    def _compute_file_range(
        self,
        file_idx: int,
        file_len: int,
        global_example_idx: int,
        split_start: int,
        split_end: int,
        use_virtual_split: bool,
        use_example_sharding: bool
    ) -> tuple[int, int, int]:
        """
        Compute the range of examples to read from this file.

        Handles:
        - Checkpoint resumption
        - Virtual split boundaries
        - Example-level sharding

        Args:
            file_idx: Current file index
            file_len: Length of the file
            global_example_idx: Global position before this file
            split_start, split_end: Split boundaries
            use_virtual_split, use_example_sharding: Flags

        Returns:
            (file_start_idx, file_end_idx, updated_global_idx):
                Range to read and updated global position
        """
        # Compute the range of examples we need from this file (for efficient random access)
        # This eliminates sequential seeking - we use .select() to jump directly to the range
        file_start_idx = 0  # Start of range to read from this file
        file_end_idx = file_len  # End of range (exclusive)

        # 1. Handle checkpoint resumption (skip examples before checkpoint position)
        if file_idx == self._current_file_index:
            # This is the file where we need to resume
            file_start_idx = max(file_start_idx, self._current_example_index)

        # 2. Handle virtual split and example-level sharding
        # These define windows in global example space that we need to intersect with this file
        if use_virtual_split or use_example_sharding:
            # Examples in this file span [global_example_idx, global_example_idx + file_len)
            file_global_start = global_example_idx
            file_global_end = global_example_idx + file_len

            # Compute the global range we want
            desired_global_start = split_start
            desired_global_end = split_end

            # If using example-level sharding, further restrict the range
            if use_example_sharding:
                # Shard boundaries are relative to split_start
                shard_global_start = split_start + self._shard_start_idx
                shard_global_end = split_start + self._shard_end_idx
                desired_global_start = max(desired_global_start, shard_global_start)
                desired_global_end = min(desired_global_end, shard_global_end)

            # Check if file intersects with desired range
            if file_global_end <= desired_global_start or file_global_start >= desired_global_end:
                # File is completely outside desired range - return empty range
                return file_start_idx, file_start_idx, global_example_idx + file_len

            # Compute intersection with desired range
            intersect_start = max(file_global_start, desired_global_start)
            intersect_end = min(file_global_end, desired_global_end)

            # Convert to file-local indices
            file_start_idx = max(file_start_idx, intersect_start - file_global_start)
            file_end_idx = min(file_end_idx, intersect_end - file_global_start)

        return file_start_idx, file_end_idx, global_example_idx

    def _load_file_range(
        self,
        arrow_file: str,
        file_len: int,
        file_start_idx: int,
        file_end_idx: int,
        file_len_cached: bool
    ) -> Dataset:
        """
        Load Arrow file and select the specified range of examples.

        Args:
            arrow_file: Path to Arrow file
            file_len: Length of the file
            file_start_idx: Start index in file
            file_end_idx: End index in file (exclusive)
            file_len_cached: Whether file length was cached (file not yet loaded)

        Returns:
            Dataset containing selected examples
        """
        # Load the Arrow file if we haven't already (deferred until after range check)
        if file_len_cached:
            ds = Dataset.from_file(arrow_file)
        else:
            # Already loaded during _get_file_length
            ds = Dataset.from_file(arrow_file)

        # Use .select() to efficiently extract just the range we need
        # This is fast - Arrow supports random access, no sequential iteration needed
        if file_start_idx > 0 or file_end_idx < file_len:
            if file_start_idx >= file_end_idx:
                # Return empty dataset
                return ds.select([])
            ds = ds.select(range(file_start_idx, file_end_idx))

        return ds

    def _handle_early_exit(
        self,
        reason: str,
        batch_buffer: list | None,
        batch_start_idx: int | None,
        worker_files: list[str]
    ):
        """
        Handle early exit from iteration (split/shard exhausted).

        Flushes any pending batch, caches length if appropriate,
        marks iteration as complete.

        Args:
            reason: Reason for exit ("split_exhausted", "shard_exhausted")
            batch_buffer: Pending batch buffer (if batched map)
            batch_start_idx: Starting index of batch
            worker_files: List of files being processed

        Yields:
            Any remaining buffered examples
        """
        # Flush pending batch buffer if needed
        if (
            self._map_batched
            and self._map_function is not None
            and batch_buffer
        ):
            yield from self._flush_batch_buffer(batch_buffer, batch_start_idx)

        # Cache exact length
        if self.length_estimate_mode in ("dynamic", "exact"):
            if self._cached_exact_length is None:
                self._cached_exact_length = self._output_count

        # Mark iteration as complete for next fresh iteration
        self._current_file_index = len(worker_files)
        self._current_example_index = 0

    def _track_counts(self, input_delta: int, output_delta: int) -> None:
        """
        Update input/output counts for length estimation.

        Only updates if length_estimate_mode is "dynamic" or "exact".

        Args:
            input_delta: Number of input examples to add
            output_delta: Number of output examples to add
        """
        if self.length_estimate_mode in ("dynamic", "exact"):
            self._input_count += input_delta
            self._output_count += output_delta

    # ========== Main Iteration Method ==========

    def _base_iter(self):
        """
        Base iteration without shuffle buffer.

        This is the core iteration logic that reads from Arrow files.
        The shuffle buffer (if enabled) wraps this iterator.

        Yields:
            Examples from the dataset
        """
        # Phase 1: Initialize iteration state
        start_input_count, start_output_count = self._prepare_iteration()
        worker_files, is_fresh_iteration, worker_id, num_workers = self._initialize_iteration_state()
        split_start, split_end, use_virtual_split, use_example_sharding, worker_sharding_enabled = \
            self._compute_iteration_boundaries(num_workers, worker_id)

        # Phase 2: Initialize batch buffer for batched map operations
        if self._map_batched and self._map_function is not None:
            batch_buffer = []
            batch_start_idx = None

        # Track global position for virtual splits and example-level sharding
        global_example_idx = 0

        # Phase 3: Main file iteration loop
        for file_idx, arrow_file in enumerate(worker_files):
            # Skip files already processed (checkpoint restoration)
            should_skip, global_idx_delta = self._skip_processed_file(
                file_idx, arrow_file, use_virtual_split, use_example_sharding
            )
            if should_skip:
                global_example_idx += global_idx_delta
                continue

            # Get file length and compute range to read
            file_len, file_len_cached = self._get_file_length(file_idx, arrow_file)
            file_start_idx, file_end_idx, global_example_idx = self._compute_file_range(
                file_idx, file_len, global_example_idx, split_start, split_end,
                use_virtual_split, use_example_sharding
            )

            # Skip if range is empty (file outside split boundaries)
            if file_start_idx >= file_end_idx:
                if use_virtual_split or use_example_sharding:
                    global_example_idx += file_len
                continue

            # Load file and select range
            ds = self._load_file_range(arrow_file, file_len, file_start_idx, file_end_idx, file_len_cached)

            # Phase 4: Iterate over examples in the file
            # Note: enumerate starts from file_start_idx to maintain correct checkpoint indices
            for local_idx, example in enumerate(ds, start=file_start_idx):
                # Check early exit conditions
                if use_virtual_split and global_example_idx >= split_end:
                    # Reached end of split
                    yield from self._handle_early_exit(
                        "split_exhausted",
                        batch_buffer if self._map_batched and self._map_function is not None else None,
                        batch_start_idx if self._map_batched and self._map_function is not None else None,
                        worker_files
                    )
                    return

                if use_example_sharding:
                    # Compute position relative to split start
                    relative_idx = global_example_idx - split_start
                    if relative_idx >= self._shard_end_idx:
                        # Reached end of shard
                        yield from self._handle_early_exit(
                            "shard_exhausted",
                            batch_buffer if self._map_batched and self._map_function is not None else None,
                            batch_start_idx if self._map_batched and self._map_function is not None else None,
                            worker_files
                        )
                        return

                # Save index for map with_indices (before incrementing)
                current_global_idx = global_example_idx

                if use_virtual_split or use_example_sharding:
                    global_example_idx += 1

                # Update position for checkpointing (for next checkpoint)
                self._current_file_index = file_idx
                self._current_example_index = local_idx + 1  # Next example to process

                # Apply map function if configured
                if self._map_function is not None:
                    if self._map_batched:
                        # Batched mode: collect examples into batch buffer
                        if batch_start_idx is None:
                            batch_start_idx = current_global_idx
                        batch_buffer.append(example)

                        # Track batch buffer size for length estimation
                        if self.length_estimate_mode in ("dynamic", "exact"):
                            self._current_batch_buffer_size = len(batch_buffer)

                        # Process batch when full
                        if len(batch_buffer) >= self._map_batch_size:
                            result_examples = self._apply_map_to_batch(
                                batch_buffer, batch_start_idx
                            )

                            # Track counts atomically before yielding
                            self._track_counts(len(batch_buffer), len(result_examples))
                            if self.length_estimate_mode in ("dynamic", "exact"):
                                self._current_batch_buffer_size = 0  # Reset buffer size

                            # Yield results
                            for result_example in result_examples:
                                yield result_example
                            batch_buffer = []
                            batch_start_idx = None
                    else:
                        # Non-batched mode: apply to individual example
                        self._track_counts(1, 0)  # Track input

                        example = self._apply_map_to_example(
                            example, current_global_idx
                        )
                        # If map function returned None, skip this example (filtering)
                        if example is None:
                            continue

                        # Track output and yield
                        self._track_counts(0, 1)  # Track output
                        yield example
                else:
                    # No map function, input = output
                    self._track_counts(1, 1)
                    yield example

            # After finishing a file, move to next file
            if file_idx >= self._current_file_index:
                self._current_example_index = 0
                self._current_file_index = file_idx + 1

        # Phase 5: Finalization - flush pending batch and cache length
        if self._map_batched and self._map_function is not None and batch_buffer:
            # Flush final partial batch
            yield from self._flush_batch_buffer(batch_buffer, batch_start_idx)

        # Cache exact length after complete iteration (in dynamic/exact mode only)
        if self.length_estimate_mode in ("dynamic", "exact"):
            if self._cached_exact_length is None and self._output_count > 0:
                self._cached_exact_length = self._output_count
            self._current_batch_buffer_size = 0  # Clear buffer tracking

        # Mark iteration as complete so next iteration starts fresh
        # ONLY if we actually processed some examples in this iteration
        # This prevents the alternating 9/0/9/0 pattern when SynchronizedDataLoader
        # stops iteration immediately (before any examples are processed)
        processed_any_examples = self._output_count > start_output_count
        if worker_files and processed_any_examples:
            self._current_file_index = len(worker_files)
            self._current_example_index = 0

    def __iter__(self):
        """
        Iterate through Arrow files sequentially with optional shuffle buffer.

        Supports:
        - File-level and example-level sharding
        - Multi-worker DataLoader (each worker gets a subset of files)
        - Checkpoint resumption (skips to saved position)
        - Example-level shuffle buffer (if configured via .shuffle(buffer_size=N))

        For StatefulDataLoader compatibility, each worker instance tracks
        its own position independently.

        IMPORTANT: We check self._current_file_index and self._current_example_index
        dynamically (not captured in local vars) so that if load_state_dict() is
        called after __iter__ but before iteration starts, we use the restored values.

        Note:
            When shuffle buffer is enabled and checkpoint resumption occurs, the shuffle
            pattern after resumption will differ from a non-interrupted run. However,
            randomness is still maintained and the seed ensures reproducibility of the
            overall shuffle behavior.
        """
        # Apply shuffle buffer if configured
        if self._shuffle_buffer_size is not None and self._shuffle_buffer_size > 0:
            # Compute effective seed for shuffle buffer (accounts for epoch)
            effective_seed = self._get_effective_seed(
                fallback_seed=self._shuffle_seed or 0
            )

            # Wrap base iterator with shuffle buffer
            yield from self._apply_shuffle_buffer(
                self._base_iter(),
                self._shuffle_buffer_size,
                effective_seed  # Use effective seed
            )
        else:
            # No shuffle buffer, use base iterator directly
            yield from self._base_iter()

    def __len__(self) -> int:
        """
        Get total number of examples in the dataset.

        For mapped datasets with N->M transformations, returns:
        - Progressive estimate during first iteration (dynamic mode)
        - Exact cached count after first complete iteration (dynamic mode)
        - Original pre-map length (static mode)
        """
        # Static mode: always return original length
        if self.length_estimate_mode == "static":
            return self._get_original_length()

        # Multi-worker synced length: use if available (takes precedence)
        # This is set by sync_dataset_state_from_dataloader() when aggregating from workers
        if (
            hasattr(self, "_synced_cached_length")
            and self._synced_cached_length is not None
        ):
            return self._synced_cached_length

        # Dynamic/exact mode: use cached exact length if available
        if self._cached_exact_length is not None:
            return self._cached_exact_length

        # Dynamic mode: estimate from observed ratio
        if self._output_count > 0 and self._input_count > 0:
            original = self._get_original_length()
            ratio = self._output_count / self._input_count
            estimated = int(original * ratio)
            return estimated

        # No data yet, return original length
        return self._get_original_length()

    def _get_original_length(self) -> int:
        """
        Compute and cache original (pre-map) length.

        Accounts for virtual splits and sharding:
        - Virtual split: Returns slice size
        - Example-level sharding: Returns shard size within split
        - File-level sharding: Returns total across assigned files within split
        """
        # Lazy computation - only calculate once
        if self._original_length is not None:
            return self._original_length

        # For example-level sharding, length is determined by shard boundaries
        # (which are relative to the split if one exists)
        if self._shard_start_idx is not None and self._shard_end_idx is not None:
            self._original_length = self._shard_end_idx - self._shard_start_idx
            return self._original_length

        # Count examples in assigned files
        files = self._get_files_for_shard()

        # Use cached file lengths if available (much faster!)
        if self.file_lengths is not None:
            # Get lengths for the files we're using
            lengths = (
                self._shuffled_lengths if self._shuffled_lengths else self.file_lengths
            )

            # For file-level sharding, sum lengths of assigned files
            if self._shard_config and self._shard_config[2] == "file":
                num_shards, shard_index = self._shard_config[0], self._shard_config[1]
                total = sum(
                    lengths[idx]
                    for idx in range(len(lengths))
                    if idx % num_shards == shard_index
                )
            else:
                total = sum(lengths)
        else:
            # Fallback: memory-map each file (slower)
            total = 0
            for arrow_file in files:
                ds = Dataset.from_file(arrow_file)
                total += len(ds)

        # Apply virtual split if present
        if self._split_start_idx is not None or self._split_end_idx is not None:
            split_start = (
                self._split_start_idx if self._split_start_idx is not None else 0
            )
            split_end = (
                self._split_end_idx if self._split_end_idx is not None else total
            )
            self._original_length = split_end - split_start
        else:
            self._original_length = total

        return self._original_length

    @property
    def column_names(self) -> List[str]:
        """
        Get dataset column names from Arrow schema.

        This is a standard HuggingFace Datasets attribute commonly used
        for operations like map(remove_columns=dataset.column_names).

        Returns:
            List of column names (e.g., ['text', 'id', 'meta'])
        """
        if self._column_names is None:
            if not self.arrow_files:
                return []

            # Load first Arrow file to get schema (fast - only metadata)
            ds = Dataset.from_file(self.arrow_files[0])
            self._column_names = ds.column_names

        return self._column_names

    @property
    def features(self):
        """
        Get dataset features (schema with type information).

        This is a standard HuggingFace Datasets attribute that provides
        detailed schema information including column types, nested structures,
        and encoding information.

        Returns:
            DatasetFeatures object describing the schema
        """
        if self._features is None:
            if not self.arrow_files:
                return None

            # Load first Arrow file to get features (fast - only metadata)
            ds = Dataset.from_file(self.arrow_files[0])
            self._features = ds.features

        return self._features

    @property
    def n_shards(self) -> int:
        """
        Number of Arrow file shards in the dataset.

        This reflects the actual number of Arrow files available,
        accounting for shuffling if applied.

        Returns:
            Number of shards (Arrow files)
        """
        files = self._shuffled_files if self._shuffled_files else self.arrow_files
        return len(files)

    def state_dict(self) -> Dict[str, Any]:
        """
        Get checkpoint state.

        For efficiency, this doesn't store the arrow_files list or file_lengths,
        which can be thousands of entries. Instead, it stores a checksum to verify
        dataset identity on restore. Similarly, shuffled arrays are reproduced
        from the shuffle seed rather than stored.

        Returns:
            Dictionary containing:
            - current_file_index: Which Arrow file we're currently in
            - current_example_index: Which example within that file
            - shuffle_seed: Random seed used for shuffling (if any)
            - shard_config: Sharding configuration (if any)
            - dataset_fingerprint: Hash of arrow_files for verification
            - num_files: Number of arrow files (for validation)
            - split_start_idx: Start index for virtual split (if any)
            - split_end_idx: End index for virtual split (if any)
            - shard_start_idx: Start index for example-level sharding (if any)
            - shard_end_idx: End index for example-level sharding (if any)
            - length_estimate_mode: Length estimation mode
            - reset_length_on_iter: Whether to reset counts on iteration
            - original_length: Cached original (pre-map) length
            - input_count: Examples consumed so far
            - output_count: Examples yielded so far
            - cached_exact_length: Exact count after full iteration
            - length_invalidated: Whether stats were invalidated
        """
        import hashlib

        # Create fingerprint of dataset (hash of file paths)
        # This verifies we're restoring to the same dataset
        file_list_str = "\n".join(self.arrow_files)
        fingerprint = hashlib.sha256(file_list_str.encode()).hexdigest()

        return {
            "current_file_index": self._current_file_index,
            "current_example_index": self._current_example_index,
            "shuffle_seed": self._shuffle_seed,  # Current effective seed
            "base_shuffle_seed": self._base_shuffle_seed,  # Base seed for epoch-based shuffling
            "epoch": self._epoch,  # Current epoch
            "shuffle_buffer_size": self._shuffle_buffer_size,
            "shard_config": self._shard_config,
            # Instead of storing thousands of file paths, store a checksum
            "dataset_fingerprint": fingerprint,
            "num_files": len(self.arrow_files),
            # No need to store shuffled arrays - we can reproduce from seed
            "split_start_idx": self._split_start_idx,
            "split_end_idx": self._split_end_idx,
            "shard_start_idx": self._shard_start_idx,
            "shard_end_idx": self._shard_end_idx,
            # Length estimation state
            "length_estimate_mode": self.length_estimate_mode,
            "reset_length_on_iter": self._reset_length_on_iter,
            "original_length": self._original_length,
            "input_count": self._input_count,
            "output_count": self._output_count,
            "cached_exact_length": self._cached_exact_length,
            "length_invalidated": self._length_invalidated,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Restore from checkpoint state.

        Args:
            state_dict: Dictionary from a previous state_dict() call (per-worker)

        This allows efficient resumption - the iterator will skip to the
        saved position without having to iterate through all previous examples.

        Note: With multi-worker DataLoader, this is called once per worker with
        that worker's specific state.
        """
        import hashlib
        import random

        self._current_file_index = state_dict["current_file_index"]
        self._current_example_index = state_dict["current_example_index"]
        self._shuffle_seed = state_dict.get("shuffle_seed")
        self._base_shuffle_seed = state_dict.get("base_shuffle_seed")  # Restore base seed
        self._epoch = state_dict.get("epoch", 0)  # Default to 0 for old checkpoints
        self._shuffle_buffer_size = state_dict.get("shuffle_buffer_size")
        self._shard_config = state_dict.get("shard_config")

        # Verify dataset fingerprint matches
        file_list_str = "\n".join(self.arrow_files)
        fingerprint = hashlib.sha256(file_list_str.encode()).hexdigest()
        saved_fingerprint = state_dict.get("dataset_fingerprint")

        if saved_fingerprint and fingerprint != saved_fingerprint:
            raise ValueError(
                f"Dataset fingerprint mismatch! Checkpoint was created with a different dataset. "
                f"Expected {saved_fingerprint}, got {fingerprint}. "
                f"This usually means the dataset files have changed."
            )

        # Verify number of files matches
        saved_num_files = state_dict.get("num_files")
        if saved_num_files and len(self.arrow_files) != saved_num_files:
            raise ValueError(
                f"Number of files mismatch! Checkpoint has {saved_num_files} files, "
                f"but current dataset has {len(self.arrow_files)} files."
            )

        # Reconstruct shuffled arrays from seed if shuffle was used
        if self._shuffle_seed is not None:
            if self.file_lengths:
                # Shuffle files and lengths together
                paired = list(zip(self.arrow_files, self.file_lengths))
                rng = random.Random(self._shuffle_seed)
                rng.shuffle(paired)
                shuffled_files, shuffled_lengths = zip(*paired)
                self._shuffled_files = list(shuffled_files)
                self._shuffled_lengths = list(shuffled_lengths)
            else:
                self._shuffled_files = self.arrow_files.copy()
                self._shuffled_lengths = None
                rng = random.Random(self._shuffle_seed)
                rng.shuffle(self._shuffled_files)
        else:
            self._shuffled_files = None
            self._shuffled_lengths = None

        self._split_start_idx = state_dict.get("split_start_idx")
        self._split_end_idx = state_dict.get("split_end_idx")
        self._shard_start_idx = state_dict.get("shard_start_idx")
        self._shard_end_idx = state_dict.get("shard_end_idx")

        # Restore length estimation state
        self.length_estimate_mode = state_dict.get("length_estimate_mode", "dynamic")
        self._reset_length_on_iter = state_dict.get("reset_length_on_iter", False)
        self._original_length = state_dict.get("original_length")
        self._input_count = state_dict.get("input_count", 0)
        self._output_count = state_dict.get("output_count", 0)
        self._cached_exact_length = state_dict.get("cached_exact_length")
        self._length_invalidated = state_dict.get("length_invalidated", False)

        # Set last_iter_epoch to match restored epoch so we don't re-shuffle
        # The files were already shuffled correctly when we reconstructed them above
        self._last_iter_epoch = self._epoch

        # Mark that we just restored from checkpoint
        # This prevents the next iteration from resetting counts
        self._restored_from_checkpoint = True

    def get_length_stats(self) -> Dict[str, Any]:
        """
        Get current length estimation statistics.

        Returns:
            Dictionary with:
            - mode: Current estimation mode
            - original_length: Pre-map length (may be None if not computed)
            - input_count: Examples consumed so far
            - output_count: Examples yielded so far
            - ratio: Current input/output ratio (if data available)
            - cached_exact: Cached exact length (if available)
            - current_estimate: What __len__() would return right now
            - batch_buffer_size: Current pending batch buffer size
            - checkpoint_position: Current file and example indices
        """
        stats = {
            "mode": self.length_estimate_mode,
            "original_length": self._original_length,
            "input_count": self._input_count,
            "output_count": self._output_count,
            "cached_exact": self._cached_exact_length,
            "invalidated": self._length_invalidated,
            "batch_buffer_size": self._current_batch_buffer_size,
            "checkpoint_file_idx": self._current_file_index,
            "checkpoint_example_idx": self._current_example_index,
            "reset_on_iter": self._reset_length_on_iter,
        }

        if self._output_count > 0 and self._input_count > 0:
            stats["ratio"] = self._output_count / self._input_count
        else:
            stats["ratio"] = None

        stats["current_estimate"] = len(self)

        return stats

    def set_length_estimate_mode(self, mode: str):
        """
        Change length estimation mode.

        Args:
            mode: 'static', 'dynamic', or 'exact'
        """
        if mode not in ("static", "dynamic", "exact"):
            raise ValueError(f"Invalid mode: {mode}")
        self.length_estimate_mode = mode

    def to_hf_iterable(self):
        """Convert to HuggingFace IterableDataset for full compatibility."""

        def gen():
            for example in self:
                yield example

        # The wrapped length may not represent the true length, which depends upon the implementation
        # of the map function, but certain APIs, like torch.Dataloader, expect the dataset to have a
        # __len__(), when their own length is queried.
        return IterableDatasetWithLength(
            HFIterableDataset.from_generator(gen), len(self)
        )


class FastDatasetLoaderSimple:
    """
    Fast dataset loader using simple generator approach.
    """

    def __init__(self, index_dir: Optional[str] = None):
        if index_dir is None:
            index_dir = os.path.expanduser("~/.cache/fast_hf_indexes_simple")

        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def _get_config_hash(
        self,
        path: str,
        name: Optional[str] = None,
        split: Optional[str] = None,
        data_files: Optional[Union[str, list]] = None,
        revision: Optional[str] = None,
        **kwargs,
    ) -> str:
        config = {
            "path": path,
            "name": name,
            "split": split,
            "data_files": data_files,
            "revision": revision,
        }
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _get_index_file(self, config_hash: str) -> Path:
        return self.index_dir / f"{config_hash}.json"

    def _get_arrow_files(self, dataset_obj: Dataset) -> Optional[list]:
        """Get Arrow file paths from dataset."""
        if hasattr(dataset_obj, "cache_files") and dataset_obj.cache_files:
            return [cf["filename"] for cf in dataset_obj.cache_files]
        if hasattr(dataset_obj, "_data_files") and dataset_obj._data_files:
            return [df["filename"] for df in dataset_obj._data_files]
        return None

    def _get_file_lengths_from_metadata(
        self, arrow_files: List[str], split: str
    ) -> Optional[List[int]]:
        """
        Try to extract file lengths from HuggingFace's dataset_info.json.

        This avoids opening each Arrow file individually to read metadata.
        For datasets with thousands of files, this is significantly faster.

        Args:
            arrow_files: List of Arrow file paths
            split: Split name (e.g., 'train', 'validation')

        Returns:
            List of file lengths if found and validated, None otherwise
        """
        if not arrow_files:
            return None

        try:
            # All Arrow files should be in the same cache directory
            cache_dir = Path(arrow_files[0]).parent

            # Look for dataset_info.json in the cache directory
            dataset_info_path = cache_dir / "dataset_info.json"
            if not dataset_info_path.exists():
                return None

            # Parse the metadata file
            with open(dataset_info_path, "r") as f:
                dataset_info = json.load(f)

            # Extract shard_lengths for this split
            splits = dataset_info.get("splits", {})
            if split not in splits:
                return None

            shard_lengths = splits[split].get("shard_lengths", [])

            # Validate that the number of shards matches number of Arrow files
            if len(shard_lengths) != len(arrow_files):
                logger.warning(
                    f"Shard count mismatch: dataset_info.json has {len(shard_lengths)} shards, "
                    f"but found {len(arrow_files)} Arrow files. Falling back to file-by-file indexing."
                )
                return None

            # Validate that it's not empty
            if not shard_lengths:
                return None

            logger.info(
                f"Loaded file lengths from dataset_info.json: {len(shard_lengths)} files, "
                f"{sum(shard_lengths):,} total examples"
            )
            return shard_lengths

        except Exception as e:
            # If anything goes wrong, fall back to reading files individually
            logger.debug(
                f"Could not load file lengths from dataset_info.json: {e}. "
                f"Falling back to file-by-file indexing."
            )
            return None

    def _is_saved_dataset_path(self, path: str) -> bool:
        """
        Check if path is a local directory containing a saved dataset.

        Saved datasets (from save_to_disk()) have this structure:
            path/
            ├── dataset_dict.json     # {"splits": ["train", ...]}
            └── train/
                ├── state.json        # {"_data_files": [...]}
                ├── dataset_info.json
                └── data-*.arrow

        Or for single-split datasets:
            path/
            ├── state.json
            ├── dataset_info.json
            └── data-*.arrow
        """
        if not path:
            return False

        dataset_path = Path(path)
        if not dataset_path.is_dir():
            return False

        # Check for multi-split format (dataset_dict.json)
        if (dataset_path / "dataset_dict.json").exists():
            return True

        # Check for single-split format (state.json in root)
        if (dataset_path / "state.json").exists():
            return True

        return False

    def _load_saved_dataset(
        self,
        path: str,
        split: str,
        force_reindex: bool = False,
        length_estimate: str = "dynamic",
        reset_length_on_iter: bool = False,
    ) -> Optional["SimpleArrowIterableDataset"]:
        """
        Load a saved dataset directly from disk, bypassing load_from_disk().

        Args:
            path: Path to saved dataset directory
            split: Split to load (e.g., 'train')
            force_reindex: Force reindexing even if cached
            length_estimate: Length estimation mode
            reset_length_on_iter: Reset length counts on each iteration

        Returns:
            SimpleArrowIterableDataset if successful, None otherwise
        """
        dataset_path = Path(path)

        # Determine split directory
        dataset_dict_path = dataset_path / "dataset_dict.json"
        if dataset_dict_path.exists():
            # Multi-split format
            with open(dataset_dict_path, "r") as f:
                dataset_dict = json.load(f)

            available_splits = dataset_dict.get("splits", [])
            if split not in available_splits:
                logger.warning(
                    f"Split '{split}' not found in saved dataset. "
                    f"Available splits: {available_splits}"
                )
                return None

            split_dir = dataset_path / split
        else:
            # Single-split format (state.json in root)
            split_dir = dataset_path

        # Read state.json to get data files
        state_path = split_dir / "state.json"
        if not state_path.exists():
            logger.warning(f"state.json not found in {split_dir}")
            return None

        with open(state_path, "r") as f:
            state = json.load(f)

        data_files = state.get("_data_files", [])
        if not data_files:
            logger.warning(f"No data files listed in {state_path}")
            return None

        # Build full paths to Arrow files
        arrow_files = [
            str(split_dir / df["filename"])
            for df in data_files
            if df.get("filename", "").endswith(".arrow")
        ]

        if not arrow_files:
            logger.warning(f"No Arrow files found in {split_dir}")
            return None

        # Verify files exist
        missing = [f for f in arrow_files if not Path(f).exists()]
        if missing:
            logger.warning(f"Missing Arrow files: {missing[:5]}...")
            return None

        num_files = len(arrow_files)
        logger.info(f"Found saved dataset with {num_files} Arrow file(s)")

        # Check for cached index
        config_hash = self._get_config_hash(path, split=split)
        if not force_reindex:
            index_data = self._load_index(config_hash)
            if index_data is not None:
                cached_files = index_data.get("arrow_files", [])
                if cached_files == arrow_files:
                    logger.info("Loading from cached index")
                    file_lengths = index_data.get("file_lengths")
                    iterable_ds = SimpleArrowIterableDataset(arrow_files, file_lengths)
                    iterable_ds.length_estimate_mode = length_estimate
                    iterable_ds._reset_length_on_iter = reset_length_on_iter
                    return iterable_ds

        # Get file lengths - try dataset_info.json first (if it has shard_lengths)
        file_lengths = self._get_file_lengths_from_metadata(arrow_files, split)

        if file_lengths is None:
            # Fall back to reading each Arrow file
            logger.info("Computing per-file example counts...")
            file_lengths = []

            use_progress = HAS_TQDM and sys.stderr.isatty()
            iterator = (
                tqdm(arrow_files, desc="Indexing files", unit="file")
                if use_progress
                else arrow_files
            )

            for arrow_file in iterator:
                ds_file = Dataset.from_file(arrow_file)
                file_lengths.append(len(ds_file))

        total_examples = sum(file_lengths)
        logger.info(f"Total examples: {total_examples:,}")

        # Save index for future use
        metadata = {
            "dataset_path": path,
            "split": split,
            "source": "saved_dataset",
            "num_arrow_files": num_files,
            "total_examples": total_examples,
        }
        self._save_index(config_hash, arrow_files, file_lengths, metadata)

        # Create dataset
        iterable_ds = SimpleArrowIterableDataset(arrow_files, file_lengths)
        iterable_ds.length_estimate_mode = length_estimate
        iterable_ds._reset_length_on_iter = reset_length_on_iter

        return iterable_ds

    def _save_index(
        self,
        config_hash: str,
        arrow_files: list,
        file_lengths: list,
        metadata: Dict[str, Any],
    ):
        index_data = {
            "version": METADATA_VERSION,
            "arrow_files": arrow_files,
            "file_lengths": file_lengths,
            "metadata": metadata,
            "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        index_file = self._get_index_file(config_hash)
        with open(index_file, "w") as f:
            json.dump(index_data, f, indent=2)

    def _load_index(self, config_hash: str) -> Optional[Dict[str, Any]]:
        index_file = self._get_index_file(config_hash)
        if not index_file.exists():
            return None

        with open(index_file, "r") as f:
            index_data = json.load(f)

        # Check version - force reindex if mismatch
        stored_version = index_data.get("version", 1)  # Default to v1 if missing
        if stored_version != METADATA_VERSION:
            logger.info(
                f"Index version mismatch (stored: v{stored_version}, current: v{METADATA_VERSION}). "
                f"Forcing reindex..."
            )
            return None

        return index_data

    def load_iterable(
        self,
        path: str,
        name: Optional[str] = None,
        split: Optional[str] = None,
        data_files: Optional[Union[str, list]] = None,
        revision: Optional[str] = None,
        force_reindex: bool = False,
        num_proc: Optional[int] = None,
        length_estimate: str = "dynamic",
        reset_length_on_iter: bool = False,
        **load_dataset_kwargs,
    ):
        """
        Load dataset as IterableDataset with instant loading after first time.

        Supports both HuggingFace Hub datasets and local saved datasets (from save_to_disk()).
        For Hub datasets, supports HuggingFace split notation like "train[10000:]" without
        triggering reindexing.

        Args:
            path: Either a HuggingFace Hub dataset path (e.g., 'allenai/c4')
                  or a local directory path to a saved dataset
            split: Split to load (e.g., 'train', 'validation')
            length_estimate: How to estimate length for mapped datasets
                'static': Never change from original (safest)
                'dynamic': Progressive estimate, lock after full iteration (default)
                'exact': Alias for 'dynamic' (kept for clarity)
            reset_length_on_iter: If True, reset counts on each new iteration.
                                   If False (default), preserve estimates across iterations.
        """
        # Check if path is a local saved dataset directory
        if self._is_saved_dataset_path(path):
            logger.info(f"Detected saved dataset at: {path}")
            # Parse split notation for saved datasets too
            base_split, slice_start, slice_end = (
                _parse_split_notation(split) if split else (split, None, None)
            )
            # Default to 'train' if no split specified
            effective_split = base_split or "train"

            result = self._load_saved_dataset(
                path=path,
                split=effective_split,
                force_reindex=force_reindex,
                length_estimate=length_estimate,
                reset_length_on_iter=reset_length_on_iter,
            )
            if result is not None:
                # Apply slice if present
                if slice_start is not None or slice_end is not None:
                    result = result.slice(slice_start, slice_end)
                return result
            else:
                logger.warning(
                    "Failed to load saved dataset, falling back to load_from_disk"
                )
                # Fall through to try load_from_disk via load_dataset

        # Parse split notation (e.g., "train[10000:]" → "train", 10000, None)
        base_split, slice_start, slice_end = (
            _parse_split_notation(split) if split else (split, None, None)
        )

        # Use base split for config hash (so "train" and "train[10000:]" share cache)
        config_hash = self._get_config_hash(
            path, name, base_split, data_files, revision
        )
        index_data = self._load_index(config_hash) if not force_reindex else None

        if index_data is not None:
            # Fast path: create iterable from indexed Arrow files
            arrow_files = index_data["arrow_files"]
            file_lengths = index_data.get("file_lengths")  # May be None for old indices
            metadata = index_data["metadata"]

            if all(Path(f).exists() for f in arrow_files):
                start_time = time.time()

                logger.debug(f"Dataset: {path}" + (f"/{name}" if name else ""))
                if split:
                    logger.debug(f"Split: {split}")

                # Create simple iterable dataset (INSTANT!)
                iterable_ds = SimpleArrowIterableDataset(arrow_files, file_lengths)

                # Set length estimation configuration
                iterable_ds.length_estimate_mode = length_estimate
                iterable_ds._reset_length_on_iter = reset_length_on_iter

                # Apply slice if present in split notation
                if slice_start is not None or slice_end is not None:
                    iterable_ds = iterable_ds.slice(slice_start, slice_end)

                elapsed = time.time() - start_time

                logger.debug(
                    f"Loaded as IterableDataset in {elapsed:.3f}s "
                    f"Arrow files: {len(arrow_files)} (natural shards)"
                )

                return iterable_ds

            else:
                logger.warning("Arrow files missing. Re-indexing...")

        # Slow path: initial load
        logger.info(
            f"{'Re-indexing' if index_data else 'First-time indexing'} dataset..."
        )
        logger.info(f"Dataset: {path}" + (f"/{name}" if name else ""))
        logger.info("This will be slow, but only happens once...")

        start_time = time.time()

        # Load with base split only (download full split, we'll slice virtually)
        ds = load_dataset(
            path,
            name=name,
            split=base_split,
            data_files=data_files,
            revision=revision,
            num_proc=num_proc,
            **load_dataset_kwargs,
        )

        load_time = time.time() - start_time
        logger.info(f"Dataset loaded in {load_time:.1f}s")

        # Get Arrow files
        arrow_files = self._get_arrow_files(ds)

        if arrow_files:
            num_files = len(arrow_files)
            logger.info(f"Found {num_files} Arrow file(s) in HF cache")

            # Compute per-file example counts
            # First try to load from dataset_info.json (fast path for HF datasets)
            file_lengths = self._get_file_lengths_from_metadata(arrow_files, base_split)

            if file_lengths is None:
                # Fall back to reading each Arrow file individually
                logger.info("Computing per-file example counts...")
                file_lengths = []

                # Use tqdm progress bar if available and connected to TTY
                use_progress = HAS_TQDM and sys.stderr.isatty()
                iterator = (
                    tqdm(arrow_files, desc="Indexing files", unit="file")
                    if use_progress
                    else arrow_files
                )

                for arrow_file in iterator:
                    ds_file = Dataset.from_file(arrow_file)
                    file_lengths.append(len(ds_file))

            total_examples = sum(file_lengths)
            logger.info(f"Total examples: {total_examples:,}")

            metadata = {
                "dataset_path": path,
                "dataset_name": name,
                "split": base_split,  # Store base split for cache consistency
                "load_time": load_time,
                "num_arrow_files": num_files,
                "total_examples": total_examples,
            }

            self._save_index(config_hash, arrow_files, file_lengths, metadata)

            total_size = sum(Path(f).stat().st_size for f in arrow_files)
            size_gb = total_size / (1024**3)

            logger.info(
                f"Index saved: {num_files} Arrow files = {num_files} natural shards, Data size: {size_gb:.2f} GB"
            )

            # Create dataset and apply slice if present
            iterable_ds = SimpleArrowIterableDataset(arrow_files, file_lengths)

            # Set length estimation configuration
            iterable_ds.length_estimate_mode = length_estimate
            iterable_ds._reset_length_on_iter = reset_length_on_iter

            if slice_start is not None or slice_end is not None:
                iterable_ds = iterable_ds.slice(slice_start, slice_end)

            return iterable_ds

        else:
            logger.warning("Could not find Arrow files")
            # Fallback: use regular to_iterable_dataset
            result_ds = ds.to_iterable_dataset(num_shards=1)
            # Note: Slice cannot be applied to regular iterable dataset fallback
            # User should ensure dataset can be indexed for slice support
            return result_ds


# Global instance
_default_loader = None


def get_default_loader() -> FastDatasetLoaderSimple:
    global _default_loader
    if _default_loader is None:
        _default_loader = FastDatasetLoaderSimple()
    return _default_loader


def fast_load_iterable_dataset(
    path: str,
    name: Optional[str] = None,
    split: Optional[str] = None,
    data_files: Optional[Union[str, list]] = None,
    revision: Optional[str] = None,
    force_reindex: bool = False,
    num_proc: Optional[int] = None,
    index_dir: Optional[str] = None,
    length_estimate: str = "dynamic",
    reset_length_on_iter: bool = False,
    **load_dataset_kwargs,
):
    """
    Fast loading as IterableDataset with proper sharding support.

    Supports both HuggingFace Hub datasets and local saved datasets (from save_to_disk()).
    First call: Slow (indexes Arrow files), subsequent calls: Instant.

    Returns a SimpleArrowIterableDataset that:
    - Loads instantly (just reads file paths from index)
    - Supports .shuffle(seed) for shard-level shuffling
    - Supports .shard(num_shards, index) for DDP
    - Supports .map(fn) for lazy transformations
    - Each Arrow file = 1 natural shard

    Args:
        path: Either a HuggingFace Hub dataset path (e.g., 'allenai/c4') or
              a local directory path to a saved dataset (from save_to_disk())
        name: Dataset configuration name (e.g., "wikitext-2-raw-v1")
        split: Split to load (e.g., "train", "train[:1000]")
        data_files: Specific data files to load
        revision: Dataset revision/version
        force_reindex: Force rebuilding the Arrow file index
        num_proc: Number of processes for indexing
        index_dir: Custom directory for storing index files
        length_estimate: How to estimate length for mapped datasets
            'static': Never change from original (safest)
            'dynamic': Progressive estimate, lock after full iteration (default)
            'exact': Alias for 'dynamic' (kept for clarity)
        reset_length_on_iter: If True, reset counts on each new iteration.
                               If False (default), preserve estimates across iterations.
        **load_dataset_kwargs: Additional kwargs passed to load_dataset()

    Example:
        # Load (instant after first time!)
        ids = fast_load_iterable_dataset("dataset", "config", split="train")

        # Shard-level shuffling
        ids = ids.shuffle(seed=42)

        # For DDP
        ids = ids.shard(num_shards=world_size, index=rank)

        # Lazy map
        ids = ids.map(tokenize)

        # Iterate
        for example in ids:
            pass
    """
    if index_dir is not None:
        loader = FastDatasetLoaderSimple(index_dir=index_dir)
    else:
        loader = get_default_loader()

    return loader.load_iterable(
        path=path,
        name=name,
        split=split,
        data_files=data_files,
        revision=revision,
        force_reindex=force_reindex,
        num_proc=num_proc,
        length_estimate=length_estimate,
        reset_length_on_iter=reset_length_on_iter,
        **load_dataset_kwargs,
    )
