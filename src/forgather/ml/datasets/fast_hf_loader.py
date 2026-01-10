"""
Fast HuggingFace Dataset Loader - Simple Generator Approach

Uses a simple generator that reads Arrow files sequentially.
No expensive interleave_datasets() calls. Just a simple, fast generator.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from datasets import load_dataset, Dataset
from datasets import IterableDataset as HFIterableDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
import torch.utils.data
import logging

logger = logging.getLogger(__name__)


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

    def __init__(self, arrow_files: List[str]):
        self.arrow_files = arrow_files
        self._shuffled_files = None
        self._shard_config = None  # (num_shards, shard_index, shard_mode)

        # Checkpoint state
        self._current_file_index = 0
        self._current_example_index = 0
        self._shuffle_seed = None
        self._total_examples = None  # Cached length

        # Virtual split boundaries (global example indices, before sharding)
        self._split_start_idx = None  # Inclusive
        self._split_end_idx = None    # Exclusive

        # Example-level sharding boundaries (global example indices, after split)
        self._shard_start_idx = None  # Inclusive
        self._shard_end_idx = None    # Exclusive

    def shuffle(self, seed: Optional[int] = None, buffer_size: int = 1000):
        """
        Shuffle at the Arrow file level (shard-level shuffling).
        """
        import random

        # Shuffle Arrow file order
        shuffled_files = self.arrow_files.copy()
        rng = random.Random(seed)
        rng.shuffle(shuffled_files)

        # Create new instance with shuffled files
        new_dataset = SimpleArrowIterableDataset(shuffled_files)
        new_dataset._shard_config = self._shard_config
        new_dataset._shuffle_seed = seed
        new_dataset._shuffled_files = shuffled_files
        new_dataset._split_start_idx = self._split_start_idx
        new_dataset._split_end_idx = self._split_end_idx
        return new_dataset

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
                if idx.endswith('%'):
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

        # Get total examples (before any splitting)
        # We need the base total, not the current split total
        if self._split_start_idx is not None or self._split_end_idx is not None:
            # Already has a split, get the original total
            # This is tricky - we need to temporarily clear the split
            temp_dataset = SimpleArrowIterableDataset(self.arrow_files)
            temp_dataset._shuffled_files = self._shuffled_files
            total_examples = len(temp_dataset)
        else:
            total_examples = len(self)

        # Parse indices
        start_idx = parse_index(start, total_examples) if start is not None else 0
        end_idx = parse_index(end, total_examples) if end is not None else total_examples

        # Validate
        if start_idx < 0 or start_idx > total_examples:
            raise ValueError(f"Start index {start_idx} out of range [0, {total_examples}]")
        if end_idx < 0 or end_idx > total_examples:
            raise ValueError(f"End index {end_idx} out of range [0, {total_examples}]")
        if start_idx >= end_idx:
            raise ValueError(f"Start index {start_idx} must be < end index {end_idx}")

        # Create new dataset with split
        new_dataset = SimpleArrowIterableDataset(self.arrow_files)
        new_dataset._shuffled_files = self._shuffled_files
        new_dataset._shuffle_seed = self._shuffle_seed
        new_dataset._shard_config = self._shard_config
        new_dataset._split_start_idx = start_idx
        new_dataset._split_end_idx = end_idx

        # Note: Don't copy old shard boundaries - sharding should happen on the sliced dataset
        return new_dataset

    def shard(self, num_shards: int, index: int, mode: str = 'auto'):
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
        new_dataset = SimpleArrowIterableDataset(self.arrow_files)
        new_dataset._shuffled_files = self._shuffled_files
        new_dataset._shuffle_seed = self._shuffle_seed

        # Determine sharding mode
        files = self._shuffled_files if self._shuffled_files else self.arrow_files
        num_files = len(files)

        if mode == 'auto':
            # Use file-level if we have enough files, otherwise example-level
            shard_mode = 'file' if num_shards <= num_files else 'example'
        else:
            shard_mode = mode

        if shard_mode == 'file':
            # File-level sharding
            if num_shards > num_files:
                logger.warning(
                    f"File-level sharding with num_shards={num_shards} > num_files={num_files}. "
                    f"Some shards will be empty. Consider using mode='example'."
                )
            new_dataset._shard_config = (num_shards, index, 'file')

        elif shard_mode == 'example':
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

            new_dataset._shard_config = (num_shards, index, 'example')
            new_dataset._shard_start_idx = shard_start
            new_dataset._shard_end_idx = shard_end
        else:
            raise ValueError(f"Invalid shard mode: {mode}. Must be 'auto', 'file', or 'example'.")

        return new_dataset

    def map(self, function, batched: bool = False, **kwargs):
        """
        Apply a map function (lazy - returns new iterable).
        """
        # Convert to HF IterableDataset for proper map support
        hf_iterable = self._to_hf_iterable()
        return hf_iterable.map(function, batched=batched, **kwargs)

    def _get_files_for_shard(self):
        """Get the Arrow files for this shard."""
        files = self._shuffled_files if self._shuffled_files else self.arrow_files

        if self._shard_config:
            shard_mode = self._shard_config[2] if len(self._shard_config) >= 3 else 'file'

            if shard_mode == 'file':
                num_shards, shard_index = self._shard_config[0], self._shard_config[1]
                # Distribute files: shard i gets files i, i+num_shards, i+2*num_shards, ...
                files = [f for idx, f in enumerate(files) if idx % num_shards == shard_index]
            elif shard_mode == 'example':
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
    
    def __iter__(self):
        """
        Iterate through Arrow files sequentially.

        Supports:
        - File-level and example-level sharding
        - Multi-worker DataLoader (each worker gets a subset of files)
        - Checkpoint resumption (skips to saved position)

        For StatefulDataLoader compatibility, each worker instance tracks
        its own position independently.

        IMPORTANT: We check self._current_file_index and self._current_example_index
        dynamically (not captured in local vars) so that if load_state_dict() is
        called after __iter__ but before iteration starts, we use the restored values.
        """

        worker_id, num_workers = self._get_worker_info()
        files = self._get_files_for_shard()

        # Further shard files among workers (for file-level sharding and multi-worker)
        # Worker i processes files: i, i+num_workers, i+2*num_workers, ...
        worker_files = [f for idx, f in enumerate(files) if idx % num_workers == worker_id]

        # Track global position for virtual splits and example-level sharding
        global_example_idx = 0
        use_virtual_split = self._split_start_idx is not None or self._split_end_idx is not None
        use_example_sharding = self._shard_start_idx is not None

        # Get split boundaries (default to full dataset)
        split_start = self._split_start_idx if self._split_start_idx is not None else 0
        split_end = self._split_end_idx if self._split_end_idx is not None else float('inf')

        for file_idx, arrow_file in enumerate(worker_files):
            # Check checkpoint position dynamically (not captured as local var)
            # This allows load_state_dict() to work even if called after __iter__
            if file_idx < self._current_file_index:
                # Skip entire file, but track global position
                if use_virtual_split or use_example_sharding:
                    ds = Dataset.from_file(arrow_file)
                    global_example_idx += len(ds)
                continue

            # Memory-map Arrow file
            ds = Dataset.from_file(arrow_file)

            for example_idx, example in enumerate(ds):
                # Skip examples before checkpoint (only for resume file)
                # Check dynamically so load_state_dict() updates are seen
                if file_idx == self._current_file_index and example_idx < self._current_example_index:
                    if use_virtual_split or use_example_sharding:
                        global_example_idx += 1
                    continue

                # Check virtual split boundaries first
                if use_virtual_split:
                    if global_example_idx < split_start:
                        global_example_idx += 1
                        continue
                    if global_example_idx >= split_end:
                        # Reached end of split, stop iteration
                        return

                # For example-level sharding, check relative position within split
                if use_example_sharding:
                    # Compute position relative to split start
                    relative_idx = global_example_idx - split_start

                    if relative_idx < self._shard_start_idx:
                        global_example_idx += 1
                        continue
                    if relative_idx >= self._shard_end_idx:
                        # Reached end of shard, stop iteration
                        return

                if use_virtual_split or use_example_sharding:
                    global_example_idx += 1

                # Update position for checkpointing (for next checkpoint)
                self._current_file_index = file_idx
                self._current_example_index = example_idx + 1  # Next example to process

                yield example

            # After finishing a file, move to next file
            if file_idx >= self._current_file_index:
                self._current_example_index = 0
                self._current_file_index = file_idx + 1

    def __len__(self) -> int:
        """
        Get total number of examples in the dataset.

        Accounts for virtual splits and sharding:
        - Virtual split: Returns slice size
        - Example-level sharding: Returns shard size within split
        - File-level sharding: Returns total across assigned files within split

        Caches the result for efficiency.
        """
        if self._total_examples is not None:
            return self._total_examples

        # For example-level sharding, length is determined by shard boundaries
        # (which are relative to the split if one exists)
        if self._shard_start_idx is not None and self._shard_end_idx is not None:
            self._total_examples = self._shard_end_idx - self._shard_start_idx
            return self._total_examples

        # Count examples in assigned files
        files = self._get_files_for_shard()
        total = 0

        for arrow_file in files:
            ds = Dataset.from_file(arrow_file)
            total += len(ds)

        # Apply virtual split if present
        if self._split_start_idx is not None or self._split_end_idx is not None:
            split_start = self._split_start_idx if self._split_start_idx is not None else 0
            split_end = self._split_end_idx if self._split_end_idx is not None else total
            self._total_examples = split_end - split_start
        else:
            self._total_examples = total

        return self._total_examples

    def state_dict(self) -> Dict[str, Any]:
        """
        Get checkpoint state.

        Returns:
            Dictionary containing:
            - current_file_index: Which Arrow file we're currently in
            - current_example_index: Which example within that file
            - shuffle_seed: Random seed used for shuffling (if any)
            - shard_config: Sharding configuration (if any)
            - arrow_files: List of Arrow file paths
            - shuffled_files: Shuffled file order (if shuffled)
            - split_start_idx: Start index for virtual split (if any)
            - split_end_idx: End index for virtual split (if any)
            - shard_start_idx: Start index for example-level sharding (if any)
            - shard_end_idx: End index for example-level sharding (if any)
        """
        return {
            "current_file_index": self._current_file_index,
            "current_example_index": self._current_example_index,
            "shuffle_seed": self._shuffle_seed,
            "shard_config": self._shard_config,
            "arrow_files": self.arrow_files,
            "shuffled_files": self._shuffled_files,
            "split_start_idx": self._split_start_idx,
            "split_end_idx": self._split_end_idx,
            "shard_start_idx": self._shard_start_idx,
            "shard_end_idx": self._shard_end_idx,
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
        self._current_file_index = state_dict["current_file_index"]
        self._current_example_index = state_dict["current_example_index"]
        self._shuffle_seed = state_dict.get("shuffle_seed")
        self._shard_config = state_dict.get("shard_config")
        self.arrow_files = state_dict["arrow_files"]
        self._shuffled_files = state_dict.get("shuffled_files")
        self._split_start_idx = state_dict.get("split_start_idx")
        self._split_end_idx = state_dict.get("split_end_idx")
        self._shard_start_idx = state_dict.get("shard_start_idx")
        self._shard_end_idx = state_dict.get("shard_end_idx")

        # Reset cached length since configuration might have changed
        self._total_examples = None


    def _to_hf_iterable(self):
        """Convert to HuggingFace IterableDataset for full compatibility."""
        def gen():
            for example in self:
                yield example

        return HFIterableDataset.from_generator(gen)


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
        **kwargs
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
        if hasattr(dataset_obj, 'cache_files') and dataset_obj.cache_files:
            return [cf['filename'] for cf in dataset_obj.cache_files]
        if hasattr(dataset_obj, '_data_files') and dataset_obj._data_files:
            return [df['filename'] for df in dataset_obj._data_files]
        return None

    def _save_index(self, config_hash: str, arrow_files: list, metadata: Dict[str, Any]):
        index_data = {
            "arrow_files": arrow_files,
            "metadata": metadata,
            "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        index_file = self._get_index_file(config_hash)
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)

    def _load_index(self, config_hash: str) -> Optional[Dict[str, Any]]:
        index_file = self._get_index_file(config_hash)
        if not index_file.exists():
            return None
        with open(index_file, 'r') as f:
            return json.load(f)

    def load_iterable(
        self,
        path: str,
        name: Optional[str] = None,
        split: Optional[str] = None,
        data_files: Optional[Union[str, list]] = None,
        revision: Optional[str] = None,
        force_reindex: bool = False,
        num_proc: Optional[int] = None,
        **load_dataset_kwargs
    ):
        """
        Load dataset as IterableDataset with instant loading after first time.
        """
        config_hash = self._get_config_hash(path, name, split, data_files, revision)
        index_data = self._load_index(config_hash) if not force_reindex else None

        if index_data is not None:
            # Fast path: create iterable from indexed Arrow files
            arrow_files = index_data['arrow_files']
            metadata = index_data['metadata']

            if all(Path(f).exists() for f in arrow_files):
                start_time = time.time()

                print(f"Dataset: {path}" + (f"/{name}" if name else ""))
                if split:
                    print(f"Split: {split}")

                # Create simple iterable dataset (INSTANT!)
                iterable_ds = SimpleArrowIterableDataset(arrow_files)

                elapsed = time.time() - start_time

                print(f"✓ Loaded as IterableDataset in {elapsed:.3f}s")
                print(f"  Arrow files: {len(arrow_files)} (natural shards)")
                print(f"  Use .shuffle(seed=42) for shard-level shuffling")
                print(f"  Use .shard(num_shards=N, index=i) for DDP")
                print(f"  Use .map(fn) for lazy transformations")
                print(f"  (Zero-copy - memory-mapped from HF cache)")

                return iterable_ds

            else:
                logger.warning("Arrow files missing. Re-indexing...")

        # Slow path: initial load
        print(f"{'Re-indexing' if index_data else 'First-time indexing'} dataset...")
        print(f"Dataset: {path}" + (f"/{name}" if name else ""))
        print(f"This will be slow, but only happens once...")

        start_time = time.time()

        ds = load_dataset(
            path,
            name=name,
            split=split,
            data_files=data_files,
            revision=revision,
            num_proc=num_proc,
            **load_dataset_kwargs
        )

        load_time = time.time() - start_time
        print(f"✓ Dataset loaded in {load_time:.1f}s")

        # Get Arrow files
        arrow_files = self._get_arrow_files(ds)

        if arrow_files:
            num_files = len(arrow_files)
            print(f"✓ Found {num_files} Arrow file(s) in HF cache")

            metadata = {
                "dataset_path": path,
                "dataset_name": name,
                "split": split,
                "load_time": load_time,
                "num_arrow_files": num_files,
            }

            self._save_index(config_hash, arrow_files, metadata)

            print(f"✓ Index saved")
            print(f"  Next load will be instant!")
            print(f"  {num_files} Arrow files = {num_files} natural shards")

            total_size = sum(Path(f).stat().st_size for f in arrow_files)
            size_gb = total_size / (1024**3)
            print(f"  Data size: {size_gb:.2f} GB")

            # Return as simple iterable (INSTANT!)
            return SimpleArrowIterableDataset(arrow_files)

        else:
            logger.warning("Could not find Arrow files")
            # Fallback: use regular to_iterable_dataset
            return ds.to_iterable_dataset(num_shards=1)


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
    **load_dataset_kwargs
):
    """
    Fast loading as IterableDataset with proper sharding support.

    First call: Slow (indexes Arrow files), subsequent calls: Instant.

    Returns a SimpleArrowIterableDataset that:
    - Loads instantly (just reads file paths from index)
    - Supports .shuffle(seed) for shard-level shuffling
    - Supports .shard(num_shards, index) for DDP
    - Supports .map(fn) for lazy transformations
    - Each Arrow file = 1 natural shard

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
        **load_dataset_kwargs
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("Fast HF Dataset Loader - Simple Generator Approach")
    print("="*80)

    # Demo
    print("\nDemo: Instant loading with simple generator")

    ids = fast_load_iterable_dataset(
        "wikitext",
        name="wikitext-2-raw-v1",
        split="train"
    )

    print(f"\n✓ Dataset loaded")
    print(f"\nTesting shuffle...")
    ids_shuffled = ids.shuffle(seed=42)

    print(f"\nIterating first 3 examples:")
    for i, example in enumerate(ids_shuffled):
        if i < 3:
            print(f"  Example {i}: {example['text'][:60]}...")
        else:
            break

    print("\n✓ Simple and fast!")
