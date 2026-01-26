# Fast HuggingFace Dataset Loader with Stateful Checkpointing

## Overview

`fast_hf_loader_simple.py` provides instant dataset loading and efficient mid-epoch checkpointing for large HuggingFace datasets.

## Performance

**Loading Speed:**
- First load: ~20 minutes (indexes Arrow files)
- Subsequent loads: <1 second (instant)
- Your dataset: 234 Arrow files, ~958GB

**Checkpoint Resumption:**
- Without stateful: Hours to iterate through millions of examples
- With stateful: <1 second to resume from any position

## Usage

### Basic Loading

```python
from forgather.ml.datasets import fast_load_iterable_dataset

# Load dataset (instant after first time)
ids = fast_load_iterable_dataset(
    "HuggingFaceTB/smollm-corpus",
    "fineweb-edu-dedup",
    split="train"
)

# Shuffle at shard level (234 Arrow files)
ids = ids.shuffle(seed=42)

# Optional: Shard for DDP
ids = ids.shard(num_shards=world_size, index=rank)

# Lazy transformations
ids = ids.map(tokenize_function)

# Iterate
for example in ids:
    # Training code
    ...
```

### Virtual Splits (Train/Val/Test)

Create train/val/test splits without copying data:

```python
from fast_hf_loader_simple import fast_load_iterable_dataset

# Load full dataset
ids = fast_load_iterable_dataset("dataset", "config", split="train")

# Create train/val/test splits (70%/15%/15%)
train_ds = ids.slice(None, 0.7)        # First 70%
val_ds = ids.slice(0.7, 0.85)          # Next 15%
test_ds = ids.slice(0.85, None)        # Last 15%

# Or using percentage strings
train_ds = ids.slice(None, "80%")      # First 80%
val_ds = ids.slice("80%", None)        # Last 20%

# Or absolute indices
subset = ids.slice(1000, 2000)         # Examples 1000-1999

# Or using HuggingFace-style select() with indices
subset = ids.select(range(1000, 2000)) # Examples 1000-1999
first_100 = ids.select(range(100))     # First 100 examples

# Combine with shuffling and sharding
train_ds = ids.shuffle(seed=42).slice(None, 0.8)
train_shard = train_ds.shard(num_shards=world_size, index=rank)
```

**Benefits:**
- No data duplication - virtual slicing is zero-copy
- Works with shuffling and sharding
- Compatible with checkpoint resumption
- Flexible indexing: percentages, absolute indices, or percentage strings

### With StatefulDataLoader Checkpointing

```python
from fast_hf_loader_simple import fast_load_iterable_dataset
from torchdata.stateful_dataloader import StatefulDataLoader

# Load and prepare dataset
ids = fast_load_iterable_dataset("dataset", "config", split="train")
ids = ids.shuffle(seed=42)
ids = ids.map(tokenize)

# Create dataloader (works with any num_workers)
dataloader = StatefulDataLoader(
    ids,
    batch_size=32,
    num_workers=4  # Use workers to keep GPU fed
)

# Training loop
for step, batch in enumerate(dataloader):
    # Forward/backward pass
    loss = model(batch)
    loss.backward()
    optimizer.step()

    # Save checkpoint periodically
    if step % 1000 == 0:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'dataloader': dataloader.state_dict(),  # Includes dataset state
            'step': step
        }
        torch.save(checkpoint, f'checkpoint_{step}.pt')

# Resume from checkpoint
checkpoint = torch.load('checkpoint_5000.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

# Create NEW dataloader and restore
ids = fast_load_iterable_dataset("dataset", "config", split="train")
ids = ids.shuffle(seed=42)
ids = ids.map(tokenize)

dataloader = StatefulDataLoader(ids, batch_size=32, num_workers=4)
dataloader.load_state_dict(checkpoint['dataloader'])

# Continue training from exact position
for step, batch in enumerate(dataloader, start=checkpoint['step']+1):
    ...
```

## Features

✅ **Instant Loading**
- First load indexes Arrow file paths
- Subsequent loads are instant (<1 second)
- No data copying - memory-maps HF cache

✅ **Shard-Level Shuffling**
- Shuffles Arrow file order for randomization
- More efficient than example-level shuffling
- 234 Arrow files = 234 natural shards

✅ **Virtual Splits**
- Create train/val/test splits without copying data
- Support percentage (0.8 or "80%") and absolute indexing
- Zero-copy slicing for memory efficiency
- Compatible with shuffling, sharding, and checkpointing

✅ **Flexible Sharding for DDP**
- File-level sharding: Each rank gets different Arrow files (efficient)
- Example-level sharding: Works even with more ranks than files
- Auto mode automatically selects the best approach
- No data duplication across ranks

✅ **Stateful Checkpointing**
- Implements `state_dict()`/`load_state_dict()` protocol
- Compatible with `torchdata.stateful_dataloader.StatefulDataLoader`
- Efficient resumption without iteration
- **Works with any num_workers value** (0, 1, 2, 4, 8, etc.)

✅ **Multi-Worker Support**
- Workers prefetch batches in parallel with training
- Keeps GPU fed during data loading
- Each worker gets a disjoint subset of Arrow files
- No data duplication across workers

## How It Works

### Indexing (First Load Only)

1. Calls `load_dataset()` to load dataset normally
2. Extracts Arrow file paths from HF cache
3. Computes per-file example counts (for fast `len()` later)
4. Saves file paths, lengths, and metadata version to index
5. Returns `SimpleArrowIterableDataset`

### Instant Loading (Subsequent Loads)

1. Reads index and checks metadata version
2. If version mismatch, automatically reindexes (ensures compatibility)
3. Creates `SimpleArrowIterableDataset` with cached file lengths
4. `len()` uses cached counts - no file access needed!
5. Memory-maps Arrow files on demand during iteration (zero-copy)

### Checkpointing

1. Tracks position as `(file_index, example_index)`
2. `state_dict()` returns current position (~1 KB)
3. `load_state_dict()` restores position
4. Iterator skips directly to saved position (no iteration needed)

### Worker Distribution

With 234 Arrow files and `num_workers=4`:
- Worker 0: files 0, 4, 8, 12, ... (~59 files)
- Worker 1: files 1, 5, 9, 13, ... (~59 files)
- Worker 2: files 2, 6, 10, 14, ... (~58 files)
- Worker 3: files 3, 7, 11, 15, ... (~58 files)

Each worker independently tracks its position for checkpointing.

### Flexible Sharding

The dataset supports two sharding modes for DDP training:

**File-Level Sharding (default when num_shards ≤ num_files):**
- Most efficient - minimal overhead during iteration
- Each shard gets a subset of Arrow files
- Shard i gets files at indices i, i+num_shards, i+2*num_shards, ...
- Example with 8 files and 2 shards:
  - Shard 0: files 0, 2, 4, 6
  - Shard 1: files 1, 3, 5, 7

**Example-Level Sharding (used when num_shards > num_files):**
- Works with any number of shards, even more than files
- Divides total examples evenly across shards
- Slightly more overhead during iteration (tracks global position)
- Example with 1 file (1000 examples) and 2 shards:
  - Shard 0: examples 0-499
  - Shard 1: examples 500-999

**Auto Mode (recommended):**
```python
# Automatically selects best mode
ids = ids.shard(num_shards=world_size, index=rank, mode='auto')

# Auto uses file-level if num_shards <= num_files
# Auto uses example-level if num_shards > num_files
```

**Explicit Mode Selection:**
```python
# Force file-level sharding
ids = ids.shard(num_shards=2, index=0, mode='file')

# Force example-level sharding
ids = ids.shard(num_shards=4, index=0, mode='example')
```

**Use Cases:**
- Training on 2 GPUs with 1-file dataset → Example-level required
- Training on 8 GPUs with 234-file dataset → File-level more efficient
- Auto mode handles both cases automatically

## Testing

Comprehensive tests verify correctness:

```bash
# Test instant loading and basic functionality
python fast_hf_loader_simple.py

# Test checkpoint functionality (num_workers=0, 1, 2)
python test_checkpoint_final.py

# Test specific checkpoint scenarios
python test_correct_comparison.py
```

All tests pass, confirming:
- ✅ Instant loading works
- ✅ Shuffling works
- ✅ Virtual splits work (percentage, absolute, train/val/test)
- ✅ File-level and example-level sharding work
- ✅ Auto mode selects appropriate sharding
- ✅ Multi-worker works
- ✅ Checkpointing works with num_workers=0, 1, 2, etc.
- ✅ Checkpointing works with both sharding modes and virtual splits

## API Reference

### `fast_load_iterable_dataset(...)`

Main loading function.

**Parameters:**
- `path`: Dataset path (e.g., "HuggingFaceTB/smollm-corpus")
- `name`: Config name (e.g., "fineweb-edu-dedup")
- `split`: Split to load (e.g., "train")
- `data_files`: Optional data files pattern
- `revision`: Optional git revision
- `force_reindex`: Force re-indexing (default: False)
- `num_proc`: Processes for initial indexing
- `index_dir`: Custom index directory
- `**load_dataset_kwargs`: Additional args for `load_dataset()`

**Returns:** `SimpleArrowIterableDataset`

### `SimpleArrowIterableDataset`

Iterable dataset with checkpointing support.

**Methods:**
- `.shuffle(seed=None)`: Shuffle Arrow file order
- `.select(indices)`: Select examples by indices (HuggingFace API compatible)
  - `indices`: Range, list, iterable, ndarray, or Series of integer indices
  - Supports only contiguous ranges (efficiently translates to `.slice()`)
  - Example: `ds.select(range(100, 200))` selects examples 100-199
- `.slice(start, end)`: Create virtual split (train/val/test)
  - `start`: Start index - int, float (percentage), string ("80%"), or None
  - `end`: End index - int, float (percentage), string ("80%"), or None
  - Returns new dataset with virtual split applied
- `.shard(num_shards, index, mode='auto')`: Shard for DDP
  - `mode='auto'`: Auto-select file or example-level sharding
  - `mode='file'`: File-level sharding (efficient, requires num_shards ≤ num_files)
  - `mode='example'`: Example-level sharding (works with any num_shards)
- `.map(function, batched=False)`: Lazy transformations
- `.__len__()`: Total examples (cached, accounts for splits and sharding)
- `.state_dict()`: Get checkpoint state
- `.load_state_dict(state_dict)`: Restore from checkpoint

## Recommendations

1. **Use num_workers≥1 for training**
   - Workers prefetch batches in parallel
   - Prevents GPU starvation
   - Recommended: `num_workers=4` for most cases

2. **Checkpoint frequently**
   - Every 1000-10000 steps
   - Keep multiple checkpoints

3. **For DDP training**
   - Each rank loads the same dataset
   - Use `.shard(num_shards=world_size, index=rank)`
   - Each rank gets different Arrow files

4. **Test checkpoint resume early**
   - Verify resumption works in your setup
   - Ensures you won't lose progress

## Example: Full Training Script

See `CHECKPOINT_GUIDE.md` for complete training examples with:
- Basic checkpoint save/restore
- DDP training
- Multi-worker DataLoader
- Production-ready patterns

## Architecture

- **No copying**: Memory-maps Arrow files from HF cache
- **Cached file lengths**: Per-file example counts cached in index for O(1) `len()`
- **Metadata versioning**: Automatic reindex when format changes
- **No expensive operations**: File access only during iteration, not for metadata
- **Minimal state**: Checkpoint is ~1 KB
- **Standard protocol**: Compatible with PyTorch ecosystem

## Summary

This implementation provides:
- 20 minutes → <1 second dataset loading
- Hours → <1 second checkpoint resumption
- Virtual splits for train/val/test without data copying
- Flexible sharding (file-level or example-level)
- Full DDP and multi-worker support
- Production-ready for long-running training jobs
- Compatible with standard PyTorch tools

Perfect for training on large datasets (100GB+) where:
- Dataset loading is a bottleneck
- Long training runs need checkpointing
- Mid-epoch resumption is critical
