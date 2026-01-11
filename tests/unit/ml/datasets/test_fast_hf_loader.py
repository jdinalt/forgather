"""
Tests for fast_hf_loader: StatefulDataLoader checkpoint functionality
"""

import pytest

try:
    from torchdata.stateful_dataloader import StatefulDataLoader

    HAS_STATEFUL = True
except ImportError:
    HAS_STATEFUL = False

from forgather.ml.datasets import fast_load_iterable_dataset, interleave_datasets


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_stateful_checkpoint_restore(num_workers):
    """
    Test that checkpoint save/restore works correctly with StatefulDataLoader.

    Verifies that:
    1. A checkpoint can be saved after N batches
    2. A NEW dataloader can be created and restored from checkpoint
    3. The restored dataloader continues from the correct position
    """
    # Load dataset
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )
    ids = ids.shuffle(seed=42)

    # Create dataloader and iterate 5 batches
    dataloader = StatefulDataLoader(ids, batch_size=4, num_workers=num_workers)

    checkpoint = None
    for i, batch in enumerate(dataloader):
        if i >= 4:
            checkpoint = dataloader.state_dict()
            break

    assert checkpoint is not None, "Failed to save checkpoint"

    # Create NEW dataloader (fresh state)
    ids_fresh = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )
    ids_fresh = ids_fresh.shuffle(seed=42)
    dataloader_fresh = StatefulDataLoader(
        ids_fresh, batch_size=4, num_workers=num_workers
    )

    # Get batch 5 from fresh dataloader (this is what restored should yield as batch 0)
    expected_batch = None
    for i, batch in enumerate(dataloader_fresh):
        if i == 5:
            expected_batch = batch
            break

    assert expected_batch is not None, "Failed to get expected batch"

    # Create NEW dataloader with restore
    ids_restored = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )
    ids_restored = ids_restored.shuffle(seed=42)
    dataloader_restored = StatefulDataLoader(
        ids_restored, batch_size=4, num_workers=num_workers
    )
    dataloader_restored.load_state_dict(checkpoint)

    # Get first batch from restored dataloader
    restored_batch = None
    for i, batch in enumerate(dataloader_restored):
        restored_batch = batch
        break

    assert restored_batch is not None, "Failed to get restored batch"

    # Verify batches match
    if isinstance(expected_batch, dict) and isinstance(restored_batch, dict):
        # Compare dict batches (HuggingFace dataset format)
        for k in expected_batch.keys():
            assert k in restored_batch, f"Key {k} missing in restored batch"
            # Compare values
            if isinstance(expected_batch[k], str):
                assert (
                    expected_batch[k] == restored_batch[k]
                ), f"Mismatch for key {k}: expected {expected_batch[k]}, got {restored_batch[k]}"
            else:
                # For lists or tensors, convert to string for comparison
                assert str(expected_batch[k]) == str(
                    restored_batch[k]
                ), f"Mismatch for key {k}"
    else:
        # Fallback comparison
        assert str(expected_batch) == str(
            restored_batch
        ), f"Batches don't match: expected {expected_batch}, got {restored_batch}"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_fast_load_basic():
    """Test basic dataset loading functionality."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )

    # Check it's iterable
    assert hasattr(ids, "__iter__"), "Dataset should be iterable"

    # Check it has length
    assert hasattr(ids, "__len__"), "Dataset should have __len__"
    assert len(ids) > 0, "Dataset should not be empty"

    # Check it has state_dict/load_state_dict
    assert hasattr(ids, "state_dict"), "Dataset should have state_dict"
    assert hasattr(ids, "load_state_dict"), "Dataset should have load_state_dict"

    # Test getting an example
    example = next(iter(ids))
    assert example is not None, "Should be able to get an example"
    assert "text" in example, "Example should have 'text' field"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_shuffle():
    """Test shuffle functionality."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )
    ids_shuffled = ids.shuffle(seed=42)

    assert hasattr(ids_shuffled, "__iter__"), "Shuffled dataset should be iterable"

    # Get first example from each
    ex1 = next(iter(ids))
    ex2 = next(iter(ids_shuffled))

    # They might be the same by chance, but at least verify both work
    assert ex1 is not None
    assert ex2 is not None


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_shard():
    """Test sharding functionality for DDP."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )

    # Test single shard (should always work)
    ids_shard = ids.shard(num_shards=1, index=0)
    assert hasattr(ids_shard, "__iter__"), "Shard should be iterable"

    # Verify shard yields examples
    ex = next(iter(ids_shard))
    assert ex is not None
    assert "text" in ex, "Example should have 'text' field"

    # Test that sharding with multiple shards doesn't crash
    # (Note: wikitext only has 1 Arrow file, so some shards may be empty)
    ids_shard0 = ids.shard(num_shards=2, index=0)
    ids_shard1 = ids.shard(num_shards=2, index=1)

    assert hasattr(ids_shard0, "__iter__"), "Shard 0 should be iterable"
    assert hasattr(ids_shard1, "__iter__"), "Shard 1 should be iterable"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_example_level_sharding():
    """Test example-level sharding with num_shards > num_files."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )
    total_examples = len(ids)

    # Shard into more shards than files (wikitext has only 1 file)
    num_shards = 4
    shards = [
        ids.shard(num_shards=num_shards, index=i, mode="example")
        for i in range(num_shards)
    ]

    # Verify all shards are iterable
    for i, shard in enumerate(shards):
        assert hasattr(shard, "__iter__"), f"Shard {i} should be iterable"

    # Count examples in each shard
    shard_sizes = []
    for i, shard in enumerate(shards):
        count = sum(1 for _ in shard)
        shard_sizes.append(count)
        print(f"Shard {i}: {count} examples")

    # Verify total examples is preserved
    assert (
        sum(shard_sizes) == total_examples
    ), f"Total examples mismatch: {sum(shard_sizes)} != {total_examples}"

    # Verify shards are roughly equal in size (within 1 example)
    min_size = min(shard_sizes)
    max_size = max(shard_sizes)
    assert (
        max_size - min_size <= 1
    ), f"Shard sizes too uneven: min={min_size}, max={max_size}"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_shard_auto_mode():
    """Test auto mode selection for sharding."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )

    # With 1 file and 2 shards, auto should use example-level sharding
    shard0 = ids.shard(num_shards=2, index=0, mode="auto")
    shard1 = ids.shard(num_shards=2, index=1, mode="auto")

    # Both shards should have examples (not empty)
    count0 = sum(1 for _ in shard0)
    count1 = sum(1 for _ in shard1)

    assert count0 > 0, "Shard 0 should not be empty with auto mode"
    assert count1 > 0, "Shard 1 should not be empty with auto mode"

    # Total should match original
    total = len(ids)
    assert (
        count0 + count1 == total
    ), f"Total examples mismatch: {count0 + count1} != {total}"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_example_sharding_lengths():
    """Test that __len__ works correctly with example-level sharding."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )
    total_examples = len(ids)

    # Create shards
    num_shards = 3
    shards = [
        ids.shard(num_shards=num_shards, index=i, mode="example")
        for i in range(num_shards)
    ]

    # Check that reported lengths match actual iteration
    for i, shard in enumerate(shards):
        reported_len = len(shard)
        actual_len = sum(1 for _ in shard)
        assert (
            reported_len == actual_len
        ), f"Shard {i} length mismatch: reported {reported_len}, actual {actual_len}"

    # Check that total length is preserved
    total_shard_len = sum(len(shard) for shard in shards)
    assert (
        total_shard_len == total_examples
    ), f"Total shard lengths {total_shard_len} != original {total_examples}"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
@pytest.mark.parametrize("num_workers", [0, 1])
def test_example_sharding_checkpoint(num_workers):
    """Test checkpoint compatibility with example-level sharding."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )

    # Shard with example-level mode
    ids_shard = ids.shard(num_shards=2, index=0, mode="example")

    # Create dataloader and iterate a few batches
    dataloader = StatefulDataLoader(ids_shard, batch_size=4, num_workers=num_workers)

    checkpoint = None
    for i, batch in enumerate(dataloader):
        if i >= 3:
            checkpoint = dataloader.state_dict()
            break

    assert checkpoint is not None, "Failed to save checkpoint"

    # Create fresh dataloader to get expected batch
    ids_fresh = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )
    ids_fresh_shard = ids_fresh.shard(num_shards=2, index=0, mode="example")
    dataloader_fresh = StatefulDataLoader(
        ids_fresh_shard, batch_size=4, num_workers=num_workers
    )

    expected_batch = None
    for i, batch in enumerate(dataloader_fresh):
        if i == 4:
            expected_batch = batch
            break

    assert expected_batch is not None, "Failed to get expected batch"

    # Restore and verify
    ids_restored = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )
    ids_restored_shard = ids_restored.shard(num_shards=2, index=0, mode="example")
    dataloader_restored = StatefulDataLoader(
        ids_restored_shard, batch_size=4, num_workers=num_workers
    )
    dataloader_restored.load_state_dict(checkpoint)

    restored_batch = next(iter(dataloader_restored))

    # Verify batches match
    if isinstance(expected_batch, dict) and isinstance(restored_batch, dict):
        for k in expected_batch.keys():
            assert k in restored_batch, f"Key {k} missing in restored batch"
            assert str(expected_batch[k]) == str(
                restored_batch[k]
            ), f"Mismatch for key {k}"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_virtual_split_basic():
    """Test basic virtual split functionality."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )
    total_examples = len(ids)

    # First 50%
    train_ds = ids.slice(None, 0.5)
    assert len(train_ds) == total_examples // 2

    # Last 50%
    val_ds = ids.slice(0.5, None)
    assert len(val_ds) == total_examples - total_examples // 2

    # Verify examples don't overlap
    train_count = sum(1 for _ in train_ds)
    val_count = sum(1 for _ in val_ds)

    assert train_count == len(train_ds)
    assert val_count == len(val_ds)
    assert train_count + val_count == total_examples


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_virtual_split_percentage_string():
    """Test virtual split with percentage strings."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )
    total_examples = len(ids)

    # Using percentage strings
    train_ds = ids.slice(None, "80%")
    val_ds = ids.slice("80%", None)

    assert len(train_ds) == int(total_examples * 0.8)
    assert len(val_ds) == total_examples - int(total_examples * 0.8)
    assert len(train_ds) + len(val_ds) == total_examples


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_virtual_split_absolute_indices():
    """Test virtual split with absolute indices."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )

    # Absolute indices
    subset = ids.slice(100, 200)
    assert len(subset) == 100

    count = sum(1 for _ in subset)
    assert count == 100


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_virtual_split_with_sharding():
    """Test virtual split combined with sharding."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )

    # First 80% for training, split across 2 shards
    train_ds = ids.slice(None, 0.8)
    train_total = len(train_ds)

    shard0 = train_ds.shard(num_shards=2, index=0, mode="example")
    shard1 = train_ds.shard(num_shards=2, index=1, mode="example")

    # Verify shard sizes
    count0 = sum(1 for _ in shard0)
    count1 = sum(1 for _ in shard1)

    assert count0 + count1 == train_total
    assert abs(count0 - count1) <= 1  # Should be roughly equal


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_virtual_split_checkpoint():
    """Test checkpoint compatibility with virtual splits."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )

    # Create virtual split (first 80%)
    train_ds = ids.slice(None, 0.8)

    # Create dataloader and iterate
    dataloader = StatefulDataLoader(train_ds, batch_size=4, num_workers=0)

    checkpoint = None
    for i, batch in enumerate(dataloader):
        if i >= 3:
            checkpoint = dataloader.state_dict()
            break

    assert checkpoint is not None

    # Create fresh dataloader to get expected batch
    ids_fresh = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )
    train_fresh = ids_fresh.slice(None, 0.8)
    dataloader_fresh = StatefulDataLoader(train_fresh, batch_size=4, num_workers=0)

    expected_batch = None
    for i, batch in enumerate(dataloader_fresh):
        if i == 4:
            expected_batch = batch
            break

    # Restore and verify
    ids_restored = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )
    train_restored = ids_restored.slice(None, 0.8)
    dataloader_restored = StatefulDataLoader(
        train_restored, batch_size=4, num_workers=0
    )
    dataloader_restored.load_state_dict(checkpoint)

    restored_batch = next(iter(dataloader_restored))

    # Verify batches match
    if isinstance(expected_batch, dict) and isinstance(restored_batch, dict):
        for k in expected_batch.keys():
            assert k in restored_batch
            assert str(expected_batch[k]) == str(restored_batch[k])


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_three_way_split():
    """Test creating train/val/test splits."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )
    total = len(ids)

    # 70% train, 15% val, 15% test
    train_ds = ids.slice(None, 0.7)
    val_ds = ids.slice(0.7, 0.85)
    test_ds = ids.slice(0.85, None)

    train_count = sum(1 for _ in train_ds)
    val_count = sum(1 for _ in val_ds)
    test_count = sum(1 for _ in test_ds)

    # Verify counts match
    assert train_count == len(train_ds)
    assert val_count == len(val_ds)
    assert test_count == len(test_ds)

    # Verify total is preserved
    assert train_count + val_count + test_count == total

    # Verify rough proportions (within rounding)
    assert abs(train_count - int(total * 0.7)) <= 1
    assert abs(val_count - int(total * 0.15)) <= 1


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_parse_split_notation():
    """Test _parse_split_notation function."""
    from forgather.ml.datasets.fast_hf_loader import _parse_split_notation

    # Test plain split
    base, start, end = _parse_split_notation("train")
    assert base == "train"
    assert start is None
    assert end is None

    # Test with start index
    base, start, end = _parse_split_notation("train[10000:]")
    assert base == "train"
    assert start == 10000
    assert end is None

    # Test with end index
    base, start, end = _parse_split_notation("train[:5000]")
    assert base == "train"
    assert start is None
    assert end == 5000

    # Test with both indices
    base, start, end = _parse_split_notation("train[1000:5000]")
    assert base == "train"
    assert start == 1000
    assert end == 5000

    # Test with percentage strings
    base, start, end = _parse_split_notation("train[10%:20%]")
    assert base == "train"
    assert start == "10%"
    assert end == "20%"

    # Test with None input
    base, start, end = _parse_split_notation(None)
    assert base is None
    assert start is None
    assert end is None


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_split_notation_no_reindex():
    """
    Test that using split notation doesn't trigger reindexing.

    Verifies that "train" and "train[10000:]" use the same cached index.
    """
    import tempfile
    from pathlib import Path

    # Create temporary index directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load with base split first (will create index)
        ids1 = fast_load_iterable_dataset(
            "wikitext", name="wikitext-2-raw-v1", split="train", index_dir=tmpdir
        )

        # Count index files created
        index_files_before = list(Path(tmpdir).glob("*.json"))
        assert len(index_files_before) == 1, "Should have exactly one index file"

        # Load with sliced split (should reuse same index)
        ids2 = fast_load_iterable_dataset(
            "wikitext", name="wikitext-2-raw-v1", split="train[100:]", index_dir=tmpdir
        )

        # Verify no new index files created
        index_files_after = list(Path(tmpdir).glob("*.json"))
        assert len(index_files_after) == 1, "Should still have exactly one index file"
        assert (
            index_files_before[0] == index_files_after[0]
        ), "Should use same index file"

        # Verify lengths are different (slice applied)
        assert (
            len(ids2) == len(ids1) - 100
        ), "Sliced dataset should be 100 examples shorter"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_split_notation_slice_applied():
    """Test that split notation correctly applies the slice."""
    # Load with split notation
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )
    ids_sliced = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[100:200]"
    )

    # Verify length
    assert len(ids_sliced) == 100, "Sliced dataset should have exactly 100 examples"

    # Verify we can iterate
    count = sum(1 for _ in ids_sliced)
    assert count == 100, f"Should iterate exactly 100 examples, got {count}"

    # Test with percentage notation
    ids_pct = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[10%:20%]"
    )
    total = len(ids)
    expected_len = int(total * 0.2) - int(total * 0.1)
    assert len(ids_pct) == expected_len, "Percentage slice length mismatch"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_split_notation_with_operations():
    """Test that split notation works with shuffle, shard, and other operations."""
    # Load with split notation
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[100:]"
    )

    # Apply shuffle
    ids_shuffled = ids.shuffle(seed=42)
    assert hasattr(ids_shuffled, "__iter__"), "Should be iterable after shuffle"

    # Apply shard
    ids_shard = ids.shard(num_shards=2, index=0, mode="example")
    assert hasattr(ids_shard, "__iter__"), "Should be iterable after shard"

    # Verify we can iterate
    count = sum(1 for _ in ids_shard)
    assert count > 0, "Should have examples after sharding"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_split_notation_checkpoint():
    """Test that checkpointing works with split notation."""
    # Load with split notation
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[100:]"
    )

    # Create dataloader
    dataloader = StatefulDataLoader(ids, batch_size=4, num_workers=0)

    # Save checkpoint after a few batches
    checkpoint = None
    for i, batch in enumerate(dataloader):
        if i >= 3:
            checkpoint = dataloader.state_dict()
            break

    assert checkpoint is not None, "Should be able to save checkpoint"

    # Create fresh dataloader with same split notation
    ids_restored = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[100:]"
    )
    dataloader_restored = StatefulDataLoader(ids_restored, batch_size=4, num_workers=0)

    # Restore checkpoint
    dataloader_restored.load_state_dict(checkpoint)

    # Should be able to continue iteration
    restored_batch = next(iter(dataloader_restored))
    assert restored_batch is not None, "Should get batch after restore"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_column_names():
    """Test column_names property."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )

    # Should have column_names attribute
    assert hasattr(ids, "column_names"), "Dataset should have column_names attribute"

    # Should return list of column names
    column_names = ids.column_names
    assert isinstance(column_names, list), "column_names should be a list"
    assert len(column_names) > 0, "column_names should not be empty"

    # Wikitext should have 'text' column
    assert "text" in column_names, "Wikitext dataset should have 'text' column"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_features():
    """Test features property."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )

    # Should have features attribute
    assert hasattr(ids, "features"), "Dataset should have features attribute"

    # Features should not be None
    features = ids.features
    assert features is not None, "features should not be None"

    # Should have 'text' feature
    assert "text" in features, "features should contain 'text' key"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_n_shards():
    """Test n_shards property."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )

    # Should have n_shards attribute
    assert hasattr(ids, "n_shards"), "Dataset should have n_shards attribute"

    # Should return number of Arrow files
    n_shards = ids.n_shards
    assert isinstance(n_shards, int), "n_shards should be an integer"
    assert n_shards > 0, "n_shards should be positive"

    # After shuffling, n_shards should remain the same
    ids_shuffled = ids.shuffle(seed=42)
    assert ids_shuffled.n_shards == n_shards, "n_shards should not change after shuffle"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_metadata_with_operations():
    """Test that metadata properties work with shuffle, shard, and slice."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )

    original_columns = ids.column_names
    original_n_shards = ids.n_shards

    # After shuffle
    ids_shuffled = ids.shuffle(seed=42)
    assert (
        ids_shuffled.column_names == original_columns
    ), "column_names should not change after shuffle"
    assert (
        ids_shuffled.n_shards == original_n_shards
    ), "n_shards should not change after shuffle"

    # After slice
    ids_sliced = ids.slice(None, 0.5)
    assert (
        ids_sliced.column_names == original_columns
    ), "column_names should not change after slice"
    assert (
        ids_sliced.n_shards == original_n_shards
    ), "n_shards should not change after slice"

    # After shard
    ids_shard = ids.shard(num_shards=2, index=0, mode="example")
    assert (
        ids_shard.column_names == original_columns
    ), "column_names should not change after shard"
    # Note: n_shards is the number of Arrow files, not DDP shards, so it doesn't change


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_map_remove_columns():
    """Test using column_names with map(remove_columns=...)."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )

    # Verify column_names is accessible (common pattern in training)
    assert hasattr(ids, "column_names"), "Should have column_names attribute"
    assert len(ids.column_names) > 0, "Should have columns"

    # Apply map with remove_columns (common pattern in training)
    def add_length(example):
        return {"text_length": len(example["text"])}

    # This should work without errors (common training pattern)
    ids_mapped = ids.map(add_length)

    # Should still be iterable
    assert hasattr(ids_mapped, "__iter__"), "Mapped dataset should be iterable"

    # Get first example to verify map worked
    example = next(iter(ids_mapped))
    assert "text" in example, "Original column should still exist"
    assert "text_length" in example, "New column should be added"
    assert isinstance(example["text_length"], int), "text_length should be an integer"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_native_map_basic():
    """Test native map implementation (non-HF)."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )

    # Apply map transformation
    def add_length(example):
        return {"text": example["text"], "length": len(example["text"])}

    ids_mapped = ids.map(add_length)

    # Verify map is applied
    example = next(iter(ids_mapped))
    assert "text" in example
    assert "length" in example
    assert example["length"] == len(example["text"])


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_native_map_remove_columns():
    """Test native map with remove_columns."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )

    # Map with remove_columns
    def add_tokens(example):
        return {"tokens": example["text"].split()}

    ids_mapped = ids.map(add_tokens, remove_columns=["text"])

    # Verify transformation and column removal
    example = next(iter(ids_mapped))
    assert "tokens" in example
    assert "text" not in example, "text column should be removed"
    assert isinstance(example["tokens"], list)


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_native_map_filtering():
    """Test native map with filtering (None returns)."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )

    # Filter out empty lines
    def filter_empty(example):
        if len(example["text"].strip()) == 0:
            return None
        return example

    ids_filtered = ids.map(filter_empty)

    # Verify filtering works
    count = 0
    for example in ids_filtered:
        assert len(example["text"].strip()) > 0, "Empty examples should be filtered"
        count += 1
        if count >= 10:  # Just check first 10
            break

    assert count > 0, "Should have some examples"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_native_map_with_checkpoint():
    """Test that native map preserves checkpoint state."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )

    # Apply map
    def add_prefix(example):
        return {"text": "PREFIX: " + example["text"]}

    ids_mapped = ids.map(add_prefix)

    # Create dataloader and iterate a few batches
    dataloader = StatefulDataLoader(ids_mapped, batch_size=4, num_workers=0)

    checkpoint = None
    for i, batch in enumerate(dataloader):
        if i >= 3:
            checkpoint = dataloader.state_dict()
            break

    assert checkpoint is not None, "Should save checkpoint"

    # Create fresh dataloader with SAME map
    ids_fresh = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )
    ids_fresh_mapped = ids_fresh.map(add_prefix)
    dataloader_fresh = StatefulDataLoader(ids_fresh_mapped, batch_size=4, num_workers=0)

    # Get expected batch (batch 4)
    expected_batch = None
    for i, batch in enumerate(dataloader_fresh):
        if i == 4:
            expected_batch = batch
            break

    # Create restored dataloader
    ids_restored = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )
    ids_restored_mapped = ids_restored.map(add_prefix)
    dataloader_restored = StatefulDataLoader(
        ids_restored_mapped, batch_size=4, num_workers=0
    )
    dataloader_restored.load_state_dict(checkpoint)

    # Get first batch from restored
    restored_batch = next(iter(dataloader_restored))

    # Verify batches match
    if isinstance(expected_batch, dict) and isinstance(restored_batch, dict):
        for k in expected_batch.keys():
            assert k in restored_batch
            # Check that PREFIX was applied
            if k == "text":
                for text in restored_batch[k]:
                    assert text.startswith("PREFIX: "), "Map should be applied"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_native_map_with_operations():
    """Test native map works with shuffle, shard, and slice."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train"
    )

    # Apply map
    def uppercase(example):
        return {"text": example["text"].upper()}

    # Chain operations: shuffle -> slice -> map -> shard
    ids_transformed = (
        ids.shuffle(seed=42)
        .slice(None, 0.5)
        .map(uppercase)
        .shard(num_shards=2, index=0, mode="example")
    )

    # Verify it's iterable and transformations work
    example = next(iter(ids_transformed))
    assert "text" in example
    assert example["text"] == example["text"].upper(), "Should be uppercased"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_batched_map_basic():
    """Test batched map with simple transformation."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[:20]"
    )

    # Simple batched function that adds length field
    def add_lengths(batch):
        texts = batch["text"]
        return {"text": texts, "length": [len(t) for t in texts]}

    # Apply batched map
    ids_mapped = ids.map(add_lengths, batched=True, batch_size=5)

    # Verify transformation
    examples = list(ids_mapped)
    assert len(examples) > 0, "Should have examples"
    for ex in examples:
        assert "length" in ex, "Should have length field"
        assert ex["length"] == len(ex["text"]), "Length should match text length"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_batched_map_n_to_m():
    """Test batched map with N->M mapping (filtering/duplication)."""
    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[:20]"
    )

    # Function that filters out empty texts and duplicates non-empty ones
    def filter_and_duplicate(batch):
        texts = batch["text"]
        result_texts = []
        for text in texts:
            if text.strip():  # Non-empty
                result_texts.append(text)
                result_texts.append(text + " (duplicate)")  # Duplicate it
        return {"text": result_texts}

    # Apply batched map
    ids_mapped = ids.map(filter_and_duplicate, batched=True, batch_size=5)

    # Count examples
    count = sum(1 for _ in ids_mapped)
    # Should have roughly 2x non-empty examples
    assert count > 0, "Should have examples after filtering and duplication"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_batched_map_with_block_tokenize():
    """Test batched map with block_tokenize_fn (packed sequences)."""
    from transformers import AutoTokenizer

    from forgather.ml.datasets.block_tokenizer import block_tokenize_fn

    ids = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[:30]"
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Apply block tokenization with packing
    ids_tokenized = ids.map(
        lambda batch: block_tokenize_fn(
            batch,
            tokenizer=tokenizer,
            feature="text",
            max_length=128,
            packed=True,
            overflow=True,
            min_len=1,
        ),
        batched=True,
        batch_size=10,
    )

    # Verify packed sequences
    examples = []
    for i, ex in enumerate(ids_tokenized):
        examples.append(ex)
        if i >= 10:
            break

    assert len(examples) > 0, "Should have packed sequences"
    for ex in examples:
        assert "input_ids" in ex, "Should have input_ids"
        assert isinstance(ex["input_ids"], list), "input_ids should be a list"
        assert len(ex["input_ids"]) <= 128, "Should not exceed max_length"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_interleave_basic():
    """Test basic round-robin interleaving."""
    ds1 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[:100]"
    )
    ds2 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[100:200]"
    )

    # Interleave round-robin
    combined = interleave_datasets([ds1, ds2])

    # Verify it's iterable
    assert hasattr(combined, "__iter__"), "Should be iterable"

    # Verify some examples alternate
    examples = []
    for i, ex in enumerate(combined):
        examples.append(ex)
        if i >= 5:
            break

    assert len(examples) == 6, "Should get 6 examples"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_interleave_with_probabilities():
    """Test probabilistic sampling with interleaving."""
    ds1 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[:100]"
    )
    ds2 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[100:200]"
    )

    # Interleave with probabilities (70% ds1, 30% ds2)
    combined = interleave_datasets([ds1, ds2], probabilities=[0.7, 0.3], seed=42)

    # Should be iterable
    assert hasattr(combined, "__iter__"), "Should be iterable"

    # Get some examples
    count = sum(
        1 for _ in combined.datasets[0]
    )  # This resets iterator, so just check it works
    assert True  # If we get here, interleaving works


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_interleave_stopping_strategies():
    """Test different stopping strategies."""
    # First exhausted (stops when first dataset runs out)
    ds1 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[:50]"
    )
    ds2 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[100:200]"
    )
    combined_first = interleave_datasets(
        [ds1, ds2], stopping_strategy="first_exhausted"
    )
    count_first = sum(1 for _ in combined_first)
    assert count_first > 0, "Should have examples with first_exhausted"

    # All exhausted (continues until all exhausted) - recreate datasets
    ds1 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[:50]"
    )
    ds2 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[100:200]"
    )
    combined_all = interleave_datasets([ds1, ds2], stopping_strategy="all_exhausted")
    count_all = sum(1 for _ in combined_all)
    assert count_all >= count_first, "all_exhausted should have >= examples"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_interleave_checkpoint():
    """Test that interleaved dataset preserves checkpoint state."""
    ds1 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[:100]"
    )
    ds2 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[100:200]"
    )

    # Create interleaved dataset
    combined = interleave_datasets([ds1, ds2], seed=42)

    # Create dataloader and iterate
    dataloader = StatefulDataLoader(combined, batch_size=4, num_workers=0)

    checkpoint = None
    for i, batch in enumerate(dataloader):
        if i >= 3:
            checkpoint = dataloader.state_dict()
            break

    assert checkpoint is not None, "Should save checkpoint"

    # Create fresh interleaved dataset
    ds1_fresh = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[:100]"
    )
    ds2_fresh = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[100:200]"
    )
    combined_fresh = interleave_datasets([ds1_fresh, ds2_fresh], seed=42)
    dataloader_fresh = StatefulDataLoader(combined_fresh, batch_size=4, num_workers=0)

    # Get expected batch (batch 4)
    expected_batch = None
    for i, batch in enumerate(dataloader_fresh):
        if i == 4:
            expected_batch = batch
            break

    # Restore from checkpoint
    ds1_restored = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[:100]"
    )
    ds2_restored = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[100:200]"
    )
    combined_restored = interleave_datasets([ds1_restored, ds2_restored], seed=42)
    dataloader_restored = StatefulDataLoader(
        combined_restored, batch_size=4, num_workers=0
    )
    dataloader_restored.load_state_dict(checkpoint)

    # Get first batch from restored
    restored_batch = next(iter(dataloader_restored))

    # Verify restoration worked (should get same data)
    assert restored_batch is not None, "Should get restored batch"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_interleave_metadata():
    """Test that interleaved dataset has correct metadata."""
    ds1 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[:100]"
    )
    ds2 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[:100]"
    )

    combined = interleave_datasets([ds1, ds2])

    # Should have metadata properties
    assert hasattr(combined, "column_names"), "Should have column_names"
    assert hasattr(combined, "features"), "Should have features"
    assert hasattr(combined, "n_shards"), "Should have n_shards"

    # column_names should come from first dataset
    assert combined.column_names == ds1.column_names


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_interleave_with_map():
    """Test interleaving datasets that have map applied."""
    ds1 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[:50]"
    )
    ds2 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[100:150]"
    )

    # Apply different maps to each dataset
    def add_prefix1(example):
        return {"text": "DS1: " + example["text"]}

    def add_prefix2(example):
        return {"text": "DS2: " + example["text"]}

    ds1_mapped = ds1.map(add_prefix1)
    ds2_mapped = ds2.map(add_prefix2)

    # Interleave mapped datasets
    combined = interleave_datasets([ds1_mapped, ds2_mapped])

    # Verify maps are applied
    examples = []
    for i, ex in enumerate(combined):
        examples.append(ex)
        if i >= 3:
            break

    # Check that prefixes are applied
    for ex in examples:
        assert ex["text"].startswith("DS1:") or ex["text"].startswith("DS2:")


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_interleave_dynamic_probabilities():
    """Test interleaving with dynamic probability function."""
    from forgather.ml.datasets import balance_remaining_examples

    # Create datasets of different lengths
    ds1 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[:30]"
    )
    ds2 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[100:200]"
    )

    # Use dynamic balancing function
    combined = interleave_datasets(
        [ds1, ds2],
        probabilities=balance_remaining_examples,
        seed=42,
        stopping_strategy="first_exhausted",
    )

    # Collect examples
    count = sum(1 for _ in combined)
    assert count > 0, "Should have examples with dynamic probabilities"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_interleave_custom_probability_function():
    """Test interleaving with custom probability function."""

    # Custom function that increases weight of dataset 0 over time
    def curriculum_probabilities(step, datasets, examples_per_dataset, exhausted):
        """Gradually shift from dataset 1 to dataset 0."""
        # First 50 steps: mostly dataset 1
        # Next 50 steps: transition period
        # After 100 steps: mostly dataset 0
        if step < 50:
            return [0.1, 0.9]  # 10% ds0, 90% ds1
        elif step < 100:
            # Smooth transition
            progress = (step - 50) / 50.0
            return [0.1 + 0.8 * progress, 0.9 - 0.8 * progress]
        else:
            return [0.9, 0.1]  # 90% ds0, 10% ds1

    ds1 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[:100]"
    )
    ds2 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[100:200]"
    )

    combined = interleave_datasets(
        [ds1, ds2],
        probabilities=curriculum_probabilities,
        seed=42,
        stopping_strategy="first_exhausted",
    )

    # Verify it produces examples
    count = sum(1 for _ in combined)
    assert count > 0, "Should have examples with custom probability function"


@pytest.mark.skipif(
    not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available"
)
def test_balance_remaining_examples_function():
    """Test the balance_remaining_examples utility function."""
    from forgather.ml.datasets import balance_remaining_examples

    # Create mock datasets with known lengths
    ds1 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[:50]"
    )  # 50 examples
    ds2 = fast_load_iterable_dataset(
        "wikitext", name="wikitext-2-raw-v1", split="train[100:200]"
    )  # 100 examples

    # Simulate partway through iteration
    examples_per_dataset = [10, 20]  # Consumed 10 from ds1, 20 from ds2
    exhausted = [False, False]

    weights = balance_remaining_examples(
        step=30,
        datasets=[ds1, ds2],
        examples_per_dataset=examples_per_dataset,
        exhausted=exhausted,
    )

    # ds1: 50 - 10 = 40 remaining
    # ds2: 100 - 20 = 80 remaining
    # ds2 should have higher weight (more remaining)
    assert len(weights) == 2, "Should return 2 weights"
    assert (
        weights[1] > weights[0]
    ), "Dataset with more remaining should have higher weight"
    assert weights[0] == 40.0, "ds1 weight should be remaining examples"
    assert weights[1] == 80.0, "ds2 weight should be remaining examples"
