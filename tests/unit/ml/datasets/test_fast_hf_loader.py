"""
Tests for fast_hf_loader: StatefulDataLoader checkpoint functionality
"""
import pytest

try:
    from torchdata.stateful_dataloader import StatefulDataLoader
    HAS_STATEFUL = True
except ImportError:
    HAS_STATEFUL = False

from forgather.ml.datasets.fast_hf_loader import fast_load_iterable_dataset


@pytest.mark.skipif(not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available")
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
    ids = fast_load_iterable_dataset('wikitext', name='wikitext-2-raw-v1', split='train')
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
    ids_fresh = fast_load_iterable_dataset('wikitext', name='wikitext-2-raw-v1', split='train')
    ids_fresh = ids_fresh.shuffle(seed=42)
    dataloader_fresh = StatefulDataLoader(ids_fresh, batch_size=4, num_workers=num_workers)

    # Get batch 5 from fresh dataloader (this is what restored should yield as batch 0)
    expected_batch = None
    for i, batch in enumerate(dataloader_fresh):
        if i == 5:
            expected_batch = batch
            break

    assert expected_batch is not None, "Failed to get expected batch"

    # Create NEW dataloader with restore
    ids_restored = fast_load_iterable_dataset('wikitext', name='wikitext-2-raw-v1', split='train')
    ids_restored = ids_restored.shuffle(seed=42)
    dataloader_restored = StatefulDataLoader(ids_restored, batch_size=4, num_workers=num_workers)
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
                assert expected_batch[k] == restored_batch[k], \
                    f"Mismatch for key {k}: expected {expected_batch[k]}, got {restored_batch[k]}"
            else:
                # For lists or tensors, convert to string for comparison
                assert str(expected_batch[k]) == str(restored_batch[k]), \
                    f"Mismatch for key {k}"
    else:
        # Fallback comparison
        assert str(expected_batch) == str(restored_batch), \
            f"Batches don't match: expected {expected_batch}, got {restored_batch}"


@pytest.mark.skipif(not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available")
def test_fast_load_basic():
    """Test basic dataset loading functionality."""
    ids = fast_load_iterable_dataset('wikitext', name='wikitext-2-raw-v1', split='train')

    # Check it's iterable
    assert hasattr(ids, '__iter__'), "Dataset should be iterable"

    # Check it has length
    assert hasattr(ids, '__len__'), "Dataset should have __len__"
    assert len(ids) > 0, "Dataset should not be empty"

    # Check it has state_dict/load_state_dict
    assert hasattr(ids, 'state_dict'), "Dataset should have state_dict"
    assert hasattr(ids, 'load_state_dict'), "Dataset should have load_state_dict"

    # Test getting an example
    example = next(iter(ids))
    assert example is not None, "Should be able to get an example"
    assert 'text' in example, "Example should have 'text' field"


@pytest.mark.skipif(not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available")
def test_shuffle():
    """Test shuffle functionality."""
    ids = fast_load_iterable_dataset('wikitext', name='wikitext-2-raw-v1', split='train')
    ids_shuffled = ids.shuffle(seed=42)

    assert hasattr(ids_shuffled, '__iter__'), "Shuffled dataset should be iterable"

    # Get first example from each
    ex1 = next(iter(ids))
    ex2 = next(iter(ids_shuffled))

    # They might be the same by chance, but at least verify both work
    assert ex1 is not None
    assert ex2 is not None


@pytest.mark.skipif(not HAS_STATEFUL, reason="torchdata.stateful_dataloader not available")
def test_shard():
    """Test sharding functionality for DDP."""
    ids = fast_load_iterable_dataset('wikitext', name='wikitext-2-raw-v1', split='train')

    # Test single shard (should always work)
    ids_shard = ids.shard(num_shards=1, index=0)
    assert hasattr(ids_shard, '__iter__'), "Shard should be iterable"

    # Verify shard yields examples
    ex = next(iter(ids_shard))
    assert ex is not None
    assert 'text' in ex, "Example should have 'text' field"

    # Test that sharding with multiple shards doesn't crash
    # (Note: wikitext only has 1 Arrow file, so some shards may be empty)
    ids_shard0 = ids.shard(num_shards=2, index=0)
    ids_shard1 = ids.shard(num_shards=2, index=1)

    assert hasattr(ids_shard0, '__iter__'), "Shard 0 should be iterable"
    assert hasattr(ids_shard1, '__iter__'), "Shard 1 should be iterable"
