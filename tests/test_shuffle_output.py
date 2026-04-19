"""
Test shuffle_output parameter to verify output sequences are randomized.
"""

from transformers import AutoTokenizer

from datasets import Dataset
from forgather.ml.datasets import block_tokenize_fn


def test_shuffle_output():
    """Test that shuffle_output randomizes the order of output sequences."""

    # Create a dataset with predictable, variable-length documents
    texts = [f"Document number {i} " * (i + 5) for i in range(50)]
    dataset = Dataset.from_dict({"text": texts})
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    max_length = 64

    print("\n" + "=" * 80)
    print("TEST: Shuffle Output Sequences")
    print("=" * 80)

    # Test 1: Without shuffle (deterministic ordering)
    print("\n1. Testing WITHOUT shuffle (should have deterministic ordering)...")
    result_no_shuffle_1 = dataset.map(
        lambda features: block_tokenize_fn(
            features,
            tokenizer=tokenizer,
            feature="text",
            max_length=max_length,
            overflow=False,
            packed=True,
            packing_strategy="best_fit",
            shuffle_output=False,  # No shuffle
            min_len=10,
            add_bos=True,
            add_eos=True,
        ),
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
    )

    result_no_shuffle_2 = dataset.map(
        lambda features: block_tokenize_fn(
            features,
            tokenizer=tokenizer,
            feature="text",
            max_length=max_length,
            overflow=False,
            packed=True,
            packing_strategy="best_fit",
            shuffle_output=False,  # No shuffle
            min_len=10,
            add_bos=True,
            add_eos=True,
        ),
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
    )

    # Compare first few sequences
    num_blocks_1 = len(result_no_shuffle_1["input_ids"])
    num_blocks_2 = len(result_no_shuffle_2["input_ids"])

    print(f"   Run 1: {num_blocks_1} blocks")
    print(f"   Run 2: {num_blocks_2} blocks")

    # Check if sequences are identical
    identical = all(
        seq1 == seq2
        for seq1, seq2 in zip(
            result_no_shuffle_1["input_ids"][:5], result_no_shuffle_2["input_ids"][:5]
        )
    )

    print(f"   First 5 sequences identical: {identical}")
    assert identical, "Without shuffle, sequences should be identical across runs"

    # Show first sequence lengths
    lengths_1 = [len(seq) for seq in result_no_shuffle_1["input_ids"][:10]]
    print(f"   First 10 sequence lengths: {lengths_1}")

    # Test 2: With shuffle but fixed seed (deterministic but different order)
    print("\n2. Testing WITH shuffle and FIXED seed (deterministic but shuffled)...")
    result_shuffle_seed1 = dataset.map(
        lambda features: block_tokenize_fn(
            features,
            tokenizer=tokenizer,
            feature="text",
            max_length=max_length,
            overflow=False,
            packed=True,
            packing_strategy="best_fit",
            shuffle_output=True,
            seed=42,  # Fixed seed
            min_len=10,
            add_bos=True,
            add_eos=True,
        ),
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
    )

    result_shuffle_seed2 = dataset.map(
        lambda features: block_tokenize_fn(
            features,
            tokenizer=tokenizer,
            feature="text",
            max_length=max_length,
            overflow=False,
            packed=True,
            packing_strategy="best_fit",
            shuffle_output=True,
            seed=42,  # Same seed
            min_len=10,
            add_bos=True,
            add_eos=True,
        ),
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
    )

    num_blocks_s1 = len(result_shuffle_seed1["input_ids"])
    num_blocks_s2 = len(result_shuffle_seed2["input_ids"])

    print(f"   Run 1 (seed=42): {num_blocks_s1} blocks")
    print(f"   Run 2 (seed=42): {num_blocks_s2} blocks")

    # With same seed, shuffled results should be identical
    identical_shuffled = all(
        seq1 == seq2
        for seq1, seq2 in zip(
            result_shuffle_seed1["input_ids"][:5], result_shuffle_seed2["input_ids"][:5]
        )
    )

    print(f"   First 5 sequences identical: {identical_shuffled}")
    assert identical_shuffled, "With same seed, shuffled sequences should be identical"

    lengths_s = [len(seq) for seq in result_shuffle_seed1["input_ids"][:10]]
    print(f"   First 10 sequence lengths: {lengths_s}")

    # Test 3: Verify shuffling actually changes order
    print("\n3. Verifying that shuffle changes the order...")

    # Compare non-shuffled vs shuffled
    different_order = any(
        seq1 != seq2
        for seq1, seq2 in zip(
            result_no_shuffle_1["input_ids"][:10],
            result_shuffle_seed1["input_ids"][:10],
        )
    )

    print(f"   Order changed by shuffling: {different_order}")
    assert different_order, "Shuffling should change the order of sequences"

    # Test 4: Different seeds produce different orders
    print("\n4. Testing different seeds produce different orders...")
    result_shuffle_seed99 = dataset.map(
        lambda features: block_tokenize_fn(
            features,
            tokenizer=tokenizer,
            feature="text",
            max_length=max_length,
            overflow=False,
            packed=True,
            packing_strategy="best_fit",
            shuffle_output=True,
            seed=99,  # Different seed
            min_len=10,
            add_bos=True,
            add_eos=True,
        ),
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
    )

    different_seeds_different_order = any(
        seq1 != seq2
        for seq1, seq2 in zip(
            result_shuffle_seed1["input_ids"][:10],
            result_shuffle_seed99["input_ids"][:10],
        )
    )

    print(
        f"   Different seeds produce different orders: {different_seeds_different_order}"
    )
    assert (
        different_seeds_different_order
    ), "Different seeds should produce different orders"

    # Test 5: Analyze length distribution before and after shuffle
    print("\n5. Analyzing length distribution...")

    lengths_no_shuffle = [len(seq) for seq in result_no_shuffle_1["input_ids"]]
    lengths_shuffled = [len(seq) for seq in result_shuffle_seed1["input_ids"]]

    # Sort both to compare distributions
    lengths_no_shuffle_sorted = sorted(lengths_no_shuffle)
    lengths_shuffled_sorted = sorted(lengths_shuffled)

    # Distributions should be identical (same sequences, just reordered)
    distributions_match = lengths_no_shuffle_sorted == lengths_shuffled_sorted

    print(f"   Length distributions match: {distributions_match}")
    print(
        f"   No shuffle - avg length: {sum(lengths_no_shuffle)/len(lengths_no_shuffle):.1f}"
    )
    print(
        f"   Shuffled   - avg length: {sum(lengths_shuffled)/len(lengths_shuffled):.1f}"
    )

    assert (
        distributions_match
    ), "Shuffling should preserve the distribution, just reorder"

    print("\n" + "=" * 80)
    print("All shuffle tests passed!")
    print("=" * 80)
    print("\nSummary:")
    print(f"- Shuffle=False produces deterministic output")
    print(f"- Shuffle=True with fixed seed is reproducible")
    print(f"- Shuffling randomizes sequence order without changing distribution")
    print(f"- Different seeds produce different random orders")


if __name__ == "__main__":
    test_shuffle_output()
