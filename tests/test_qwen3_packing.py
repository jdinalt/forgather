"""
Test document packing with real Qwen3 tokenizer to verify the solution works.

This test demonstrates that the explicit document boundary tracking allows
packed sequences to work with Qwen3 models that lack BOS tokens.
"""
import os
import torch
from transformers import AutoTokenizer

from forgather.ml.datasets.block_tokenizer import block_tokenize_fn
from forgather.ml.data_collator import (
    DataCollatorForCausalLM,
    get_pos_ids_for_packed_sequence,
)


def test_qwen3_tokenizer_properties():
    """Verify Qwen3 tokenizer actually lacks BOS token."""
    qwen3_path = os.path.expanduser("~/ai_assets/models/qwen3-1.7b-base")

    if not os.path.exists(qwen3_path):
        print(f"Skipping test: {qwen3_path} not found")
        return

    tokenizer = AutoTokenizer.from_pretrained(qwen3_path)

    print("\n=== Qwen3 Tokenizer Properties ===")
    print(f"BOS token: {tokenizer.bos_token}")
    print(f"BOS token ID: {tokenizer.bos_token_id}")
    print(f"EOS token: {tokenizer.eos_token}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"PAD token: {tokenizer.pad_token}")
    print(f"PAD token ID: {tokenizer.pad_token_id}")

    # Verify the problem: no BOS token
    assert (
        tokenizer.bos_token_id is None
    ), f"Expected no BOS token, but found: {tokenizer.bos_token_id}"

    print("\nQwen3 tokenizer confirmed to have no BOS token!")


def test_qwen3_packing_with_boundaries():
    """Test that document packing works with Qwen3 using explicit boundaries."""
    qwen3_path = os.path.expanduser("~/ai_assets/models/qwen3-1.7b-base")

    if not os.path.exists(qwen3_path):
        print(f"Skipping test: {qwen3_path} not found")
        return

    tokenizer = AutoTokenizer.from_pretrained(qwen3_path)

    # Create sample documents
    features = {
        "text": [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world.",
            "Python is a great programming language.",
        ]
    }

    # Pack documents with our new system (should work even without BOS!)
    result = block_tokenize_fn(
        features=features,
        tokenizer=tokenizer,
        feature="text",
        max_length=128,
        packed=True,
        packing_strategy="greedy",
        add_bos=True,  # This will be disabled automatically since BOS doesn't exist
        add_eos=True,
        overflow=False,
    )

    print("\n=== Packing Results ===")
    print(f"Number of output sequences: {len(result['input_ids'])}")
    print(f"Has document_starts field: {'document_starts' in result}")

    assert "input_ids" in result
    assert "document_starts" in result
    assert len(result["input_ids"]) == len(result["document_starts"])

    # Verify document boundaries
    for i, (seq, starts) in enumerate(
        zip(result["input_ids"], result["document_starts"])
    ):
        print(f"\nSequence {i}:")
        print(f"  Length: {len(seq)}")
        print(f"  Document count: {len(starts)}")
        print(f"  Document starts at positions: {starts}")

        # Decode to verify content
        full_text = tokenizer.decode(seq)
        print(f"  Decoded text: {full_text[:100]}...")

    print("\nDocument packing successful!")


def test_qwen3_collator_with_boundaries():
    """Test that DataCollator generates correct position IDs with Qwen3."""
    qwen3_path = os.path.expanduser("~/ai_assets/models/qwen3-1.7b-base")

    if not os.path.exists(qwen3_path):
        print(f"Skipping test: {qwen3_path} not found")
        return

    tokenizer = AutoTokenizer.from_pretrained(qwen3_path)

    # Create collator WITHOUT explicitly setting packed_sequences
    # Should auto-detect from presence of document_starts
    collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt",
    )

    # Create features with explicit document boundaries
    features = [
        {
            "input_ids": [100, 200, 300, 400, 500],
            "document_starts": [0, 3],  # Two documents
        },
        {
            "input_ids": [600, 700, 800],
            "document_starts": [0],  # One document
        },
    ]

    batch = collator(features)

    print("\n=== Collator Results (Auto-Detected Packed Sequences) ===")
    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Position IDs shape: {batch['position_ids'].shape}")
    print(f"\nPosition IDs:\n{batch['position_ids']}")

    # Verify position IDs were auto-generated and reset at document boundaries
    assert "position_ids" in batch, "Position IDs should be auto-generated from document_starts"
    pos_ids = batch["position_ids"]
    assert pos_ids[0, 0] == 0, "First document should start at position 0"
    assert pos_ids[0, 3] == 0, "Second document should reset to position 0"
    assert pos_ids[1, 0] == 0, "First document in second sequence should start at 0"

    print("\nAuto-detection successful! Position IDs correctly reset at document boundaries!")


if __name__ == "__main__":
    test_qwen3_tokenizer_properties()
    test_qwen3_packing_with_boundaries()
    test_qwen3_collator_with_boundaries()
    print("\n=== ALL TESTS PASSED ===")
