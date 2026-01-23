"""
Unit tests for document boundary tracking in packed sequences.

Tests the new explicit document boundary tracking system that allows
packed sequences to work with tokenizers that don't have BOS/EOS tokens
(e.g., Qwen3).
"""

import torch
from transformers import PreTrainedTokenizerFast

from forgather.ml.data_collator import (
    DataCollatorForCausalLM,
    _pos_ids_from_boundaries,
    get_pos_ids_for_packed_sequence,
)
from forgather.ml.datasets.block_tokenizer import (
    Bin,
    Document,
    InputTokenBlock,
    OutputTokenBlock,
    block_tokenize_fn,
    pack_sequences_optimized,
)
from tokenizers import Tokenizer, models, pre_tokenizers


def create_simple_tokenizer():
    """Create a simple tokenizer for testing without special tokens."""
    # Create a basic tokenizer without BOS/EOS
    vocab = {"[PAD]": 0, "hello": 1, "world": 2, "test": 3}
    tokenizer_obj = Tokenizer(models.WordLevel(vocab))
    tokenizer_obj.pre_tokenizer = pre_tokenizers.Whitespace()

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # Explicitly set eos_token_id to None (like Qwen3 BOS)
    tokenizer.eos_token_id = None
    tokenizer.bos_token_id = None

    return tokenizer


def test_bin_get_document_starts():
    """Test Bin.get_document_starts() method."""
    bin = Bin(max_length=100)

    # Add first document
    doc1 = Document(input_ids=[1, 2, 3], length=3, original_index=0)
    bin.add_document(doc1)

    # Add second document
    doc2 = Document(input_ids=[4, 5], length=2, original_index=1)
    bin.add_document(doc2)

    # Add third document
    doc3 = Document(input_ids=[6, 7, 8, 9], length=4, original_index=2)
    bin.add_document(doc3)

    starts = bin.get_document_starts()
    assert starts == [0, 3, 5], f"Expected [0, 3, 5], got {starts}"
    assert bin.get_ids() == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_bin_get_document_starts_empty():
    """Test Bin.get_document_starts() with empty bin."""
    bin = Bin(max_length=100)
    starts = bin.get_document_starts()
    assert starts == [], f"Expected empty list, got {starts}"


def test_pack_sequences_optimized_returns_boundaries():
    """Test that pack_sequences_optimized returns document boundaries."""
    documents = [
        Document(input_ids=[1, 2, 3], length=3, original_index=0),
        Document(input_ids=[4, 5], length=2, original_index=1),
        Document(input_ids=[6, 7, 8], length=3, original_index=2),
    ]

    sequences, document_starts = pack_sequences_optimized(
        documents=documents,
        max_length=10,
        min_len=1,
        stride=0,
        overflow=False,
        strategy="best_fit",
        bos_token_id=None,
        shuffle_output=False,
    )

    # Should pack all three documents into one sequence
    assert len(sequences) == 1, f"Expected 1 sequence, got {len(sequences)}"
    # best_fit sorts by length, so order will be: [3,3,2] -> doc0, doc2, doc1
    assert sequences[0] == [1, 2, 3, 6, 7, 8, 4, 5]

    # Document boundaries should match the packed order
    assert len(document_starts) == 1
    assert document_starts[0] == [
        0,
        3,
        6,
    ], f"Expected [0, 3, 6], got {document_starts[0]}"


def test_output_token_block_tracks_boundaries():
    """Test that OutputTokenBlock tracks document starts correctly."""
    block = OutputTokenBlock(max_length=100)

    # Add first document
    input1 = InputTokenBlock([1, 2, 3], length=3)
    block.append(input1)

    # Add second document
    input2 = InputTokenBlock([4, 5], length=2)
    block.append(input2)

    # Add third document
    input3 = InputTokenBlock([6, 7, 8, 9], length=4)
    block.append(input3)

    assert block.get_ids() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert block.get_document_starts() == [0, 3, 5]


def test_input_token_block_is_document_start():
    """Test InputTokenBlock.is_document_start() method."""
    block = InputTokenBlock([1, 2, 3, 4, 5], length=5)

    # Should be True at the start
    assert block.is_document_start() == True

    # Read some tokens
    block.read(2)

    # Should be False after reading
    assert block.is_document_start() == False


def test_pos_ids_from_boundaries():
    """Test _pos_ids_from_boundaries helper function."""
    # Create a batch with 2 sequences
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],  # Sequence 1: 2 docs at positions 0 and 4
            [9, 10, 11, 12, 13, 14, 15, 16],  # Sequence 2: 3 docs at positions 0, 3, 6
        ]
    )

    # Document starts: padded with -1 for sequences with fewer documents
    document_starts = torch.tensor(
        [
            [0, 4, -1],  # Seq 1: 2 documents
            [0, 3, 6],  # Seq 2: 3 documents
        ]
    )

    pos_ids = _pos_ids_from_boundaries(input_ids, document_starts)

    # Expected position IDs
    expected = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1, 2, 3],  # Reset at position 4
            [0, 1, 2, 0, 1, 2, 0, 1],  # Reset at positions 3 and 6
        ]
    )

    assert torch.equal(pos_ids, expected), f"Expected:\n{expected}\n\nGot:\n{pos_ids}"


def test_get_pos_ids_with_boundaries():
    """Test get_pos_ids_for_packed_sequence with explicit boundaries."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    document_starts = torch.tensor([[0, 3]])

    pos_ids = get_pos_ids_for_packed_sequence(
        input_ids, document_starts=document_starts
    )

    expected = torch.tensor([[0, 1, 2, 0, 1, 2]])
    assert torch.equal(pos_ids, expected)


def test_block_tokenize_fn_returns_boundaries():
    """Test that block_tokenize_fn returns document_starts field."""
    tokenizer = create_simple_tokenizer()

    features = {"text": ["hello world", "test", "hello"]}

    # Test with greedy packing
    result = block_tokenize_fn(
        features=features,
        tokenizer=tokenizer,
        feature="text",
        max_length=10,
        packed=True,
        packing_strategy="greedy",
        add_bos=False,
        add_eos=False,
    )

    assert "input_ids" in result
    assert "document_starts" in result
    assert len(result["input_ids"]) == len(result["document_starts"])

    # Test with optimized packing
    result_opt = block_tokenize_fn(
        features=features,
        tokenizer=tokenizer,
        feature="text",
        max_length=10,
        packed=True,
        packing_strategy="best_fit",
        add_bos=False,
        add_eos=False,
    )

    assert "input_ids" in result_opt
    assert "document_starts" in result_opt
    assert len(result_opt["input_ids"]) == len(result_opt["document_starts"])


def test_data_collator_with_boundaries():
    """Test DataCollatorForCausalLM with explicit document boundaries."""
    tokenizer = create_simple_tokenizer()

    # Create collator with packed_sequences enabled
    collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        packed_sequences=True,
        padding="longest",
        return_tensors="pt",
    )

    # Create features with document_starts
    features = [
        {"input_ids": [1, 2, 3, 4, 5], "document_starts": [0, 3]},  # 2 docs
        {"input_ids": [6, 7, 8], "document_starts": [0]},  # 1 doc
    ]

    batch = collator(features)

    assert "input_ids" in batch
    assert "position_ids" in batch
    assert "labels" in batch

    # Check that position IDs reset at document boundaries
    # Sequence 1: [1, 2, 3, 4, 5, PAD, PAD, PAD] -> positions [0, 1, 2, 0, 1, ...]
    # Sequence 2: [6, 7, 8, PAD, PAD, PAD, PAD, PAD] -> positions [0, 1, 2, ...]
    pos_ids = batch["position_ids"]
    assert pos_ids[0, 0] == 0  # First document starts at 0
    assert pos_ids[0, 3] == 0  # Second document resets at position 3
    assert pos_ids[1, 0] == 0  # First (and only) document starts at 0


def test_data_collator_auto_detect():
    """Test that DataCollatorForCausalLM auto-detects packed sequences from document_starts."""
    tokenizer = create_simple_tokenizer()

    # Create collator WITHOUT explicitly setting packed_sequences (should auto-detect)
    collator = DataCollatorForCausalLM(
        tokenizer=tokenizer, padding="longest", return_tensors="pt"
    )

    # Create features with document_starts (should trigger auto-detection)
    features = [
        {"input_ids": [1, 2, 3, 4, 5], "document_starts": [0, 3]},  # 2 docs
        {"input_ids": [6, 7, 8], "document_starts": [0]},  # 1 doc
    ]

    batch = collator(features)

    # Should automatically generate position_ids because document_starts is present
    assert "position_ids" in batch
    pos_ids = batch["position_ids"]
    assert pos_ids[0, 0] == 0  # First document starts at 0
    assert pos_ids[0, 3] == 0  # Second document resets at position 3


def test_data_collator_explicit_disable():
    """Test that packed_sequences=False disables position ID generation even with document_starts."""
    tokenizer = create_simple_tokenizer()

    # Explicitly disable packed sequences
    collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        packed_sequences=False,
        padding="longest",
        return_tensors="pt",
    )

    # Create features with document_starts
    features = [
        {"input_ids": [1, 2, 3, 4, 5], "document_starts": [0, 3]},
        {"input_ids": [6, 7, 8], "document_starts": [0]},
    ]

    batch = collator(features)

    # Should NOT generate position_ids because packed_sequences is explicitly False
    assert "position_ids" not in batch


def test_data_collator_fallback_to_token_based():
    """Test that DataCollatorForCausalLM falls back to token-based detection."""
    # Create tokenizer WITH EOS token
    vocab = {"[PAD]": 0, "hello": 1, "world": 2, "test": 3, "[EOS]": 99}
    tokenizer_obj = Tokenizer(models.WordLevel(vocab))
    tokenizer_obj.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
    tokenizer.add_special_tokens({"pad_token": "[PAD]", "eos_token": "[EOS]"})

    collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        packed_sequences=True,
        padding="longest",
        return_tensors="pt",
    )

    # Create features WITHOUT document_starts (should fall back to EOS detection)
    features = [
        {"input_ids": [1, 2, 99, 3, 4, 99]},  # Two documents separated by EOS
    ]

    batch = collator(features)

    assert "position_ids" in batch
    # Position should reset after each EOS token
    pos_ids = batch["position_ids"][0]
    # This should work with the legacy token-based method


if __name__ == "__main__":
    # Run all tests
    test_bin_get_document_starts()
    test_bin_get_document_starts_empty()
    test_pack_sequences_optimized_returns_boundaries()
    test_output_token_block_tracks_boundaries()
    test_input_token_block_is_document_start()
    test_pos_ids_from_boundaries()
    test_get_pos_ids_with_boundaries()
    test_block_tokenize_fn_returns_boundaries()
    test_data_collator_with_boundaries()
    test_data_collator_auto_detect()
    test_data_collator_explicit_disable()
    test_data_collator_fallback_to_token_based()

    print("All tests passed!")
