"""
Test to understand HuggingFace dataset.map() batch behavior and data loss at boundaries.

This test explores:
1. How partial sequences at end of batch are handled
2. Quantifying data loss with different batch sizes
3. Understanding when the map function is called and with what data
"""
from datasets import Dataset
from transformers import AutoTokenizer
from forgather.ml.datasets import block_tokenize_fn


def test_batch_boundary_behavior():
    """Test what happens to partial sequences at batch boundaries."""

    # Create a small dataset with known sizes
    # Assuming ~5 tokens per word, create documents of specific lengths
    dataset = Dataset.from_dict({
        "text": [
            "Short doc",  # ~2 tokens
            "Medium length document with some more words here",  # ~10 tokens
            "A longer document with quite a bit more content to make it bigger and test splitting behavior",  # ~18 tokens
            "Another short one",  # ~3 tokens
            "Final medium document with reasonable length",  # ~7 tokens
        ]
    })

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Track what batches we receive
    call_count = 0
    batch_sizes = []
    total_input_docs = 0
    total_output_blocks = 0

    def instrumented_block_tokenize(features, **kwargs):
        nonlocal call_count, total_input_docs, total_output_blocks
        call_count += 1
        batch_size = len(features["text"])
        batch_sizes.append(batch_size)
        total_input_docs += batch_size

        print(f"\n=== Batch {call_count} ===")
        print(f"Input batch size: {batch_size}")
        print(f"Input texts: {features['text']}")

        result = block_tokenize_fn(
            features,
            tokenizer=tokenizer,
            feature="text",
            max_length=20,
            overflow=True,
            packed=True,
            min_len=5,
            add_bos=True,
            add_eos=True,
        )

        num_outputs = len(result["input_ids"])
        total_output_blocks += num_outputs
        print(f"Output blocks: {num_outputs}")
        print(f"Output block sizes: {[len(ids) for ids in result['input_ids']]}")

        return result

    # Test with different batch sizes
    print("\n" + "="*80)
    print("TEST 1: Batch size = 2")
    print("="*80)
    call_count = 0
    batch_sizes = []
    total_input_docs = 0
    total_output_blocks = 0

    result1 = dataset.map(
        instrumented_block_tokenize,
        batched=True,
        batch_size=2,
        remove_columns=["text"],
    )

    print(f"\nSummary for batch_size=2:")
    print(f"  Total map() calls: {call_count}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Total input docs: {total_input_docs}")
    print(f"  Total output blocks: {total_output_blocks}")
    print(f"  Final dataset size: {len(result1)}")

    print("\n" + "="*80)
    print("TEST 2: Batch size = 5 (entire dataset)")
    print("="*80)
    call_count = 0
    batch_sizes = []
    total_input_docs = 0
    total_output_blocks = 0

    result2 = dataset.map(
        instrumented_block_tokenize,
        batched=True,
        batch_size=5,
        remove_columns=["text"],
    )

    print(f"\nSummary for batch_size=5:")
    print(f"  Total map() calls: {call_count}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Total input docs: {total_input_docs}")
    print(f"  Total output blocks: {total_output_blocks}")
    print(f"  Final dataset size: {len(result2)}")

    print("\n" + "="*80)
    print("TEST 3: Batch size = 1000 (larger than dataset)")
    print("="*80)
    call_count = 0
    batch_sizes = []
    total_input_docs = 0
    total_output_blocks = 0

    result3 = dataset.map(
        instrumented_block_tokenize,
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
    )

    print(f"\nSummary for batch_size=1000:")
    print(f"  Total map() calls: {call_count}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Total input docs: {total_input_docs}")
    print(f"  Total output blocks: {total_output_blocks}")
    print(f"  Final dataset size: {len(result3)}")

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Batch size 2 produced {len(result1)} blocks")
    print(f"Batch size 5 produced {len(result2)} blocks")
    print(f"Batch size 1000 produced {len(result3)} blocks")

    if len(result2) != len(result3):
        print("\nWARNING: Different batch sizes produce different numbers of output blocks!")
        print("This indicates data loss at batch boundaries.")
    else:
        print("\nBatch sizes 5 and 1000 produce same output (both process full dataset in one batch)")


def test_packing_efficiency():
    """Measure packing efficiency with greedy algorithm."""

    # Create dataset with variable-length documents
    import random
    random.seed(42)

    # Generate documents of various lengths
    def generate_doc(num_words):
        words = ["word"] * num_words
        return " ".join(words)

    # Mix of short, medium, and long documents
    num_docs = 100
    doc_lengths = []
    texts = []
    for _ in range(num_docs):
        length = random.choice([5, 10, 15, 20, 25, 30, 40, 50])  # words
        doc_lengths.append(length)
        texts.append(generate_doc(length))

    dataset = Dataset.from_dict({"text": texts})
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    max_length = 64

    # Test greedy packing
    result = dataset.map(
        lambda features: block_tokenize_fn(
            features,
            tokenizer=tokenizer,
            feature="text",
            max_length=max_length,
            overflow=True,
            packed=True,
            min_len=10,
            add_bos=True,
            add_eos=True,
        ),
        batched=True,
        batch_size=1000,  # Process all at once
        remove_columns=["text"],
    )

    # Analyze packing efficiency
    block_sizes = [len(ids) for ids in result["input_ids"]]
    total_tokens = sum(block_sizes)
    num_blocks = len(block_sizes)
    avg_block_size = total_tokens / num_blocks if num_blocks > 0 else 0
    utilization = avg_block_size / max_length * 100

    print("\n" + "="*80)
    print("PACKING EFFICIENCY ANALYSIS")
    print("="*80)
    print(f"Input documents: {num_docs}")
    print(f"Document length distribution: {sorted(set(doc_lengths))}")
    print(f"Max block length: {max_length}")
    print(f"Output blocks: {num_blocks}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average block size: {avg_block_size:.1f}")
    print(f"Space utilization: {utilization:.1f}%")
    print(f"Min block size: {min(block_sizes)}")
    print(f"Max block size: {max(block_sizes)}")
    print(f"Block sizes: {sorted(block_sizes)}")


if __name__ == "__main__":
    test_batch_boundary_behavior()
    test_packing_efficiency()
