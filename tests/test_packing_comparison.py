"""
Integration test comparing greedy vs optimized packing strategies.

Tests the full block_tokenize_fn with different packing strategies to measure
improvements in packing efficiency.
"""

import random

from transformers import AutoTokenizer

from datasets import Dataset
from forgather.ml.datasets import block_tokenize_fn


def generate_variable_length_dataset(num_docs=200, seed=42):
    """Generate a dataset with variable-length documents."""
    random.seed(seed)

    def generate_doc(num_words):
        words = [f"word{i%100}" for i in range(num_words)]
        return " ".join(words)

    # Realistic distribution: many short docs, some medium, few long
    doc_lengths = []
    texts = []

    for _ in range(num_docs):
        # Weighted distribution favoring shorter documents
        choice = random.random()
        if choice < 0.4:  # 40% short (5-15 words)
            length = random.randint(5, 15)
        elif choice < 0.7:  # 30% medium (20-40 words)
            length = random.randint(20, 40)
        elif choice < 0.9:  # 20% long (50-100 words)
            length = random.randint(50, 100)
        else:  # 10% very long (150-250 words)
            length = random.randint(150, 250)

        doc_lengths.append(length)
        texts.append(generate_doc(length))

    return Dataset.from_dict({"text": texts}), doc_lengths


def analyze_packing(result, max_length, strategy_name):
    """Analyze packing efficiency."""
    block_sizes = [len(ids) for ids in result["input_ids"]]
    num_blocks = len(block_sizes)

    if num_blocks == 0:
        return {
            "strategy": strategy_name,
            "num_blocks": 0,
            "total_tokens": 0,
            "avg_block_size": 0,
            "utilization_pct": 0,
            "min_block_size": 0,
            "max_block_size": 0,
            "std_dev": 0,
        }

    total_tokens = sum(block_sizes)
    avg_block_size = total_tokens / num_blocks
    utilization = avg_block_size / max_length * 100

    # Calculate standard deviation
    variance = sum((size - avg_block_size) ** 2 for size in block_sizes) / num_blocks
    std_dev = variance**0.5

    return {
        "strategy": strategy_name,
        "num_blocks": num_blocks,
        "total_tokens": total_tokens,
        "avg_block_size": avg_block_size,
        "utilization_pct": utilization,
        "min_block_size": min(block_sizes),
        "max_block_size": max(block_sizes),
        "std_dev": std_dev,
        "block_sizes": sorted(block_sizes),
    }


def compare_strategies(dataset, max_length, tokenizer, batch_size=1000):
    """Compare all three packing strategies."""
    results = {}

    for strategy in ["greedy", "best_fit", "first_fit"]:
        print(f"\nTesting {strategy} strategy...")

        result = dataset.map(
            lambda features: block_tokenize_fn(
                features,
                tokenizer=tokenizer,
                feature="text",
                max_length=max_length,
                overflow=True,
                packed=True,
                packing_strategy=strategy,
                min_len=10,
                add_bos=True,
                add_eos=True,
            ),
            batched=True,
            batch_size=batch_size,
            remove_columns=["text"],
        )

        results[strategy] = analyze_packing(result, max_length, strategy)

    return results


def print_comparison(results):
    """Print formatted comparison of results."""
    print("\n" + "=" * 100)
    print("PACKING STRATEGY COMPARISON")
    print("=" * 100)

    strategies = ["greedy", "best_fit", "first_fit"]

    # Header
    print(f"{'Metric':<25} {'Greedy':>20} {'Best Fit':>20} {'First Fit':>20}")
    print("-" * 100)

    # Metrics
    metrics = [
        ("Num blocks", "num_blocks", "{:>20d}"),
        ("Total tokens", "total_tokens", "{:>20d}"),
        ("Avg block size", "avg_block_size", "{:>20.1f}"),
        ("Utilization %", "utilization_pct", "{:>20.1f}"),
        ("Min block size", "min_block_size", "{:>20d}"),
        ("Max block size", "max_block_size", "{:>20d}"),
        ("Std dev", "std_dev", "{:>20.1f}"),
    ]

    for label, key, fmt in metrics:
        values = [results[s][key] for s in strategies]
        formatted = [fmt.format(v) for v in values]
        print(f"{label:<25} {formatted[0]} {formatted[1]} {formatted[2]}")

    print("=" * 100)

    # Calculate improvements
    greedy_blocks = results["greedy"]["num_blocks"]
    best_fit_blocks = results["best_fit"]["num_blocks"]
    first_fit_blocks = results["first_fit"]["num_blocks"]

    if greedy_blocks > 0:
        best_fit_improvement = ((greedy_blocks - best_fit_blocks) / greedy_blocks) * 100
        first_fit_improvement = (
            (greedy_blocks - first_fit_blocks) / greedy_blocks
        ) * 100

        print("\nIMPROVEMENTS OVER GREEDY:")
        print(f"  Best Fit:  {best_fit_improvement:>6.1f}% fewer blocks")
        print(f"  First Fit: {first_fit_improvement:>6.1f}% fewer blocks")

    # Utilization improvements
    greedy_util = results["greedy"]["utilization_pct"]
    best_fit_util = results["best_fit"]["utilization_pct"]
    first_fit_util = results["first_fit"]["utilization_pct"]

    print(f"\nUTILIZATION IMPROVEMENTS:")
    print(f"  Best Fit:  {best_fit_util - greedy_util:>+6.1f} percentage points")
    print(f"  First Fit: {first_fit_util - greedy_util:>+6.1f} percentage points")


def test_overflow_true():
    """Test with overflow=True (split long documents)."""
    print("\n" + "#" * 100)
    print("# TEST 1: overflow=True (split long documents)")
    print("#" * 100)

    dataset, doc_lengths = generate_variable_length_dataset(num_docs=200)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    max_length = 128

    results = compare_strategies(dataset, max_length, tokenizer)
    print_comparison(results)


def test_overflow_false():
    """Test with overflow=False (truncate long documents)."""
    print("\n" + "#" * 100)
    print("# TEST 2: overflow=False (truncate long documents)")
    print("#" * 100)

    dataset, doc_lengths = generate_variable_length_dataset(num_docs=200)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    max_length = 128

    results = {}
    for strategy in ["greedy", "best_fit", "first_fit"]:
        print(f"\nTesting {strategy} strategy...")

        result = dataset.map(
            lambda features: block_tokenize_fn(
                features,
                tokenizer=tokenizer,
                feature="text",
                max_length=max_length,
                overflow=False,  # Truncate instead of split
                packed=True,
                packing_strategy=strategy,
                min_len=10,
                add_bos=True,
                add_eos=True,
            ),
            batched=True,
            batch_size=1000,
            remove_columns=["text"],
        )

        results[strategy] = analyze_packing(result, max_length, strategy)

    print_comparison(results)


def test_small_batches():
    """Test impact of batch size on packing efficiency."""
    print("\n" + "#" * 100)
    print("# TEST 3: Impact of batch size (greedy only)")
    print("#" * 100)

    dataset, doc_lengths = generate_variable_length_dataset(num_docs=100)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    max_length = 128

    batch_sizes = [10, 50, 100, 1000]

    print(f"\n{'Batch Size':>15} {'Num Blocks':>15} {'Utilization %':>15}")
    print("-" * 50)

    for batch_size in batch_sizes:
        result = dataset.map(
            lambda features: block_tokenize_fn(
                features,
                tokenizer=tokenizer,
                feature="text",
                max_length=max_length,
                overflow=True,
                packed=True,
                packing_strategy="greedy",
                min_len=10,
                add_bos=True,
                add_eos=True,
            ),
            batched=True,
            batch_size=batch_size,
            remove_columns=["text"],
        )

        stats = analyze_packing(result, max_length, "greedy")
        print(
            f"{batch_size:>15d} {stats['num_blocks']:>15d} {stats['utilization_pct']:>15.1f}"
        )

    print("\nNote: Smaller batch sizes result in more blocks due to")
    print("      partial sequences at end of each batch.")


def test_realistic_pretraining_scenario():
    """Test with realistic pretraining parameters."""
    print("\n" + "#" * 100)
    print("# TEST 4: Realistic pretraining scenario")
    print("# (4096 token sequences, mixed document lengths)")
    print("#" * 100)

    # Larger dataset, longer sequences
    dataset, doc_lengths = generate_variable_length_dataset(num_docs=500)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    max_length = 4096

    results = compare_strategies(dataset, max_length, tokenizer, batch_size=1000)
    print_comparison(results)


if __name__ == "__main__":
    print("Running packing strategy comparison tests...")
    print("This will download the GPT-2 tokenizer if not already cached.")

    test_overflow_true()
    test_overflow_false()
    test_small_batches()
    test_realistic_pretraining_scenario()

    print("\n" + "=" * 100)
    print("ALL TESTS COMPLETED")
    print("=" * 100)
