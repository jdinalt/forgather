"""
Test harness for fast_load_iterable_dataset() and preprocess_dataset().

This script demonstrates and tests various features of Forgather's dataset loading:
- Fast loading of HuggingFace datasets using Arrow file caching
- Virtual splits for efficient train/val/test partitioning
- Example-level sharding for distributed training (DDP)
- Shuffle with configurable buffer size
- Tokenization with batched processing
- Performance timing for startup and iteration

Usage Examples:
    # Basic usage - load and tokenize dataset
    python fast_hf_loader.py ~/path/to/tokenizer

    # Test sharding performance (compare shard 0 vs shard 1)
    python fast_hf_loader.py ~/path/to/tokenizer --shard-index 0
    python fast_hf_loader.py ~/path/to/tokenizer --shard-index 1

    # Test virtual splits
    python fast_hf_loader.py ~/path/to/tokenizer --select-range "10%:20%"

    # Test shuffle with custom buffer
    python fast_hf_loader.py ~/path/to/tokenizer --shuffle --shuffle-buffer 10000

    # Custom dataset
    python fast_hf_loader.py ~/path/to/tokenizer \\
        --dataset wikitext \\
        --dataset-name wikitext-2-raw-v1 \\
        --split "train[:1000]"

    # Print multiple examples
    python fast_hf_loader.py ~/path/to/tokenizer --num-examples 5
"""

import argparse
import os
import time
from argparse import RawTextHelpFormatter
from functools import partial
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from forgather.ml.datasets import (
    default_tokenize_map_fn,
    fast_load_iterable_dataset,
    preprocess_dataset,
)


class Args(argparse.Namespace):
    """Command-line arguments for the test harness."""

    # Required arguments
    tokenizer_id_or_path: str

    # Dataset selection
    dataset_id_or_path: str
    dataset_name: Optional[str]
    split: str

    # Sharding (for distributed training testing)
    num_shards: int
    shard_index: int
    shard_mode: str

    # Virtual splits (for train/val/test partitioning)
    select_range: Optional[str]

    # Shuffling
    shuffle: bool
    shuffle_seed: Optional[int]
    shuffle_buffer: int

    # Tokenization
    batch_size: int
    add_eos: bool

    # Output control
    num_examples: int
    timing: bool


def main(args: Args):
    """
    Main test harness function.

    This function demonstrates the complete workflow:
    1. Load tokenizer
    2. Load dataset with fast Arrow file caching
    3. Apply preprocessing (sharding, slicing, shuffling, tokenization)
    4. Iterate and display results with timing information
    """
    # ========================================================================
    # Step 1: Load Tokenizer
    # ========================================================================
    print("=" * 70)
    print("Fast HuggingFace Dataset Loader - Test Harness")
    print("=" * 70)
    print()

    print(f"Loading tokenizer from: {args.tokenizer_id_or_path}")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        args.tokenizer_id_or_path
    )
    print(f"  Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"  Vocabulary size: {tokenizer.vocab_size:,}")
    print()

    # ========================================================================
    # Step 2: Load Dataset
    # ========================================================================
    # fast_load_iterable_dataset uses Arrow file caching for instant loading
    # on subsequent runs. The dataset is memory-mapped, not loaded into RAM.
    print(f"Loading dataset: {args.dataset_id_or_path}")
    if args.dataset_name:
        print(f"  Dataset name: {args.dataset_name}")
    print(f"  Split: {args.split}")
    print()

    start_time = time.time()
    raw_dataset = fast_load_iterable_dataset(
        path=args.dataset_id_or_path,
        name=args.dataset_name,
        split=args.split,
    )
    load_time = time.time() - start_time

    print(f"Dataset loaded in {load_time:.3f}s")
    print(f"  Type: {type(raw_dataset).__name__}")
    try:
        total_examples = len(raw_dataset)
        print(f"  Total examples: {total_examples:,}")
    except Exception:
        print("  Total examples: Unknown (IterableDataset)")
    print()

    # ========================================================================
    # Step 3: Preprocess Dataset
    # ========================================================================
    # preprocess_dataset applies transformations in order:
    # 1. select_range - Virtual split (train/val/test partitioning)
    # 2. shuffle - Shuffles data for training
    # 3. shard_dataset - Splits data across GPUs for DDP
    # 4. Tokenization - Converts text to token IDs

    print("Applying preprocessing...")

    # Build shard_dataset config
    shard_config = None
    if args.num_shards > 1:
        shard_config = {
            "num_shards": args.num_shards,
            "index": args.shard_index,
            "mode": args.shard_mode,
        }
        print(f"  Sharding: {args.num_shards} shards, using shard {args.shard_index}")
        print(f"    Mode: {args.shard_mode}")

    # Build select_range config
    if args.select_range:
        print(f"  Select range: {args.select_range}")

    # Build shuffle config
    shuffle_config = None
    if args.shuffle:
        shuffle_config = {
            "seed": args.shuffle_seed,
            "buffer_size": args.shuffle_buffer,
        }
        print(f"  Shuffle: enabled")
        if args.shuffle_seed is not None:
            print(f"    Seed: {args.shuffle_seed}")
        print(f"    Buffer size: {args.shuffle_buffer:,}")

    print(f"  Tokenization batch size: {args.batch_size}")
    print(f"  Add EOS token: {args.add_eos}")
    print()

    # Apply preprocessing
    # This is lazy - transformations happen during iteration, not here
    start_time = time.time()
    tokenized_dataset = preprocess_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        map_fn=partial(default_tokenize_map_fn, add_eos=args.add_eos),
        map_kwargs=dict(batch_size=args.batch_size),
        shard_dataset=shard_config,
        select_range=args.select_range,
        shuffle=shuffle_config if args.shuffle else False,
    )
    preprocess_time = time.time() - start_time

    print(f"Preprocessing configured in {preprocess_time:.3f}s")
    print(f"  Type: {type(tokenized_dataset).__name__}")
    print()

    # ========================================================================
    # Step 4: Iterate and Display Results
    # ========================================================================
    # This is where the actual work happens - iteration triggers:
    # - Loading Arrow files
    # - Applying virtual splits (if configured)
    # - Shuffling (if configured)
    # - Sharding (if configured)
    # - Tokenization
    print("=" * 70)
    print("Iterating Dataset")
    print("=" * 70)
    print()

    # Time to first example (measures startup overhead)
    print("Getting first example...")
    start_time = time.time()
    dataset_iterator = iter(tokenized_dataset)
    first_example = next(dataset_iterator)
    first_example_time = time.time() - start_time

    print(f"Time to first example: {first_example_time:.3f}s")
    if first_example_time < 0.1:
        print("  ✓ Fast startup (efficient seeking working correctly)")
    elif first_example_time < 1.0:
        print("  ✓ Good startup time")
    else:
        print("  ⚠ Slow startup - may indicate performance issue")
    print()

    # Display examples
    print(f"Displaying first {args.num_examples} example(s):")
    print("-" * 70)

    for i in range(args.num_examples):
        if i == 0:
            example = first_example
        else:
            try:
                example = next(dataset_iterator)
            except StopIteration:
                print(f"\nReached end of dataset after {i} examples")
                break

        print(f"\nExample {i + 1}:")
        print(f"  Input IDs shape: {len(example['input_ids'])}")
        print(f"  First 20 token IDs: {example['input_ids'][:20]}")

        # Decode and print text (truncated for readability)
        decoded_text = tokenizer.decode(example["input_ids"])
        max_display_chars = 200
        if len(decoded_text) > max_display_chars:
            decoded_text = decoded_text[:max_display_chars] + "..."
        print(f"  Decoded text: {decoded_text!r}")

    print()
    print("=" * 70)
    print("Test Complete")
    print("=" * 70)

    # Display timing summary if requested
    if args.timing:
        print()
        print("Timing Summary:")
        print(f"  Dataset load: {load_time:.3f}s")
        print(f"  Preprocessing config: {preprocess_time:.3f}s")
        print(f"  Time to first example: {first_example_time:.3f}s")
        print(f"  Total: {load_time + preprocess_time + first_example_time:.3f}s")


def parse_args() -> Args:
    """
    Parse command-line arguments.

    Returns:
        Args object with parsed arguments
    """
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description=__doc__,  # Use module docstring
        epilog=(
            "Performance Testing:\n"
            "  # Compare shard 0 vs shard 1 startup time\n"
            "  time python %(prog)s ~/tokenizer --num-shards 2 --shard-index 0\n"
            "  time python %(prog)s ~/tokenizer --num-shards 2 --shard-index 1\n"
            "\n"
            "  # Test shuffle buffer performance\n"
            "  time python %(prog)s ~/tokenizer --shuffle --shuffle-buffer 1000\n"
            "  time python %(prog)s ~/tokenizer --shuffle --shuffle-buffer 10000\n"
            "\n"
            "  # Test virtual split performance\n"
            "  time python %(prog)s ~/tokenizer --select-range ':10%%'\n"
            "  time python %(prog)s ~/tokenizer --select-range '90%%:'\n"
        ),
    )

    # ========================================================================
    # Required Arguments
    # ========================================================================
    parser.add_argument(
        "tokenizer_id_or_path",
        type=os.path.expanduser,
        help="Path or HuggingFace model ID of tokenizer\n"
        "Examples:\n"
        "  ~/ai_assets/tokenizers/my_tokenizer/\n"
        "  google/flan-t5-small\n",
    )

    # ========================================================================
    # Dataset Selection
    # ========================================================================
    dataset_group = parser.add_argument_group(
        "Dataset Selection", "Configure which dataset and split to load"
    )

    dataset_group.add_argument(
        "--dataset",
        dest="dataset_id_or_path",
        default="HuggingFaceTB/smollm-corpus",
        help="Dataset path or HuggingFace ID\n"
        "(default: %(default)s)\n"
        "Examples:\n"
        "  wikitext\n"
        "  HuggingFaceTB/smollm-corpus\n"
        "  ~/data/my_dataset/\n",
    )

    dataset_group.add_argument(
        "--dataset-name",
        dest="dataset_name",
        default="fineweb-edu-dedup",
        help="Dataset config name (for datasets with multiple configs)\n"
        "(default: %(default)s)\n"
        "Examples:\n"
        "  wikitext-2-raw-v1\n"
        "  fineweb-edu-dedup\n",
    )

    dataset_group.add_argument(
        "--split",
        default="train[10000:]",
        help="Dataset split with optional slice notation\n"
        "(default: %(default)s)\n"
        "Examples:\n"
        "  train            - Full training set\n"
        "  train[:1000]     - First 1000 examples\n"
        "  train[1000:]     - All except first 1000\n"
        "  train[10000:20000] - Examples 10k-20k\n"
        "  validation       - Validation set\n",
    )

    # ========================================================================
    # Sharding (Distributed Training)
    # ========================================================================
    shard_group = parser.add_argument_group(
        "Sharding",
        "Configure sharding for distributed training (DDP)\n"
        "Sharding splits the dataset across multiple GPUs/processes.\n"
        "Use this to test sharding performance or simulate multi-GPU training.",
    )

    shard_group.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of shards to split dataset into\n"
        "(default: %(default)s = no sharding)\n"
        "Examples:\n"
        "  1  - No sharding (single GPU)\n"
        "  2  - Split across 2 GPUs\n"
        "  4  - Split across 4 GPUs\n"
        "  8  - Split across 8 GPUs\n",
    )

    shard_group.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Index of this shard (0 to num_shards-1)\n"
        "(default: %(default)s)\n"
        "Use different indices to test different shards:\n"
        "  0  - First shard (should be fast)\n"
        "  1  - Second shard (tests seeking performance)\n"
        "  N-1 - Last shard (tests maximum seeking)\n",
    )

    shard_group.add_argument(
        "--shard-mode",
        choices=["auto", "file", "example"],
        default="auto",
        help="Sharding mode\n"
        "(default: %(default)s)\n"
        "  auto    - Choose best mode automatically\n"
        "  file    - Shard at file level (faster, requires num_shards <= num_files)\n"
        "  example - Shard at example level (works with any num_shards)\n",
    )

    # ========================================================================
    # Virtual Splits
    # ========================================================================
    split_group = parser.add_argument_group(
        "Virtual Splits",
        "Select a range of examples without copying data\n"
        "This is efficient for train/val/test splits.",
    )

    split_group.add_argument(
        "--select-range",
        help="Select a range of examples using slice notation\n"
        "Examples:\n"
        "  :80%%         - First 80%% (training)\n"
        "  80%%:90%%     - Examples 80%%-90%% (validation)\n"
        "  90%%:         - Last 10%% (test)\n"
        "  :1000        - First 1000 examples\n"
        "  1000:2000    - Examples 1000-2000\n"
        "  50%%:         - Last 50%%\n",
    )

    # ========================================================================
    # Shuffling
    # ========================================================================
    shuffle_group = parser.add_argument_group(
        "Shuffling", "Configure dataset shuffling for training"
    )

    shuffle_group.add_argument(
        "--shuffle",
        action="store_true",
        help="Enable shuffling (disabled by default)\n"
        "Shuffles both file order and example order within a buffer.\n",
    )

    shuffle_group.add_argument(
        "--shuffle-seed",
        type=int,
        help="Random seed for shuffling (default: random)\n"
        "Use a fixed seed for reproducibility:\n"
        "  42  - Common seed for reproducible results\n",
    )

    shuffle_group.add_argument(
        "--shuffle-buffer",
        type=int,
        default=1000,
        help="Shuffle buffer size for example-level shuffling\n"
        "(default: %(default)s)\n"
        "Larger buffers give better randomization but use more memory:\n"
        "  100   - Minimal shuffling\n"
        "  1000  - Good balance (default)\n"
        "  10000 - Strong shuffling\n"
        "  0     - Disable example-level shuffle (file-level only)\n",
    )

    # ========================================================================
    # Tokenization
    # ========================================================================
    token_group = parser.add_argument_group("Tokenization", "Configure tokenization")

    token_group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for tokenization\n"
        "(default: %(default)s)\n"
        "Larger batches are faster but use more memory:\n"
        "  8   - Small batches\n"
        "  32  - Good balance (default)\n"
        "  128 - Large batches (faster)\n",
    )

    token_group.add_argument(
        "--no-eos",
        dest="add_eos",
        action="store_false",
        help="Don't add EOS token to end of sequences\n"
        "(default: EOS token is added)\n",
    )

    # ========================================================================
    # Output Control
    # ========================================================================
    output_group = parser.add_argument_group("Output Control", "Control output display")

    output_group.add_argument(
        "--num-examples",
        type=int,
        default=1,
        help="Number of examples to display\n" "(default: %(default)s)\n",
    )

    output_group.add_argument(
        "--timing",
        action="store_true",
        help="Show detailed timing information\n",
    )

    return parser.parse_args(namespace=Args())


if __name__ == "__main__":
    main(parse_args())
