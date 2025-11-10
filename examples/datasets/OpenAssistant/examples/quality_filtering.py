#!/usr/bin/env python3
"""
Quality filtering example for OpenAssistant dataset.

This example demonstrates:
- Filtering by quality threshold
- Comparing different quality settings
- Analyzing conversation characteristics
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openassistant import OpenAssistantDatasetDict
from transformers import AutoTokenizer


def show_examples(dataset, num_examples=2, prefix=""):
    """Helper to show examples from a dataset."""
    for i, example in enumerate(dataset):
        if i >= num_examples:
            break

        print(f"\n{prefix}Example {i+1}:")
        print("-" * 60)
        text = example["text"]
        if len(text) > 300:
            text = text[:300] + "..."
        print(text)
        print("-" * 60)


def main():
    print("=" * 60)
    print("OpenAssistant Dataset - Quality Filtering Example")
    print("=" * 60)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer_path = Path.home() / "ai_assets/models/fg_mistral/"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    # Create dataset with NO quality filter
    print("\n2. Creating dataset WITHOUT quality filter...")
    dataset_no_filter = OpenAssistantDatasetDict(
        tokenizer=tokenizer,
        chat_template="",
        languages=["en"],
        min_quality=None,  # No quality filter
        min_thread_length=2,
        max_thread_length=7,
        seed=42,
    )
    print("   Min quality: None (all qualities accepted)")

    train_no_filter = dataset_no_filter["train"]
    print("\n   Showing examples (no quality filter):")
    show_examples(train_no_filter, num_examples=2, prefix="   ")

    # Create dataset with HIGH quality filter
    print("\n3. Creating dataset WITH high quality filter...")
    dataset_high_quality = OpenAssistantDatasetDict(
        tokenizer=tokenizer,
        chat_template="",
        languages=["en"],
        min_quality=0.7,  # High quality threshold
        min_thread_length=3,  # Slightly longer conversations
        max_thread_length=7,
        seed=42,
    )
    print("   Min quality: 0.7 (high quality only)")
    print("   Min thread length: 3 (longer conversations)")

    train_high_quality = dataset_high_quality["train"]
    print("\n   Showing examples (high quality filter):")
    show_examples(train_high_quality, num_examples=2, prefix="   ")

    # Create dataset with temperature variation
    print("\n4. Comparing branch temperature settings...")
    print("\n   Low temperature (0.3) - more deterministic branching:")
    dataset_low_temp = OpenAssistantDatasetDict(
        tokenizer=tokenizer,
        chat_template="",
        languages=["en"],
        branch_temperature=0.3,  # More deterministic
        seed=100,
    )
    train_low_temp = dataset_low_temp["train"]
    show_examples(train_low_temp, num_examples=1, prefix="   ")

    print("\n   High temperature (2.0) - more random branching:")
    dataset_high_temp = OpenAssistantDatasetDict(
        tokenizer=tokenizer,
        chat_template="",
        languages=["en"],
        branch_temperature=2.0,  # More random
        seed=100,  # Same seed for comparison
    )
    train_high_temp = dataset_high_temp["train"]
    show_examples(train_high_temp, num_examples=1, prefix="   ")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("\nKey insights:")
    print("- Higher min_quality filters out lower quality conversations")
    print("- Lower branch_temperature favors higher quality branches")
    print("- Higher branch_temperature gives more uniform sampling")
    print("=" * 60)


if __name__ == "__main__":
    main()
