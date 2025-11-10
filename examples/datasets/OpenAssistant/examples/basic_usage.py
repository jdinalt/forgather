#!/usr/bin/env python3
"""
Basic usage example for OpenAssistant dataset.

This example demonstrates:
- Loading the dataset with default configuration
- Accessing different splits (train, validation, test)
- Iterating through examples
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openassistant import OpenAssistantDatasetDict, OpenAssistantConfig
from transformers import AutoTokenizer


def main():
    print("=" * 60)
    print("OpenAssistant Dataset - Basic Usage Example")
    print("=" * 60)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer_path = Path.home() / "ai_assets/models/fg_mistral/"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    print(f"   Loaded tokenizer from {tokenizer_path}")

    # Create configuration
    print("\n2. Creating configuration...")
    config = OpenAssistantConfig(
        languages=["en"],
        min_quality=0.5,
        min_thread_length=2,
        max_thread_length=7,
        exclude_deleted=True,
        exclude_synthetic=True,
        branch_temperature=1.0,
        seed=42,
        val_split=10,
        test_split=10,
    )
    print(f"   Languages: {config.languages}")
    print(f"   Min quality: {config.min_quality}")

    # Create dataset dict
    print("\n3. Creating dataset dict...")
    dataset_dict = OpenAssistantDatasetDict(
        tokenizer=tokenizer,
        chat_template="",  # Use tokenizer's chat template
        **config.__dict__,
    )
    print(f"   Available splits: {list(dataset_dict.tree_databases.keys())}")

    # Access splits
    print("\n4. Accessing splits...")
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["validation"]
    test_dataset = dataset_dict["test"]
    print("   ✓ Train split created")
    print("   ✓ Validation split created")
    print("   ✓ Test split created")

    # Iterate through examples
    print("\n5. Generating examples from train split...")
    print("-" * 60)
    for i, example in enumerate(train_dataset):
        if i >= 3:  # Only show first 3 examples
            break
        print(f"\nExample {i+1}:")
        print(
            example["text"][:300] + "..."
            if len(example["text"]) > 300
            else example["text"]
        )
        print("-" * 60)

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
