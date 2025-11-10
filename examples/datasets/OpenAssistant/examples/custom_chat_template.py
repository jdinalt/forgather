#!/usr/bin/env python3
"""
Custom chat template example for OpenAssistant dataset.

This example demonstrates:
- Creating a custom chat template
- Using the template with the dataset
- Configuring dataset parameters
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openassistant import OpenAssistantDatasetDict
from transformers import AutoTokenizer


def main():
    print("=" * 60)
    print("OpenAssistant Dataset - Custom Chat Template Example")
    print("=" * 60)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer_path = Path.home() / "ai_assets/models/fg_mistral/"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    # Define custom template
    print("\n2. Creating custom chat template...")
    custom_template = """{% for message in messages %}
<|{{ message['role'] }}|>
{{ message['content'] }}
<|end|>
{% endfor %}"""

    # Save template to file
    template_path = Path(__file__).parent / "custom_template.jinja"
    template_path.write_text(custom_template)
    print(f"   Saved template to: {template_path}")

    # Create dataset with custom template
    print("\n3. Creating dataset with custom configuration...")
    dataset_dict = OpenAssistantDatasetDict(
        tokenizer=tokenizer,
        chat_template=str(template_path),
        languages=["en", "es"],  # Multiple languages
        min_quality=0.6,  # Higher quality threshold
        branch_temperature=0.5,  # More deterministic branching
        seed=123,
    )
    print(f"   Languages: {dataset_dict.config.languages}")
    print(f"   Min quality: {dataset_dict.config.min_quality}")
    print(f"   Branch temperature: {dataset_dict.config.branch_temperature}")

    # Access train split
    print("\n4. Generating examples...")
    print("-" * 60)
    train_dataset = dataset_dict["train"]

    for i, example in enumerate(train_dataset):
        if i >= 2:  # Only show first 2 examples
            break
        print(f"\nExample {i+1}:")
        print(
            example["text"][:400] + "..."
            if len(example["text"]) > 400
            else example["text"]
        )
        print("-" * 60)

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)

    # Clean up template file
    template_path.unlink()


if __name__ == "__main__":
    main()
