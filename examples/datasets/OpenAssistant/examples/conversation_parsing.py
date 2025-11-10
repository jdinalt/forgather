#!/usr/bin/env python3
"""
Conversation parsing example for OpenAssistant dataset.

This example demonstrates:
- Parsing formatted text output
- Inspecting conversation structure
- Working with multi-turn dialogues
- Analyzing conversation patterns
"""

import sys
from pathlib import Path
import re

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openassistant import OpenAssistantDatasetDict


def parse_conversation(text):
    """Parse ChatML formatted text into messages."""
    # Split by role markers
    pattern = r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>"
    matches = re.findall(pattern, text, re.DOTALL)

    messages = []
    for role, content in matches:
        messages.append({"role": role, "content": content.strip()})

    return messages


def main():
    print("=" * 60)
    print("OpenAssistant Dataset - Conversation Parsing Example")
    print("=" * 60)

    # Create dataset with default template
    print("\n1. Creating dataset...")
    dataset_dict = OpenAssistantDatasetDict(
        tokenizer=None,  # Not needed for this example
        chat_template="",  # Uses default ChatML template
        languages=["en"],
        seed=42,
    )
    print("   Dataset created with default ChatML formatting")

    # Access train split
    print("\n2. Accessing and parsing conversations...")
    train_dataset = dataset_dict["train"]

    # Iterate through examples
    print("\n3. Showing parsed conversations...")
    for i, example in enumerate(train_dataset):
        if i >= 3:  # Only show first 3 examples
            break

        print("\n" + "=" * 60)
        print(f"Conversation {i+1}:")
        print("=" * 60)

        # Parse the formatted text
        text = example["text"]
        messages = parse_conversation(text)

        print(f"Number of turns: {len(messages)}")
        print("-" * 60)

        for j, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]

            # Truncate long content
            if len(content) > 200:
                content = content[:200] + "..."

            print(f"\nTurn {j+1} [{role}]:")
            print(content)

        print("-" * 60)

    print("\n" + "=" * 60)
    print("Example complete!")
    print("\nNote: This example shows how to parse ChatML formatted output")
    print("into structured messages for analysis or custom processing.")
    print("=" * 60)


if __name__ == "__main__":
    main()
