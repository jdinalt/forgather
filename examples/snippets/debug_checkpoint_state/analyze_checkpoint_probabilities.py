#!/usr/bin/env python3
"""
Analyze checkpoint state and compute soft_sequential probabilities.

This script demonstrates how missing examples_per_dataset in checkpoints
causes incorrect probability calculations on checkpoint resume for
InterleavedDataset with stopping_strategy='soft_sequential'.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def soft_sequential_probabilities(
    datasets_info: List[Dict[str, Any]], examples_per_dataset: List[int]
) -> Tuple[List[float], List[float]]:
    """
    Compute soft_sequential probabilities given dataset info.

    This implements the same logic as InterleavedDataset._update_probabilities_soft_sequential().

    Args:
        datasets_info: List of dicts with 'estimated_length' keys
        examples_per_dataset: List of how many examples already yielded from each

    Returns:
        Tuple of (probabilities, weights)
    """
    weights = []
    remaining_prob = 1.0

    for i, (info, count) in enumerate(zip(datasets_info, examples_per_dataset)):
        estimated_length = info["estimated_length"]
        remaining = max(0, estimated_length - count)

        if estimated_length > 0:
            proportion = remaining / estimated_length
        else:
            proportion = 0.0

        weight = remaining_prob * proportion
        weights.append(weight)
        remaining_prob *= 1.0 - proportion

    # Normalize to probabilities
    total = sum(weights)
    if total > 0:
        probs = [w / total for w in weights]
    else:
        probs = [0.0] * len(weights)

    return probs, weights


def extract_dataset_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract dataset state from checkpoint, handling wrapped and unwrapped formats."""
    # Try StatefulDataLoader snapshot structure first
    if "_snapshot" in state:
        snapshot = state["_snapshot"]
        worker_snapshots = snapshot.get("_worker_snapshots", {})
        worker_0 = worker_snapshots.get("worker_0", {})
        dataset_state = worker_0.get("dataset_state", {})
        return dataset_state

    # Otherwise assume it's already unwrapped
    return state


def extract_estimated_length(child_state: Dict[str, Any]) -> int:
    """
    Extract estimated length from a child dataset state.

    Args:
        child_state: Child dataset state (either SimpleArrow or nested Interleaved)

    Returns:
        Estimated length in examples
    """
    if child_state is None:
        return 0

    # Check if this is a nested InterleavedDataset
    if "child_states" in child_state:
        # Sum the estimated lengths of nested children
        nested_children = child_state.get("child_states", [])
        return sum(extract_estimated_length(nested) for nested in nested_children)

    # SimpleArrowIterableDataset
    input_count = child_state.get("input_count", 0)
    output_count = child_state.get("output_count", 0)
    original_length = child_state.get("original_length", 0)

    if output_count > 0 and input_count > 0 and original_length > 0:
        ratio = output_count / input_count
        return int(original_length * ratio)

    return 0


def extract_examples_yielded(child_state: Dict[str, Any]) -> int:
    """
    Extract the number of examples already yielded from a child dataset.

    Args:
        child_state: Child dataset state (either SimpleArrow or nested Interleaved)

    Returns:
        Number of examples yielded
    """
    if child_state is None:
        return 0

    # Nested InterleavedDataset tracks this directly
    if "current_example_count" in child_state:
        return child_state["current_example_count"]

    # SimpleArrowIterableDataset uses output_count
    return child_state.get("output_count", 0)


def analyze_checkpoint_probabilities(
    checkpoint_dir: str, rank: int = 0, format_type: str = "text"
) -> int:
    """
    Load checkpoint and analyze probability calculations.

    Args:
        checkpoint_dir: Path to checkpoint directory
        rank: Rank number for multi-process checkpoints
        format_type: Output format ("text", "json", or "summary")

    Returns:
        Exit code (0 for success)
    """
    checkpoint_path = Path(checkpoint_dir)
    dataset_state_file = checkpoint_path / f"dataset_state_rank_{rank}.pt"

    if not dataset_state_file.exists():
        # Try without rank suffix
        dataset_state_file = checkpoint_path / "dataset_state.pt"

    if not dataset_state_file.exists():
        print(f"ERROR: Dataset state file not found at either:", file=sys.stderr)
        print(f"  {checkpoint_path / f'dataset_state_rank_{rank}.pt'}", file=sys.stderr)
        print(f"  {checkpoint_path / 'dataset_state.pt'}", file=sys.stderr)
        return 1

    if format_type == "text":
        print(f"Loading dataset state from: {dataset_state_file}")
        print("=" * 80)

    try:
        state = torch.load(dataset_state_file, map_location="cpu")
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}", file=sys.stderr)
        return 1

    # Extract dataset state from wrapper
    dataset_state = extract_dataset_state(state)

    # Get top-level info
    stopping_strategy = dataset_state.get("stopping_strategy", "unknown")
    child_states = dataset_state.get("child_states", [])

    # Check if this checkpoint even uses InterleavedDataset
    if not child_states:
        if format_type == "text":
            print("This checkpoint does not appear to use InterleavedDataset.")
            print("No child_states found in dataset state.")
        return 0

    if format_type == "text":
        print(f"\n### DATASET STRUCTURE ###")
        print(f"Stopping strategy: {stopping_strategy}\n")

    # Parse child datasets
    datasets_info = []
    correct_examples_per_dataset = []

    for i, child_state in enumerate(child_states):
        if child_state is None:
            continue

        estimated_length = extract_estimated_length(child_state)
        examples_yielded = extract_examples_yielded(child_state)

        # Determine dataset type/name
        if "child_states" in child_state:
            name = f"Dataset {i} (nested InterleavedDataset)"
        else:
            # Try to identify by packing ratio
            input_count = child_state.get("input_count", 0)
            output_count = child_state.get("output_count", 0)
            if output_count > 0 and input_count > 0:
                ratio = output_count / input_count
                if ratio < 0.1:
                    name = f"Dataset {i} (high packing - ratio {ratio:.4f})"
                elif ratio > 0.9:
                    name = f"Dataset {i} (low packing - ratio {ratio:.4f})"
                else:
                    name = f"Dataset {i} (packing ratio {ratio:.4f})"
            else:
                name = f"Dataset {i}"

        datasets_info.append(
            {
                "name": name,
                "estimated_length": estimated_length,
            }
        )
        correct_examples_per_dataset.append(examples_yielded)

        if format_type == "text":
            print(f"{name}")
            print(f"  Estimated length: {estimated_length:,}")
            print(f"  Already yielded: {examples_yielded:,}")

            if estimated_length > 0:
                pct_consumed = (examples_yielded / estimated_length) * 100
                print(f"  Consumed: {pct_consumed:.2f}%")
            print()

    # Check if examples_per_dataset was saved
    has_examples_per_dataset = "examples_per_dataset" in dataset_state

    if format_type == "json":
        output = {
            "checkpoint_path": str(dataset_state_file),
            "stopping_strategy": stopping_strategy,
            "has_examples_per_dataset": has_examples_per_dataset,
            "datasets": [
                {
                    "name": info["name"],
                    "estimated_length": info["estimated_length"],
                    "examples_yielded": correct_examples_per_dataset[i],
                }
                for i, info in enumerate(datasets_info)
            ],
        }

        if stopping_strategy == "soft_sequential":
            probs_correct, _ = soft_sequential_probabilities(
                datasets_info, correct_examples_per_dataset
            )
            buggy_examples = [0] * len(datasets_info)
            probs_buggy, _ = soft_sequential_probabilities(
                datasets_info, buggy_examples
            )

            output["correct_probabilities"] = probs_correct
            output["buggy_probabilities"] = (
                probs_buggy if not has_examples_per_dataset else None
            )

        print(json.dumps(output, indent=2))
        return 0

    if format_type == "summary":
        print(f"Checkpoint: {checkpoint_path.name}")
        print(f"  Strategy: {stopping_strategy}")
        print(
            f"  examples_per_dataset: {'SAVED' if has_examples_per_dataset else 'NOT SAVED'}"
        )

        if stopping_strategy == "soft_sequential" and not has_examples_per_dataset:
            print(f"  WARNING: Bug will affect probability calculations on resume!")

        return 0

    # Full text analysis
    if stopping_strategy != "soft_sequential":
        print("=" * 80)
        print(f"\nStopping strategy is '{stopping_strategy}', not 'soft_sequential'.")
        print("This analysis is specific to soft_sequential behavior.")
        return 0

    print("=" * 80)
    print("\n### PROBABILITY ANALYSIS ###\n")

    # Scenario 1: Correct behavior
    print("SCENARIO 1: Correct behavior (if examples_per_dataset was saved)")
    print("-" * 80)

    probs_correct, weights_correct = soft_sequential_probabilities(
        datasets_info, correct_examples_per_dataset
    )

    for info, count, prob, weight in zip(
        datasets_info, correct_examples_per_dataset, probs_correct, weights_correct
    ):
        remaining = info["estimated_length"] - count
        print(f"{info['name']}:")
        print(f"  Estimated length: {info['estimated_length']:,}")
        print(f"  Already yielded: {count:,}")
        print(f"  Remaining: {remaining:,}")

        if info["estimated_length"] > 0:
            proportion = remaining / info["estimated_length"]
            print(f"  Proportion remaining: {proportion:.4f}")

        print(f"  Weight: {weight:.6f}")
        print(f"  Probability: {prob:.4f} ({prob*100:.2f}%)")
        print()

    # Scenario 2: Bug - examples_per_dataset reset to zeros
    if not has_examples_per_dataset:
        print("\nSCENARIO 2: BUG - examples_per_dataset reset to [0, 0, ...]")
        print("-" * 80)
        print("WARNING: examples_per_dataset was NOT saved in this checkpoint!")
        print("On resume, it will be initialized to all zeros.\n")

        buggy_examples_per_dataset = [0] * len(datasets_info)

        probs_buggy, weights_buggy = soft_sequential_probabilities(
            datasets_info, buggy_examples_per_dataset
        )

        for i, (info, prob, weight) in enumerate(
            zip(datasets_info, probs_buggy, weights_buggy)
        ):
            remaining = info["estimated_length"]  # All examples appear to remain
            print(f"{info['name']}:")
            print(f"  Estimated length: {info['estimated_length']:,}")
            print(
                f"  Already yielded: 0 (INCORRECT - should be {correct_examples_per_dataset[i]:,})"
            )
            print(f"  Remaining: {remaining:,} (INCORRECT)")
            print(f"  Proportion remaining: 1.0000 (INCORRECT)")
            print(f"  Weight: {weight:.6f}")
            print(f"  Probability: {prob:.4f} ({prob*100:.2f}%)")
            print()

        print("\n" + "=" * 80)
        print("\n### IMPACT ###\n")

        print("The bug causes dramatic differences in sampling probabilities:\n")
        for i, info in enumerate(datasets_info):
            correct_prob = probs_correct[i]
            buggy_prob = probs_buggy[i]

            print(f"{info['name']}:")
            print(f"  Correct probability: {correct_prob*100:.2f}%")
            print(f"  Buggy probability: {buggy_prob*100:.2f}%")

            if correct_prob > 0.0001:
                change_ratio = buggy_prob / correct_prob
                print(f"  Change: {change_ratio:.2f}x")
            else:
                print(f"  Change: {buggy_prob*100:.2f}% (from near zero)")

            # Highlight nearly-exhausted datasets that get revived
            if correct_examples_per_dataset[i] > 0:
                pct_consumed = (
                    correct_examples_per_dataset[i] / max(1, info["estimated_length"])
                ) * 100
                if pct_consumed > 95.0:
                    print(
                        f"\n  WARNING: This dataset was {pct_consumed:.2f}% consumed before resume"
                    )
                    print(
                        f"  but will get {buggy_prob*100:.2f}% sampling probability after resume!"
                    )
                    print(f"  This can cause sudden shifts in training dynamics.\n")
            else:
                print()

    else:
        print("\nGood news: examples_per_dataset WAS saved in this checkpoint.")
        print("Probability calculations will be correct on resume.")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Analyze checkpoint probabilities for InterleavedDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script analyzes how InterleavedDataset computes sampling probabilities
with stopping_strategy='soft_sequential', and demonstrates the bug where
missing examples_per_dataset causes incorrect calculations on checkpoint resume.

Examples:
  # Basic usage
  %(prog)s /path/to/checkpoint-36621

  # Analyze specific rank
  %(prog)s /path/to/checkpoint-36621 --rank 1

  # JSON output for programmatic processing
  %(prog)s /path/to/checkpoint-36621 --format json

  # Quick summary
  %(prog)s /path/to/checkpoint-36621 --format summary

For detailed context on this bug, see:
  CHECKPOINT_BUG_ANALYSIS.md
        """,
    )

    parser.add_argument("checkpoint_dir", help="Path to checkpoint directory")
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank number for multi-process checkpoints (default: 0)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "summary"],
        default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()

    return analyze_checkpoint_probabilities(args.checkpoint_dir, args.rank, args.format)


if __name__ == "__main__":
    sys.exit(main())
