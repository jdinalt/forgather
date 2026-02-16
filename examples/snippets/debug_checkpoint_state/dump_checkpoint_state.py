#!/usr/bin/env python3
"""
Dump and analyze dataset checkpoint state.

This script loads the dataset checkpoint state and displays all the critical
information needed to debug checkpoint-related issues, particularly with
InterleavedDataset and packing ratio estimation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def extract_dataset_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract dataset state from checkpoint, handling both wrapped and unwrapped formats.

    Args:
        state: Raw checkpoint state dict

    Returns:
        Extracted dataset state dict
    """
    # Try StatefulDataLoader snapshot structure first
    if "_snapshot" in state:
        snapshot = state["_snapshot"]
        worker_snapshots = snapshot.get("_worker_snapshots", {})
        worker_0 = worker_snapshots.get("worker_0", {})
        dataset_state = worker_0.get("dataset_state", {})
        return dataset_state

    # Otherwise assume it's already unwrapped
    return state


def format_number(n: Optional[int]) -> str:
    """Format a number with thousands separators or return 'None'."""
    if n is None:
        return "None"
    return f"{n:,}"


def print_simple_arrow_state(
    state: Dict[str, Any], indent: str = "", format_type: str = "text"
):
    """
    Print SimpleArrowIterableDataset state with analysis.

    Args:
        state: Dataset state dict
        indent: Indentation prefix for output
        format_type: Output format ("text" or "json")
    """
    if format_type == "json":
        # For JSON, just return the relevant fields
        return {
            "current_file_index": state.get("current_file_index"),
            "current_example_index": state.get("current_example_index"),
            "num_files": state.get("num_files"),
            "length_estimate_mode": state.get("length_estimate_mode"),
            "input_count": state.get("input_count", 0),
            "output_count": state.get("output_count", 0),
            "original_length": state.get("original_length"),
            "cached_exact_length": state.get("cached_exact_length"),
            "shuffle_seed": state.get("shuffle_seed"),
            "base_shuffle_seed": state.get("base_shuffle_seed"),
            "epoch": state.get("epoch"),
        }

    # Text format
    print(f"{indent}current_file_index: {state.get('current_file_index')}")
    print(f"{indent}current_example_index: {state.get('current_example_index')}")
    print(f"{indent}num_files: {state.get('num_files')}")

    # Length estimation state - CRITICAL for debugging packing issues
    input_count = state.get("input_count", 0)
    output_count = state.get("output_count", 0)
    original_length = state.get("original_length")
    cached_exact_length = state.get("cached_exact_length")

    print(f"\n{indent}Length Estimation State:")
    print(f"{indent}  length_estimate_mode: {state.get('length_estimate_mode')}")
    print(f"{indent}  input_count: {format_number(input_count)}")
    print(f"{indent}  output_count: {format_number(output_count)}")
    print(f"{indent}  original_length: {format_number(original_length)}")
    print(f"{indent}  cached_exact_length: {format_number(cached_exact_length)}")

    # Compute the estimated length and packing ratio
    if output_count > 0 and input_count > 0 and original_length is not None:
        ratio = output_count / input_count
        estimated_length = int(original_length * ratio)
        print(f"{indent}  packing_ratio: {ratio:.4f}")
        print(f"{indent}  estimated_length: {format_number(estimated_length)}")

        # Warning about examples_per_dataset not being saved
        if estimated_length > 0:
            print(
                f"\n{indent}NOTE: If examples_per_dataset is not saved in parent InterleavedDataset:"
            )
            print(
                f"{indent}  Remaining examples would be computed as: {format_number(estimated_length)} - 0 = {format_number(estimated_length)}"
            )
            print(
                f"{indent}  This would make soft_sequential think all examples remain!"
            )

    print(f"\n{indent}Shuffle state:")
    print(f"{indent}  shuffle_seed: {state.get('shuffle_seed')}")
    print(f"{indent}  base_shuffle_seed: {state.get('base_shuffle_seed')}")
    print(f"{indent}  epoch: {state.get('epoch')}")


def analyze_interleaved_dataset(
    dataset_state: Dict[str, Any], indent: str = "", depth: int = 0
):
    """
    Recursively analyze InterleavedDataset state.

    Args:
        dataset_state: InterleavedDataset state dict
        indent: Indentation prefix for output
        depth: Current nesting depth
    """
    prefix = "Nested " * depth if depth > 0 else "Top-level "

    print(f"{indent}{prefix}InterleavedDataset state:")
    print(
        f"{indent}  current_dataset_index: {dataset_state.get('current_dataset_index')}"
    )
    print(
        f"{indent}  current_example_count: {format_number(dataset_state.get('current_example_count'))}"
    )
    print(f"{indent}  datasets_exhausted: {dataset_state.get('datasets_exhausted')}")
    print(f"{indent}  probabilities: {dataset_state.get('probabilities')}")
    print(f"{indent}  seed: {dataset_state.get('seed')}")
    print(f"{indent}  stopping_strategy: {dataset_state.get('stopping_strategy')}")

    # Check if examples_per_dataset is saved (critical for soft_sequential)
    if "examples_per_dataset" in dataset_state:
        examples = dataset_state["examples_per_dataset"]
        print(f"{indent}  examples_per_dataset: {examples}")
    else:
        print(f"{indent}  examples_per_dataset: NOT SAVED IN CHECKPOINT!")
        print(
            f"{indent}  WARNING: This will cause incorrect probability calculations on resume"
        )
        print(f"{indent}           for stopping_strategy='soft_sequential'")

    print()


def dump_checkpoint_state(
    checkpoint_dir: str,
    rank: int = 0,
    format_type: str = "text",
    show_trainer_state: bool = True,
):
    """
    Load and dump dataset state from checkpoint.

    Args:
        checkpoint_dir: Path to checkpoint directory
        rank: Rank number for multi-process checkpoints
        format_type: Output format ("text", "json", or "summary")
        show_trainer_state: Whether to also show trainer state
    """
    checkpoint_path = Path(checkpoint_dir)

    # Load the dataset state for specified rank
    dataset_state_file = checkpoint_path / f"dataset_state_rank_{rank}.pt"

    if not dataset_state_file.exists():
        # Try without rank suffix (single-process checkpoint)
        dataset_state_file = checkpoint_path / "dataset_state.pt"

    if not dataset_state_file.exists():
        print(f"ERROR: Dataset state file not found at either:", file=sys.stderr)
        print(f"  {checkpoint_path / f'dataset_state_rank_{rank}.pt'}", file=sys.stderr)
        print(f"  {checkpoint_path / 'dataset_state.pt'}", file=sys.stderr)
        return 1

    if format_type == "text":
        print(f"Loading dataset state from: {dataset_state_file}")
        print("=" * 80)

    # Load the state
    try:
        state = torch.load(dataset_state_file, map_location="cpu")
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}", file=sys.stderr)
        return 1

    if format_type == "json":
        # JSON output - just dump the raw state
        output = {
            "checkpoint_path": str(dataset_state_file),
            "raw_state": state,
        }
        print(json.dumps(output, indent=2, default=str))
        return 0

    # Extract dataset state from wrapper if needed
    dataset_state = extract_dataset_state(state)

    if format_type == "summary":
        # Compact summary format
        print(f"Checkpoint: {checkpoint_path.name}")
        print(
            f"  current_example_count: {format_number(dataset_state.get('current_example_count'))}"
        )
        print(
            f"  examples_per_dataset: {'SAVED' if 'examples_per_dataset' in dataset_state else 'NOT SAVED'}"
        )
        print(f"  num_children: {len(dataset_state.get('child_states', []))}")
        return 0

    # Full text output
    print("\n### RAW CHECKPOINT STATE ###")
    print(json.dumps(state, indent=2, default=str))
    print("\n" + "=" * 80)

    print("\n### ANALYSIS ###\n")

    # Determine if this is wrapped in StatefulDataLoader
    if state != dataset_state:
        print("(Extracted dataset_state from StatefulDataLoader snapshot)\n")

    # Analyze top-level InterleavedDataset
    analyze_interleaved_dataset(dataset_state)

    # Analyze child datasets
    child_states = dataset_state.get("child_states", [])
    print(f"Number of child datasets: {len(child_states)}")
    print()

    for i, child_state in enumerate(child_states):
        print(f"### Child Dataset {i} ###")

        if child_state is None:
            print("  State: None")
            continue

        # Check if this is another InterleavedDataset (nested)
        if "child_states" in child_state:
            print("  Type: InterleavedDataset (nested)")
            analyze_interleaved_dataset(child_state, indent="  ", depth=1)

            # Check nested children
            nested_children = child_state.get("child_states", [])
            print(f"  Number of nested children: {len(nested_children)}")

            for j, nested_state in enumerate(nested_children):
                print(f"\n  ### Nested Child {j} ###")
                if nested_state is None:
                    print("    State: None")
                else:
                    print("    Type: SimpleArrowIterableDataset")
                    print_simple_arrow_state(
                        nested_state, indent="    ", format_type="text"
                    )
        else:
            print("  Type: SimpleArrowIterableDataset")
            print_simple_arrow_state(child_state, indent="  ", format_type="text")

        print()

    # Show trainer state if requested
    if show_trainer_state:
        trainer_state_file = checkpoint_path / "trainer_state.pt"
        if trainer_state_file.exists():
            print("\n" + "=" * 80)
            print("### TRAINER STATE ###")
            try:
                trainer_state = torch.load(trainer_state_file, map_location="cpu")
                print(f"global_step: {format_number(trainer_state.get('global_step'))}")
                print(f"epoch: {trainer_state.get('epoch')}")
                print(f"total_steps: {format_number(trainer_state.get('total_steps'))}")
            except Exception as e:
                print(f"Warning: Could not load trainer state: {e}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Dump and analyze dataset checkpoint state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - analyze checkpoint from rank 0
  %(prog)s /path/to/checkpoint-36621

  # Analyze specific rank
  %(prog)s /path/to/checkpoint-36621 --rank 1

  # JSON output for programmatic processing
  %(prog)s /path/to/checkpoint-36621 --format json

  # Summary output for quick overview
  %(prog)s /path/to/checkpoint-36621 --format summary

  # Skip trainer state display
  %(prog)s /path/to/checkpoint-36621 --no-trainer-state

For more context on checkpoint debugging, see:
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
    parser.add_argument(
        "--no-trainer-state", action="store_true", help="Do not display trainer state"
    )

    args = parser.parse_args()

    return dump_checkpoint_state(
        args.checkpoint_dir,
        args.rank,
        args.format,
        show_trainer_state=not args.no_trainer_state,
    )


if __name__ == "__main__":
    sys.exit(main())
