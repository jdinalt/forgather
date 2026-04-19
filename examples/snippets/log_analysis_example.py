#!/usr/bin/env python
"""Example: Analyzing training logs programmatically.

This example demonstrates how to use the log analysis API to:
1. Load training logs
2. Compute summary statistics
3. Generate plots
4. Compare multiple training runs
"""

from pathlib import Path

from forgather.ml.analysis import (
    TrainingLog,
    compute_summary_statistics,
    plot_training_metrics,
)
from forgather.ml.analysis.log_parser import find_log_files
from forgather.ml.analysis.metrics import format_summary_text
from forgather.ml.analysis.plotting import plot_loss_curves


def analyze_single_run(log_path: str):
    """Analyze a single training run."""
    print(f"\n{'='*60}")
    print("Analyzing Single Training Run")
    print(f"{'='*60}\n")

    # Load log
    log = TrainingLog.from_file(log_path)
    print(f"Loaded log: {log.run_name}")
    print(f"Total records: {len(log.records)}")

    # Compute summary statistics
    summary = compute_summary_statistics(log)

    # Print formatted summary
    print("\n" + format_summary_text(summary))

    # Extract specific metrics
    print("\nExtracting specific metrics:")
    train_records = log.get_training_records()
    print(f"  Training records: {len(train_records)}")
    print(f"  Evaluation records: {len(log.get_eval_records())}")

    # Find best checkpoint
    best_step, best_loss = log.find_best_step("eval_loss", mode="min")
    print(f"  Best checkpoint: step {best_step}, eval_loss {best_loss:.4f}")


def compare_multiple_runs(log_paths: list[str], output_dir: str):
    """Compare multiple training runs."""
    print(f"\n{'='*60}")
    print("Comparing Multiple Training Runs")
    print(f"{'='*60}\n")

    # Load all logs
    logs = []
    for path in log_paths:
        log = TrainingLog.from_file(path)
        logs.append(log)
        print(f"Loaded: {log.run_name}")

    # Generate comparison plot
    output_path = Path(output_dir) / "comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating comparison plot...")
    plot_loss_curves(
        logs=logs,
        smooth_window=5,
        output_path=str(output_path),
        show=False,
    )
    print(f"Saved to: {output_path}")

    # Compare summary statistics
    print("\nComparison Summary:")
    print(f"{'Run Name':<30} {'Best Loss':<12} {'Best Eval Loss':<15} {'Samples/sec':<12}")
    print("-" * 75)

    for log in logs:
        summary = compute_summary_statistics(log)
        run_name = summary.get("run_name", "Unknown")[:28]
        best_loss = summary.get("best_loss", float("nan"))
        best_eval = summary.get("best_eval_loss", float("nan"))
        samples_sec = summary.get("train_samples_per_second", float("nan"))

        print(
            f"{run_name:<30} {best_loss:<12.4f} {best_eval:<15.4f} {samples_sec:<12.2f}"
        )


def find_and_analyze_latest(project_dir: str):
    """Find and analyze the latest training log in a project."""
    print(f"\n{'='*60}")
    print("Finding Latest Training Run")
    print(f"{'='*60}\n")

    # Find all logs
    log_files = find_log_files(project_dir)

    if not log_files:
        print(f"No training logs found in {project_dir}")
        return

    print(f"Found {len(log_files)} training logs")
    print(f"Latest: {log_files[0]}\n")

    # Analyze latest
    analyze_single_run(str(log_files[0]))


def main():
    """Run all examples."""
    # Example 1: Analyze single run
    example_log = "examples/tiny_experiments/ddp_trainer/output_models/default_model/runs/iterable_2026-01-27T06-25-07/trainer_logs.json"

    if Path(example_log).exists():
        analyze_single_run(example_log)
    else:
        print(f"Example log not found: {example_log}")

    # Example 2: Find and analyze latest in project
    project_dir = "examples/tiny_experiments/ddp_trainer"
    if Path(project_dir).exists():
        find_and_analyze_latest(project_dir)

    # Example 3: Compare multiple runs
    log_paths = [
        "examples/tiny_experiments/ddp_trainer/output_models/default_model/runs/iterable_2026-01-27T06-25-07/trainer_logs.json",
        "examples/tiny_experiments/ddp_trainer/output_models/default_model/runs/sharded_fast_2026-01-26T11-09-33/trainer_logs.json",
    ]

    existing_logs = [p for p in log_paths if Path(p).exists()]
    if len(existing_logs) >= 2:
        compare_multiple_runs(existing_logs, "/tmp/log_analysis_output")
    else:
        print("\nSkipping comparison example (insufficient logs found)")


if __name__ == "__main__":
    main()
