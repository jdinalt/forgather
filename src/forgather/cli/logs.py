"""CLI commands for training log analysis."""

import argparse
import json
import os
import sys
from pathlib import Path

from forgather.ml.analysis import (
    TrainingLog,
    compute_summary_statistics,
    plot_training_metrics,
)
from forgather.ml.analysis.log_parser import find_log_files
from forgather.ml.analysis.metrics import (
    format_summary_markdown,
    format_summary_oneline,
    format_summary_text,
)
from forgather.ml.analysis.plotting import plot_loss_curves

from .utils import _open_in_editor


def summary_cmd(args):
    """Generate summary statistics from training logs."""
    # Determine which logs to process
    if args.all:
        # Process all logs in project
        if args.log_path:
            print("Warning: Ignoring log_path argument when --all is specified", file=sys.stderr)
        log_files = find_log_files(args.project_dir)
        if not log_files:
            print("Error: No training logs found in project.")
            print(f"Searched in: {Path(args.project_dir) / 'output_models'}")
            sys.exit(1)
        log_paths = log_files
    elif args.log_path:
        log_paths = [Path(args.log_path)]
    else:
        # Try to find latest log in current project
        log_files = find_log_files(args.project_dir)
        if not log_files:
            print("Error: No training logs found in project.")
            print(f"Searched in: {Path(args.project_dir) / 'output_models'}")
            sys.exit(1)
        log_paths = [log_files[0]]
        if args.format != "one-line":
            print(f"Using latest log: {log_paths[0]}")
            print()

    # Process each log
    outputs = []
    for log_path in log_paths:
        # Load log
        try:
            if log_path.is_dir():
                log = TrainingLog.from_run_dir(log_path)
            else:
                log = TrainingLog.from_file(log_path)
        except Exception as e:
            print(f"Error loading log {log_path}: {e}", file=sys.stderr)
            continue

        # Compute summary
        summary = compute_summary_statistics(log)

        # Format output
        if args.format == "json":
            output = json.dumps(summary, indent=2)
        elif args.format == "md":
            output = format_summary_markdown(summary)
        elif args.format == "one-line":
            output = format_summary_oneline(summary)
        else:  # text
            output = format_summary_text(summary)

        outputs.append(output)

    # Combine outputs
    if args.format == "one-line":
        # Add header for one-line format
        if len(outputs) > 1 or args.all:
            header = f"{'Run Name':<32} | {'Steps':<11} | {'Time':<12} | Loss     | Eval     | Throughput"
            separator = "-" * len(header)
            final_output = "\n".join([header, separator] + outputs)
        else:
            final_output = "\n".join(outputs)
    elif args.format == "json" and len(outputs) > 1:
        # Combine JSON outputs into an array
        final_output = "[\n" + ",\n".join(outputs) + "\n]"
    else:
        # Separate multiple outputs with blank lines
        final_output = "\n\n".join(outputs)

    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(final_output)
        print(f"Summary written to: {output_path}")
    else:
        print(final_output)


def plot_cmd(args):
    """Generate plots from training logs."""
    # Determine log paths
    log_paths = []

    if args.log_paths:
        # Use specified log paths
        log_paths = [Path(p) for p in args.log_paths]
    elif args.compare:
        # Compare mode with explicit paths
        log_paths = [Path(p) for p in args.compare]
    else:
        # Try to find latest log in current project
        log_files = find_log_files(args.project_dir)
        if not log_files:
            print("Error: No training logs found in project.")
            print(f"Searched in: {Path(args.project_dir) / 'output_models'}")
            sys.exit(1)
        log_paths = [log_files[0]]

    # Load logs
    logs = []
    labels = getattr(args, "labels", None) or []
    for idx, log_path in enumerate(log_paths):
        try:
            if log_path.is_dir():
                log = TrainingLog.from_run_dir(log_path)
            else:
                log = TrainingLog.from_file(log_path)
            # Apply custom label if provided
            if idx < len(labels):
                log.label = labels[idx]
            logs.append(log)
        except Exception as e:
            print(f"Error loading log {log_path}: {e}")
            sys.exit(1)

    print(f"Plotting {len(logs)} training run(s)...")

    # Parse metrics
    if args.metrics:
        metrics = args.metrics.split(",")
    else:
        # Default metrics based on plot type
        metrics = None  # Will use defaults in plot function

    # Determine output path following forgather conventions
    if args.output:
        # Explicit output path specified
        output_path = args.output
        # Add extension if needed
        if not output_path.endswith((".png", ".svg", ".pdf")):
            output_path = f"{output_path}.{args.format}"
    else:
        # Auto-generate filename in tmp/ directory (following forgather convention)
        temp_dir = "./tmp"
        os.makedirs(temp_dir, exist_ok=True)

        # Generate descriptive filename based on plot type
        config_prefix = ""
        if hasattr(args, "config_template") and args.config_template:
            config_name = os.path.basename(args.config_template)
            if config_name.endswith(".yaml"):
                config_name = config_name[:-5]
            config_prefix = f"{config_name}_"

        if args.loss_curves:
            base_name = f"{config_prefix}loss_curves"
        elif args.metrics:
            metric_name = args.metrics.replace(",", "_")
            base_name = f"{config_prefix}plot_{metric_name}"
        else:
            base_name = f"{config_prefix}training_plot"

        output_path = os.path.join(temp_dir, f"{base_name}.{args.format}")

    # Generate plot (never show interactively - doesn't work on remote SSH)
    plot_title = getattr(args, "title", None)
    try:
        if args.loss_curves:
            # Special loss curves plot with LR on secondary axis
            fig = plot_loss_curves(
                logs=logs,
                x_axis=args.x_axis,
                smooth_window=args.smooth,
                output_path=output_path,
                show=False,  # Never show - use editor instead
                title=plot_title,
            )
        else:
            # General metrics plot
            fig = plot_training_metrics(
                logs=logs,
                metrics=metrics,
                x_axis=args.x_axis,
                smooth_window=args.smooth,
                log_scale=args.log_scale,
                output_path=output_path,
                show=False,  # Never show - use editor instead
                title=plot_title,
            )

        print(f"Plot saved to: {output_path}")

        # Open in editor if requested
        if args.edit:
            print(f"Opening in editor: {output_path}")
            _open_in_editor(output_path)

    except Exception as e:
        print(f"Error generating plot: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def list_cmd(args):
    """List available training logs in project."""
    log_files = find_log_files(args.project_dir)

    if not log_files:
        print("No training logs found.")
        print(f"Searched in: {Path(args.project_dir) / 'output_models'}")
        return

    print(f"Found {len(log_files)} training log(s):\n")

    for i, log_path in enumerate(log_files, 1):
        # Extract run info from path
        parts = log_path.parts
        if "runs" in parts:
            runs_idx = parts.index("runs")
            if runs_idx > 0:
                model_name = parts[runs_idx - 1]
                run_name = parts[runs_idx + 1] if runs_idx + 1 < len(parts) else "unknown"
            else:
                model_name = "unknown"
                run_name = parts[runs_idx + 1] if runs_idx + 1 < len(parts) else "unknown"
        else:
            model_name = "unknown"
            run_name = "unknown"

        # Get file modification time
        mtime = log_path.stat().st_mtime
        import datetime

        mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

        print(f"{i}. {model_name}/{run_name}")
        print(f"   Path: {log_path}")
        print(f"   Modified: {mtime_str}")
        print()


def logs_cmd(args):
    """Main entry point for logs command."""
    if args.logs_subcommand == "summary":
        summary_cmd(args)
    elif args.logs_subcommand == "plot":
        plot_cmd(args)
    elif args.logs_subcommand == "list":
        list_cmd(args)
    else:
        print("Error: No subcommand specified.")
        print("Use 'forgather logs --help' for usage information.")
        sys.exit(1)
