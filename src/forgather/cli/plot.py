"""CLI commands for diagnostic plot generation."""

import os
import sys
from pathlib import Path

from forgather.ml.analysis.heatmap import plot_parameter_heatmap
from forgather.ml.analysis.log_parser import find_diagnostic_logs

from .utils import _open_in_editor

# Diagnostic log filenames to search for when auto-detecting
_DIAGNOSTIC_FILENAMES = ["parameter_norms.json", "gradient_norms.json"]


def _find_latest_diagnostic_log(project_dir: str) -> Path | None:
    """Find the most recently modified diagnostic log in the project."""
    all_logs = []
    for filename in _DIAGNOSTIC_FILENAMES:
        all_logs.extend(find_diagnostic_logs(project_dir, filename))

    if not all_logs:
        return None

    # Already sorted by mtime (most recent first)
    return all_logs[0]


def heatmap_cmd(args):
    """Generate per-parameter heatmap from diagnostic logs."""
    # Determine log path
    if args.log_path:
        log_path = Path(args.log_path)
    else:
        log_path = _find_latest_diagnostic_log(args.project_dir)
        if log_path is None:
            print("Error: No diagnostic log files found in project.")
            print(f"Searched in: {Path(args.project_dir) / 'output_models'}")
            print(f"Looking for: {', '.join(_DIAGNOSTIC_FILENAMES)}")
            sys.exit(1)
        print(f"Using latest log: {log_path}")

    if not log_path.exists():
        print(f"Error: File not found: {log_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
        if not output_path.endswith((".png", ".svg", ".pdf")):
            output_path = f"{output_path}.{args.format}"
    else:
        temp_dir = "./tmp"
        os.makedirs(temp_dir, exist_ok=True)

        # Name based on source file
        stem = log_path.stem
        metric_suffix = f"_{args.metric}" if args.metric else ""
        output_path = os.path.join(
            temp_dir, f"heatmap_{stem}{metric_suffix}.{args.format}"
        )

    # Parse figsize
    figsize = tuple(args.figsize) if args.figsize else None

    try:
        plot_parameter_heatmap(
            log_path=log_path,
            metric=args.metric,
            step_stride=args.step_stride,
            filter_pattern=getattr(args, "filter", None),
            log_scale=args.log_scale,
            vmin=args.vmin,
            vmax=args.vmax,
            title=args.title,
            output_path=output_path,
            figsize=figsize,
        )
        print(f"Heatmap saved to: {output_path}")

        if args.edit:
            print(f"Opening in editor: {output_path}")
            _open_in_editor(output_path)

    except Exception as e:
        print(f"Error generating heatmap: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def plot_cmd(args):
    """Main entry point for plot command."""
    if args.plot_subcommand == "heatmap":
        heatmap_cmd(args)
    else:
        print("Error: No subcommand specified.")
        print("Use 'forgather plot --help' for usage information.")
        sys.exit(1)
