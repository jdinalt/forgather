"""Argument parser for logs command."""

import argparse
import os
from argparse import RawTextHelpFormatter

path_type = lambda x: os.path.normpath(os.path.expanduser(x))


def create_logs_parser(_global_args):
    """Create parser for logs command."""
    parser = argparse.ArgumentParser(
        prog="forgather logs",
        description="Analyze and visualize training logs",
        formatter_class=RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="logs_subcommand", help="Logs subcommands")

    # summary subcommand
    summary_parser = subparsers.add_parser(
        "summary",
        help="Generate summary statistics from training logs",
        formatter_class=RawTextHelpFormatter,
    )
    summary_parser.add_argument(
        "log_path",
        nargs="?",
        type=path_type,
        help="Path to trainer_logs.json or run directory (default: latest run in project)",
    )
    summary_parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "md", "one-line"],
        default="text",
        help="Output format",
    )
    summary_parser.add_argument(
        "--all",
        action="store_true",
        help="Process all logs in project (not just latest)",
    )
    summary_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (default: print to stdout)",
    )

    # plot subcommand
    plot_parser = subparsers.add_parser(
        "plot",
        help="Generate plots from training logs",
        formatter_class=RawTextHelpFormatter,
        description="Generate plots from training logs.\n\n"
        "Default behavior: Saves plot to tmp/ directory (gitignored).\n"
        "Use --output FILE to specify a different location.\n"
        "Use -e/--edit to open the plot in your editor (works with remote sessions).",
    )
    plot_parser.add_argument(
        "log_paths",
        nargs="*",
        type=path_type,
        help="Paths to trainer_logs.json or run directories (default: latest run in project)",
    )
    plot_parser.add_argument(
        "--metrics",
        type=str,
        help="Comma-separated list of metrics to plot (e.g., 'loss,eval_loss,learning_rate')",
    )
    plot_parser.add_argument(
        "--x-axis",
        type=str,
        choices=["step", "epoch", "time"],
        default="step",
        help="X-axis variable (default: step)",
    )
    plot_parser.add_argument(
        "--smooth",
        type=int,
        help="Apply smoothing with specified window size",
    )
    plot_parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log scale for y-axis",
    )
    plot_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path for plot (extension optional, determined by --format)",
    )
    plot_parser.add_argument(
        "--format",
        type=str,
        choices=["png", "svg", "pdf"],
        default="png",
        help="Plot format (default: png)",
    )
    plot_parser.add_argument(
        "--compare",
        nargs="+",
        type=path_type,
        help="Compare multiple runs (provide paths)",
    )
    plot_parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        help="Custom labels for runs (one per log path, in same order)",
    )
    plot_parser.add_argument(
        "--title",
        type=str,
        help="Custom plot title",
    )
    plot_parser.add_argument(
        "--loss-curves",
        action="store_true",
        help="Generate loss curves plot with LR on secondary axis",
    )
    plot_parser.add_argument(
        "--grad-norm",
        action="store_true",
        help="Plot gradient norm across training steps",
    )
    plot_parser.add_argument(
        "--ignore-outliers",
        dest="ignore_outliers",
        action="store_true",
        default=True,
        help="Auto-scale y-axis using 5th/95th percentiles so large early-training\n"
        "values do not squash the tail of the curve (default: on)",
    )
    plot_parser.add_argument(
        "--no-ignore-outliers",
        dest="ignore_outliers",
        action="store_false",
        help="Disable outlier-aware y-auto-scaling and use the full (min, max) range",
    )
    plot_parser.add_argument(
        "--perplexity",
        action="store_true",
        help="Plot loss-like metrics as perplexity (exp(loss))",
    )
    plot_parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Clip plotted data and x-axis at this lower bound (in the current --x-axis units)",
    )
    plot_parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Clip plotted data and x-axis at this upper bound (in the current --x-axis units)",
    )
    plot_parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Override the lower y-axis bound (takes precedence over --ignore-outliers)",
    )
    plot_parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Override the upper y-axis bound (takes precedence over --ignore-outliers)",
    )
    plot_parser.add_argument(
        "-e",
        "--edit",
        action="store_true",
        help="Open plot in editor after generation (works with VS Code remote sessions)",
    )

    # list subcommand (registered for side effect on `subparsers`)
    subparsers.add_parser(
        "list",
        help="List available training logs in project",
        formatter_class=RawTextHelpFormatter,
    )

    return parser
