"""Argument parser for logs command."""

import argparse
import os
from argparse import RawTextHelpFormatter

path_type = lambda x: os.path.normpath(os.path.expanduser(x))


def create_logs_parser(global_args):
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
        "--loss-curves",
        action="store_true",
        help="Generate loss curves plot with LR on secondary axis",
    )
    plot_parser.add_argument(
        "-e",
        "--edit",
        action="store_true",
        help="Open plot in editor after generation (works with VS Code remote sessions)",
    )

    # list subcommand
    list_parser = subparsers.add_parser(
        "list",
        help="List available training logs in project",
        formatter_class=RawTextHelpFormatter,
    )

    return parser
