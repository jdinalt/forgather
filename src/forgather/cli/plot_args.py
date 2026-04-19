"""Argument parser for plot command."""

import argparse
import os
from argparse import RawTextHelpFormatter

path_type = lambda x: os.path.normpath(os.path.expanduser(x))


def create_plot_parser(global_args):
    """Create parser for plot command."""
    parser = argparse.ArgumentParser(
        prog="forgather plot",
        description="Generate diagnostic plots from training data",
        formatter_class=RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="plot_subcommand", help="Plot subcommands")

    # heatmap subcommand
    heatmap_parser = subparsers.add_parser(
        "heatmap",
        help="Generate per-parameter heatmap from diagnostic logs",
        formatter_class=RawTextHelpFormatter,
        description=(
            "Generate a per-parameter heatmap from diagnostic log files.\n\n"
            "Supported log files:\n"
            "  parameter_norms.json  - Per-parameter L2 and spectral norms\n"
            "  gradient_norms.json   - Per-parameter gradient norms\n\n"
            "The metric is auto-detected from the file content unless\n"
            "explicitly specified with --metric."
        ),
    )
    heatmap_parser.add_argument(
        "log_path",
        nargs="?",
        type=path_type,
        help=(
            "Path to a diagnostic log file (parameter_norms.json or "
            "gradient_norms.json). If omitted, searches the project for "
            "the most recent file."
        ),
    )
    heatmap_parser.add_argument(
        "--metric",
        type=str,
        choices=["norm", "spectral_norm", "grad_norm"],
        help="Which metric to plot (auto-detected from file if omitted)",
    )
    heatmap_parser.add_argument(
        "--step-stride",
        type=int,
        default=1,
        help="Plot every Nth step (default: 1, all eval steps)",
    )
    heatmap_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (default: ./tmp/heatmap.png)",
    )
    heatmap_parser.add_argument(
        "--format",
        type=str,
        choices=["png", "svg", "pdf"],
        default="png",
        help="Plot format (default: png)",
    )
    heatmap_parser.add_argument(
        "-e",
        "--edit",
        action="store_true",
        help="Open plot in editor after generation",
    )
    heatmap_parser.add_argument(
        "--title",
        type=str,
        help="Custom plot title",
    )
    heatmap_parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log scale for color mapping",
    )
    heatmap_parser.add_argument(
        "--vmin",
        type=float,
        help="Manual minimum for color range",
    )
    heatmap_parser.add_argument(
        "--vmax",
        type=float,
        help="Manual maximum for color range",
    )
    heatmap_parser.add_argument(
        "--filter",
        "-f",
        type=str,
        help="Regex to filter parameter FQN names (only matching names are plotted)",
    )
    heatmap_parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        help="Figure size in inches (default: auto-scaled)",
    )

    return parser
