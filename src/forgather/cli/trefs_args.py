"""Argument parser for trefs command."""

import argparse
from argparse import RawTextHelpFormatter

from .utils import add_editor_arg, add_output_arg


def create_trefs_parser(global_args):
    """Create parser for trefs command."""
    parser = argparse.ArgumentParser(
        prog="forgather trefs",
        description="List referenced templates",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["md", "files", "tree", "dot", "svg"],
        default="files",
        help="Output format: md (markdown), files (file list), tree (hierarchical), dot (graphviz), svg (render SVG).",
    )
    add_output_arg(parser)
    add_editor_arg(parser)
    return parser
