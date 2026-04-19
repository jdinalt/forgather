"""Argument parser for checkpoint command."""

import argparse
import os
from argparse import RawTextHelpFormatter

from .dynamic_args import parse_dynamic_args

path_type = lambda x: os.path.normpath(os.path.expanduser(x))


def create_checkpoint_parser(global_args):
    """Create parser for checkpoint command."""
    parser = argparse.ArgumentParser(
        prog="forgather cp",
        description="Checkpoint tools",
        formatter_class=RawTextHelpFormatter,
    )

    parse_dynamic_args(parser, global_args)

    subparsers = parser.add_subparsers(
        dest="cp_subcommand", help="Checkpoint subcommands"
    )

    # cp project subcommand
    link_parser = subparsers.add_parser(
        "link",
        help="Add synlinks to latest checkpoint to output directory ",
        formatter_class=RawTextHelpFormatter,
    )
    link_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just show what the command would do, without doing it.",
    )
    link_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite, if existing.",
    )
    link_parser.add_argument(
        "--output-path",
        type=path_type,
        help="Just show the generated commandline, without actually executing it.",
    )

    # cp inspect subcommand
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect checkpoint structure and validate integrity",
        formatter_class=RawTextHelpFormatter,
    )
    inspect_parser.add_argument(
        "checkpoint_path",
        type=path_type,
        help="Path to checkpoint directory to inspect",
    )
    inspect_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed information including file lists",
    )
    inspect_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate that all expected files exist",
    )

    # cp components subcommand
    components_parser = subparsers.add_parser(
        "components",
        help="List available components in a checkpoint",
        formatter_class=RawTextHelpFormatter,
        description="List all available checkpoint components.\n\n"
        "Components are identified by the file prefix (e.g., 'optimizer' for optimizer_state.pt).\n"
        "Common components: optimizer, scheduler, model, dataset, rng, trainer\n",
    )
    components_parser.add_argument(
        "checkpoint_path",
        type=path_type,
        help="Path to checkpoint directory",
    )
    components_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show individual file details for each component",
    )

    # cp list subcommand
    list_parser = subparsers.add_parser(
        "list",
        help="List modifiable parameters in a checkpoint component",
        formatter_class=RawTextHelpFormatter,
        description="List modifiable parameters in a checkpoint component.\n\n"
        "The component name is the file prefix (e.g., 'optimizer' for optimizer_state.pt).\n"
        "Use 'forgather checkpoint components CHECKPOINT' to see all available components.\n",
    )
    list_parser.add_argument(
        "checkpoint_path",
        type=path_type,
        help="Path to checkpoint directory or file",
    )
    list_parser.add_argument(
        "--component",
        type=str,
        default="optimizer",
        metavar="NAME",
        help="Component name (file prefix, e.g., 'optimizer', 'scheduler'). Default: optimizer",
    )
    list_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show value types and detailed information",
    )

    # cp modify subcommand
    modify_parser = subparsers.add_parser(
        "modify",
        help="Modify parameters in a checkpoint component",
        formatter_class=RawTextHelpFormatter,
        description="Modify optimizer, scheduler, or other component parameters in a checkpoint.\n\n"
        "The component name is the file prefix (e.g., 'optimizer' for optimizer_state.pt).\n"
        "Use 'forgather checkpoint components CHECKPOINT' to see all available components.\n\n"
        "Examples:\n"
        "  forgather checkpoint modify checkpoint-1000 --component optimizer --set weight_decay=0.01\n"
        "  forgather checkpoint modify checkpoint-1000 --component optimizer --scale lr=0.5\n"
        "  forgather checkpoint modify checkpoint-1000 --component scheduler --set last_epoch=100\n",
    )
    modify_parser.add_argument(
        "checkpoint_path",
        type=path_type,
        help="Path to checkpoint directory or file",
    )
    modify_parser.add_argument(
        "--component",
        type=str,
        default="optimizer",
        metavar="NAME",
        help="Component name (file prefix, e.g., 'optimizer', 'scheduler'). Default: optimizer",
    )
    modify_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Set parameter to exact value (can be used multiple times)",
    )
    modify_parser.add_argument(
        "--scale",
        action="append",
        default=[],
        metavar="KEY=FACTOR",
        help="Multiply parameter by factor (can be used multiple times)",
    )
    modify_parser.add_argument(
        "--param-group",
        type=int,
        metavar="INDEX",
        help="For optimizer: target specific param group (default: all)",
    )
    modify_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without saving",
    )
    modify_parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup creation (still uses atomic operations)",
    )
    modify_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Skip confirmation prompts",
    )
    modify_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Detailed logging",
    )
    modify_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    return parser
