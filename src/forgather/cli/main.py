"""Main CLI controller and argument parsing."""

import argparse
from argparse import RawTextHelpFormatter
import os
import sys

from .commands import (
    index_cmd,
    ls_cmd,
    meta_cmd,
    targets_cmd,
    template_list,
    graph_cmd,
    trefs_cmd,
    pp_cmd,
    tb_cmd,
    code_cmd,
    construct_cmd,
    train_cmd,
    ws_cmd,
)

from .dynamic_args import (
    parse_dynamic_args,
    partition_args,
    get_dynamic_args,
)


def parse_global_args(args=None):
    """Parse global arguments and return global args + remaining args for subcommand."""
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Forgather CLI",
        epilog=(""),
        add_help=False,  # We'll handle help at the subcommand level
    )

    parser.add_argument(
        "-p",
        "--project-dir",
        type=str,
        default=".",
    )

    parser.add_argument(
        "-t",
        "--config-template",
        type=str,
        default=None,
        help="Configuration Template Name",
    )

    parser.add_argument(
        "--no-dyn",
        action="store_true",
        help="Disable processing of dynamic args defined in configuration templates",
    )

    # Parse known args to separate global from subcommand args
    global_args, remaining_args = parser.parse_known_args(args)

    return global_args, remaining_args


def get_subcommand_registry():
    """Registry of all available subcommands and their argument parsers."""
    return {
        "index": create_index_parser,
        "ls": create_ls_parser,
        "meta": create_meta_parser,
        "targets": create_targets_parser,
        "tlist": create_tlist_parser,
        "graph": create_graph_parser,
        "trefs": create_trefs_parser,
        "pp": create_pp_parser,
        "tb": create_tb_parser,
        "code": create_code_parser,
        "construct": create_construct_parser,
        "train": create_train_parser,
        "dataset": create_dataset_parser,
        "ws": create_ws_parser,
    }


def create_index_parser(global_args):
    """Create parser for index command."""
    parser = argparse.ArgumentParser(
        prog="forgather index",
        description="Show project index",
        formatter_class=RawTextHelpFormatter,
    )
    return parser


def create_ls_parser(global_args):
    """Create parser for ls command."""
    parser = argparse.ArgumentParser(
        prog="forgather ls",
        description="List available configurations",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search for project in all sub-directories and list them.",
    )
    parser.add_argument(
        "project",
        type=str,
        nargs="*",
        help="List configurations in projects",
    )
    return parser


def create_meta_parser(global_args):
    """Create parser for meta command."""
    parser = argparse.ArgumentParser(
        prog="forgather meta",
        description="Show meta configuration",
        formatter_class=RawTextHelpFormatter,
    )
    return parser


def create_targets_parser(global_args):
    """Create parser for targets command."""
    parser = argparse.ArgumentParser(
        prog="forgather targets",
        description="Show output targets",
        formatter_class=RawTextHelpFormatter,
    )
    return parser


def create_tlist_parser(global_args):
    """Create parser for tlist command."""
    parser = argparse.ArgumentParser(
        prog="forgather tlist",
        description="List available templates",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["md", "files"],
        default="files",
        help="Output format.",
    )
    return parser


def create_graph_parser(global_args):
    """Create parser for graph command."""
    parser = argparse.ArgumentParser(
        prog="forgather graph",
        description="Preprocess and parse into node graph",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["none", "repr", "yaml", "fconfig", "python"],
        default="yaml",
        help="Graph format",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Shows details of template preprocessing to assist with debugging",
    )
    return parser


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
        choices=["md", "files"],
        default="files",
        help="Output format.",
    )
    return parser


def create_pp_parser(global_args):
    """Create parser for pp command."""
    parser = argparse.ArgumentParser(
        prog="forgather pp",
        description="Preprocess configuration",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Shows details of template preprocessing to assist with debugging",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Shows additional debug information",
    )

    parse_dynamic_args(parser, global_args)
    return parser


def create_tb_parser(global_args):
    """Create parser for tb command."""
    parser = argparse.ArgumentParser(
        prog="forgather tb",
        description="Start Tensorboard for project",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Configure TB to watch all model directories",
    )
    parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="All arguments after -- will be forwarded as Tensorboard arguments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just show the generated commandline, without actually executing it.",
    )
    return parser


def create_code_parser(global_args):
    """Create parser for code command."""
    parser = argparse.ArgumentParser(
        prog="forgather code",
        description="Output configuration as Python code",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--target",
        type=str,
        default="main",
        help="Output target name",
    )
    return parser


def create_construct_parser(global_args):
    """Create parser for construct command."""
    parser = argparse.ArgumentParser(
        prog="forgather construct",
        description="Materialize and print a target",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--target",
        type=str,
        default="main",
        help="Output target name",
    )
    parser.add_argument(
        "--call",
        action="store_true",
        help="Call the materialized object",
    )
    return parser


def create_train_parser(global_args):
    """Create parser for train command."""
    parser = argparse.ArgumentParser(
        prog="forgather train",
        description="Run configuration with train script",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--devices",
        type=str,
        default=None,
        help='CUDA Visible Devices e.g. "0,1"',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just show the generated commandline, without actually executing it.",
    )
    parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="All arguments after -- will be forwarded as torchrun arguments.",
    )
    parse_dynamic_args(parser, global_args)
    return parser


def create_dataset_parser(global_args):
    """Create parser for dataset command."""
    parser = argparse.ArgumentParser(
        prog="forgather dataset",
        description="Dataset preprocessing and testing",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-T",
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer to test",
    )
    parser.add_argument(
        "--pp",
        action="store_true",
        help="Show preprocessed configuration",
    )
    parser.add_argument(
        "-H",
        "--histogram",
        action="store_true",
        help="Generate dataset token length historgram and statistics",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="train_dataset_split",
        help='The dataset to sample from; see "forgather targets"',
    )
    parser.add_argument(
        "--histogram-samples",
        type=int,
        default=1000,
        help="Number of samples to use for histogram",
    )
    # parser.add_argument(
    #    "-c",
    #    "--chat-template",
    #    type=str,
    #    default=None,
    #    help="Path to chat template",
    # )
    parser.add_argument(
        "-n",
        "--examples",
        type=int,
        default=None,
        help="Number of examples to print",
    )
    parser.add_argument(
        "-s",
        "--tokenized",
        action="store_true",
        help="The split is already tokenized",
    )
    parse_dynamic_args(parser, global_args)
    return parser


def create_ws_parser(global_args):
    """Create parser for targets command."""
    parser = argparse.ArgumentParser(
        prog="forgather ws",
        description="Forgather workspace",
        formatter_class=RawTextHelpFormatter,
    )
    return parser


def show_main_help():
    """Show the main help message with available subcommands."""
    print("Forgather CLI")
    print()
    print("Usage: forgather [global options] <subcommand> [subcommand options]")
    print()
    print("Global options:")
    print("  -p, --project-dir DIR    Project directory (default: current directory)")
    print("  -t, --config-template T  Configuration template name")
    print("  --help                   Show this help message")
    print()
    print("Available subcommands:")
    registry = get_subcommand_registry()
    for cmd_name in sorted(registry.keys()):
        # Create a dummy global_args for the registry call
        dummy_global_args = argparse.Namespace(project_dir=".", config_template=None)
        try:
            parser = registry[cmd_name](dummy_global_args)
            print(f"  {cmd_name:<12} {parser.description}")
        except:
            print(f"  {cmd_name:<12} [Error loading description]")
    print()
    print("Use 'forgather <subcommand> --help' for help on a specific subcommand.")


def parse_args(args=None):
    """Parse arguments with dynamic subcommand handling."""
    global_args, remaining_args = parse_global_args(args)

    # Handle case where no subcommand is provided or --help is requested globally
    if not remaining_args or (remaining_args and remaining_args[0] in ["--help", "-h"]):
        show_main_help()
        sys.exit(0)

    # Extract subcommand name
    subcommand = remaining_args[0]
    subcommand_args = remaining_args[1:]

    # Get subcommand registry
    registry = get_subcommand_registry()

    # Check if subcommand exists
    if subcommand not in registry:
        print(f"Error: Unknown subcommand '{subcommand}'")
        print()
        show_main_help()
        sys.exit(1)

    # Create subcommand parser and parse its arguments
    subcommand_parser = registry[subcommand](global_args)

    try:
        sub_args = subcommand_parser.parse_args(subcommand_args)
    except SystemExit:
        # argparse calls sys.exit on help or error - let it through
        raise

    # Get dynamic argument names from the parser (if available)
    dynamic_arg_names = getattr(subcommand_parser, "_dynamic_arg_names", [])

    # Partition the subcommand arguments
    if dynamic_arg_names:
        built_in_sub_args, dynamic_sub_args = partition_args(
            sub_args, dynamic_arg_names
        )
    else:
        built_in_sub_args = vars(sub_args)
        dynamic_sub_args = {}

    # Combine global and built-in subcommand args into a single namespace
    combined_args = argparse.Namespace()

    # Add global args
    for key, value in vars(global_args).items():
        setattr(combined_args, key, value)

    # Add built-in subcommand args
    for key, value in built_in_sub_args.items():
        setattr(combined_args, key, value)

    # Add the command name
    combined_args.command = subcommand

    # Store dynamic args separately for easy access
    combined_args._dynamic_args = dynamic_sub_args

    return combined_args


def main():
    """Main CLI entry point."""
    try:
        args = parse_args()
        match args.command:
            case "index":
                index_cmd(args)
            case "ls":
                ls_cmd(args)
            case "meta":
                meta_cmd(args)
            case "targets":
                targets_cmd(args)
            case "tlist":
                template_list(args)
            case "graph":
                graph_cmd(args)
            case "trefs":
                trefs_cmd(args)
            case "pp":
                pp_cmd(args)
            case "tb":
                tb_cmd(args)
            case "code":
                code_cmd(args)
            case "construct":
                construct_cmd(args)
            case "train":
                train_cmd(args)
            case "dataset":
                from .dataset import dataset_cmd

                dataset_cmd(args)
            case "ws":
                ws_cmd(args)
            case _:
                index_cmd(args)
    except SystemExit:
        # Let argparse's sys.exit calls through (for help, errors, etc.)
        raise
    except KeyboardInterrupt:
        sys.exit(1)
