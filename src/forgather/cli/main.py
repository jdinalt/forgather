"""Main CLI controller and argument parsing."""

import argparse
from argparse import RawTextHelpFormatter
import logging
import os
import sys

from .commands import (
    index_cmd,
    ls_cmd,
    meta_cmd,
    targets_cmd,
    template_list,
    graph_cmd,
    pp_cmd,
    tb_cmd,
    code_cmd,
    construct_cmd,
)

from .dynamic_args import (
    parse_dynamic_args,
    partition_args,
)

from .utils import add_output_arg, add_editor_arg


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

    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start interactive shell with tab completion",
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
        "control": create_control_parser,
        "model": create_model_parser,
    }


def create_index_parser(global_args):
    """Create parser for index command."""
    parser = argparse.ArgumentParser(
        prog="forgather index",
        description="Show project index",
        formatter_class=RawTextHelpFormatter,
    )
    add_output_arg(parser)
    add_editor_arg(parser)
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
        "--debug",
        "-d",
        action="store_true",
        help="Debug meta-data parsing",
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
    add_output_arg(parser)
    add_editor_arg(parser)
    return parser


def create_targets_parser(global_args):
    """Create parser for targets command."""
    parser = argparse.ArgumentParser(
        prog="forgather targets",
        description="Show output targets",
        formatter_class=RawTextHelpFormatter,
    )
    add_output_arg(parser)
    add_editor_arg(parser)
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
    add_output_arg(parser)
    add_editor_arg(parser)
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
    add_output_arg(parser)
    add_editor_arg(parser)
    parse_dynamic_args(parser, global_args)
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
        choices=["md", "files", "tree", "dot", "svg"],
        default="files",
        help="Output format: md (markdown), files (file list), tree (hierarchical), dot (graphviz), svg (render SVG).",
    )
    add_output_arg(parser)
    add_editor_arg(parser)
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
    add_output_arg(parser)
    add_editor_arg(parser)
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
    parse_dynamic_args(parser, global_args)
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
    add_output_arg(parser)
    add_editor_arg(parser)
    parse_dynamic_args(parser, global_args)
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
    add_output_arg(parser)
    add_editor_arg(parser)
    parse_dynamic_args(parser, global_args)
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
        type=os.path.expanduser,
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
    parser.add_argument(
        "-n",
        "--examples",
        type=int,
        default=None,
        help="Number of examples to print",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="*",
        help="Features to show",
    )
    parser.add_argument(
        "-s",
        "--tokenized",
        action="store_true",
        help="The split is already tokenized",
    )
    add_output_arg(parser)
    add_editor_arg(parser)
    parse_dynamic_args(parser, global_args)
    return parser


def create_ws_parser(global_args):
    path_type = lambda x: os.path.normpath(os.path.expanduser(x))

    """Create parser for workspace command."""
    parser = argparse.ArgumentParser(
        prog="forgather ws",
        description="Forgather workspace management",
        formatter_class=RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="ws_subcommand", help="Workspace subcommands"
    )

    # create subcommand
    init_parser = subparsers.add_parser(
        "create",
        help="Initialize a new forgather workspace",
        formatter_class=RawTextHelpFormatter,
    )
    init_parser.add_argument("--name", required=True, help="Workspace name")
    init_parser.add_argument(
        "--description", required=True, help="Workspace description"
    )
    init_parser.add_argument(
        "--forgather-dir",
        required=True,
        type=path_type,
        help="Path to forgather installation directory",
    )
    init_parser.add_argument(
        "search_paths",
        nargs="*",
        type=path_type,
        help="Additional search paths for templates",
    )
    init_parser.add_argument(
        "--no-defaults",
        action="store_true",
        help="Don't include default forgather search paths",
    )

    # project subcommand
    project_parser = subparsers.add_parser(
        "project",
        help="Create a new forgather project in the workspace",
        formatter_class=RawTextHelpFormatter,
    )
    project_parser.add_argument("--name", required=True, help="Project name")
    project_parser.add_argument(
        "--description", required=True, help="Project description"
    )
    project_parser.add_argument(
        "--config-prefix",
        default="configs",
        help="Configuration prefix (default: configs)",
    )
    project_parser.add_argument(
        "--default-config",
        default="default.yaml",
        help="Default configuration name (default: default)",
    )
    project_parser.add_argument(
        "project_dir",
        nargs="?",
        help="Project directory name (defaults to project name with spaces replaced by underscores)",
    )

    return parser


def create_control_parser(global_args):
    """Create parser for control command."""
    parser = argparse.ArgumentParser(
        prog="forgather control",
        description="Control running training jobs",
        formatter_class=RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="control_subcommand", help="Control subcommands"
    )

    # list subcommand
    list_parser = subparsers.add_parser(
        "list",
        help="List discoverable training jobs",
        formatter_class=RawTextHelpFormatter,
    )
    list_parser.add_argument(
        "--remote",
        type=str,
        metavar="HOST:PORT",
        help="Query remote host for jobs (e.g., compute-node-01:8947)",
    )

    # status subcommand
    status_parser = subparsers.add_parser(
        "status",
        help="Get status of a training job",
        formatter_class=RawTextHelpFormatter,
    )
    status_parser.add_argument("job_id", help="Job ID to query")

    # stop subcommand
    stop_parser = subparsers.add_parser(
        "stop",
        help="Send graceful stop command to a training job (saves final checkpoint)",
        formatter_class=RawTextHelpFormatter,
    )
    stop_parser.add_argument("job_id", help="Job ID to stop")

    # abort subcommand
    abort_parser = subparsers.add_parser(
        "abort",
        help="Abort training job WITHOUT saving checkpoint",
        formatter_class=RawTextHelpFormatter,
    )
    abort_parser.add_argument("job_id", help="Job ID to abort")
    abort_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )

    # save subcommand
    save_parser = subparsers.add_parser(
        "save",
        help="Trigger checkpoint save in a training job",
        formatter_class=RawTextHelpFormatter,
    )
    save_parser.add_argument("job_id", help="Job ID to save")

    # save-stop subcommand
    save_stop_parser = subparsers.add_parser(
        "save-stop",
        help="Save checkpoint and stop training job",
        formatter_class=RawTextHelpFormatter,
    )
    save_stop_parser.add_argument("job_id", help="Job ID to save and stop")

    # cleanup subcommand
    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Remove endpoint files for dead training jobs",
        formatter_class=RawTextHelpFormatter,
    )
    cleanup_parser.add_argument(
        "--force", action="store_true", help="Remove all job files without confirmation"
    )

    return parser


def create_model_parser(global_args):
    """Create parser for train command."""
    parser = argparse.ArgumentParser(
        prog="forgather model",
        description="Test a model definition",
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--device", type=str, default="meta", help="Device to construct model on"
    )

    subparsers = parser.add_subparsers(
        dest="model_subcommand", help="Model subcommands"
    )

    # construct subcommand
    construct_parser = subparsers.add_parser(
        "construct",
        help="Construct a model",
        formatter_class=RawTextHelpFormatter,
    )

    # test subcommand
    test_parser = subparsers.add_parser(
        "test",
        help="Test model forward and backward",
        formatter_class=RawTextHelpFormatter,
    )
    test_parser.add_argument("--batch-size", type=int, default="2", help="Batch size")
    test_parser.add_argument(
        "--sequence-length", type=int, default="512", help="Sequence length"
    )

    add_output_arg(parser)
    add_editor_arg(parser)
    parse_dynamic_args(parser, global_args)
    return parser


def show_main_help():
    """Show the main help message with available subcommands."""
    print("Forgather CLI")
    print()
    print("Usage: forgather [global options] <subcommand> [subcommand options]")
    print(
        "       forgather -i                      # Interactive mode with tab completion"
    )
    print()
    print("Global options:")
    print("  -p, --project-dir DIR    Project directory (default: current directory)")
    print("  -t, --config-template T  Configuration template name")
    print("  -i, --interactive        Start interactive shell with tab completion")
    print("  --no-dyn                 Disable dynamic help (from config meta-data)")
    print("  --help                   Show this help message")
    print()
    print("Available subcommands:")
    registry = get_subcommand_registry()
    for cmd_name in sorted(registry.keys()):
        # Create a dummy global_args for the registry call
        dummy_global_args = argparse.Namespace(
            project_dir=".", config_template=None, no_dyn=True
        )
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

    # Handle interactive mode
    if global_args.interactive:
        from .interactive import interactive_main

        interactive_main(global_args.project_dir)
        sys.exit(0)

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
    logging.basicConfig(level=logging.WARNING)
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
                from .trefs import trefs_cmd

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
                from .train import train_cmd

                train_cmd(args)
            case "dataset":
                from .dataset import dataset_cmd

                dataset_cmd(args)
            case "ws":
                from .workspace import ws_cmd

                ws_cmd(args)
            case "control":
                from .control import control_cmd

                control_cmd(args)
            case "model":
                from .model import model_cmd

                model_cmd(args)
            case _:
                index_cmd(args)
    except SystemExit:
        # Let argparse's sys.exit calls through (for help, errors, etc.)
        raise
    except KeyboardInterrupt:
        sys.exit(1)
