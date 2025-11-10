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
        "project": create_project_parser,
        "inf": create_inf_parser,
        "convert": create_convert_parser,
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
    create_parser = subparsers.add_parser(
        "create",
        help="Initialize a new forgather workspace",
        formatter_class=RawTextHelpFormatter,
    )
    create_parser.add_argument("--name", required=True, help="Workspace name")
    create_parser.add_argument(
        "--description", required=True, help="Workspace description"
    )
    create_parser.add_argument(
        "--forgather-dir",
        required=True,
        type=path_type,
        help="Path to forgather installation directory",
    )
    create_parser.add_argument(
        "--lib",
        "-l",
        action="append",
        type=str,
        help="Forgather template library name (in forgather/templatelib)",
    )
    create_parser.add_argument(
        "--search-path",
        action="append",
        type=path_type,
        help="Additional search paths for templates",
    )
    create_parser.add_argument(
        "workspace_dir",
        nargs="?",
        help="Workspace directory name (defaults to project name with spaces replaced by underscores)",
    )

    return parser


def create_project_parser(global_args):
    path_type = lambda x: os.path.normpath(os.path.expanduser(x))

    """Create parser for workspace command."""
    parser = argparse.ArgumentParser(
        prog="forgather project",
        description="Forgather project management",
        formatter_class=RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="project_subcommand", help="Project subcommands"
    )

    # project subcommand
    create_parser = subparsers.add_parser(
        "create",
        help="Create a new forgather project in the workspace",
        formatter_class=RawTextHelpFormatter,
    )
    create_parser.add_argument("--name", required=True, help="Project name")
    create_parser.add_argument(
        "--description", required=True, help="Project description"
    )
    create_parser.add_argument(
        "--config-prefix",
        default="configs",
        help="Configuration prefix (default: configs)",
    )
    create_parser.add_argument(
        "--default-config",
        default="default.yaml",
        help="Default configuration name (default: default)",
    )
    create_parser.add_argument(
        "--project-dir-name",
        help="Project directory name (defaults to project name with spaces replaced by underscores)",
    )
    create_parser.add_argument(
        "copy_from",
        nargs="?",
        type=path_type,
        help="Source configuration (filepath) to copy as default config",
    )

    # Show subcommand
    show_parser = subparsers.add_parser(
        "show",
        help="Show project info",
        formatter_class=RawTextHelpFormatter,
    )

    # config subcommand
    new_config_parser = subparsers.add_parser(
        "new_config",
        help="Create new config",
        formatter_class=RawTextHelpFormatter,
    )

    new_config_parser.add_argument(
        "config_name",
        type=str,
        help="Configuration name (relative to project configs dir)",
    )

    new_config_parser.add_argument(
        "copy_from",
        nargs="?",
        type=path_type,
        help="Source configuration (filepath) to copy",
    )

    new_config_parser.add_argument(
        "--type",
        type=str,
        choices=["config", "project", "ws"],
        default="config",
        help="Create new config in: config=configs dir, project=project level, ws=workspace level",
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
    parser.add_argument(
        "--dtype",
        type=str,
        help="Construct with default torch dtype",
    )
    parser.add_argument(
        "--no-init-weights",
        action="store_true",
        help="Construct with no_init_weights() context manager",
    )
    parser.add_argument(
        "--load-from-checkpoint",
        type=os.path.expanduser,
        default=None,
        help="Load model weights from checkpoint (path)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
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
    test_parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    test_parser.add_argument(
        "--sequence-length", type=int, default=512, help="Sequence length"
    )
    test_parser.add_argument(
        "--steps", type=int, default=1, help="Number of train steps"
    )
    test_parser.add_argument(
        "--dataset-project",
        type=os.path.expanduser,
        default=None,
        help="Path to dataset project",
    )
    test_parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset config name",
    )
    test_parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-2,
        help="Learning rate",
    )

    add_output_arg(parser)
    add_editor_arg(parser)
    parse_dynamic_args(parser, global_args)
    return parser


def create_inf_parser(global_args):
    """Create parser for inference command."""
    parser = argparse.ArgumentParser(
        prog="forgather inf",
        description="Run inference server or client\n\n"
        "Usage:\n"
        "  forgather inf server [args...]  - Start inference server\n"
        "  forgather inf client [args...]  - Start inference client\n\n"
        "All arguments after 'server' or 'client' are forwarded to the respective script.",
        formatter_class=RawTextHelpFormatter,
        add_help=True,
    )
    # Capture subcommand as required positional
    parser.add_argument(
        "subcommand",
        choices=["server", "client"],
        help="Subcommand: 'server' or 'client'",
    )
    # Use REMAINDER to capture all following args (including flags)
    parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments to forward to the script",
    )

    return parser


def create_convert_parser(global_args):
    """Create parser for convert command."""
    # Note: We use add_help=False because we want --help to be forwarded to the script
    parser = argparse.ArgumentParser(
        prog="forgather convert",
        description="Convert between HuggingFace and Forgather model formats\n\n"
        "All arguments are forwarded to scripts/convert_llama.py",
        formatter_class=RawTextHelpFormatter,
        add_help=False,
    )
    # Add a dummy positional to enable REMAINDER to work
    parser.add_argument(
        "dummy",
        nargs="?",
        default="",
        help=argparse.SUPPRESS,  # Hide from help
    )
    # Use REMAINDER to capture all following args (including flags)
    parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments to forward to convert_llama.py",
    )

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
    # Special handling for commands that have flag conflicts with global args
    # - 'inf' command: --interactive/-i conflicts with global interactive mode
    # - 'convert' command: -t conflicts with global --config-template
    args_list = args if args is not None else sys.argv[1:]

    # Track which flags to restore and workarounds needed
    inf_interactive_workaround = False
    convert_t_workaround = False
    convert_original_args = None  # Save original args for convert command
    removed_flags = []

    # Handle 'inf' command with --interactive/-i
    if "inf" in args_list:
        inf_idx = args_list.index("inf")
        remaining_after_inf = args_list[inf_idx + 1 :]
        if "--interactive" in remaining_after_inf or "-i" in remaining_after_inf:
            inf_interactive_workaround = True
            args_for_global = args_list.copy()
            if "--interactive" in remaining_after_inf:
                args_for_global.remove("--interactive")
                removed_flags.append(
                    (
                        "--interactive",
                        inf_idx + remaining_after_inf.index("--interactive") + 1,
                    )
                )
            if "-i" in remaining_after_inf:
                args_for_global.remove("-i")
                removed_flags.append(
                    ("-i", inf_idx + remaining_after_inf.index("-i") + 1)
                )
            args_list = args_for_global

    # Handle 'convert' command with -t
    # Save original args for convert, then remove -t to prevent global parser from consuming it
    if "convert" in args_list:
        convert_idx = args_list.index("convert")
        remaining_after_convert = args_list[convert_idx + 1 :]
        # Save the original args after 'convert' for later use
        convert_original_args = remaining_after_convert.copy()
        # Check if -t appears after 'convert' (not before, which would be global -t)
        if "-t" in remaining_after_convert:
            convert_t_workaround = True
            args_for_global = args_list.copy()
            # Find -t after convert and remove it along with its value for global parsing
            t_idx_in_remaining = remaining_after_convert.index("-t")
            t_idx_in_full = convert_idx + 1 + t_idx_in_remaining
            args_for_global.pop(t_idx_in_full)  # Remove -t
            # Also remove the value after -t if it exists and doesn't start with -
            if t_idx_in_full < len(args_for_global) and not args_for_global[
                t_idx_in_full
            ].startswith("-"):
                args_for_global.pop(t_idx_in_full)
            args_list = args_for_global

    # Parse global args with potentially modified args_list
    global_args, remaining_args = parse_global_args(args_list)

    # Restore removed flags to remaining_args
    if removed_flags:
        # Sort by position to restore in correct order
        removed_flags.sort(key=lambda x: x[1])
        # Find the subcommand in remaining_args to know where to insert
        subcommand = None
        if remaining_args:
            subcommand = remaining_args[0]

        if subcommand in ["inf", "convert"]:
            subcommand_idx = remaining_args.index(subcommand)
            # Insert flags after the subcommand (and after any positional arg for inf)
            insert_pos = (
                subcommand_idx + 2
                if subcommand == "inf" and len(remaining_args) > subcommand_idx + 1
                else subcommand_idx + 1
            )
            for flag, _ in removed_flags:
                remaining_args.insert(insert_pos, flag)
                insert_pos += 1

    # Handle interactive mode (but skip if it was the inf command workaround)
    if global_args.interactive and not inf_interactive_workaround:
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
        # For convert command, just pass all args as remainder without parsing
        if subcommand == "convert":
            # Create a minimal namespace with remainder containing all original args
            sub_args = argparse.Namespace()
            # Use saved original args if available (when -t was present), otherwise use subcommand_args
            sub_args.remainder = (
                convert_original_args
                if convert_original_args is not None
                else subcommand_args
            )
            sub_args.dummy = ""
        else:
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

            case "project":
                from .project import project_cmd

                project_cmd(args)
            case "control":
                from .control import control_cmd

                control_cmd(args)
            case "model":
                from .model import model_cmd

                model_cmd(args)
            case "inf":
                from .inference import inf_cmd

                inf_cmd(args)
            case "convert":
                from .convert import convert_cmd

                convert_cmd(args)
            case _:
                index_cmd(args)
    except SystemExit:
        # Let argparse's sys.exit calls through (for help, errors, etc.)
        raise
    except KeyboardInterrupt:
        sys.exit(1)
