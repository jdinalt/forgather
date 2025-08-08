"""Main CLI controller and argument parsing."""

import argparse
from argparse import RawTextHelpFormatter
import os

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
)


def parse_args(args=None):
    # Common args
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Forgather CLI",
        epilog=(""),
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

    # Sub-commands
    subparsers = parser.add_subparsers(dest="command", help="subcommand help")

    """ index """
    index_parser = subparsers.add_parser("index", help="Show project index")

    """ ls """
    ls_parser = subparsers.add_parser("ls", help="List available configurations")

    ls_parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search for project in all sub-directories and list them.",
    )

    """ meta """
    meta_parser = subparsers.add_parser("meta", help="Show meta configuration")

    """ targets """
    targets_parser = subparsers.add_parser("targets", help="Show output targets")

    """ tlist """
    all_templates_parser = subparsers.add_parser(
        "tlist", help="List available templates."
    )

    all_templates_parser.add_argument(
        "--format",
        type=str,
        choices=["md", "files"],
        default="files",
        help="Output format.",
    )

    """ graph """
    graph_parser = subparsers.add_parser(
        "graph", help="Preprocess and parse into node graph"
    )

    graph_parser.add_argument(
        "--format",
        type=str,
        choices=["none", "repr", "yaml", "fconfig", "python"],
        default="yaml",
        help="Graph format",
    )

    """ trefs """
    referenced_templates_parser = subparsers.add_parser(
        "trefs", help="List referenced templates"
    )

    referenced_templates_parser.add_argument(
        "--format",
        type=str,
        choices=["md", "files"],
        default="files",
        help="Output format.",
    )

    """ pp """
    pp_parser = subparsers.add_parser("pp", help="Preprocess configuration")

    """ tb """
    tb_parser = subparsers.add_parser("tb", help="Start Tensorboard for project")

    tb_parser.add_argument(
        "--all",
        action="store_true",
        help="Configure TB to watch all model directories",
    )

    tb_parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="All arguments after -- will be forwarded as Tensorboard arguments.",
    )

    tb_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just show the generated commandline, without actually executing it.",
    )

    """ code """
    code_parser = subparsers.add_parser(
        "code", help="Output configuration as Python code"
    )

    code_parser.add_argument(
        "--target",
        type=str,
        default="main",
        help="Output target name",
    )

    """ construct """
    construct_parser = subparsers.add_parser(
        "construct", help="Materialize and print a target"
    )

    construct_parser.add_argument(
        "--target",
        type=str,
        default="main",
        help="Output target name",
    )

    construct_parser.add_argument(
        "--call",
        action="store_true",
        help="Call the materialized object",
    )

    """ train """
    train_parser = subparsers.add_parser(
        "train", help="Run configuration with train script"
    )

    train_parser.add_argument(
        "-d",
        "--devices",
        type=str,
        default=None,
        help='CUDA Visible Devices e.g. "0,1"',
    )

    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just show the generated commandline, without actually executing it.",
    )

    train_parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="All arguments after -- will be forwarded as torchrun arguments.",
    )

    return parser.parse_args(args)


def main():
    """Main CLI entry point."""
    args = parse_args()
    if args.config_template:
        args.config_template = os.path.basename(args.config_template)
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
        case _:
            index_cmd(args)
