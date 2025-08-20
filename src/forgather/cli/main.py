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
    subparsers.add_parser("index", help="Show project index")

    """ ls """
    ls_parser = subparsers.add_parser("ls", help="List available configurations")

    ls_parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search for project in all sub-directories and list them.",
    )

    """ meta """
    subparsers.add_parser("meta", help="Show meta configuration")

    """ targets """
    subparsers.add_parser("targets", help="Show output targets")

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

    graph_parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Shows details of template preprocessing to assist with debugging",
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

    pp_parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Shows details of template preprocessing to assist with debugging",
    )

    pp_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Shows additional debug information",
    )

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

    """ dataset """
    ds_parser = subparsers.add_parser(
        "dataset", help="Dataset preprocessing and testing"
    )

    ds_parser.add_argument(
        "-T",
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer to test",
    )

    ds_parser.add_argument(
        "--pp",
        action="store_true",
        help="Show preprocessed configuration",
    )

    ds_parser.add_argument(
        "-H",
        "--histogram",
        action="store_true",
        help="Generate dataset token length historgram and statistics",
    )

    ds_parser.add_argument(
        "--target",
        type=str,
        default="train_dataset_split",
        help="The dataset to sample from; see \"forgather targets\"",
    )

    ds_parser.add_argument(
        "--histogram-samples",
        type=int,
        default=1000,
        help="Number of samples to use for histogram",
    )

    ds_parser.add_argument(
        "-c",
        "--chat-template",
        type=str,
        default=None,
        help="Path to chat template",
    )

    ds_parser.add_argument(
        "-n",
        "--examples",
        type=int,
        default=None,
        help="Number of examples to print",
    )

    ds_parser.add_argument(
        "-s",
        "--tokenized",
        action="store_true",
        help="The split is already tokenized",
    )

    return parser.parse_args(args)


def main():
    """Main CLI entry point."""
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
        case _:
            index_cmd(args)
