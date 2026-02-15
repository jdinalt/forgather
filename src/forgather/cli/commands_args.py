"""Argument parsers for simple commands in commands.py."""

import argparse
from argparse import RawTextHelpFormatter

from .dynamic_args import parse_dynamic_args
from .utils import add_editor_arg, add_output_arg


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
