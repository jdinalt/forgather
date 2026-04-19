"""Argument parser for workspace command."""

import argparse
import os
from argparse import RawTextHelpFormatter

path_type = lambda x: os.path.normpath(os.path.expanduser(x))


def create_ws_parser(global_args):
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
