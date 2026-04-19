"""Argument parser for project command."""

import argparse
import os
from argparse import RawTextHelpFormatter

path_type = lambda x: os.path.normpath(os.path.expanduser(x))


def create_project_parser(global_args):
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
