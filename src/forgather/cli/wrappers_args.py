"""Argument parsers for wrapper commands that forward to external scripts."""

import argparse
from argparse import RawTextHelpFormatter


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


def create_finalize_parser(global_args):
    """Create parser for finalize command."""
    # add_help=False so --help is forwarded to the script for its own help.
    parser = argparse.ArgumentParser(
        prog="forgather finalize",
        description="Finalize a trained Forgather model into a clean directory\n\n"
        "All arguments are forwarded to tools/finalize_model/finalize_model.py",
        formatter_class=RawTextHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        "dummy",
        nargs="?",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments to forward to finalize_model.py",
    )

    return parser
