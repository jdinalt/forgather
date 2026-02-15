"""Argument parser for train command."""

import argparse
from argparse import RawTextHelpFormatter

from .dynamic_args import parse_dynamic_args


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
