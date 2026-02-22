"""Argument parser for model command."""

import argparse
import os
from argparse import RawTextHelpFormatter

from .dynamic_args import parse_dynamic_args
from .utils import add_editor_arg, add_output_arg


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
    parser.add_argument(
        "--fuse-optim-with-backward",
        action="store_true",
        help="Combine backward with optimizer step to save memory",
    )
    parser.add_argument(
        "--refresh-model",
        "-r",
        action="store_true",
        help="Force regeneration of fresh model from sources by deleting output_dir"
    )

    subparsers = parser.add_subparsers(
        dest="model_subcommand", help="Model subcommands"
    )
    parser.add_argument("--save-checkpoint", action="store_true", help="Save model checkpoint")
    parser.add_argument("--safetensors", action="store_true", help="Save using safetensors")

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
        "--packed",
        action="store_true",
        help="Enable packed sequences in data collator",
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
