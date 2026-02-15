"""Argument parser for dataset command."""

import argparse
import os
from argparse import RawTextHelpFormatter

from .dynamic_args import parse_dynamic_args
from .utils import add_editor_arg, add_output_arg


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
    parser.add_argument(
        "--num-shards",
        type=int,
        help="Split the dataset into N shards (for distributed processing)",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="The shard to select, out of `num-shards` (for distributed processing)",
    )
    parser.add_argument(
        "--select-range",
        help="Select dataset range. eg. '100:500', '10%%:', ':0.1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Shuffle with seed",
    )
    add_output_arg(parser)
    add_editor_arg(parser)
    parse_dynamic_args(parser, global_args)
    return parser
