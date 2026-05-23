"""Argument parser for the `eval` subcommand."""

import argparse
import os
from argparse import RawTextHelpFormatter

path_type = lambda x: os.path.normpath(os.path.expanduser(x))


def create_eval_parser(_global_args):
    parser = argparse.ArgumentParser(
        prog="forgather eval",
        description="Evaluate a model on a named eval config",
        formatter_class=RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="eval_subcommand", help="Eval subcommands")

    # list
    subparsers.add_parser(
        "list",
        help="List available eval configs",
        formatter_class=RawTextHelpFormatter,
    )

    # show
    show = subparsers.add_parser(
        "show",
        help="Show details of an eval config",
        formatter_class=RawTextHelpFormatter,
    )
    show.add_argument("name", help="Eval config name (e.g., c4, tinystories)")
    show.add_argument(
        "--pp",
        action="store_true",
        help="Show the preprocessed YAML",
    )

    # test
    test = subparsers.add_parser(
        "test",
        help="Run an eval and write results to {model}/evals/",
        formatter_class=RawTextHelpFormatter,
    )
    test.add_argument("name", help="Eval config name")
    test.add_argument(
        "-M",
        "--model",
        type=path_type,
        default=None,
        help="Path to model directory. Defaults to current project's output_dir.",
    )
    test.add_argument(
        "-d",
        "--devices",
        type=str,
        default=None,
        help='CUDA_VISIBLE_DEVICES, e.g. "0,1"',
    )
    test.add_argument(
        "--trainer",
        choices=["ddp", "simple", "pipeline"],
        default="ddp",
        help="Trainer backend (default: ddp)",
    )
    test.add_argument("--checkpoint", type=path_type, default=None)
    test.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Use from_pretrained on the model dir instead of resuming a checkpoint",
    )
    test.add_argument("--batch-size", type=int, default=None)
    test.add_argument("--max-length", type=int, default=None)
    test.add_argument("--max-steps", type=int, default=-1)
    test.add_argument("--dtype", default="bfloat16")
    test.add_argument("--attn-implementation", default="sdpa")
    test.add_argument("--compile", action="store_true")
    test.add_argument(
        "--output-dir",
        type=path_type,
        default=None,
        help="Override where evals/ is written (default: model path)",
    )
    test.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the torchrun command without executing it",
    )
    test.add_argument(
        "--enqueue",
        action="store_true",
        help="Submit to the forgather-server queue instead of running locally.",
    )
    test.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Queue priority for --enqueue (default: 0).",
    )
    test.add_argument(
        "--server",
        type=str,
        default=None,
        metavar="URL",
        help="forgather-server URL for --enqueue (default: $FORGATHER_SERVER_URL or http://127.0.0.1:8765).",
    )

    return parser
