"""Argument parser for train command."""

import argparse
from argparse import RawTextHelpFormatter

from .dynamic_args import parse_dynamic_args
from .submit_orch import add_locality_args


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
        "-n",
        "--nproc",
        type=str,
        default=None,
        metavar="N",
        help=(
            "Override --nproc-per-node passed to torchrun (default: from"
            " config's nproc_per_node, typically 'gpu'). Use this on"
            " CPU-only hosts (e.g. '--nproc 1') or to limit the rank count."
            " Accepts the same values as torchrun: an integer, 'gpu', or"
            " 'auto'."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just show the generated commandline, without actually executing it.",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help=(
            "Submit to the forgather-server scheduler instead of running\n"
            "locally. Runs in the background by default; --foreground attaches\n"
            "and streams the job's output. (See also: forgather submit.)"
        ),
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help=(
            "With --schedule, attach to the scheduled job and stream its\n"
            "output until it exits (Ctrl-C detaches without stopping it)."
        ),
    )
    # Deprecated alias of --schedule (kept so existing scripts/examples work).
    parser.add_argument(
        "--enqueue",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="SOURCE",
        help=(
            "Dataset source for the scheduled job: 'auto' (cluster routing),\n"
            "'local' (in-process loader), or 'server:<id>'. Unset = mode-aware\n"
            "(auto when the server is in cluster mode, else local). Only\n"
            "applies with --schedule."
        ),
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Scheduler priority for --schedule (default: 0).",
    )
    parser.add_argument(
        "--requested-gpus",
        type=int,
        default=None,
        metavar="N",
        help="GPUs to request when scheduling (default: nproc_per_node from config).",
    )
    parser.add_argument(
        "--server",
        "--via-server",
        dest="via_server",
        type=str,
        default=None,
        metavar="URL",
        help=(
            "forgather-server URL for --schedule (default: $FORGATHER_SERVER_URL"
            " or http://127.0.0.1:8765)."
        ),
    )
    add_locality_args(parser)
    parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="All arguments after -- will be forwarded as torchrun arguments.",
    )
    parse_dynamic_args(parser, global_args)
    return parser
