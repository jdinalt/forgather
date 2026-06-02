"""Argument parser for the `submit` command.

`forgather submit` is shorthand for submitting the current project's config to
the forgather-server scheduler (the explicit, memorable spelling of
`forgather train --schedule`). `--global` promotes it to a multi-node fan-out
(absorbing the deprecated `cluster submit`).
"""

import argparse
from argparse import RawTextHelpFormatter

from .dynamic_args import parse_dynamic_args
from .submit_orch import add_locality_args, add_via_server_arg


def create_submit_parser(global_args):
    parser = argparse.ArgumentParser(
        prog="forgather submit",
        description="Submit the current project's config to the scheduler",
        formatter_class=RawTextHelpFormatter,
        epilog=(
            "Project + config come from the global forgather flags -p / -t,\n"
            "like `forgather train`. Single-node by default (background);\n"
            "--foreground attaches, --global fans out across the cluster."
        ),
    )

    # Single-node (default) knobs — shared with `train --schedule`.
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Attach to the scheduled job and stream its output (single-node).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="SOURCE",
        help=(
            "Dataset source: 'auto' (cluster routing), 'local' (in-process),\n"
            "or 'server:<id>'. Unset = mode-aware (auto under cluster mode)."
        ),
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Scheduler priority (default: 0).",
    )
    parser.add_argument(
        "--requested-gpus",
        type=int,
        default=None,
        metavar="N",
        help="GPUs to request (single-node; default: config's nproc_per_node).",
    )
    add_via_server_arg(parser)
    add_locality_args(parser)

    # Multi-node fan-out (--global; formerly `cluster submit`).
    parser.add_argument(
        "--global",
        dest="run_global",
        action="store_true",
        help="Fan out across the cluster (multi-node). Requires -p <abs-path>.",
    )
    parser.add_argument(
        "--member",
        "--members",
        dest="member",
        action="append",
        default=[],
        metavar="HOST:GPUS[:IFACE]",
        help=(
            "--global: per-node spec, repeatable or comma-separated. HOST is\n"
            "the member hostname, GPUS the requested count, IFACE an optional\n"
            "NCCL socket interface. Omit to use every reachable member's GPUs."
        ),
    )
    parser.add_argument(
        "--rdzv-host",
        default=None,
        metavar="HOSTNAME",
        help="--global: rendezvous host (default: cluster master).",
    )
    parser.add_argument(
        "--rdzv-port",
        type=int,
        default=None,
        help="--global: rendezvous port (default: 29400).",
    )
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="--global: skip the cross-peer forgather/torch/nccl version check.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="--global: poll the bundle until terminal, exit non-zero on failure.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=3600,
        help="--global: seconds to wait when --wait is set (default: 3600).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="--global: seconds between polls when --wait is set (default: 10).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="--global: emit the submit response JSON instead of the table view.",
    )

    parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are forwarded (single-node, like train).",
    )
    parse_dynamic_args(parser, global_args)
    return parser
