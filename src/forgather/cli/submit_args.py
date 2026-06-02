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
        help="--global / --diloco-server: emit the response as JSON.",
    )

    # DiLoCo worker opt-in (formerly `forgather diloco worker`). Selecting a
    # DiLoCo param-server (or --resume-workers) submits the project as one or
    # more DiLoCo workers that join that server, instead of a plain training
    # job. This is a different parallelism axis from --global (independent
    # local-SGD replicas vs one rendezvous), so the two are mutually exclusive.
    diloco = parser.add_argument_group("DiLoCo worker (opt-in)")
    diloco.add_argument(
        "--diloco-server",
        dest="server",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "Join this DiLoCo param-server as a worker: a server id/label/\n"
            "host:port. When omitted but --count/--worker-id is given, the\n"
            "single running server is used automatically."
        ),
    )
    diloco.add_argument(
        "--count",
        type=int,
        default=1,
        help="Launch N identical DiLoCo workers as scheduled jobs (default: 1).",
    )
    diloco.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Worker id (auto-generated if not provided).",
    )
    diloco.add_argument(
        "--resume-workers",
        dest="resume_workers",
        action="store_true",
        help=(
            "Re-launch every stopped worker the server knows (reusing each id\n"
            "so it resumes its checkpoint). Can't be combined with\n"
            "--worker-id / --count."
        ),
    )
    diloco.add_argument(
        "--heartbeat-interval",
        type=float,
        default=30.0,
        help="Seconds between worker heartbeats to the server (default: 30).",
    )
    diloco.add_argument(
        "--gpus-per-worker",
        type=int,
        default=1,
        help="GPUs the scheduler reserves per DiLoCo worker (default: 1).",
    )
    diloco.add_argument(
        "-d",
        "--devices",
        type=str,
        default=None,
        help='CUDA_VISIBLE_DEVICES for the direct/foreground worker, e.g. "0,1".',
    )
    diloco.add_argument(
        "--dry-run",
        action="store_true",
        help="Direct/foreground DiLoCo worker: show the command without running.",
    )

    parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are forwarded (single-node, like train).",
    )
    parse_dynamic_args(parser, global_args)
    return parser
