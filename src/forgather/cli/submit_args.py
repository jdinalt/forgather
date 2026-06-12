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
        "--dry-run",
        action="store_true",
        help="Show what would be submitted (or the command) without doing it.",
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
        help=(
            "GPUs to request per job/worker (single-node default: config's\n"
            "nproc_per_node; DiLoCo default: 1). --global sizes nodes via\n"
            "--member instead."
        ),
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

    # DiLoCo worker opt-in (formerly `forgather diloco worker`). --diloco submits
    # the project as one or more DiLoCo workers (independent local-SGD replicas
    # joining a param-server) instead of a plain training job. Plain --diloco
    # without --global is a different parallelism axis (independent replicas vs
    # one rendezvous), so the two are mutually exclusive — but combining --global
    # with --diloco-server composes them: the multi-node bundle is one logical
    # DiLoCo worker group (e.g. multi-node Pipeline Parallel averaged with
    # another such group via DiLoCo). Per-worker GPUs come from --requested-gpus
    # (default 1).
    diloco = parser.add_argument_group("DiLoCo worker (opt-in)")
    diloco.add_argument(
        "--diloco",
        action="store_true",
        help="Submit as DiLoCo worker(s) (joins a param-server; see flags below).",
    )
    diloco.add_argument(
        "--diloco-server",
        dest="diloco_server",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "DiLoCo param-server to join: a server id/label/host:port. When\n"
            "omitted, the single running server is used automatically. Implies\n"
            "--diloco. Combine with --global to make the multi-node bundle one\n"
            "logical DiLoCo worker group."
        ),
    )
    diloco.add_argument(
        "--diloco-worker-count",
        dest="count",
        type=int,
        default=1,
        metavar="N",
        help="Launch N identical DiLoCo workers as scheduled jobs (default: 1).",
    )
    diloco.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help=(
            "Worker id (auto-generated if not provided). With --global +\n"
            "--diloco-server this is the *base* id shared by every rank;\n"
            "the PP callback appends ``_pp<rank>``."
        ),
    )
    diloco.add_argument(
        "--resume-workers",
        dest="resume_workers",
        action="store_true",
        help=(
            "Re-launch every stopped worker the server knows (reusing each id\n"
            "so it resumes its checkpoint). Implies --diloco; can't be combined\n"
            "with --worker-id / --diloco-worker-count."
        ),
    )
    diloco.add_argument(
        "--heartbeat-interval",
        type=float,
        default=30.0,
        help="Seconds between worker heartbeats to the server (default: 30).",
    )
    diloco.add_argument(
        "--backend",
        choices=("http", "shared_memory"),
        default=None,
        help=(
            "Dev/debug only: the worker's sync backend, honored solely with\n"
            "--local-only (a direct foreground launch with no server to query).\n"
            "The orchestrated path does NOT take this — the backend is declared\n"
            "once on the param server ('diloco server --backend …') and derived\n"
            "at launch from /info, so workers can't disagree. 'collective' is a\n"
            "launch topology, selected with --diloco-replicate (not here)."
        ),
    )
    diloco.add_argument(
        "--env",
        dest="worker_env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra environment variable for the worker process(es), repeatable\n"
            "(KEY=VALUE). Forwarded to each scheduled worker job's env. Intended\n"
            "for debug/tuning knobs (e.g. DILOCO_DEBUG_STEP_DELAY); submit one\n"
            "worker per distinct env to build a heterogeneous group. Plain\n"
            "--diloco worker submits only (not --global compose / collective)."
        ),
    )
    diloco.add_argument(
        "--diloco-replicate",
        dest="replicate",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Select the collective topology: N independent replicas in one\n"
            "torchrun job (nproc_per_node = N, also the GPU reservation). The\n"
            "replicas all-reduce pseudo-gradients among themselves; the\n"
            "coordinator provides /info + the dataset shard dispatch. The param\n"
            "server must declare --backend collective. Single-host; not\n"
            "compatible with --global. Default: 1 (no collective)."
        ),
    )

    parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are forwarded (single-node, like train).",
    )
    parse_dynamic_args(parser, global_args)
    return parser
