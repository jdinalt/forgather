"""Argument parser for the `forgather cluster` subcommand."""

import argparse
from argparse import RawTextHelpFormatter


def create_cluster_parser(global_args):
    parser = argparse.ArgumentParser(
        prog="forgather cluster",
        description=(
            "Manage multi-node operation against a forgather-server "
            "running with --cluster. Hostnames in --member flags "
            "resolve to the cluster's UUIDs via the membership table, "
            "so you never have to handle UUIDs yourself."
        ),
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        metavar="URL",
        help=(
            "forgather-server base URL "
            "(default: $FORGATHER_SERVER_URL or http://127.0.0.1:8765)"
        ),
    )

    sub = parser.add_subparsers(dest="cluster_subcommand", help="Cluster subcommands")

    nodes_parser = sub.add_parser(
        "nodes",
        help="List cluster members (hostname, address, reachability, GPUs)",
        formatter_class=RawTextHelpFormatter,
    )
    nodes_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw /api/cluster/members JSON instead of the table view",
    )

    jobs_parser = sub.add_parser(
        "jobs",
        help="List multi-node bundles (or show one)",
        formatter_class=RawTextHelpFormatter,
    )
    jobs_parser.add_argument(
        "cluster_job_id",
        nargs="?",
        default=None,
        help="If set, print details for one bundle; otherwise list all",
    )
    jobs_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON instead of the table view",
    )

    submit_parser = sub.add_parser(
        "submit",
        help="Submit a multi-node training fan-out",
        formatter_class=RawTextHelpFormatter,
        epilog=(
            "Project + config come from the global forgather flags "
            "``-p`` / ``-t``, just like ``forgather train``. So:\n"
            "  forgather -p <project> -t <config> cluster submit ...\n"
            "Required arguments are checked at dispatch time, not by "
            "argparse, because they live in the global namespace."
        ),
    )
    submit_parser.add_argument(
        "--member",
        action="append",
        default=[],
        metavar="HOST:GPUS[:IFACE]",
        help=(
            "Per-node spec, repeatable. ``HOST`` is the cluster member's "
            "hostname (looked up in the membership table), ``GPUS`` is "
            "the requested GPU count for that node, and ``IFACE`` is "
            "an optional NCCL socket interface name. If --member is "
            "omitted, every reachable member contributes all of its "
            "available GPUs."
        ),
    )
    submit_parser.add_argument(
        "--rdzv-host",
        default=None,
        metavar="HOSTNAME",
        help=(
            "Hostname of the rendezvous host (default: cluster master). "
            "Resolved to a node_id via the membership table."
        ),
    )
    submit_parser.add_argument(
        "--rdzv-port",
        type=int,
        default=None,
        help="Rendezvous port (default: 29400)",
    )
    submit_parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Per-rank queue priority (default: 0)",
    )
    submit_parser.add_argument(
        "--dynamic-arg",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help=(
            "Dynamic argument forwarded to every rank, repeatable. "
            "Same as the Submit dialog's dynamic-args panel."
        ),
    )
    submit_parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help=(
            "Skip the cross-peer version check. The server otherwise "
            "returns HTTP 409 when forgather/torch/nccl/transformers "
            "versions diverge across the selected set."
        ),
    )
    submit_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the submit response JSON instead of the table view",
    )
    submit_parser.add_argument(
        "--wait",
        action="store_true",
        help=(
            "Poll the bundle until it reaches a terminal state, then "
            "exit with code 0 on done / non-zero on failed/cancelled. "
            "Useful for shell-driven smoke tests."
        ),
    )
    submit_parser.add_argument(
        "--wait-timeout",
        type=int,
        default=3600,
        help="Seconds to wait when --wait is set (default: 3600)",
    )
    submit_parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Seconds between polls when --wait is set (default: 10)",
    )

    cancel_parser = sub.add_parser(
        "cancel",
        help="Fan out cancel to every participant of a bundle",
        formatter_class=RawTextHelpFormatter,
    )
    cancel_parser.add_argument("cluster_job_id", help="Bundle id to cancel")

    return parser
