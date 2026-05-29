"""Argument parser for diloco command."""

import argparse
import os
from argparse import RawTextHelpFormatter

from forgather.ml.diloco.auth import add_auth_args
from forgather.tls.runtime import add_server_tls_args


def create_diloco_parser(global_args):
    """Create parser for diloco command."""
    parser = argparse.ArgumentParser(
        prog="forgather diloco",
        description="DiLoCo distributed training (Local-SGD with outer optimizer)",
        formatter_class=RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="diloco_subcommand", help="DiLoCo subcommands"
    )

    # server subcommand
    server_parser = subparsers.add_parser(
        "server",
        help="Start DiLoCo parameter server",
        formatter_class=RawTextHelpFormatter,
    )
    server_parser.add_argument(
        "--output-dir",
        "-o",
        type=os.path.expanduser,
        required=True,
        help="Training output directory. This should be a model directory, if not --from-checkpoint",
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=8512,
        help="Server port (default: 8512)",
    )
    server_parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        required=True,
        help="Number of expected workers",
    )
    server_parser.add_argument(
        "--outer-lr",
        type=float,
        default=0.7,
        help="Outer optimizer learning rate (default: 0.7)",
    )
    server_parser.add_argument(
        "--outer-momentum",
        type=float,
        default=0.9,
        help="Outer optimizer momentum (default: 0.9)",
    )
    server_parser.add_argument(
        "--no-nesterov",
        action="store_true",
        help="Disable Nesterov momentum for outer optimizer",
    )
    server_parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save server state every N sync rounds (default: 10); set to 0 to disable automatic save",
    )
    server_parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help=(
            "Maximum number of checkpoints to keep. Oldest are deleted\n"
            "when the limit is exceeded. 0 = keep all. (default: 3)"
        ),
    )
    server_parser.add_argument(
        "--from-checkpoint",
        "-c",
        default=None,
        type=os.path.expanduser,
        help="Load model from specified checkpoint path. Overrides loading from newest.",
    )
    server_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1). Use 0.0.0.0 for remote access.",
    )
    server_parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Enable asynchronous mode (workers don't wait for each other)",
    )
    server_parser.add_argument(
        "--dn-buffer-size",
        type=int,
        default=0,
        help=(
            "Delayed Nesterov buffer size. In async mode, buffer this many\n"
            "submissions before applying momentum. Between buffered steps,\n"
            "apply simple gradient descent. 0 = disabled (default: 0)"
        ),
    )
    server_parser.add_argument(
        "--dylu",
        action="store_true",
        help="Enable Dynamic Local Updates (DyLU) - adapt sync frequency per worker",
    )
    server_parser.add_argument(
        "--dylu-base-sync-every",
        type=int,
        default=500,
        help="DyLU base sync_every for the fastest worker (default: 500)",
    )
    server_parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=120.0,
        help=(
            "Seconds since last heartbeat before a worker is considered dead\n"
            "and evicted. Set to 0 to disable health monitoring. (default: 120)"
        ),
    )
    server_parser.add_argument(
        "--min-workers",
        type=int,
        default=1,
        help=(
            "Minimum workers required to proceed with sync. If the number\n"
            "of registered workers drops below this, the barrier will not\n"
            "release. (default: 1)"
        ),
    )
    server_parser.add_argument(
        "--default-work-units",
        type=int,
        default=1024,
        help=(
            "Number of work units per (dataset_id, shuffle_seed) queue when\n"
            "workers opt into work-unit dispatch. Server-wide; persists for\n"
            "the lifetime of the server. (default: 1024)"
        ),
    )

    # Security (issue #90): bearer-token auth + TLS. Mirrors the
    # operator-facing flags on dataset_server and inference_server so
    # the surface is identical across servers.
    add_auth_args(server_parser)
    add_server_tls_args(server_parser)

    # Two-port bulk plane (issue #90). When ``--bulk-port`` is set the
    # large pseudo-gradient / global-params endpoints move to a
    # separate listener. The defaults for that listener are
    # opt-out-everything (no TLS, no auth) — matching torch.distributed
    # on a trusted LAN. Operators who want both ports fully secured
    # can pass ``--bulk-tls`` and (implicitly via the control auth
    # token) ``--bulk-auth``. RCE protection is independent: every
    # inbound tensor blob is loaded with ``weights_only=True``.
    server_parser.add_argument(
        "--bulk-port",
        type=int,
        default=None,
        help=(
            "Run /submit_pseudograd, /submit_fragment_pseudograd, and\n"
            "/global_params on a separate listener at this port. Workers\n"
            "learn the URL from the X-Forgather-Bulk-Url header on\n"
            "/register. Leaves the control port for the small JSON wire."
        ),
    )
    bulk_tls_group = server_parser.add_mutually_exclusive_group()
    bulk_tls_group.add_argument(
        "--bulk-tls",
        dest="bulk_tls",
        action="store_const",
        const=True,
        default=None,
        help=(
            "Force TLS on the bulk listener. Default with --bulk-port "
            "is cleartext (matching torch.distributed)."
        ),
    )
    bulk_tls_group.add_argument(
        "--no-bulk-tls",
        dest="bulk_tls",
        action="store_const",
        const=False,
        help="Force TLS off on the bulk listener (the default when --bulk-port is set).",
    )
    bulk_auth_group = server_parser.add_mutually_exclusive_group()
    bulk_auth_group.add_argument(
        "--bulk-auth",
        dest="bulk_auth",
        action="store_const",
        const=True,
        default=None,
        help=(
            "Require the same bearer token on the bulk listener. "
            "Default with --bulk-port is no-auth — the opt-out for "
            "throughput on a trusted LAN."
        ),
    )
    bulk_auth_group.add_argument(
        "--no-bulk-auth",
        dest="bulk_auth",
        action="store_const",
        const=False,
        help="Run the bulk listener without bearer auth (the default when --bulk-port is set).",
    )

    # status subcommand
    status_parser = subparsers.add_parser(
        "status",
        help="Get DiLoCo server status",
        formatter_class=RawTextHelpFormatter,
    )
    status_parser.add_argument(
        "--server",
        type=str,
        default="localhost:8512",
        help="Server address as host:port (default: localhost:8512)",
    )
    status_parser.add_argument(
        "--auth-token",
        default=None,
        help=(
            "Bearer token for authenticated servers. When omitted, the "
            "client falls back to the FORGATHER_DILOCO_SERVER_TOKEN env "
            "var, then to the per-port loopback file. Remote servers "
            "without one of those configured will see 401."
        ),
    )
    status_parser.add_argument(
        "--no-verify-tls",
        action="store_true",
        help=(
            "Skip TLS certificate verification on the upstream server. "
            "Intended for SSH-tunneled remotes where the trust boundary "
            "is external."
        ),
    )

    # worker subcommand
    worker_parser = subparsers.add_parser(
        "worker",
        help="Run training as a DiLoCo worker",
        formatter_class=RawTextHelpFormatter,
    )
    worker_parser.add_argument(
        "--server",
        type=str,
        required=True,
        help="DiLoCo server address as host:port",
    )
    # NOTE: sync_every / bf16_comm / dylu / num_fragments are NOT worker
    # flags. They must match across the group, so the server is their sole
    # authority — the worker reads them from /info at startup. See
    # DiLoCoCallback (server-authoritative settings).
    worker_parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Worker ID (auto-generated if not provided)",
    )
    worker_parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=30.0,
        help=(
            "Seconds between heartbeats to server. Enables server-side\n"
            "health monitoring and DyLU speed reporting. 0 = disabled.\n"
            "Client-local; validated against the server's heartbeat-timeout.\n"
            "(default: 30)"
        ),
    )
    worker_parser.add_argument(
        "-d",
        "--devices",
        type=str,
        default=None,
        help='CUDA Visible Devices e.g. "0,1"',
    )
    worker_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the generated command without executing",
    )
    worker_parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Remaining arguments forwarded to the training script",
    )

    return parser
