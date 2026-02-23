"""Argument parser for diloco command."""

import argparse
from argparse import RawTextHelpFormatter


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
        "-m",
        "--model-path",
        type=str,
        required=True,
        help="Path to model directory (loads state_dict for initial global params)",
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
        "--save-dir",
        type=str,
        default=None,
        help="Directory for periodic server state saves",
    )
    server_parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save server state every N sync rounds (default: 10)",
    )
    server_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to server state file to resume from",
    )
    server_parser.add_argument(
        "--from-checkpoint",
        "-c",
        action="store_true",
        help="Load model from Forgather checkpoint format",
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
        "--no-dashboard",
        action="store_true",
        help="Disable the web dashboard (served at /dashboard by default)",
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
    worker_parser.add_argument(
        "--sync-every",
        type=int,
        default=500,
        help="Number of local optimizer steps between syncs (default: 500)",
    )
    worker_parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Worker ID (auto-generated if not provided)",
    )
    worker_parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bfloat16 communication (send full precision pseudo-gradients)",
    )
    worker_parser.add_argument(
        "--dylu",
        action="store_true",
        help="Enable Dynamic Local Updates - adapt sync_every based on server recommendations",
    )
    worker_parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=30.0,
        help=(
            "Seconds between heartbeats to server. Enables server-side\n"
            "health monitoring and DyLU speed reporting. 0 = disabled.\n"
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
        "--num-fragments",
        type=int,
        default=1,
        help=(
            "Number of fragments for streaming sync. When > 1, splits the model\n"
            "into N fragments that sync at staggered intervals in background\n"
            "threads, overlapping communication with computation. (default: 1)"
        ),
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
