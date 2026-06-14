"""Argument parser for diloco command."""

import argparse
import os
from argparse import RawTextHelpFormatter

from forgather.ml.diloco.auth import add_auth_args
from forgather.tls.runtime import add_server_tls_args

_TOKEN_SCALE = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "g": 1_000_000_000}


def _parse_token_count(s):
    """Parse a token count that may carry a K/M/B suffix.

    A bare number is raw tokens — back-compatible with scripts and the
    webui/orchestrator, which emit raw counts. A case-insensitive K/M/B (or G)
    suffix scales it, decimals allowed: ``2.08B`` = 2_080_000_000, ``2080M`` =
    same, ``500K`` = 500_000, ``8000000`` = 8_000_000. Returns an int.
    """
    s = str(s).strip()
    if not s:
        raise argparse.ArgumentTypeError("empty token count")
    mult = 1
    if s[-1].lower() in _TOKEN_SCALE:
        mult = _TOKEN_SCALE[s[-1].lower()]
        s = s[:-1].strip()
    try:
        val = float(s) * mult
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid token count {s!r} — use a number, optionally with a "
            "K/M/B suffix (e.g. 2.08B, 2080M, 8000000)"
        )
    if val < 0:
        raise argparse.ArgumentTypeError("token count must be >= 0")
    return int(val)


def create_diloco_parser(global_args):
    """Create parser for diloco command."""
    parser = argparse.ArgumentParser(
        prog="forgather diloco",
        description="DiLoCo: distributed low-communication training (local-SGD with an outer optimizer)",
        formatter_class=RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="diloco_subcommand", help="DiLoCo subcommands"
    )

    # Locality flags: the forgather server is the default, required path.
    # --local-fallback degrades to a direct/foreground action only when the
    # server is unreachable; --local-only skips the server entirely. Without
    # either, an unreachable server is an error (no silent local degrade).
    # Shared with train --schedule / submit / eval via submit_orch.
    from .submit_orch import add_locality_args as _add_locality_args

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
        "--run-name",
        default=None,
        help=(
            "Short label for this run's stats log dir "
            "(<output_dir>/runs/<timestamp>_<run-name>, holding the JSONL\n"
            "stream + TensorBoard events). Defaults to the hostname. A resume\n"
            "from checkpoint continues the prior run's dir regardless."
        ),
    )
    server_parser.add_argument(
        "-H",
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
        "--grace-period",
        type=float,
        default=0.0,
        help=(
            "Async grace window in seconds (Liu et al. 2024, Sec. 3). When a\n"
            "worker submits, the server holds the response and aggregates any\n"
            "other workers that submit within the window into ONE outer step,\n"
            "so near-simultaneous workers resync against the same model.\n"
            "0 = disabled (default: 0.0). Only used with --async."
        ),
    )
    server_parser.add_argument(
        "--token-budget",
        type=_parse_token_count,
        default=0,
        metavar="TOKENS",
        help=(
            "Global training-token budget. When the aggregated cross-worker\n"
            "token count reaches it, the server relays save_and_stop to every\n"
            "worker — the controlling stop for open-ended runs (esp. async,\n"
            "where uneven worker speeds make a per-worker step budget a poor\n"
            "proxy). Workers run open-ended. Accepts a K/M/B suffix\n"
            "(e.g. 2.08B, 2080M) or a bare token count. 0 = no budget (default: 0)."
        ),
    )
    server_parser.add_argument(
        "--verbose-sync",
        action="store_true",
        help=(
            "Log every sync round (server outer step + each worker's sync line) "
            "at INFO. Off by default — routine progress rides the per-step "
            "sync/up_mb/dn_mb/sync_s log columns. Server-authoritative: the "
            "workers adopt this from /info. A targeted DiLoCo diagnostic."
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
    # Group-wide worker settings (issue #53 follow-up). These must match
    # across every worker, so they're owned by the server and the workers
    # adopt them from /info — there are no corresponding worker flags.
    server_parser.add_argument(
        "--sync-every",
        type=int,
        default=500,
        help=(
            "Local optimizer steps between syncs (H), applied by every\n"
            "worker. Under --dylu the DyLU base rate is used instead.\n"
            "(default: 500)"
        ),
    )
    server_parser.add_argument(
        "--num-fragments",
        type=int,
        default=1,
        help=(
            "Streaming-sync fragments every worker splits the model into.\n"
            "1 = no streaming. Must be uniform across the group, so it's\n"
            "set here, not per worker. (default: 1)"
        ),
    )
    server_parser.add_argument(
        "--fragment-assignment",
        choices=["strided", "sequential"],
        default="strided",
        help=(
            "How transformer blocks are assigned to streaming fragments\n"
            "(Streaming DiLoCo, arXiv:2501.18512): 'strided' (block i ->\n"
            "fragment i %% N, the paper's mild preference) or 'sequential'\n"
            "(contiguous runs of blocks). Server-authoritative. (default: strided)"
        ),
    )
    # Wire precision (issue #130). Four server-authoritative knobs
    # advertised via /info and adopted by every worker. The legacy
    # ``--no-bf16`` flag is preserved as an alias for
    # ``--upload-dtype fp32`` so older operator scripts keep working.
    server_parser.add_argument(
        "--upload-dtype",
        dest="upload_dtype",
        choices=("fp32", "bf16"),
        default=None,
        help=(
            "Wire dtype for the worker → server pseudo-gradient leg.\n"
            "Centralized: the group's upload precision is set on the\n"
            "server and adopted by every worker. (default: bf16)"
        ),
    )
    server_parser.add_argument(
        "--upload-sr",
        dest="upload_sr",
        action="store_true",
        help=(
            "Use stochastic rounding for the fp32 → bf16 upload cast.\n"
            "No effect when --upload-dtype=fp32 or when both snapshot\n"
            "and live weights are already bf16. (default: off)"
        ),
    )
    server_parser.add_argument(
        "--download-dtype",
        dest="download_dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help=(
            "Wire dtype for the server → worker averaged-params leg.\n"
            "``bf16`` halves return-path bandwidth — convergence impact\n"
            "is the open research question (issue #130). (default: fp32)"
        ),
    )
    server_parser.add_argument(
        "--download-sr",
        dest="download_sr",
        action="store_true",
        help=(
            "Use stochastic rounding for the fp32 → bf16 download cast\n"
            "on the server. Only meaningful with --download-dtype=bf16.\n"
            "(default: off)"
        ),
    )
    server_parser.add_argument(
        "--backend",
        dest="backend",
        choices=("http", "shared_memory", "collective"),
        default="http",
        help=(
            "Sync backend the worker group must use (issue #154). Declared\n"
            "here and advertised via /info; the launcher derives each worker's\n"
            "backend from this at launch, and every running worker also\n"
            "validates its own against it and fails loud on disagreement — so a\n"
            "group can't be launched with workers that disagree on how to\n"
            "communicate. The single source of truth: `forgather submit` does\n"
            "not take --backend on the orchestrated path. (default: http)"
        ),
    )
    server_parser.add_argument(
        "--wire-format",
        dest="wire_format",
        choices=("pickle", "safetensors"),
        default="pickle",
        help=(
            "Bulk-tensor wire codec for both sync legs (issue #154).\n"
            "``safetensors`` drops pickle for an explicit typed, zero-copy\n"
            "frame (no arbitrary-code deserialization); ``pickle`` is the\n"
            "back-compatible default for an older worker. (default: pickle)"
        ),
    )
    server_parser.add_argument(
        "--grpc",
        dest="grpc_enabled",
        action="store_true",
        help=(
            "Serve the bulk legs over a streaming gRPC listener (issue #154)\n"
            "instead of the HTTP control port. Advertised via /info so workers\n"
            "negotiate it; HTTP stays the fallback. Supersedes --bulk-cleartext\n"
            "(gRPC is the single bulk fast-path). Cleartext/trusted-LAN today;\n"
            "TLS parity is a follow-up. (default: off)"
        ),
    )
    server_parser.add_argument(
        "--no-bf16",
        dest="bf16_comm",
        action="store_false",
        default=None,
        help=(
            "DEPRECATED alias for ``--upload-dtype fp32``. Kept so older\n"
            "operator scripts keep working. Mutually exclusive with\n"
            "--upload-dtype."
        ),
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

    # Cleartext bulk plane (issue #90). When enabled, the large
    # pseudo-gradient / global-params endpoints move to a separate
    # cleartext, unauthenticated listener on a server-picked ephemeral
    # port — its sole purpose is to bypass TLS for throughput on a
    # trusted LAN. There is nothing to configure: a TLS bulk plane would
    # defeat the point (just use the control port), and a bearer over a
    # sniffable socket is theater. Workers learn the ephemeral port from
    # the X-Forgather-Bulk-Url header on /register (over the TLS control
    # plane). RCE protection is independent: every inbound tensor blob is
    # loaded with ``weights_only=True``.
    server_parser.add_argument(
        "--bulk-cleartext",
        dest="bulk_cleartext",
        action="store_true",
        default=False,
        help=(
            "Bypass TLS for bulk data: serve /submit_pseudograd,\n"
            "/submit_fragment_pseudograd, and /global_params on a\n"
            "separate cleartext listener on a server-assigned ephemeral\n"
            "port (workers learn it from the X-Forgather-Bulk-Url header\n"
            "on /register). Trades on-wire confidentiality of the bulk\n"
            "tensors for throughput; use only on a trusted network."
        ),
    )

    # Launch-as-scheduled-job knobs. By default `server` enqueues a
    # diloco_server job through the forgather server; --local-only runs it in
    # the foreground (also how the scheduler spawns the real server, so it
    # doesn't re-enqueue), --local-fallback runs foreground only if the
    # server is down.
    _add_locality_args(server_parser)
    server_parser.add_argument(
        "--server",
        "--via-server",
        dest="server",
        type=str,
        default=None,
        metavar="URL",
        help="forgather-server base URL to enqueue on (default: env / http://127.0.0.1:8765).",
    )
    server_parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Scheduler priority for the enqueued server job (default: 0).",
    )
    server_parser.add_argument(
        "--json",
        action="store_true",
        help="When enqueuing, emit the created queue item as JSON.",
    )

    # status subcommand
    status_parser = subparsers.add_parser(
        "status",
        help="Get DiLoCo server status",
        formatter_class=RawTextHelpFormatter,
    )
    status_parser.add_argument(
        "--diloco-server",
        dest="diloco_server",
        type=str,
        default=None,
        help=(
            "DiLoCo server: a server id/label/host:port. When omitted, the\n"
            "single running server is used automatically (ambiguous if more\n"
            "than one); falls back to localhost:8512 when the forgather\n"
            "server can't be consulted."
        ),
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
    status_parser.add_argument(
        "--queues",
        action="store_true",
        help=(
            "Also show work-unit dispatch: per-queue dataset label, row "
            "count, issued/completed, and a per-worker issued/completed "
            "breakdown (no per-unit heatmap — see the webui for that)."
        ),
    )
    status_parser.add_argument(
        "--watch",
        "-w",
        action="store_true",
        help=(
            "Refresh the status in place every --interval seconds until\n"
            "Ctrl-C (like `watch`, but in-process — reuses the connection\n"
            "and works in the interactive CLI). Not compatible with --json."
        ),
    )
    status_parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Refresh interval for --watch (default: 2.0).",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the merged status (status + info + workers [+ queues]) as JSON.",
    )
    status_parser.add_argument(
        "--server",
        "--via-server",
        dest="server",
        type=str,
        default=None,
        metavar="URL",
        help=(
            "forgather-server base URL to route through (default: "
            "$FORGATHER_SERVER_URL or http://127.0.0.1:8765). When the "
            "server is reachable and knows this target, status is read "
            "through it (central token/TLS handling)."
        ),
    )
    _add_locality_args(status_parser)

    # servers subcommand — discovery via the forgather server.
    servers_parser = subparsers.add_parser(
        "servers",
        help="List DiLoCo servers the forgather server knows (local + registered)",
        formatter_class=RawTextHelpFormatter,
    )
    servers_parser.add_argument(
        "--server",
        "--via-server",
        dest="server",
        type=str,
        default=None,
        metavar="URL",
        help="forgather-server base URL (default: env / http://127.0.0.1:8765).",
    )
    servers_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the server list as JSON.",
    )

    # logs subcommand — dump / tail a worker's or server's captured TTY.
    logs_parser = subparsers.add_parser(
        "logs",
        help="Dump or follow a DiLoCo worker/server job's captured TTY log",
        formatter_class=RawTextHelpFormatter,
    )
    logs_parser.add_argument(
        "job",
        type=str,
        help=(
            "Job to read: a queue_id, a local DiLoCo server id/label, or a\n"
            "worker_id. Resolved to the underlying job via the forgather\n"
            "server. (Raw queue_ids also work with 'forgather job tail'.)"
        ),
    )
    logs_parser.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="Stream new output until the job exits or Ctrl-C.",
    )
    logs_parser.add_argument(
        "--path",
        action="store_true",
        help=(
            "Instead of printing the log, print the path to the captured TTY\n"
            "file (on the forgather server's host) and exit — e.g. for\n"
            '`tail -f "$(forgather diloco logs <job> --path)"`. Takes\n'
            "precedence over --follow."
        ),
    )
    logs_parser.add_argument(
        "--server",
        "--via-server",
        dest="server",
        type=str,
        default=None,
        metavar="URL",
        help="forgather-server base URL (default: env / http://127.0.0.1:8765).",
    )

    # Shared client-connection args for the control-plane subcommands.
    def _add_client_conn_args(p):
        p.add_argument(
            "--diloco-server",
            dest="diloco_server",
            type=str,
            default=None,
            help=(
                "DiLoCo server: a server id/label/host:port. When omitted,\n"
                "the single running server is used automatically (ambiguous\n"
                "if more than one); falls back to localhost:8512 when the\n"
                "forgather server can't be consulted."
            ),
        )
        p.add_argument(
            "--auth-token",
            default=None,
            help=(
                "Bearer token for authenticated servers. When omitted, the "
                "client falls back to the FORGATHER_DILOCO_SERVER_TOKEN env "
                "var, then to the per-port loopback file."
            ),
        )
        p.add_argument(
            "--no-verify-tls",
            action="store_true",
            help="Skip TLS certificate verification on the upstream server.",
        )
        p.add_argument(
            "--server",
            "--via-server",
            dest="server",
            type=str,
            default=None,
            metavar="URL",
            help=(
                "forgather-server base URL to route through (default: env / "
                "http://127.0.0.1:8765). When the server is up and knows this "
                "target, the action goes through it (central token/TLS)."
            ),
        )
        _add_locality_args(p)

    # control subcommand — relay a trainer-control command to workers.
    control_parser = subparsers.add_parser(
        "control",
        help="Relay save / save-stop / abort to one or all workers",
        formatter_class=RawTextHelpFormatter,
    )
    control_parser.add_argument(
        "action",
        choices=["save", "save-stop", "abort"],
        help=(
            "save      request a checkpoint on every (or one) worker\n"
            "save-stop save a final checkpoint then stop the worker(s)\n"
            "abort     stop immediately without saving"
        ),
    )
    control_parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Target a single worker by id. Omitted = all registered workers.",
    )
    _add_client_conn_args(control_parser)

    # token-budget subcommand — show or set the server's global token budget.
    budget_parser = subparsers.add_parser(
        "token-budget",
        help="Show or set the server's global token budget at runtime",
        formatter_class=RawTextHelpFormatter,
    )
    budget_parser.add_argument(
        "value",
        type=_parse_token_count,
        nargs="?",
        default=None,
        metavar="TOKENS",
        help=(
            "New global token budget (0 = open-ended). Accepts a K/M/B suffix\n"
            "(e.g. 2.08B, 2080M) or a bare token count. Omit to show the current\n"
            "value. Lowering it below the tokens trained so far stops the\n"
            "workers on their next heartbeat."
        ),
    )
    _add_client_conn_args(budget_parser)

    # shutdown subcommand — stop the server (clean by default).
    shutdown_parser = subparsers.add_parser(
        "shutdown",
        help="Stop the DiLoCo server (cleanly stops workers first by default)",
        formatter_class=RawTextHelpFormatter,
    )
    shutdown_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Stop the server immediately without stopping workers first.\n"
            "Workers will fail on their next sync. Default is a clean\n"
            "shutdown: save-stop all workers, wait for them to exit,\n"
            "checkpoint the server, then stop it."
        ),
    )
    shutdown_parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help=(
            "Clean shutdown only: max seconds to wait for workers to stop\n"
            "before giving up (default: 600). On timeout the server is left\n"
            "running so you can troubleshoot."
        ),
    )
    _add_client_conn_args(shutdown_parser)

    # register / unregister — manage external DiLoCo servers in the
    # forgather server's registry (orchestrator-only).
    register_parser = subparsers.add_parser(
        "register",
        help="Register an external DiLoCo server with the forgather server",
        formatter_class=RawTextHelpFormatter,
    )
    register_parser.add_argument(
        "url",
        type=str,
        help="Base URL of the external DiLoCo server, e.g. https://host:8512",
    )
    register_parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Human-friendly label (defaults to the base URL).",
    )
    register_parser.add_argument(
        "--auth-token",
        type=str,
        default=None,
        help="Bearer token the proxy uses upstream. Omit for a --no-auth server.",
    )
    register_parser.add_argument(
        "--no-verify-tls",
        action="store_true",
        help="Skip TLS chain validation for this entry (SSH-tunneled remotes).",
    )
    register_parser.add_argument(
        "--server",
        "--via-server",
        dest="server",
        type=str,
        default=None,
        metavar="URL",
        help="forgather-server base URL (default: env / http://127.0.0.1:8765).",
    )
    register_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the created registry entry as JSON.",
    )

    unregister_parser = subparsers.add_parser(
        "unregister",
        help="Remove a registered external DiLoCo server",
        formatter_class=RawTextHelpFormatter,
    )
    unregister_parser.add_argument(
        "entry_id",
        type=str,
        help="Registry entry id (accepts the 'registered:<id>' form from 'servers').",
    )
    unregister_parser.add_argument(
        "--server",
        "--via-server",
        dest="server",
        type=str,
        default=None,
        metavar="URL",
        help="forgather-server base URL (default: env / http://127.0.0.1:8765).",
    )

    return parser
