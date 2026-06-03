"""Argument parser for diloco command."""

import argparse
import os
import sys
from argparse import RawTextHelpFormatter

from forgather.ml.diloco.auth import add_auth_args
from forgather.tls.runtime import add_server_tls_args


def _diloco_subcommand(argv=None):
    """Best-effort: the ``diloco`` sub-subcommand being invoked (e.g.
    ``"worker"``), read from argv, or ``None``.

    Used only to decide whether to pay the config load for dynamic-arg
    discovery: ``create_diloco_parser`` builds every subparser eagerly, but
    only ``worker`` consumes/forwards a config's dynamic args, so the other
    verbs (status / servers / logs / control / shutdown) shouldn't load a
    project just to build their parser. The first non-flag token after
    ``diloco`` is the sub-subcommand (global flags precede ``diloco``)."""
    argv = sys.argv[1:] if argv is None else argv
    if "diloco" not in argv:
        return None
    for tok in argv[argv.index("diloco") + 1 :]:
        if not tok.startswith("-"):
            return tok
    return None


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
        "--no-bf16",
        dest="bf16_comm",
        action="store_false",
        help=(
            "Send full-precision pseudo-gradients instead of bfloat16.\n"
            "Centralized: the group's wire precision is set on the server\n"
            "and adopted by every worker. (default: bf16 enabled)"
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
        "--via-server",
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
        "--server",
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
        "--via-server",
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
        "--via-server",
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
        "--via-server",
        type=str,
        default=None,
        metavar="URL",
        help="forgather-server base URL (default: env / http://127.0.0.1:8765).",
    )

    # Shared client-connection args for the control-plane subcommands.
    def _add_client_conn_args(p):
        p.add_argument(
            "--server",
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
            "--via-server",
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

    # worker subcommand
    worker_parser = subparsers.add_parser(
        "worker",
        help="Run training as a DiLoCo worker",
        description=(
            "Run training as a DiLoCo worker (deprecated alias of "
            "`forgather submit --diloco`).\n"
            "Enqueues worker(s) through the forgather server by default; "
            "--local-only runs one in the foreground."
        ),
        formatter_class=RawTextHelpFormatter,
    )
    worker_parser.add_argument(
        "--server",
        type=str,
        default=None,
        help=(
            "DiLoCo server the worker connects to: a server id/label/host:port.\n"
            "When omitted, the single running server is used automatically\n"
            "(ambiguous if more than one); the direct/foreground path falls\n"
            "back to localhost:8512."
        ),
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
        "--dry-run",
        action="store_true",
        help="Show the generated command without executing (direct path only)",
    )
    # Launch-as-scheduled-job knobs (orchestrator path). When the forgather
    # server is up, the worker is enqueued as a training job; these control
    # how many, how they're named, and where data comes from.
    worker_parser.add_argument(
        "--count",
        type=int,
        default=1,
        help=(
            "Launch N identical workers as scheduled jobs with auto-generated\n"
            "names (requires the forgather server). Default: 1."
        ),
    )
    worker_parser.add_argument(
        "--resume-workers",
        dest="resume_workers",
        action="store_true",
        help=(
            "Re-launch every stopped worker the server knows (reusing each\n"
            "id, so it resumes its checkpoint) — the way to bring a worker set\n"
            "back after a shutdown/stop. Requires the forgather server; can't\n"
            "be combined with --worker-id / --count. Honors --dataset and\n"
            "dynamic args for the relaunched jobs. (Named to avoid clashing\n"
            "with configs' own --resume dynamic arg.)"
        ),
    )
    worker_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="SOURCE",
        help=(
            "Dataset source for the worker job(s): 'auto' (cluster routing),\n"
            "'local' (in-process loader), or 'server:<id>' for a specific\n"
            "dataset server (id from 'forgather diloco servers' or the\n"
            "dataset-server registry). When unset, the default is mode-aware\n"
            "(matching the webui): 'auto' if the forgather server is in\n"
            "cluster mode, otherwise 'local'."
        ),
    )
    worker_parser.add_argument(
        "--gpus-per-worker",
        type=int,
        default=1,
        help="GPUs the scheduler reserves per worker job (default: 1).",
    )
    worker_parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Scheduler priority for the enqueued worker job(s) (default: 0).",
    )
    worker_parser.add_argument(
        "--via-server",
        type=str,
        default=None,
        metavar="URL",
        help="forgather-server base URL to enqueue on (default: env / http://127.0.0.1:8765).",
    )
    _add_locality_args(worker_parser)
    worker_parser.add_argument(
        "--json",
        action="store_true",
        help="When enqueuing, emit the launched worker/queue-id list as JSON.",
    )
    worker_parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Remaining arguments forwarded to the training script (direct path)",
    )

    # Dynamic/template args: build argparse options + help from the config's
    # ``dynamic_args`` metadata (the established pattern — see train_args.py),
    # so a config's args show up under `diloco worker --help`, parse, and
    # forward. Like `forgather train`, this uses the selected config or the
    # project's DEFAULT config when ``-t`` is omitted (parse_dynamic_args
    # resolves the default via ``Project(config_name=None)``) — so
    # ``forgather diloco worker --compile no`` works from a project dir
    # without an explicit ``-t``.
    #
    # Gated to an actual ``worker`` invocation (not just "a config is
    # selected"): create_diloco_parser builds every subparser eagerly, so
    # without this gate every `diloco <sub>` (status / servers / logs / …)
    # would pay a config load — and error noisily ("Loading dynamic args
    # failed!") when run outside a project. Only `worker` forwards them.
    #
    # NOTE: we deliberately do NOT propagate ``_dynamic_arg_names`` to the
    # top-level diloco parser. main.py's partition is global to the chosen
    # subcommand, and the dynamic-arg set includes framework-standard names
    # like ``output_dir`` — smearing them across the namespace would strip
    # ``--output-dir`` from a sibling like ``diloco server``. ``_worker_cmd``
    # collects the worker's dynamic args from the config schema itself
    # (see ``_load_dynamic_schema`` / ``_worker_dynamic_args``).
    if _diloco_subcommand() == "worker" and not getattr(global_args, "no_dyn", False):
        from .dynamic_args import parse_dynamic_args

        parse_dynamic_args(worker_parser, global_args)

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
        "--via-server",
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
        "--via-server",
        type=str,
        default=None,
        metavar="URL",
        help="forgather-server base URL (default: env / http://127.0.0.1:8765).",
    )

    return parser
