"""
CLI entry: ``python -m tools.dataset_server [options]``.

Starts a foreground dataset server. With ``--allow-load`` it accepts
``POST /v1/load`` requests so a client running with
``FORGATHER_DATASET_SERVER=http://host:port`` can transparently route
its ``fast_load_iterable_dataset(...)`` calls through the server.

Examples
--------
Run a load-on-demand server on the default port::

    python -m tools.dataset_server --allow-load

Bind to all interfaces on a custom port::

    python -m tools.dataset_server --host 0.0.0.0 --port 9000 --allow-load

Run a registry-only server (preregister via Python; no HTTP load)::

    python -m tools.dataset_server --port 8765
"""

from __future__ import annotations

import argparse
import logging
import sys

from .server import DatasetServer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.dataset_server",
        description="Run a forgather dataset server (proof of concept).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help=(
            "Port to listen on (default: 8766). NOTE: 8765 is the "
            "forgather orchestration server's port — picked 8766 here "
            "to avoid the collision."
        ),
    )
    parser.add_argument(
        "--allow-load",
        action="store_true",
        help=(
            "Enable POST /v1/load — clients can request the server to "
            "lazily load HuggingFace datasets via fast_load_iterable_dataset. "
            "Required for transparent FORGATHER_DATASET_SERVER routing."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Server log level (default: INFO)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    srv = DatasetServer(host=args.host, port=args.port, allow_load=args.allow_load)
    srv.start()

    print(
        f"Dataset server listening on {srv.url} "
        f"(allow_load={args.allow_load}). Ctrl-C to stop.",
        flush=True,
    )

    try:
        # Hold the main thread; the HTTP server runs on a daemon thread.
        srv._thread.join()  # type: ignore[union-attr]
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
        srv.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
