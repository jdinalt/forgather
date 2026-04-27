#!/usr/bin/env python3
"""Forgather web server entry point.

Usage (typically via the CLI shim):
    forgather server -H 127.0.0.1 -p 8765
    python tools/forgather_server/server.py -H 127.0.0.1 -p 8765

The server is intended to be a single-user, localhost-first prototype. It
binds to 127.0.0.1 by default; anyone who wants to expose it on a LAN should
do so behind an SSH tunnel or a reverse proxy of their choosing.
"""

import argparse
import logging
import sys
from argparse import RawTextHelpFormatter
from pathlib import Path

# Support both `python -m forgather_server.server` and standalone execution.
if __name__ == "__main__" and __package__ is None:
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from forgather_server.app import create_app
else:
    from .app import create_app

import uvicorn


def main():
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Forgather web server (prototype)",
    )
    parser.add_argument("-H", "--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("-p", "--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument(
        "-l",
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload (development)",
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger("forgather_server").setLevel(log_level)

    app = create_app()

    logging.info(f"Starting Forgather server on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=True,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
