#!/usr/bin/env python3
"""
forgather dataset server — uvicorn-hosted FastAPI app.

Entry point. Parses CLI arguments, configures logging, builds a
:class:`ServerState` from the parsed flags, mints/loads a bearer
token, and hands the FastAPI app to ``uvicorn.run``.

Runs as:

    tools/dataset_server/server.py [args...]      # standalone (chmod +x)
    python tools/dataset_server/server.py [...]   # standalone via interpreter
    python -m tools.dataset_server [...]          # module form
    forgather dataset-server start [...]          # via the forgather CLI

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import sys
from pathlib import Path
from typing import Optional

from forgather.preprocess import forgather_config_dir

# Support both standalone (`./server.py …`, `python server.py …`) and
# module (`python -m tools.dataset_server`, `from tools.dataset_server …`)
# execution. When running as a script there's no parent package, so we
# patch sys.path to make `dataset_server.*` importable. Mirrors the
# pattern used by tools/inference_server/server.py.
if __name__ == "__main__" and __package__ is None:
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from dataset_server.app import create_app
    from dataset_server.auth import standalone_token_file, write_standalone_token
    from dataset_server.state import ServerState
else:
    from .app import create_app
    from .auth import standalone_token_file, write_standalone_token
    from .state import ServerState

import uvicorn

from forgather.tls import (
    TLSRequiredError,
    enforce_non_loopback_policy,
)
from forgather.tls.runtime import (
    add_server_tls_args,
)
from forgather.tls.runtime import uvicorn_ssl_kwargs as tls_uvicorn_ssl_kwargs

_SERVICE_NAME = "dataset_server"


def default_config_file() -> Path:
    """Path checked for a default YAML config when ``--config`` is omitted.

    Lives next to the per-port token files at
    ``<forgather_config_dir>/dataset_server/config.yaml`` (Linux:
    ``~/.config/forgather/dataset_server/config.yaml``) so a user's
    operator-managed dataset_server settings live alongside the rest of
    that tool's state. Existence is checked at startup; if absent, the
    server falls back to pure CLI defaults — no error.
    """
    return Path(forgather_config_dir()) / "dataset_server" / "config.yaml"


class _HelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Combine raw-text wrapping with auto-appended (default: …) suffixes,
    skipping boolean flags where the default tag is just noise."""

    def _get_help_string(self, action):
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            return action.help or ""
        return super()._get_help_string(action)


def _parse_local(value: str) -> tuple[str, str]:
    """``foo=/abs/path`` -> ``("foo", "/abs/path")``."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--local must be NAME=PATH (got {value!r})")
    name, path = value.split("=", 1)
    name = name.strip()
    path = os.path.expanduser(path.strip())
    if not name:
        raise argparse.ArgumentTypeError(f"--local name is empty in {value!r}")
    if "/" in name:
        raise argparse.ArgumentTypeError(f"--local name {name!r} must not contain '/'")
    if not path:
        raise argparse.ArgumentTypeError(f"--local path is empty in {value!r}")
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"--local path does not exist: {path}")
    return name, os.path.abspath(path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataset_server",
        formatter_class=_HelpFormatter,
        description=(
            "forgather dataset server — serves HuggingFace datasets and\n"
            "named local datasets to remote clients via the\n"
            "FORGATHER_DATASET_SERVER env var (or directly via /v1/load).\n"
        ),
        epilog=(
            "Examples:\n"
            "\n"
            "Cache-only HF + a couple of named locals:\n"
            "  dataset_server --local stories=/data/tinystories \\\n"
            "                 --local mycorpus=/data/saved_corpus\n"
            "\n"
            "Lock down to local mappings only (no HF, no path access):\n"
            "  dataset_server --no-hf --local foo=/data/foo\n"
            "\n"
            "Trusted-LAN with auth disabled:\n"
            "  dataset_server -H 0.0.0.0 --no-auth\n"
        ),
    )
    parser.add_argument("-H", "--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("-p", "--port", type=int, default=8766, help="Bind port")
    parser.add_argument(
        "-l",
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    # Auth
    auth_group = parser.add_mutually_exclusive_group()
    auth_group.add_argument(
        "--auth-token",
        default=None,
        help=(
            "Bearer token clients must present in 'Authorization: Bearer "
            "<token>'. Auto-generated if neither this nor --auth-token-file "
            "is given."
        ),
    )
    auth_group.add_argument(
        "--auth-token-file",
        default=None,
        type=os.path.expanduser,
        help=(
            "Read the bearer token from this file (mode 0600 expected). "
            "Avoids exposing the token via argv (visible in 'ps')."
        ),
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help=(
            "Disable bearer-token auth. Any host able to reach the bind "
            "port becomes able to query the server — only set this when "
            "the network is already trusted."
        ),
    )
    parser.add_argument(
        "--regen-token",
        action="store_true",
        help=(
            "Generate a fresh auth token at startup, overwriting the "
            "persisted per-port token file. Existing clients (peers "
            "running training jobs against this server) will start "
            "getting 401 until they pick up the new token. Mirrors "
            "'forgather server --regen-token'."
        ),
    )
    parser.add_argument(
        "--quiet-tokens",
        action="store_true",
        help=(
            "Don't print the bearer token (or token-bearing curl example) "
            "to stderr at launch. Token is still written to its per-port "
            "file when auto-generated, so peers can still discover it. "
            "Intended for demo / public-exposure setups where the TTY "
            "log is visible to untrusted callers."
        ),
    )

    # Loading policy
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help=(
            "Disable HF cache loading entirely. Only datasets registered "
            "via --local will be servable."
        ),
    )
    parser.add_argument(
        "--allow-paths",
        action="store_true",
        help=(
            "Allow clients to request loading by absolute filesystem path. "
            "Off by default — the preferred method is named --local "
            "mappings, which avoid leaking server-side paths to clients."
        ),
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help=(
            "Allow HF dataset downloads when the cache is missing the "
            "requested dataset. Off by default — typical use is to serve "
            "what's already cached."
        ),
    )
    parser.add_argument(
        "--local",
        action="append",
        default=[],
        metavar="NAME=PATH",
        type=_parse_local,
        help=(
            "Register a local dataset accessible as 'local/NAME'. PATH "
            "must exist. Repeatable: --local foo=/p1 --local bar=/p2"
        ),
    )

    add_server_tls_args(parser)
    parser.add_argument(
        "--config",
        default=None,
        type=os.path.expanduser,
        metavar="FILE",
        help=(
            "Optional YAML config file. Keys mirror the CLI flag names "
            "with '-' replaced by '_' (e.g. 'no_auth: true', 'allow_paths: "
            "true'). The 'local' key takes a mapping of NAME -> PATH. CLI "
            "flags override file values. When omitted, the server looks "
            "for a default at <forgather_config_dir>/dataset_server/"
            "config.yaml (Linux: ~/.config/forgather/dataset_server/"
            "config.yaml) and loads it if present. See the README's "
            "'Configuration file' section for an example."
        ),
    )

    return parser


def _load_yaml_config(path: str) -> dict:
    """Load and validate a YAML config file."""
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(f"--config requires PyYAML; pip install pyyaml ({exc})")
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    except OSError as exc:
        raise SystemExit(f"Could not read --config {path}: {exc}")
    except yaml.YAMLError as exc:
        raise SystemExit(f"Invalid YAML in --config {path}: {exc}")
    if not isinstance(data, dict):
        raise SystemExit(f"--config {path} must contain a YAML mapping at root")
    return data


def _merge_config(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config: dict,
) -> None:
    """Apply ``config`` values to ``args`` for keys still at their default.

    CLI flags always win — values that the user supplied on the command
    line keep their value. The ``local`` key is special-cased: in the
    YAML it is a mapping ``{name: path, ...}``; we translate it to the
    same ``[(name, path), ...]`` shape that ``--local`` produces.
    """
    defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}
    known_keys = set(defaults) - {"config"}

    # Collect locals defined on the CLI so we can merge with file
    # locals rather than overwrite (a user-supplied --local should
    # add to, not replace, the file's local mappings).
    cli_locals = list(args.local or [])

    for raw_key, value in config.items():
        key = raw_key.replace("-", "_")
        if key not in known_keys:
            raise SystemExit(
                f"Unknown key in --config: {raw_key!r} "
                f"(allowed: {sorted(known_keys)})"
            )
        if key == "local":
            # YAML form: {name: path, ...} -> [(name, abspath), ...]
            if not isinstance(value, dict):
                raise SystemExit(
                    f"--config 'local' must be a mapping (got {type(value).__name__})"
                )
            file_locals = []
            for name, path in value.items():
                if not isinstance(name, str) or "/" in name:
                    raise SystemExit(
                        f"--config local: invalid name {name!r} "
                        "(must be string without '/')"
                    )
                if not isinstance(path, str):
                    raise SystemExit(f"--config local[{name!r}]: path must be a string")
                resolved = os.path.abspath(os.path.expanduser(path))
                if not os.path.exists(resolved):
                    raise SystemExit(
                        f"--config local[{name!r}]: path does not exist: {resolved}"
                    )
                file_locals.append((name, resolved))
            # Locals: union of file + CLI, with CLI winning on conflict
            # (matches "CLI overrides config" precedence).
            cli_names = {n for n, _ in cli_locals}
            merged = [(n, p) for n, p in file_locals if n not in cli_names]
            merged.extend(cli_locals)
            args.local = merged
            continue
        # For non-local keys: only override if arg is still at its default.
        if getattr(args, key) == defaults[key]:
            setattr(args, key, value)


def _resolve_auth_token(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[Optional[str], str]:
    """Resolve the effective bearer token from CLI args.

    Returns ``(token, source)``. ``token`` is ``None`` for ``--no-auth``;
    ``source`` is one of:

    - ``"cli"``         — explicit ``--auth-token``
    - ``"file"``        — explicit ``--auth-token-file``
    - ``"persisted"``   — loaded from the per-port file at
                          ``<config>/dataset_server/<port>.token``
    - ``"generated"``   — freshly minted; entry point should persist it
                          and emit the "wrote per-port token" banner
    - ``"regenerated"`` — freshly minted because ``--regen-token`` was
                          passed (same persistence as ``"generated"``,
                          but the banner is louder so the operator
                          notices any peers that just got invalidated)

    Persistence lives in the per-port file: this mirrors the
    forgather-server's auth_token model. A long-running peer that
    pulled the token last week keeps working after a restart — until
    the operator runs with ``--regen-token``.
    """
    if args.no_auth:
        return None, "cli"
    if args.auth_token:
        return args.auth_token.strip(), "cli"
    if args.auth_token_file:
        try:
            text = Path(args.auth_token_file).read_text().strip()
        except OSError as exc:
            parser.error(f"could not read --auth-token-file: {exc}")
        if not text:
            parser.error(f"auth-token-file is empty: {args.auth_token_file}")
        return text, "file"
    if args.regen_token:
        return secrets.token_hex(32), "regenerated"
    token_path = standalone_token_file(args.port)
    if token_path.is_file():
        try:
            text = token_path.read_text().strip()
        except OSError as exc:
            parser.error(f"could not read persisted token at {token_path}: {exc}")
        if text:
            return text, "persisted"
        # Empty file is treated as "missing" — fall through to mint.
    return secrets.token_hex(32), "generated"


def _format_auth_mode(args: argparse.Namespace, token_source: Optional[str]) -> str:
    """Human-readable auth line built from the *resolved* token source.

    ``token_source`` is the value ``_resolve_auth_token`` returned
    (cli / file / persisted / generated / regenerated) or ``None``
    when auth is disabled. We render the actual outcome rather than
    re-deriving it from argv precedence — important for distinguishing
    "first run, minted a fresh token" (generated) from "reused the
    existing per-port file" (persisted), which look identical in argv.
    """
    if args.no_auth or token_source is None:
        return "disabled (--no-auth)"
    if token_source == "cli":
        return "token via --auth-token"
    if token_source == "file":
        return f"token from file: {args.auth_token_file}"
    if token_source == "persisted":
        return "persisted per-port (reused existing token file)"
    if token_source == "generated":
        return "generated (minted + persisted to per-port file)"
    if token_source == "regenerated":
        return "regenerated (--regen-token; per-port file overwritten)"
    return token_source  # forward-compat: unknown source


def _log_effective_config(
    logger: logging.Logger,
    args: argparse.Namespace,
    config_path: Optional[str],
    config_path_source: str,
    token_source: Optional[str],
) -> None:
    """Dump every effective setting at startup for postmortem visibility.

    Renders one ``key=value`` per log line so grep + tail-of-log is
    enough to answer "what did this instance start with?". Auth-token
    *source* is included (cli / file / persisted / regenerated /
    generated) so a token-rotation incident is reconstructible from
    logs alone — the actual token never lands here.
    """
    auth_mode = _format_auth_mode(args, token_source)

    if config_path:
        cfg_line = f"{config_path} ({config_path_source})"
    else:
        cfg_line = "<none>"

    locals_lines = (
        [f"{name}={path}" for name, path in args.local] if args.local else ["<none>"]
    )

    logger.info("effective configuration:")
    logger.info("  host             = %s", args.host)
    logger.info("  port             = %d", int(args.port))
    logger.info("  log_level        = %s", args.log_level)
    logger.info("  config_file      = %s", cfg_line)
    logger.info("  auth             = %s", auth_mode)
    logger.info("  hf_cache_enabled = %s", not args.no_hf)
    logger.info("  allow_paths      = %s", bool(args.allow_paths))
    logger.info("  allow_downloads  = %s", bool(args.allow_downloads))
    logger.info("  locals (%d):", len(args.local))
    for line in locals_lines:
        logger.info("    %s", line)


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # If --config was not provided, fall back to the default location.
    # Missing default is silently ignored so the server still starts
    # cleanly on a fresh install; an explicit --config that points at a
    # missing path still surfaces as a load error below.
    config_path: Optional[str] = args.config
    config_path_source: str = "cli"
    if config_path is None:
        default_path = default_config_file()
        if default_path.is_file():
            config_path = str(default_path)
            config_path_source = "default"
    if config_path:
        config = _load_yaml_config(config_path)
        _merge_config(parser, args, config)
        args.config = config_path

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger(_SERVICE_NAME)
    logger.setLevel(log_level)

    if config_path and config_path_source == "default":
        logger.info("loaded default config: %s", config_path)
    elif config_path:
        logger.info("loaded config: %s", config_path)
    else:
        logger.info("no config file (CLI flags + built-in defaults only)")

    # Build state from policy flags + local mappings.
    state = ServerState(
        hf_cache_enabled=not args.no_hf,
        allow_paths=args.allow_paths,
        allow_downloads=args.allow_downloads,
    )
    for name, path in args.local:
        try:
            state.add_local(name, path)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))

    auth_token, token_source = _resolve_auth_token(parser, args)
    state.auth_required = bool(auth_token)

    # Effective-configuration dump: everything the operator could have
    # influenced via flags or the YAML config, plus the *resolved*
    # auth-token source (persisted vs generated vs regenerated) so a
    # "why did the token change?" report has the data right next to
    # the startup banner. Token *value* is logged separately on stderr
    # below (one place to copy from); this dump never echoes the secret.
    _log_effective_config(
        logger,
        args,
        config_path,
        config_path_source,
        token_source if auth_token else None,
    )

    if not auth_token:
        print(
            "!! dataset_server is running with --no-auth — any host that "
            "can reach the bind port can query datasets",
            file=sys.stderr,
            flush=True,
        )
    else:
        # Loud header line for --regen-token so any peer using the
        # previous token gets a visible heads-up in the operator's
        # terminal that they're about to start 401-ing.
        if token_source == "regenerated":
            print(
                "!! --regen-token: replacing the persisted per-port token. "
                "Existing peers will need to re-pull.",
                file=sys.stderr,
                flush=True,
            )
        # --quiet-tokens suppresses the actual value (and the curl example
        # that embeds it) for demo / public-exposure setups; the source +
        # persistence message still goes out so operators know auth is on.
        if args.quiet_tokens:
            print(
                "dataset_server auth: bearer-token enabled "
                "(value suppressed by --quiet-tokens)",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"dataset_server auth token: {auth_token}",
                file=sys.stderr,
                flush=True,
            )
            print(
                "clients must send 'Authorization: Bearer <token>'",
                file=sys.stderr,
                flush=True,
            )
            # Scheme is decided below once TLS state is known; emit a
            # placeholder here and let the launch sequence print the final
            # URL line. Keep this curl snippet under the bearer-token
            # banner so operators see auth + URL together.
            try:
                from forgather.tls import is_enabled as _tls_enabled

                _scheme = "https" if _tls_enabled() else "http"
            except Exception:
                _scheme = "http"
            print(
                f'curl -H "Authorization: Bearer {auth_token}" '
                f"{_scheme}://{args.host}:{args.port}/v1/datasets",
                file=sys.stderr,
                flush=True,
            )

        # Persist the per-port token file for any auto-discovered token.
        # Explicit --auth-token / --auth-token-file paths intentionally
        # don't touch the per-port file — operator-managed tokens stay
        # wherever the operator put them. The file is NOT deleted on
        # exit; it survives restarts (mirrors forgather-server's
        # auth_token persistence) and rotates only on --regen-token.
        if token_source in ("generated", "regenerated"):
            try:
                token_path = write_standalone_token(args.port, auth_token)
            except OSError as exc:
                logger.warning(
                    "could not write standalone-server token file: %s "
                    "(client auto-discovery disabled)",
                    exc,
                )
            else:
                print(
                    f"persisted token file: {token_path}",
                    file=sys.stderr,
                    flush=True,
                )
        elif token_source == "persisted":
            print(
                f"reusing persisted token at: {standalone_token_file(args.port)}",
                file=sys.stderr,
                flush=True,
            )

    app = create_app(state, auth_token=auth_token)

    logger.info(
        "starting dataset_server on %s:%d (hf=%s, allow_paths=%s, "
        "allow_downloads=%s, locals=%d, auth=%s)",
        args.host,
        args.port,
        state.hf_cache_enabled,
        state.allow_paths,
        state.allow_downloads,
        len(state.local_datasets),
        "on" if auth_token else "off",
    )

    try:
        ssl_kwargs = tls_uvicorn_ssl_kwargs(args)
    except FileNotFoundError as exc:
        logger.error("TLS config error: %s", exc)
        return 2
    tls_on = bool(ssl_kwargs)
    try:
        from forgather.tls import load_config as _tls_load_config

        enforce_non_loopback_policy(
            args.host,
            tls_enabled=tls_on,
            insecure=args.insecure,
            service="dataset_server",
            cfg=_tls_load_config(),
        )
    except TLSRequiredError as exc:
        logger.error("%s", exc)
        return 2
    if tls_on:
        scheme = "https"
        logger.info("TLS: serving HTTPS from %s", ssl_kwargs["ssl_certfile"])
    else:
        scheme = "http"
    print(
        f"dataset_server URL: {scheme}://{args.host}:{args.port}/",
        file=sys.stderr,
        flush=True,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=True,
        **ssl_kwargs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
