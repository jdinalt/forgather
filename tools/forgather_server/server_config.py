"""Server config file: YAML defaults for the ``forgather server`` CLI.

The server resolves its config path in this order:

1. ``--config PATH`` on the command line (explicit override).
2. Default at ``<forgather_config_dir>/server/server_config.yaml``.

If the resolved file does not exist, a template is written with every
known option commented out so the operator can see the defaults and
uncomment what they want to change. Once loaded, top-level ``args:``
keys supply defaults for ``argparse`` — any value also passed on the
command line still wins.

Only the ``args:`` section is consumed today; other top-level sections
are reserved for future use (per-feature config blocks).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from . import paths

log = logging.getLogger("forgather_server.config")


_DEFAULT_TEMPLATE = """\
# Forgather server config.
#
# Top-level ``args:`` keys override the CLI argument defaults. Anything
# passed on the command line still takes precedence over this file.
# Comment out a key to fall back to the built-in default shown.
#
# Other top-level sections may be added in the future for per-feature
# configuration; for now only ``args:`` is consumed.

args:
  # host: 127.0.0.1
  # port: 8765
  # log_level: INFO
  # reload: false
  # no_auth: false
  # regen_token: false
  # persist_sessions: false   # keep browser sessions across restarts (dev)
  # cluster: default       # cluster name (always-on; mDNS scoping unit)
  # cluster_address: []
  # lock_inference_proxy: false
  # docs_landing: null        # Path the Docs view opens by default.
                              # Absolute or relative to the repo root.
                              # null = built-in (docs/README.md).
  #
  # TLS options (see ``forgather tls --help``):
  # insecure: false
  # tls: null              # path to TLS config file

# AI agent (right-sidebar assistant). This block only BOOTSTRAPS a profile
# on first run; manage agent profiles in the webui thereafter (they persist
# in agent_profiles.json and hot-swap without a restart). Uses the Anthropic
# SDK for both Claude and local vLLM models — point ``base_url`` at a vLLM
# server that serves the Anthropic Messages API (``/v1/messages``) for local
# models, or omit it for Claude.
# agent:
#   provider: anthropic        # only adapter today
#   model: claude-sonnet-4-6   # or a vLLM --served-model-name alias; blank = auto
#   base_url: null             # e.g. https://kitt:8000 for local vLLM; null = Claude
#   api_key_env: ANTHROPIC_API_KEY   # env var holding the key (or vLLM bearer)
#   # api_key: null            # explicit key/bearer (overrides api_key_env)
#   # verify_tls: true         # false = accept any cert (LAN self-signed); or
#   #                          #   import the cert via the webui for verified TLS
#   # max_tokens: 4096
#   # max_iterations: 12       # tool-use loop cap per user message
"""


# Filled in by ``load`` so other modules (and the API) can report which
# file backed this run's startup.
_LOADED_PATH: Optional[Path] = None


def default_config_path() -> Path:
    """Default config path under the per-user server state directory."""
    return paths.server_state_dir() / "server_config.yaml"


def _ensure_template(path: Path) -> None:
    """Create a commented template at ``path`` if it doesn't exist.

    Best-effort. If the parent directory isn't writable we log and
    continue; load() will return an empty dict.
    """
    if path.exists():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DEFAULT_TEMPLATE)
        # Match the rest of the per-user server state.
        try:
            path.chmod(0o600)
        except OSError:
            pass
        log.info("wrote template server config: %s", path)
    except OSError as e:
        log.warning("could not create template server config %s: %s", path, e)


def load(explicit_path: Optional[str]) -> Tuple[Path, Dict[str, Any]]:
    """Resolve the config path and read it.

    Returns ``(path, data)`` where ``path`` is the resolved location and
    ``data`` is the parsed YAML as a dict (``{}`` when missing, empty,
    or unparseable). If ``explicit_path`` is None, the default path is
    used and a template is written when missing. If ``explicit_path``
    is provided but doesn't exist, raise ``FileNotFoundError`` — the
    operator asked for a specific file and we shouldn't silently fall
    back.
    """
    global _LOADED_PATH
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"server config not found: {path}")
    else:
        path = default_config_path()
        _ensure_template(path)

    data: Dict[str, Any] = {}
    if path.exists():
        try:
            with open(path) as f:
                loaded = yaml.safe_load(f)
            if loaded is None:
                data = {}
            elif isinstance(loaded, dict):
                data = loaded
            else:
                log.warning(
                    "server config %s did not parse as a mapping; ignoring",
                    path,
                )
        except (OSError, yaml.YAMLError) as e:
            log.warning("failed to read server config %s: %s", path, e)

    _LOADED_PATH = path
    return path, data


def loaded_path() -> Optional[Path]:
    """Resolved config path from the most recent ``load()`` call.

    Returns ``None`` if ``load()`` hasn't been called (e.g., before
    server startup completed).
    """
    return _LOADED_PATH


def args_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the ``args:`` block, normalizing the keys to argparse dest names.

    YAML uses kebab-case or snake_case naturally; argparse dests are
    snake_case. Convert dashes to underscores so either spelling works
    in the file.
    """
    raw = data.get("args") or {}
    if not isinstance(raw, dict):
        log.warning("server config 'args:' is not a mapping; ignoring")
        return {}
    return {str(k).replace("-", "_"): v for k, v in raw.items()}
