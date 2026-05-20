"""
Shared configuration utilities for inference server and client.
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union


def load_config_from_yaml(config_path: str, use_logging: bool = True) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file
        use_logging: If True, use logging module; if False, use print

    Returns:
        Configuration dictionary

    Raises:
        Exception: If config file cannot be loaded (when use_logging=True)
        SystemExit: If config file cannot be loaded (when use_logging=False)
    """
    try:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        message = f"Loaded configuration from: {config_path}"
        if use_logging:
            logging.info(message)
        else:
            print(message)

        return config or {}
    except Exception as e:
        error_message = f"Failed to load configuration from {config_path}: {e}"
        if use_logging:
            logging.error(error_message)
            raise
        else:
            print(f"Error: {error_message}")
            sys.exit(1)


def merge_config_with_args(
    config: Dict[str, Any], args: argparse.Namespace, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    """
    Merge YAML config with command line arguments, with CLI args taking precedence.

    Args:
        config: Configuration dictionary from YAML file
        args: Parsed command-line arguments
        parser: ArgumentParser instance used to parse args

    Returns:
        Updated args namespace with config values merged in
    """
    # Convert config keys to match argument names (replace - with _)
    normalized_config = {}
    for key, value in config.items():
        normalized_key = key.replace("-", "_")
        normalized_config[normalized_key] = value

    # Get default values from parser to detect which args were actually set
    defaults = {}
    for action in parser._actions:
        if action.dest not in ("help", "config"):
            defaults[action.dest] = action.default

    # For each config value, set it if the argument uses the default value
    for key, value in normalized_config.items():
        if hasattr(args, key):
            current_value = getattr(args, key)
            default_value = defaults.get(key)

            # Only override if the current value is the default (wasn't explicitly set)
            if current_value == default_value:
                # Special handling for specific argument types
                if key == "stop_sequences":
                    # Server: handle stop_sequences which uses nargs="*"
                    setattr(
                        args,
                        key,
                        (
                            value
                            if isinstance(value, list)
                            else [value] if value else None
                        ),
                    )
                elif key == "stop":
                    # Client: handle stop sequences list
                    if isinstance(value, list):
                        setattr(args, key, value)
                    else:
                        setattr(args, key, value)
                elif key == "model":
                    # Server: -m is action="append"; YAML may carry a scalar
                    # "model: PATH" (legacy) or a list. Normalize to a list.
                    if isinstance(value, list):
                        setattr(args, key, list(value))
                    elif value:
                        setattr(args, key, [value])
                elif key in ("echo", "no_echo"):
                    # Client: handle boolean flags correctly
                    if isinstance(value, bool):
                        setattr(args, key, value)
                    else:
                        setattr(args, key, value)
                else:
                    setattr(args, key, value)

    return args


def _parse_model_arg(raw: str) -> Dict[str, Any]:
    """Parse a single -m argument into a ``{name, path}`` dict.

    Accepts either ``PATH`` (name = basename of normpath) or ``NAME=PATH``.
    ``~`` is expanded; the path is not required to exist (the loader will
    error later with a more informative message).
    """
    if "=" in raw:
        name, _, path = raw.partition("=")
        name = name.strip()
        path = path.strip()
        if not name:
            raise ValueError(f"--model {raw!r}: empty name before '='")
        if not path:
            raise ValueError(f"--model {raw!r}: empty path after '='")
    else:
        path = raw.strip()
        if not path:
            raise ValueError("--model: empty path")
        # Derive name from path basename (after normalizing trailing slashes).
        name = os.path.basename(os.path.normpath(path))
        if not name:
            raise ValueError(f"--model {raw!r}: cannot derive name from path")
    return {"name": name, "path": os.path.expanduser(path)}


_VALID_ENTRY_KEYS = {
    "name",
    "path",
    "dtype",
    "attn_implementation",
    "chat_template",
    "stop_sequences",
    "compile_args",
    "cache_implementation",
    "use_cache",
}


def _parse_yaml_model_entry(idx: int, entry: Any) -> Dict[str, Any]:
    """Validate and normalize one YAML ``models:`` list entry."""
    if not isinstance(entry, dict):
        raise ValueError(
            f"models[{idx}]: expected a mapping, got {type(entry).__name__}"
        )
    # Allow "model_path" or "path"; "model" too for forgiveness.
    path = entry.get("path") or entry.get("model_path") or entry.get("model")
    if not path:
        raise ValueError(f"models[{idx}]: missing 'path' (or 'model_path') field")
    name = entry.get("name")
    if not name:
        name = os.path.basename(os.path.normpath(str(path)))
        if not name:
            raise ValueError(f"models[{idx}]: cannot derive name from path {path!r}")

    out: Dict[str, Any] = {"name": str(name), "path": os.path.expanduser(str(path))}
    for key in _VALID_ENTRY_KEYS:
        if key in ("name", "path"):
            continue
        if key in entry:
            out[key] = entry[key]
    # Surface unknown keys early — typos in YAML are easy to miss.
    unknown = set(entry.keys()) - _VALID_ENTRY_KEYS - {"model_path", "model"}
    if unknown:
        raise ValueError(
            f"models[{idx}] ({name!r}): unknown keys: {sorted(unknown)}. "
            f"Allowed: {sorted(_VALID_ENTRY_KEYS)}"
        )
    return out


def merge_model_entries(
    cli_models: Optional[List[str]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Reconcile CLI ``-m`` args and YAML ``models:`` list into specs.

    Precedence:
      1. If CLI passed any ``-m``: use those (config's ``models:`` is
         ignored — operator-mode wins over file-mode).
      2. Else if config has ``models:``: use that.
      3. Else: empty list (the caller errors out: --model required).

    Each returned spec is a dict with at least ``name`` and ``path``;
    YAML entries may also carry per-model overrides
    (``dtype``, ``stop_sequences``, ``chat_template``, etc.).

    Raises ``ValueError`` on malformed input or duplicate names.
    """
    specs: List[Dict[str, Any]] = []

    if cli_models:
        for raw in cli_models:
            specs.append(_parse_model_arg(raw))
    else:
        yaml_models = config.get("models")
        if yaml_models:
            if not isinstance(yaml_models, list):
                raise ValueError(
                    f"config 'models': expected a list, got {type(yaml_models).__name__}"
                )
            for idx, entry in enumerate(yaml_models):
                specs.append(_parse_yaml_model_entry(idx, entry))

    # Detect duplicate names early — easier to diagnose here than at load time.
    names = [s["name"] for s in specs]
    if len(set(names)) != len(names):
        seen: set = set()
        dups: List[str] = []
        for n in names:
            if n in seen and n not in dups:
                dups.append(n)
            seen.add(n)
        raise ValueError(
            f"duplicate model name(s): {dups}. "
            "Use NAME=PATH form to disambiguate."
        )

    return specs
