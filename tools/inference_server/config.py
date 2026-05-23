"""
Shared configuration utilities for inference server and client.
"""

import argparse
import logging
import sys
from typing import Any, Dict


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
                elif key in ("echo", "no_echo"):
                    # Client: handle boolean flags correctly
                    if isinstance(value, bool):
                        setattr(args, key, value)
                    else:
                        setattr(args, key, value)
                else:
                    setattr(args, key, value)

    return args
