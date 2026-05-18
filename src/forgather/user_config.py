"""User-level configuration loaded from ``<forgather_config_dir>/config.yaml``.

Lazy, best-effort reader. Missing file or parse errors return an empty dict so
callers can always treat the result as a dict.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from forgather.preprocess import forgather_config_dir


def user_config_path() -> Path:
    """Path to the user's forgather config file (may or may not exist)."""
    return Path(forgather_config_dir()) / "config.yaml"


def load_user_config() -> Dict[str, Any]:
    """Load the user's ``config.yaml``, returning {} if absent or unreadable."""
    path = user_config_path()
    if not path.is_file():
        return {}
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return {}
    return data or {}


def eval_search_paths(forgather_dir: str) -> list[str]:
    """Resolve the list of directories to scan for eval-config projects.

    Default: ``{forgather_dir}/examples/evaluation``. Users can extend or
    replace the default via ``eval.search_paths`` and
    ``eval.replace_default`` in ``<forgather_config_dir>/config.yaml``.
    """
    cfg = load_user_config().get("eval", {}) or {}
    extra = cfg.get("search_paths") or []
    if isinstance(extra, str):
        extra = [extra]
    extra = [os.path.expanduser(p) for p in extra]

    default = os.path.join(forgather_dir, "examples", "evaluation")
    if cfg.get("replace_default"):
        paths = list(extra)
    else:
        paths = [default] + list(extra)

    # De-dup while preserving order.
    seen = set()
    out = []
    for p in paths:
        p = os.path.abspath(p)
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out
