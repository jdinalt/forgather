"""
Enumerate the HuggingFace datasets cache.

Walks ``~/.cache/huggingface/datasets/`` (overridable via the
``HF_DATASETS_CACHE`` env var) and reports what's available without
hitting the network. Used by ``GET /v1/cache/hf`` and the matching
``forgather dataset-server cache`` CLI verb.

Cache directory layout (HF datasets v3.x):

    {cache_root}/
        {namespace}___{repo_name}/
            {config_name}/
                {version}/
                    {hash}/
                        dataset_info.json
                        *.arrow
                        ...

Single-config datasets and ``Dataset.save_to_disk`` outputs use a
slightly different layout; the walker tolerates missing levels and
just reports what it finds.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def hf_cache_root() -> Path:
    """Resolve the active HF datasets cache directory.

    Checks ``HF_DATASETS_CACHE``, then ``HF_HOME/datasets``, then
    ``~/.cache/huggingface/datasets`` — same precedence as the
    ``datasets`` library uses internally.
    """
    explicit = os.environ.get("HF_DATASETS_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "datasets"
    return Path.home() / ".cache" / "huggingface" / "datasets"


def _repo_id_from_dir(name: str) -> Optional[str]:
    """``allenai___c4`` -> ``allenai/c4``.

    Returns None for directories that don't match the expected pattern
    (e.g. lock files, ``downloads/``). HF replaces ``/`` with ``___``
    in repo ids when materializing them as filesystem paths.
    """
    if "___" not in name:
        return None
    if name.endswith(".lock"):
        return None
    parts = name.split("___", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return f"{parts[0]}/{parts[1]}"


def _read_info(info_path: Path) -> Dict[str, Any]:
    try:
        with open(info_path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("could not read %s: %s", info_path, exc)
        return {}


def _dir_size_bytes(path: Path) -> int:
    """Total size of regular files under ``path`` (best-effort)."""
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _scan_dataset_dir(repo_dir: Path, repo_id: str) -> Dict[str, Any]:
    """Walk a single ``namespace___repo`` directory and aggregate
    config/version/hash info."""
    configs: List[Dict[str, Any]] = []
    total_size = 0

    # Two layout possibilities:
    # (a) repo_dir/<config>/<version>/<hash>/dataset_info.json (HF Hub)
    # (b) repo_dir/dataset_info.json (single-flat datasets)
    flat_info = repo_dir / "dataset_info.json"
    if flat_info.exists():
        info = _read_info(flat_info)
        size = _dir_size_bytes(repo_dir)
        total_size += size
        configs.append(
            {
                "config": info.get("config_name") or "default",
                "version": (
                    info.get("version", {}).get("version_str")
                    if isinstance(info.get("version"), dict)
                    else info.get("version")
                ),
                "splits": _extract_splits(info),
                "size_bytes": size,
            }
        )
    else:
        for config_entry in sorted(repo_dir.iterdir()) if repo_dir.exists() else []:
            if not config_entry.is_dir():
                continue
            config_name = config_entry.name
            for version_entry in sorted(config_entry.iterdir()):
                if not version_entry.is_dir():
                    continue
                for hash_entry in sorted(version_entry.iterdir()):
                    if not hash_entry.is_dir():
                        continue
                    info = _read_info(hash_entry / "dataset_info.json")
                    size = _dir_size_bytes(hash_entry)
                    total_size += size
                    configs.append(
                        {
                            "config": config_name,
                            "version": version_entry.name,
                            "splits": _extract_splits(info),
                            "size_bytes": size,
                        }
                    )

    return {
        "repo": repo_id,
        "configs": configs,
        "size_bytes": total_size,
    }


def _extract_splits(info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull a compact splits list out of a dataset_info.json blob."""
    splits = info.get("splits") or {}
    out: List[Dict[str, Any]] = []
    if isinstance(splits, dict):
        for name, entry in splits.items():
            if not isinstance(entry, dict):
                out.append({"name": name})
                continue
            out.append(
                {
                    "name": entry.get("name", name),
                    "num_examples": entry.get("num_examples"),
                    "num_bytes": entry.get("num_bytes"),
                }
            )
    return out


def inspect_local_path(path: Path) -> Dict[str, Any]:
    """Scan a registered ``local/`` dataset path and return a summary.

    Recognized layouts:

    - **DatasetDict** (the ``save_to_disk()`` default for multi-split
      datasets): a ``dataset_dict.json`` listing splits at the root,
      with a ``<split>/dataset_info.json`` under each. We read the
      first split's info file to get the full ``splits`` dict (it
      contains entries for *every* split) plus the dataset-level
      ``features`` / ``config_name`` / ``dataset_name``.
    - **Flat Dataset**: a single ``dataset_info.json`` at the root.
      Same as the HF-cache flat layout — reuse ``_extract_splits``.
    - **Unknown**: anything else (raw parquet, jsonl, an arrow file).
      The caller can still load it on demand, but split metadata isn't
      available until ``POST /v1/load`` reports it.

    Returns a JSON-friendly dict with at least ``layout``, ``splits``,
    and ``size_bytes``. Missing fields (no features etc.) are simply
    absent — the webui's tree degrades gracefully.
    """
    out: Dict[str, Any] = {"path": str(path)}

    if not path.exists() or not path.is_dir():
        out["layout"] = "missing"
        out["splits"] = []
        out["size_bytes"] = 0
        return out

    dict_json = path / "dataset_dict.json"
    flat_info = path / "dataset_info.json"

    if dict_json.is_file():
        try:
            ddict = json.loads(dict_json.read_text())
            split_names = list(ddict.get("splits") or [])
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("could not read %s: %s", dict_json, exc)
            split_names = []

        splits_out: List[Dict[str, Any]] = []
        features_seen: Optional[List[str]] = None
        config_name: Optional[str] = None
        dataset_name: Optional[str] = None
        for split_name in split_names:
            split_info = _read_info(path / split_name / "dataset_info.json")
            # Per-split dataset_info.json carries a ``splits`` mapping
            # for the WHOLE dataset; grab our own entry's metadata.
            entry = (split_info.get("splits") or {}).get(split_name, {})
            if isinstance(entry, dict):
                splits_out.append(
                    {
                        "name": split_name,
                        "num_examples": entry.get("num_examples"),
                        "num_bytes": entry.get("num_bytes"),
                    }
                )
            else:
                splits_out.append({"name": split_name})
            if features_seen is None and isinstance(split_info.get("features"), dict):
                features_seen = list(split_info["features"].keys())
            if config_name is None and split_info.get("config_name"):
                config_name = str(split_info["config_name"])
            if dataset_name is None and split_info.get("dataset_name"):
                dataset_name = str(split_info["dataset_name"])

        out["layout"] = "dataset_dict"
        out["splits"] = splits_out
        if features_seen is not None:
            out["features"] = features_seen
        if config_name is not None:
            out["config_name"] = config_name
        if dataset_name is not None:
            out["dataset_name"] = dataset_name
    elif flat_info.is_file():
        info = _read_info(flat_info)
        out["layout"] = "dataset"
        out["splits"] = _extract_splits(info)
        feats = info.get("features")
        if isinstance(feats, dict):
            out["features"] = list(feats.keys())
        if info.get("config_name"):
            out["config_name"] = str(info["config_name"])
        if info.get("dataset_name"):
            out["dataset_name"] = str(info["dataset_name"])
    else:
        out["layout"] = "unknown"
        out["splits"] = []

    out["size_bytes"] = _dir_size_bytes(path)
    return out


def list_hf_cache(
    cache_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """List the datasets present in the HF cache.

    Returns a dict with ``cache_root`` (the path scanned) and
    ``datasets`` (a list of repo entries — each with configs/splits/
    sizes). If the cache root doesn't exist, returns an empty list
    rather than erroring — operators should be able to call this
    against a fresh node.
    """
    root = cache_root or hf_cache_root()
    out: Dict[str, Any] = {
        "cache_root": str(root),
        "datasets": [],
    }
    if not root.exists() or not root.is_dir():
        return out

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        repo_id = _repo_id_from_dir(entry.name)
        if repo_id is None:
            continue
        out["datasets"].append(_scan_dataset_dir(entry, repo_id))

    return out
