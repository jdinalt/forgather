"""Stable short hash for dataset identity.

The DiLoCo work-unit dispatch design
(``docs/design/diloco-work-unit-dispatch.md``) keys per-dataset work
queues on ``(dataset_id, shuffle_seed)``, where ``dataset_id`` is a
deterministic hash of the dataset's load identity — the same
``{path, name, split, data_files, revision}`` tuple that
``fast_load_iterable_dataset`` and ``_remote_load_iterable_dataset``
already accept.

Two workers loading the same dataset must compute the same id
independently (no server round-trip), so the hash must:

- Normalize away argument-order / None-vs-missing differences.
- Sort list-typed fields (``data_files``) for order-independence.
- Strip whitespace from string fields where leading/trailing
  whitespace would be a typo, not a meaningful identifier.

A 16-hex-character (64-bit) prefix of sha256 is short enough to log
and long enough for collision-free use at the scale we care about
(operators run on the order of dozens of distinct datasets per server,
not millions).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, List, Optional, Union


def _normalize_data_files(
    value: Optional[Union[str, List[str], dict]],
) -> Any:
    """Sort list values for order-independent hashing.

    ``data_files`` in the HuggingFace ``datasets`` API can be:
      - None (no specific files)
      - a single string ("foo.json")
      - a list of strings (sorted for canonical form)
      - a dict mapping split → file(s)
    """
    if value is None:
        return None
    if isinstance(value, list):
        return sorted(str(x) for x in value)
    if isinstance(value, dict):
        # Per-split mapping: each value normalized recursively, keys
        # stay as-is (split names are themselves identifiers).
        return {k: _normalize_data_files(v) for k, v in sorted(value.items())}
    return str(value)


def _normalize_str(value: Optional[str]) -> Optional[str]:
    """Strip whitespace; treat empty-after-strip as None."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def compute_dataset_id(
    path: str,
    name: Optional[str] = None,
    split: Optional[str] = None,
    data_files: Optional[Union[str, List[str], dict]] = None,
    revision: Optional[str] = None,
) -> str:
    """Return the canonical 16-hex-character dataset_id.

    ``path`` is the only required field — every other input is allowed
    to be None (mirroring the optional fields on
    ``fast_load_iterable_dataset``). Equal inputs (after normalization)
    produce equal ids; ordering of ``data_files`` list entries doesn't
    matter.

    The returned id is suitable for use as a queue key on the DiLoCo
    server and as a stable identifier in log messages.
    """
    if not path or not str(path).strip():
        raise ValueError("path is required and cannot be empty")
    canonical = {
        "path": str(path).strip(),
        "name": _normalize_str(name),
        "split": _normalize_str(split),
        "data_files": _normalize_data_files(data_files),
        "revision": _normalize_str(revision),
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
