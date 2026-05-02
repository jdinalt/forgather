"""Backport tensorflow/tensorboard@29f809f4 onto an installed
tensorboard tree.

Why: setuptools 82 (Feb 2026) removed `pkg_resources`. tensorboard
2.20.0 (the latest release as of writing) still imports
`pkg_resources` at module load, so `tensorboard --bind_all` exits
with `ModuleNotFoundError: No module named 'pkg_resources'`.

Upstream replaced `pkg_resources` with `importlib.metadata` + the
`packaging` package in commit 29f809f4 (March 2026), but no
release contains the fix yet. This script applies the two-file
diff in-place against the installed tensorboard package.

Properties:
    * Idempotent. Already-patched files are silently skipped.
    * Loud on drift. If the expected pre-patch text is missing
      (e.g. tensorboard was upgraded to a release containing the
      fix, or a different unrelated change moved the relevant
      code), the script raises `SystemExit` with a clear message
      so the Docker build fails and the patch can be removed.
    * Scope-limited. Touches only `default.py` and
      `data/server_ingester.py` inside the tensorboard package.

Remove this script once Forgather pins a tensorboard version
that contains the upstream fix.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# --------- replacement table (from upstream 29f809f4) -------------

PATCHES: dict[str, list[tuple[str, str]]] = {
    "tensorboard/default.py": [
        (
            "import pkg_resources\n",
            "from importlib import metadata\n",
        ),
        (
            "    return [\n"
            "        entry_point.resolve()\n"
            "        for entry_point in pkg_resources.iter_entry_points(\n"
            '            "tensorboard_plugins"\n'
            "        )\n"
            "    ]\n",
            "    return [\n"
            "        entry_point.load()\n"
            '        for entry_point in _iter_entry_points("tensorboard_plugins")\n'
            "    ]\n"
            "\n"
            "\n"
            "def _iter_entry_points(group):\n"
            '    """Returns entry points for a given group across Python versions."""\n'
            "    entry_points = metadata.entry_points()\n"
            "    # In newer Python versions, `metadata.entry_points()` returns an\n"
            "    # `EntryPoints` object with a `select()` method.\n"
            '    # Before "selectable" entry points existed, it would return a dictionary.\n'
            '    if hasattr(entry_points, "select"):\n'
            "        return entry_points.select(group=group)\n"
            "    return entry_points.get(group, ())\n",
        ),
    ],
    "tensorboard/data/server_ingester.py": [
        (
            "import pkg_resources\n",
            "from packaging import version as packaging_version\n",
        ),
        (
            "        self._version = (\n"
            "            pkg_resources.parse_version(version)\n"
            "            if version is not None\n"
            "            else version\n"
            "        )\n",
            "        self._version = (\n"
            "            packaging_version.parse(version) if version is not None else version\n"
            "        )\n",
        ),
        (
            "        return self._version >= pkg_resources.parse_version(required_version)\n",
            "        return self._version >= packaging_version.parse(required_version)\n",
        ),
    ],
}

# Marker tokens that, if already present in the file, indicate the
# upstream fix has landed (or this script already ran). We skip the
# file in that case rather than failing.
ALREADY_PATCHED_MARKERS: dict[str, str] = {
    "tensorboard/default.py": "from importlib import metadata\n",
    "tensorboard/data/server_ingester.py": "from packaging import version as packaging_version\n",
}


def tensorboard_root() -> Path:
    spec = importlib.util.find_spec("tensorboard")
    if spec is None or spec.origin is None:
        raise SystemExit("tensorboard is not importable from this Python environment")
    return Path(spec.origin).resolve().parent.parent


def apply(file_path: Path, edits: list[tuple[str, str]]) -> bool:
    text = file_path.read_text()
    new_text = text
    for old, new in edits:
        if old not in new_text:
            raise SystemExit(
                f"patch failed: expected text not found in {file_path}\n"
                f"--- expected ---\n{old}\n--- end ---\n"
                f"This usually means tensorboard was upgraded to a "
                f"release containing the upstream pkg_resources fix. "
                f"Drop docker/patches/fix_tensorboard_pkg_resources.py "
                f"and the matching RUN step from the Dockerfile."
            )
        new_text = new_text.replace(old, new, 1)
    if new_text == text:
        return False
    file_path.write_text(new_text)
    return True


def main() -> int:
    root = tensorboard_root()
    any_changed = False
    for rel, edits in PATCHES.items():
        path = root / rel
        if not path.exists():
            print(f"[fix_tb_pkg_resources] skip (missing): {rel}", file=sys.stderr)
            continue
        marker = ALREADY_PATCHED_MARKERS[rel]
        if marker in path.read_text():
            print(f"[fix_tb_pkg_resources] already patched: {rel}", file=sys.stderr)
            continue
        if apply(path, edits):
            print(f"[fix_tb_pkg_resources] patched: {rel}", file=sys.stderr)
            any_changed = True
    if not any_changed:
        print("[fix_tb_pkg_resources] nothing to do", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
