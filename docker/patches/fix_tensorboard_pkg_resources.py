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
    * Version-gated. If the installed tensorboard version is newer
      than the highest known-broken release (`KNOWN_BROKEN_CEILING`),
      the script exits non-zero so the Docker build fails — that's
      the operator's signal to drop this patch from the Dockerfile.
    * Idempotent on the broken range. Already-patched files within
      a known-broken version are silently skipped (re-running the
      build is fine).
    * Loud on drift. If the expected pre-patch text is missing on
      a known-broken version (e.g. an unrelated change moved the
      relevant code), the script raises `SystemExit` with a clear
      message so the Docker build fails.
    * Scope-limited. Touches only `default.py` and
      `data/server_ingester.py` inside the tensorboard package.

Remove this script once Forgather pins a tensorboard version
that contains the upstream fix.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

# Highest tensorboard release known to still need this patch. Any
# version > this should already contain the upstream fix; if such a
# version is installed, exit non-zero to force the operator to remove
# the patch from the Dockerfile rather than letting the build silently
# no-op.
KNOWN_BROKEN_CEILING = "2.20.0"

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


def _parse_version(v: str) -> tuple:
    """Parse a version string into a tuple for comparison.

    Prefers `packaging.version.Version` if available; falls back to a
    naive int-tuple split on '.' so this script works in stripped-down
    environments (the dev image installs `packaging`, but the patch is
    invoked early enough in the build that we don't want to depend on
    its exact transitive deps).
    """
    try:
        from packaging.version import Version

        return ("packaging", Version(v))
    except Exception:
        parts: list[int] = []
        for chunk in v.split("."):
            digits = ""
            for ch in chunk:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if not digits:
                break
            parts.append(int(digits))
        return ("naive", tuple(parts))


def check_version_in_broken_range() -> str:
    """Return the installed tensorboard version, exiting non-zero if
    it's newer than the known-broken ceiling.
    """
    try:
        installed = importlib.metadata.version("tensorboard")
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(
            "tensorboard is not installed; cannot apply pkg_resources patch"
        )
    installed_key = _parse_version(installed)
    ceiling_key = _parse_version(KNOWN_BROKEN_CEILING)
    if installed_key[0] != ceiling_key[0]:
        # Mismatched parser flavor — shouldn't happen, but bail safe.
        raise SystemExit(
            f"version-parse flavor mismatch ({installed_key[0]} vs "
            f"{ceiling_key[0]}); refusing to compare"
        )
    if installed_key[1] > ceiling_key[1]:
        raise SystemExit(
            f"tensorboard {installed} > known-broken ceiling "
            f"{KNOWN_BROKEN_CEILING}: upstream tensorboard now contains "
            f"the pkg_resources fix; this patch is no longer needed and "
            f"should be removed from the Dockerfile (drop the matching "
            f"RUN step and delete docker/patches/"
            f"fix_tensorboard_pkg_resources.py)."
        )
    return installed


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
    installed = check_version_in_broken_range()
    print(
        f"[fix_tb_pkg_resources] tensorboard {installed} (<= "
        f"{KNOWN_BROKEN_CEILING}); applying patch",
        file=sys.stderr,
    )
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
