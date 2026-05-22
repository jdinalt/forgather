"""Walk ``docs/``, expand directives, write the rendered tree to ``docs/.built/``.

The build is incremental: a page is rewritten only if it doesn't yet
exist in the output tree or if any of its dependencies — the source
markdown itself and the Python files the directives resolved against —
is newer than the cached output.

Pages with no directives are skipped entirely. The point of the output
tree is to host expanded copies, not to mirror every markdown file in
the repo.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .directives import ALL_DIRECTIVES, Directive
from .directives.base import DirectiveContext

log = logging.getLogger("forgather.docs_build.builder")

_OUTPUT_DIR_NAME = ".built"
_DEPS_FILENAME = ".deps.json"
_FENCE_RE = re.compile(r"^(```|~~~)")


@dataclass
class BuildReport:
    """Summary of what a build run did."""

    built: list[Path] = field(default_factory=list)
    skipped_no_directives: list[Path] = field(default_factory=list)
    skipped_up_to_date: list[Path] = field(default_factory=list)
    errors: list[tuple[Path, str]] = field(default_factory=list)
    output_dir: Optional[Path] = None
    cleaned: bool = False

    @property
    def total_considered(self) -> int:
        return (
            len(self.built)
            + len(self.skipped_no_directives)
            + len(self.skipped_up_to_date)
            + len(self.errors)
        )


def build(
    docs_dir: Path,
    *,
    repo_root: Path,
    clean: bool = False,
    subset: Optional[Path] = None,
    check_only: bool = False,
) -> BuildReport:
    """Build the docs cache.

    Parameters
    ----------
    docs_dir
        The ``docs/`` directory to read sources from (and write
        ``.built/`` into).
    repo_root
        Repository root; used to set griffe's search path
        (``<repo_root>/src``) so it matches mkdocs.yml.
    clean
        Remove the output tree before building.
    subset
        Restrict the build to files under this directory (relative or
        absolute). Useful for iterating on a single page.
    check_only
        Don't write any output; just report which files would be
        rebuilt. Exit-non-zero behaviour is the CLI's responsibility.
    """
    docs_dir = docs_dir.resolve()
    repo_root = repo_root.resolve()
    out_root = docs_dir / _OUTPUT_DIR_NAME
    report = BuildReport(output_dir=out_root)

    if clean and not check_only and out_root.exists():
        shutil.rmtree(out_root)
        report.cleaned = True

    subset_resolved: Optional[Path] = subset.resolve() if subset else None

    sources = list(_walk_sources(docs_dir, out_root=out_root, subset=subset_resolved))
    if not sources:
        return report

    context = _make_context(repo_root)
    deps_map = _load_deps(out_root) if out_root.exists() else {}

    for src in sources:
        rel = src.relative_to(docs_dir)
        out = out_root / rel
        try:
            text = src.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            report.errors.append((src, f"read failed: {exc}"))
            continue

        if not _has_any_directive(text):
            report.skipped_no_directives.append(src)
            continue

        if _is_up_to_date(src, out, deps_map.get(str(rel)), repo_root):
            report.skipped_up_to_date.append(src)
            continue

        try:
            rendered, deps = _expand_text(text, context=context)
        except (
            Exception
        ) as exc:  # noqa: BLE001 — directive errors return inline; this is for programmer bugs
            log.exception("docs_build: unexpected failure expanding %s", src)
            report.errors.append((src, f"expand failed: {exc}"))
            continue

        if check_only:
            report.built.append(src)
            continue

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")

        deps_map[str(rel)] = [_relpath(p, repo_root) for p in deps]
        report.built.append(src)

    if not check_only:
        out_root.mkdir(parents=True, exist_ok=True)
        _save_deps(out_root, deps_map)

    return report


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


def _walk_sources(
    docs_dir: Path, *, out_root: Path, subset: Optional[Path]
) -> Iterable[Path]:
    base = subset if subset and subset.is_dir() else docs_dir
    if subset and subset.is_file():
        yield subset
        return
    for p in sorted(base.rglob("*.md")):
        if out_root in p.parents:
            continue
        # Skip files that are symlinks to outside docs_dir. The link-
        # rewriting story for those is owned by docs_hooks.py; trying
        # to build them here would write the built copy in a place
        # that doesn't share the symlink's relative-path context.
        try:
            real = p.resolve()
            if docs_dir not in real.parents and real != p:
                continue
        except OSError:
            continue
        yield p


# ---------------------------------------------------------------------------
# Directive scanning + expansion
# ---------------------------------------------------------------------------


def _has_any_directive(text: str) -> bool:
    in_fence = False
    fence_marker: Optional[str] = None
    for line in text.splitlines():
        m = _FENCE_RE.match(line)
        if m:
            if not in_fence:
                in_fence = True
                fence_marker = m.group(1)
            elif fence_marker and line.startswith(fence_marker):
                in_fence = False
                fence_marker = None
            continue
        if in_fence:
            continue
        for d in ALL_DIRECTIVES:
            if d.pattern.match(line):
                return True
    return False


def _expand_text(text: str, *, context: DirectiveContext) -> tuple[str, list[Path]]:
    deps: set[Path] = set()
    out_lines: list[str] = []
    in_fence = False
    fence_marker: Optional[str] = None
    for line in text.splitlines():
        m = _FENCE_RE.match(line)
        if m:
            if not in_fence:
                in_fence = True
                fence_marker = m.group(1)
            elif fence_marker and line.startswith(fence_marker):
                in_fence = False
                fence_marker = None
            out_lines.append(line)
            continue
        if in_fence:
            out_lines.append(line)
            continue
        expansion = _try_match(line, context=context, deps=deps)
        if expansion is None:
            out_lines.append(line)
        else:
            out_lines.append(expansion)
    final = "\n".join(out_lines)
    if not final.endswith("\n"):
        final += "\n"
    return final, sorted(deps)


def _try_match(
    line: str, *, context: DirectiveContext, deps: set[Path]
) -> Optional[str]:
    for d in ALL_DIRECTIVES:
        m = d.pattern.match(line)
        if not m:
            continue
        result = d.expand(m, context=context)
        for dep in result.deps:
            deps.add(dep)
        return result.markdown
    return None


# ---------------------------------------------------------------------------
# Staleness + deps sidecar
# ---------------------------------------------------------------------------


def _is_up_to_date(
    src: Path,
    out: Path,
    recorded_deps: Optional[list[str]],
    repo_root: Path,
) -> bool:
    if not out.exists():
        return False
    out_mtime = out.stat().st_mtime
    if src.stat().st_mtime > out_mtime:
        return False
    for rel in recorded_deps or []:
        dep_path = Path(rel)
        if not dep_path.is_absolute():
            dep_path = repo_root / dep_path
        try:
            if dep_path.stat().st_mtime > out_mtime:
                return False
        except OSError:
            # Dep file disappeared — treat as stale so the next build
            # regenerates and updates the deps map.
            return False
    return True


def _load_deps(out_root: Path) -> dict[str, list[str]]:
    path = out_root / _DEPS_FILENAME
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_deps(out_root: Path, deps_map: dict[str, list[str]]) -> None:
    path = out_root / _DEPS_FILENAME
    payload = json.dumps(deps_map, indent=2, sort_keys=True)
    path.write_text(payload + "\n", encoding="utf-8")


def _relpath(target: Path, base: Path) -> str:
    """Path string relative to ``base`` when possible, else absolute.

    Used to write the deps sidecar relative to the repo root so the
    file stays meaningful when the working tree is moved.
    """
    try:
        return str(target.relative_to(base))
    except ValueError:
        return str(target)


# ---------------------------------------------------------------------------
# Griffe loader (deferred to keep ``import forgather.docs_build`` cheap)
# ---------------------------------------------------------------------------


def _make_context(repo_root: Path) -> DirectiveContext:
    loader = _make_griffe_loader(repo_root)
    return DirectiveContext(repo_root=repo_root, griffe_loader=loader)


def _make_griffe_loader(repo_root: Path):
    try:
        import griffe
    except ImportError as exc:  # pragma: no cover — griffe ships with mkdocstrings
        raise RuntimeError(
            "docs_build requires griffe (install mkdocstrings[python])."
        ) from exc
    return griffe.GriffeLoader(
        search_paths=[str(repo_root / "src")],
        docstring_parser=griffe.Parser("numpy"),
    )
