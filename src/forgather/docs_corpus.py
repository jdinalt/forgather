"""Enumerate the docs corpus, with the rendered ``.built`` overlay.

Shared by the webui keyword/vector search (``tools/forgather_server``) and the
``forgather docs index`` builder so both see exactly the same set of pages and
the same per-page content. Takes ``repo_root`` explicitly rather than importing
a server module, so the CLI (under ``src/forgather``) can use it too.

``forgather docs build`` renders ``:::`` mkdocstrings directives into a *sparse*
``docs/.built/`` tree (only pages that had directives). So we walk the source
``docs/`` tree and, per page, read the rendered ``.built/<rel>`` version when it
exists and is not older than source (matching the Docs viewer's staleness
check), falling back to source otherwise — rather than swapping the whole root,
which would miss every directive-free page.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

DOC_SUFFIXES = (".md", ".markdown")
BUILT_SUBDIR = ".built"


def built_overlay(rel: Path, docs_dir: Path, built_dir: Path) -> Path:
    """Rendered ``.built/<rel>`` page when present and not stale, else source."""
    candidate = built_dir / rel
    source = docs_dir / rel
    if candidate.is_file():
        try:
            if candidate.stat().st_mtime >= source.stat().st_mtime:
                return candidate
        except OSError:
            pass
    return source


def iter_doc_files(
    repo_root: Path, *, include_agent_docs: bool
) -> List[Tuple[Path, str]]:
    """Yield ``(file_to_read, rel_label)`` for each doc page.

    ``rel_label`` is the path relative to ``docs/`` (what the Docs viewer opens),
    or the repo-relative path for CLAUDE.md / CLAUDE.d entries. ``file_to_read``
    is the rendered overlay when usable. ``include_agent_docs=False`` excludes
    CLAUDE.md / CLAUDE.d (user-facing corpus).
    """
    repo_root = Path(repo_root)
    docs_dir = repo_root / "docs"
    built_dir = docs_dir / BUILT_SUBDIR
    out: List[Tuple[Path, str]] = []

    if docs_dir.is_dir():
        for p in sorted(docs_dir.rglob("*")):
            # Skip the overlay tree itself; we reach it via built_overlay.
            if built_dir == p or built_dir in p.parents:
                continue
            if p.is_file() and p.suffix.lower() in DOC_SUFFIXES:
                rel = p.relative_to(docs_dir)
                out.append((built_overlay(rel, docs_dir, built_dir), str(rel)))

    if include_agent_docs:
        for extra in (repo_root / "CLAUDE.md", repo_root / "CLAUDE.d"):
            if extra.is_file() and extra.suffix.lower() in DOC_SUFFIXES:
                out.append((extra, extra.name))
            elif extra.is_dir():
                for p in sorted(extra.rglob("*")):
                    if p.is_file() and p.suffix.lower() in DOC_SUFFIXES:
                        out.append((p, str(p.relative_to(repo_root))))
    return out
