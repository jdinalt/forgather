"""Shared keyword search over the docs corpus.

Used by both the agent's ``search_docs`` tool and the webui Docs-view search
endpoint, so they rank identically. Deliberately simple substring scoring (no
embeddings yet) — the caller reads the excerpts and decides relevance.

Built-docs overlay: ``forgather docs build`` renders ``:::`` mkdocstrings
directives into a *sparse* ``docs/.built/`` tree (only the pages that had
directives). So we walk the source ``docs/`` tree and, per page, read the
rendered ``.built/<rel>`` version when it exists (real API text instead of an
unexpanded ``:::`` line) and fall back to source otherwise — rather than
swapping the whole root, which would miss every page without a directive.

The contract (``{path, rel, score, excerpt}`` per hit) is stable so embeddings
can replace the scorer later without touching callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import search_roots

_DOC_SUFFIXES = (".md", ".markdown")
DEFAULT_MAX_HITS = 8
DEFAULT_EXCERPT_RADIUS = 240
# Skip pathologically large docs so a single huge file can't blow up a search
# request (matches the /docs/file read cap).
_MAX_DOC_BYTES = 25 * 1024 * 1024


def _repo_root() -> Path:
    return Path(search_roots.forgather_repo_root())


def _built_overlay(rel: Path, docs_dir: Path, built_dir: Path) -> Path:
    """Prefer the rendered ``.built/<rel>`` page when present AND not stale.

    Mirrors ``routes/docs.py::_maybe_built_variant``: if the source was edited
    after the last build, search the source so a hit always matches the page the
    viewer will actually open (which makes the same stale-check).
    """
    candidate = built_dir / rel
    source = docs_dir / rel
    if candidate.is_file():
        try:
            if candidate.stat().st_mtime >= source.stat().st_mtime:
                return candidate
        except OSError:
            pass
    return source


def iter_doc_files(*, include_agent_docs: bool) -> List[Tuple[Path, str]]:
    """Yield ``(file_to_read, rel_label)`` for each doc page.

    ``rel_label`` is the path relative to ``docs/`` (what the Docs viewer opens),
    or the repo-relative path for CLAUDE.md / CLAUDE.d entries. ``file_to_read``
    is the rendered overlay when available. When ``include_agent_docs`` is False
    (user-facing Docs search), CLAUDE.md / CLAUDE.d are excluded.
    """
    repo = _repo_root()
    docs_dir = repo / "docs"
    built_dir = docs_dir / ".built"
    out: List[Tuple[Path, str]] = []

    if docs_dir.is_dir():
        for p in sorted(docs_dir.rglob("*")):
            # Skip the overlay tree itself; we reach it via _built_overlay.
            if built_dir == p or built_dir in p.parents:
                continue
            if p.is_file() and p.suffix.lower() in _DOC_SUFFIXES:
                rel = p.relative_to(docs_dir)
                out.append((_built_overlay(rel, docs_dir, built_dir), str(rel)))

    if include_agent_docs:
        for extra in (repo / "CLAUDE.md", repo / "CLAUDE.d"):
            if extra.is_file() and extra.suffix.lower() in _DOC_SUFFIXES:
                out.append((extra, extra.name))
            elif extra.is_dir():
                for p in sorted(extra.rglob("*")):
                    if p.is_file() and p.suffix.lower() in _DOC_SUFFIXES:
                        out.append((p, str(p.relative_to(repo))))
    return out


def search(
    query: str,
    *,
    include_agent_docs: bool = True,
    max_hits: int = DEFAULT_MAX_HITS,
    excerpt_radius: int = DEFAULT_EXCERPT_RADIUS,
) -> Dict[str, Any]:
    """Rank doc pages by summed term frequency; return top excerpts.

    Returns ``{"query", "hits": [{path, rel, score, excerpt}, ...]}``.
    Raises ``ValueError`` on an empty query.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query is empty")
    terms = [t.lower() for t in query.split() if t]

    scored: List[Dict[str, Any]] = []
    for path, rel in iter_doc_files(include_agent_docs=include_agent_docs):
        try:
            if path.stat().st_size > _MAX_DOC_BYTES:
                continue
            text = path.read_text(errors="replace")
        except OSError:
            continue
        low = text.lower()
        score = sum(low.count(t) for t in terms)
        if score == 0:
            continue
        # Excerpt around the first matching term.
        idx = min((low.find(t) for t in terms if low.find(t) >= 0), default=-1)
        if idx < 0:
            continue
        start = max(0, idx - excerpt_radius)
        end = min(len(text), idx + excerpt_radius)
        scored.append(
            {
                "path": str(path),
                "rel": rel,
                "score": score,
                "excerpt": text[start:end].strip(),
            }
        )

    scored.sort(key=lambda h: h["score"], reverse=True)
    return {"query": query, "hits": scored[:max_hits]}
