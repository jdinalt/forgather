"""Doc search over the docs corpus — keyword now, vector/hybrid when enabled.

Used by both the agent's ``search_docs`` tool and the webui Docs-view search
endpoint, so they rank identically. The corpus walk (with the rendered
``.built`` overlay) is shared with the ``forgather docs index`` builder via
``forgather.docs_corpus``.

Modes:
- ``keyword`` (default): substring term-frequency scoring.
- ``vector`` / ``hybrid``: require a prebuilt index (``forgather docs index``)
  and sentence-transformers; fall back to keyword when either is missing.

The hit contract — ``{path, rel, score, excerpt}`` — is stable across modes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from forgather.docs_corpus import iter_doc_files

from . import search_roots

log = logging.getLogger("forgather_server.docs_search")

DEFAULT_MAX_HITS = 8
DEFAULT_EXCERPT_RADIUS = 240
# Skip pathologically large docs so a single huge file can't blow up a search
# request (matches the /docs/file read cap).
_MAX_DOC_BYTES = 25 * 1024 * 1024


def _repo_root() -> Path:
    return Path(search_roots.forgather_repo_root())


def keyword_search(
    query: str,
    *,
    include_agent_docs: bool = True,
    max_hits: int = DEFAULT_MAX_HITS,
    excerpt_radius: int = DEFAULT_EXCERPT_RADIUS,
) -> Dict[str, Any]:
    """Rank doc pages by summed term frequency; return top excerpts.

    Returns ``{"query", "mode": "keyword", "hits": [{path, rel, score, excerpt}]}``.
    Raises ``ValueError`` on an empty query.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query is empty")
    terms = [t.lower() for t in query.split() if t]

    scored: List[Dict[str, Any]] = []
    for path, rel in iter_doc_files(_repo_root(), include_agent_docs=include_agent_docs):
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
    return {"query": query, "mode": "keyword", "hits": scored[:max_hits]}


def search(
    query: str,
    *,
    include_agent_docs: bool = True,
    max_hits: int = DEFAULT_MAX_HITS,
    excerpt_radius: int = DEFAULT_EXCERPT_RADIUS,
    mode: str = "keyword",
) -> Dict[str, Any]:
    """Search the docs corpus in the requested ``mode``.

    ``mode`` is ``keyword`` (default), ``vector``, or ``hybrid``. Vector/hybrid
    require a prebuilt index + sentence-transformers; when unavailable they fall
    back to keyword (the returned ``mode`` reflects what actually ran). The
    result always carries ``{query, mode, hits}``.
    """
    if mode in ("vector", "hybrid"):
        # Implemented in the vector-search integration; until the index +
        # embedder are wired, transparently fall back to keyword.
        from . import docs_vector  # local import keeps ST/index deps lazy

        result = docs_vector.search(
            query,
            include_agent_docs=include_agent_docs,
            max_hits=max_hits,
            excerpt_radius=excerpt_radius,
            hybrid=(mode == "hybrid"),
        )
        if result is not None:
            return result
        # Fall through to keyword on any unavailability.
    return keyword_search(
        query,
        include_agent_docs=include_agent_docs,
        max_hits=max_hits,
        excerpt_radius=excerpt_radius,
    )
