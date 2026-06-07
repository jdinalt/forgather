"""Runtime vector / hybrid docs search (optional, behind a prebuilt index).

Loads the index produced by ``forgather docs index`` and answers vector or
hybrid (vector + keyword RRF) queries. Everything here is best-effort: if the
index is missing, sentence-transformers can't import, or embedding fails, the
public ``search`` returns ``None`` and ``docs_search.search`` falls back to
keyword. The index + embedder are cached and reloaded when the index file
changes on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from forgather import docs_index
from forgather.docs_corpus import iter_doc_files

from . import search_roots

log = logging.getLogger("forgather_server.docs_vector")

# RRF constant — standard default; larger = flatter rank weighting.
_RRF_K = 60
_EXCERPT_CAP = 480  # hard cap on a chunk excerpt for display

_cache: Dict[str, Any] = {"key": None, "index": None, "embedder": None}


def _repo_root() -> Path:
    return Path(search_roots.forgather_repo_root())


def reset_cache() -> None:
    _cache.update(key=None, index=None, embedder=None)


def index_available() -> bool:
    """Cheap check (no vector load) of whether a prebuilt index exists."""
    return (docs_index.index_dir(_repo_root()) / "meta.json").is_file()


def _load() -> Optional[docs_index.LoadedIndex]:
    """Return the loaded index, reloading when the on-disk meta changes."""
    repo = _repo_root()
    meta = docs_index.index_dir(repo) / "meta.json"
    try:
        mtime = meta.stat().st_mtime
    except OSError:
        reset_cache()
        return None
    # Key on (repo_root, mtime) so a repo-root change can't serve a stale index.
    key = (str(repo), mtime)
    if _cache["index"] is None or _cache["key"] != key:
        idx = docs_index.load_index(repo)
        _cache.update(key=key, index=idx, embedder=None)
    return _cache["index"]


def _embed_query(idx: docs_index.LoadedIndex, query: str):
    emb = _cache.get("embedder")
    if emb is None or emb.model_name != idx.model_name:
        emb = docs_index.Embedder(idx.model_name)
        _cache["embedder"] = emb
    return emb.embed([query])[0]


def _is_agent_doc(rel: str) -> bool:
    return rel == "CLAUDE.md" or rel.startswith("CLAUDE.d")


def _excerpt(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _EXCERPT_CAP else text[:_EXCERPT_CAP].rstrip() + "…"


def search(
    query: str,
    *,
    include_agent_docs: bool = True,
    max_hits: int = 8,
    excerpt_radius: int = 240,  # accepted for signature parity; chunks self-size
    hybrid: bool = False,
) -> Optional[Dict[str, Any]]:
    """Vector (or hybrid) search; ``None`` when the index/embedder is unavailable.

    Returns ``{query, mode, hits:[{path, rel, score, excerpt}]}`` on success.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query is empty")

    idx = _load()
    if idx is None or idx.vectors.size == 0:
        return None
    try:
        qvec = _embed_query(idx, query)
    except Exception:  # noqa: BLE001 — any embed failure → keyword fallback
        log.warning("docs vector embed failed; falling back to keyword", exc_info=True)
        return None

    # Pull a generous set of chunks, then collapse to best-per-page.
    raw = docs_index.query(idx, qvec, top_k=max_hits * 6)
    best_by_rel: Dict[str, Dict[str, Any]] = {}
    for ci, score in raw:
        chunk = idx.chunks[ci]
        rel = chunk.get("rel", "")
        if not include_agent_docs and _is_agent_doc(rel):
            continue
        cur = best_by_rel.get(rel)
        if cur is None or score > cur["score"]:
            best_by_rel[rel] = {"score": score, "text": chunk.get("text", "")}
    # Vector ranking of pages (descending score).
    vec_ranked = sorted(best_by_rel.items(), key=lambda kv: kv[1]["score"], reverse=True)
    vec_rels = [rel for rel, _ in vec_ranked]

    mode = "vector"
    kw_excerpt: Dict[str, str] = {}
    if hybrid:
        from . import docs_search  # local import avoids a circular import at load

        kw = docs_search.keyword_search(
            query,
            include_agent_docs=include_agent_docs,
            max_hits=max_hits * 4,
            excerpt_radius=excerpt_radius,
        )
        kw_rels = [h["rel"] for h in kw["hits"]]
        kw_excerpt = {h["rel"]: h["excerpt"] for h in kw["hits"]}
        fused = _rrf([vec_rels, kw_rels])
        ranked_rels = [rel for rel, _ in fused]
        score_by_rel = dict(fused)
        mode = "hybrid"
    else:
        ranked_rels = vec_rels
        score_by_rel = {rel: best_by_rel[rel]["score"] for rel in vec_rels}

    # Reconstruct openable paths (overlay-aware) for the surviving pages.
    file_map = {
        rel: path
        for path, rel in iter_doc_files(_repo_root(), include_agent_docs=include_agent_docs)
    }
    repo = _repo_root()
    hits: List[Dict[str, Any]] = []
    for rel in ranked_rels[:max_hits]:
        path = file_map.get(rel) or (repo / rel)
        excerpt = (
            _excerpt(best_by_rel[rel]["text"])
            if rel in best_by_rel
            else kw_excerpt.get(rel, "")
        )
        hits.append(
            {
                "path": str(path),
                "rel": rel,
                "score": round(float(score_by_rel.get(rel, 0.0)), 4),
                "excerpt": excerpt,
            }
        )
    return {"query": query, "mode": mode, "hits": hits}


def _rrf(rankings: List[List[str]]):
    """Reciprocal-rank fusion over rel rankings -> [(rel, score)] desc."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, rel in enumerate(ranking):
            scores[rel] = scores.get(rel, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
