"""Runtime vector / hybrid docs search + the endpoint mode param.

Uses a fake embedder + a real on-disk index (built into a temp repo) so nothing
downloads a model.
"""

from __future__ import annotations

import numpy as np
import pytest

from forgather import docs_index
from forgather_server import docs_search, docs_vector

VOCAB = ["frobnicates", "capacitor", "basics", "widget", "intro", "details"]


class FakeEmbedder:
    def __init__(self, *_a, **_k):
        self.model_name = "fake"

    @property
    def dim(self):
        return len(VOCAB)

    def embed(self, texts):
        out = []
        for t in texts:
            low = t.lower()
            v = np.array([1.0 if w in low else 0.0 for w in VOCAB], dtype="float32")
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            out.append(v)
        return np.asarray(out, dtype="float32")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "docs" / "api").mkdir(parents=True)
    (tmp_path / "docs" / "guides").mkdir(parents=True)
    (tmp_path / "docs" / "api" / "widget.md").write_text(
        "# Widget\n\nThe Widget frobnicates the flux capacitor.\n"
    )
    (tmp_path / "docs" / "guides" / "intro.md").write_text(
        "# Intro\n\nGetting started with basics.\n"
    )
    monkeypatch.setattr(
        docs_vector.search_roots, "forgather_repo_root", lambda: str(tmp_path)
    )
    # Query-time embedder is the fake one too.
    monkeypatch.setattr(docs_index, "Embedder", lambda *a, **k: FakeEmbedder())
    docs_vector.reset_cache()
    return tmp_path


def _build(repo):
    docs_index.build_index(repo, model_name="fake", embedder=FakeEmbedder())
    docs_vector.reset_cache()


def test_index_available(repo):
    assert docs_vector.index_available() is False
    _build(repo)
    assert docs_vector.index_available() is True


def test_vector_search_ranks_by_similarity(repo):
    _build(repo)
    out = docs_vector.search("capacitor", include_agent_docs=False, max_hits=5)
    assert out is not None
    assert out["mode"] == "vector"
    assert out["hits"][0]["rel"] == "api/widget.md"
    # Source path is reconstructed (overlay-aware), not the index dir.
    assert out["hits"][0]["path"].endswith("/docs/api/widget.md")


def test_hybrid_search_fuses(repo):
    _build(repo)
    out = docs_vector.search("basics", include_agent_docs=False, max_hits=5, hybrid=True)
    assert out is not None and out["mode"] == "hybrid"
    rels = [h["rel"] for h in out["hits"]]
    assert "guides/intro.md" in rels


def test_no_index_returns_none(repo):
    # No build -> unavailable -> None (caller falls back to keyword).
    assert docs_vector.search("capacitor", max_hits=5) is None


def test_docs_search_mode_falls_back_to_keyword(repo):
    # mode="vector" with no index transparently runs keyword.
    out = docs_search.search("capacitor", include_agent_docs=False, mode="vector")
    assert out["mode"] == "keyword"
    assert any(h["rel"] == "api/widget.md" for h in out["hits"])


def test_docs_search_mode_vector_when_built(repo):
    _build(repo)
    out = docs_search.search("capacitor", include_agent_docs=False, mode="vector")
    assert out["mode"] == "vector"


def test_index_reloads_on_rebuild_without_reset(repo):
    # The runtime cache is keyed on meta.json's mtime, so a rebuilt index is
    # picked up on the next query with NO reset_cache()/restart. This is why
    # `forgather docs index` needs no server-side reload endpoint.
    import os

    _build(repo)
    out1 = docs_vector.search("capacitor", include_agent_docs=False, max_hits=5)
    assert out1 is not None and "api/gizmo.md" not in [h["rel"] for h in out1["hits"]]

    # Add a new matching page and rebuild WITHOUT touching the cache.
    (repo / "docs" / "api" / "gizmo.md").write_text(
        "# Gizmo\n\nThe Gizmo also frobnicates the flux capacitor.\n"
    )
    docs_index.build_index(repo, model_name="fake", embedder=FakeEmbedder())
    # Advance the meta mtime in case the rebuild landed in the same fs tick.
    meta = docs_index.index_dir(repo) / "meta.json"
    bumped = os.path.getmtime(meta) + 5
    os.utime(meta, (bumped, bumped))

    out2 = docs_vector.search("capacitor", include_agent_docs=False, max_hits=5)
    assert "api/gizmo.md" in [h["rel"] for h in out2["hits"]]


# ---- endpoint --------------------------------------------------------------


def test_endpoint_mode_and_availability(repo):
    from forgather_server.routes import docs as docs_routes

    _build(repo)
    resp = docs_routes.docs_search_endpoint(q="capacitor", mode="vector")
    assert resp.mode == "vector"
    assert resp.vector_available is True
    assert resp.hits[0].rel == "api/widget.md"


def test_endpoint_rejects_bad_mode(repo):
    from fastapi import HTTPException

    from forgather_server.routes import docs as docs_routes

    with pytest.raises(HTTPException) as ei:
        docs_routes.docs_search_endpoint(q="x", mode="bogus")
    assert ei.value.status_code == 400
