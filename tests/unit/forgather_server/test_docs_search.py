"""Shared doc-search backend: keyword scoring + the .built overlay."""

from __future__ import annotations

import pytest

from forgather_server import docs_search


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A tiny fake repo: docs/ source, a sparse docs/.built/ overlay, CLAUDE.*"""
    (tmp_path / "docs" / "api").mkdir(parents=True)
    (tmp_path / "docs" / "guides").mkdir(parents=True)
    (tmp_path / "docs" / ".built" / "api").mkdir(parents=True)
    # api/widget.md: source has the unexpanded directive; .built has real text.
    (tmp_path / "docs" / "api" / "widget.md").write_text("::: forgather.widget\n")
    (tmp_path / "docs" / ".built" / "api" / "widget.md").write_text(
        "# Widget\nThe Widget frobnicates the flux capacitor.\n"
    )
    # guides/intro.md: no directive, no overlay — searched from source.
    (tmp_path / "docs" / "guides" / "intro.md").write_text(
        "Getting started with frobnicate basics.\n"
    )
    (tmp_path / "CLAUDE.md").write_text("Agent guidance: frobnicate carefully.\n")
    monkeypatch.setattr(
        docs_search.search_roots, "forgather_repo_root", lambda: str(tmp_path)
    )
    return tmp_path


def test_overlay_prefers_built_page(repo):
    hits = docs_search.search("frobnicates")["hits"]
    rels = [h["rel"] for h in hits]
    assert "api/widget.md" in rels
    # The matched content came from the rendered overlay, not the ::: source.
    widget = next(h for h in hits if h["rel"] == "api/widget.md")
    assert "flux capacitor" in widget["excerpt"]
    assert ":::" not in widget["excerpt"]
    # And the file read was the .built copy.
    assert ".built" in widget["path"]


def test_source_page_without_overlay_is_searched(repo):
    rels = [h["rel"] for h in docs_search.search("basics")["hits"]]
    assert "guides/intro.md" in rels


def test_include_agent_docs_toggle(repo):
    with_agent = [h["rel"] for h in docs_search.search("frobnicate")["hits"]]
    assert "CLAUDE.md" in with_agent
    user_only = [
        h["rel"]
        for h in docs_search.search("frobnicate", include_agent_docs=False)["hits"]
    ]
    assert "CLAUDE.md" not in user_only
    # docs pages still present in the user-facing corpus.
    assert "guides/intro.md" in user_only


def test_ranking_and_excerpt(repo):
    # "frobnicate" appears in intro (1) and CLAUDE.md (1); ranking is by count.
    out = docs_search.search("frobnicate")
    assert out["hits"] == sorted(out["hits"], key=lambda h: h["score"], reverse=True)
    assert all("excerpt" in h and h["excerpt"] for h in out["hits"])


def test_empty_query_raises(repo):
    with pytest.raises(ValueError, match="empty"):
        docs_search.search("   ")


def test_built_tree_not_double_counted(repo):
    # Walking docs/ must skip the .built subtree, so api/widget.md appears once.
    rels = [h["rel"] for h in docs_search.search("frobnicate")["hits"]]
    assert rels.count("api/widget.md") <= 1


# ---- /api/docs/search endpoint ---------------------------------------------


def test_docs_search_endpoint_returns_source_paths(repo):
    from forgather_server.routes import docs as docs_routes

    resp = docs_routes.docs_search_endpoint(q="frobnicates", limit=8)
    assert resp.query == "frobnicates"
    hit = next(h for h in resp.hits if h.rel == "api/widget.md")
    # Endpoint returns the SOURCE path (viewer serves the .built variant itself).
    assert hit.path.endswith("/docs/api/widget.md")
    assert ".built" not in hit.path
    # Excerpt content came from the rendered overlay.
    assert "flux capacitor" in hit.excerpt


def test_docs_search_endpoint_excludes_agent_docs(repo):
    from forgather_server.routes import docs as docs_routes

    rels = [h.rel for h in docs_routes.docs_search_endpoint(q="frobnicate").hits]
    assert "CLAUDE.md" not in rels
    assert "guides/intro.md" in rels


def test_docs_search_endpoint_blank_query(repo):
    from forgather_server.routes import docs as docs_routes

    resp = docs_routes.docs_search_endpoint(q="   ")
    assert resp.query == "" and resp.hits == []


def test_overlay_skipped_when_source_newer_than_built(repo):
    # If the source was edited after the last build, search the source (so a hit
    # always matches the page the viewer opens, which makes the same check).
    import os

    src = repo / "docs" / "api" / "widget.md"
    built = repo / "docs" / ".built" / "api" / "widget.md"
    bt = built.stat().st_mtime
    os.utime(src, (bt + 10, bt + 10))  # source newer than built
    # "frobnicates" lives only in the built copy; with the overlay skipped it
    # is no longer matched (source has the unexpanded ::: directive).
    rels = [h["rel"] for h in docs_search.search("frobnicates")["hits"]]
    assert "api/widget.md" not in rels
