"""`forgather docs search` CLI — the shared ranker + fallback diagnostics.

Hermetic: a tiny fake docs/ repo, with the shared corpus walk pointed at it via
``search_roots`` (the same helper docs_search and docs_vector both read). The
vector path is exercised through its *fallback* (no index in the fake repo), so
these tests need neither a prebuilt index nor sentence-transformers.
"""

from __future__ import annotations

import json

import pytest

from forgather.cli.docs import docs_cmd
from forgather.cli.docs_args import create_docs_parser


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "docs" / "guides").mkdir(parents=True)
    (tmp_path / "docs" / "guides" / "intro.md").write_text(
        "Getting started with frobnicate basics.\n"
    )
    (tmp_path / "CLAUDE.md").write_text("Agent guidance: frobnicate carefully.\n")
    # The CLI resolves the repo root for sys.path / the import guard via the
    # forgather package (the real checkout); the corpus it actually searches
    # comes from search_roots, which we point at the fake repo here.
    import forgather_server.docs_search as ds

    monkeypatch.setattr(ds.search_roots, "forgather_repo_root", lambda: str(tmp_path))
    return tmp_path


def _run(argv):
    return docs_cmd(create_docs_parser(None).parse_args(argv))


def test_keyword_search_text_output(repo, capsys):
    rc = _run(["search", "--mode", "keyword", "frobnicate"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "guides/intro.md" in out
    assert "CLAUDE.md" in out  # agent docs included by default (like search_docs)


def test_no_agent_docs_excludes_claude(repo, capsys):
    rc = _run(["search", "--no-agent-docs", "frobnicate"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "guides/intro.md" in out
    assert "CLAUDE.md" not in out


def test_json_output_carries_diagnostics(repo, capsys):
    rc = _run(["search", "--json", "frobnicate"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "keyword"
    diag = payload["diagnostics"]
    assert diag["requested_mode"] == "keyword"
    assert diag["ran_mode"] == "keyword"
    assert diag["hit_count"] >= 1
    assert diag["fell_back_to_keyword"] is False


def test_vector_without_index_falls_back(repo, capsys):
    # No index in the fake repo: vector transparently falls back to keyword and
    # the diagnostics record exactly that (no embedder/model needed).
    rc = _run(["search", "--mode", "vector", "--json", "frobnicate"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "keyword"
    diag = payload["diagnostics"]
    assert diag["requested_mode"] == "vector"
    assert diag["fell_back_to_keyword"] is True
    assert diag["vector_index_available"] is False


def test_vector_fallback_warns_on_stderr(repo, capsys):
    rc = _run(["search", "--mode", "vector", "frobnicate"])
    err = capsys.readouterr().err
    assert rc == 0
    # The text path explains the fallback (and points at the index builder).
    assert "ran keyword" in err and "forgather docs index" in err


def test_verbose_reports_actual_searched_root(repo, capsys):
    # Verbose diagnostics must name the root the backend actually searched
    # (search_roots), not a separately-derived one — so the line can't lie.
    rc = _run(["search", "--mode", "vector", "-v", "frobnicate"])
    err = capsys.readouterr().err
    assert rc == 0
    assert str(repo) in err
    assert "vector index: absent" in err
    assert "requested=vector ran=keyword" in err


def test_empty_query_exits_nonzero(repo, capsys):
    rc = _run(["search", "   "])
    assert rc == 2
    assert "empty" in capsys.readouterr().err
