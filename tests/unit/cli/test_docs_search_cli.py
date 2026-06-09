"""`forgather docs search` / `forgather search` CLI.

Covers the local ranker + fallback diagnostics, the server-backed vector path
(mocked), the `forgather search` alias namespace, and the main.py subcommand-
position helper that keeps a query word like "server" from triggering another
command's arg-rewriting.

Local tests are hermetic: a tiny fake docs/ repo with the shared corpus walk
pointed at it via ``search_roots``. Vector is exercised through its *fallback*
(no index in the fake repo), so they need neither an index nor
sentence-transformers.
"""

from __future__ import annotations

import json

import pytest

from forgather.cli import server_client
from forgather.cli.docs import docs_cmd
from forgather.cli.docs_args import create_docs_parser, create_search_parser
from forgather.cli.main import _subcommand_token_index


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "docs" / "guides").mkdir(parents=True)
    (tmp_path / "docs" / "guides" / "intro.md").write_text(
        "Getting started with frobnicate basics.\n"
    )
    (tmp_path / "CLAUDE.md").write_text("Agent guidance: frobnicate carefully.\n")
    import forgather_server.docs_search as ds

    monkeypatch.setattr(ds.search_roots, "forgather_repo_root", lambda: str(tmp_path))
    return tmp_path


def _run(argv):
    """Run via the `docs` subcommand parser (`forgather docs search …`)."""
    return docs_cmd(create_docs_parser(None).parse_args(argv))


def _run_alias(argv):
    """Run via the top-level alias parser (`forgather search …`)."""
    args = create_search_parser(None).parse_args(argv)
    args.docs_action = "search"  # main.py sets this before dispatching to docs_cmd
    return docs_cmd(args)


# ---- local keyword path ----------------------------------------------------


def test_keyword_is_default_and_local(repo, capsys):
    rc = _run(["search", "frobnicate"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "guides/intro.md" in out
    assert "CLAUDE.md" in out  # agent docs included by default (like search_docs)


def test_keyword_flag_explicit(repo, capsys):
    rc = _run(["search", "-k", "frobnicate"])
    assert rc == 0
    assert "guides/intro.md" in capsys.readouterr().out


def test_alias_matches_subcommand(repo, capsys):
    rc = _run_alias(["frobnicate"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "guides/intro.md" in out


def test_no_agent_docs_excludes_claude(repo, capsys):
    rc = _run(["search", "--no-agent-docs", "frobnicate"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "guides/intro.md" in out
    assert "CLAUDE.md" not in out


def test_json_carries_source_and_diagnostics(repo, capsys):
    rc = _run(["search", "--json", "frobnicate"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "keyword"
    diag = payload["diagnostics"]
    assert diag["source"] == "local"
    assert diag["ran_mode"] == "keyword" and diag["fell_back_to_keyword"] is False
    assert diag["hit_count"] >= 1


def test_empty_query_exits_nonzero(repo, capsys):
    rc = _run(["search", "   "])
    assert rc == 2
    assert "empty" in capsys.readouterr().err


# ---- local vector path (forced with --local; falls back without an index) --


def test_local_vector_without_index_falls_back(repo, capsys):
    rc = _run(["search", "--local", "--vector", "--json", "frobnicate"])
    assert rc == 0
    diag = json.loads(capsys.readouterr().out)["diagnostics"]
    assert diag["source"] == "local"
    assert diag["requested_mode"] == "vector"
    assert diag["fell_back_to_keyword"] is True
    assert diag["vector_index_available"] is False


def test_local_verbose_reports_searched_root(repo, capsys):
    rc = _run(["search", "--local", "--vector", "--verbose", "frobnicate"])
    err = capsys.readouterr().err
    assert rc == 0
    assert str(repo) in err  # the actual searched root, not a separately-derived one
    assert "vector index: absent" in err
    assert "requested=vector ran=keyword" in err


# ---- server-backed vector path (mocked client) -----------------------------


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _install_fake_client(monkeypatch, *, payload=None, raises=None, base="https://fake:1"):
    class _FakeClient:
        def __init__(self):
            self.base = base

        @classmethod
        def from_args(cls, args):
            return cls()

        def _get(self, path, params=None):
            if raises is not None:
                raise raises
            return _FakeResp(payload)

    monkeypatch.setattr(server_client, "ServerClient", _FakeClient)


def test_vector_defaults_to_server(monkeypatch, capsys):
    _install_fake_client(
        monkeypatch,
        payload={
            "query": "tls",
            "mode": "vector",
            "vector_available": True,
            "hits": [
                {"path": "/r/docs/operations/tls.md", "rel": "operations/tls.md",
                 "score": 0.91, "excerpt": "mutual TLS setup"}
            ],
        },
    )
    rc = _run(["search", "--vector", "--json", "tls"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "vector"
    diag = payload["diagnostics"]
    assert diag["source"] == "server" and diag["server"] == "https://fake:1"
    assert diag["hit_count"] == 1


def test_server_unreachable_reports_and_hints_local(monkeypatch, capsys):
    _install_fake_client(
        monkeypatch, raises=server_client.ServerUnreachable("could not reach server")
    )
    rc = _run(["search", "--vector", "tls"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "could not reach server" in err
    assert "--local" in err  # points the user at the in-process fallback


# ---- main.py subcommand-position helper (query-word safety) -----------------


def test_subcommand_token_index_skips_query_words():
    # "server" as a docs-search query word must NOT look like the subcommand.
    assert _subcommand_token_index(["docs", "search", "inference", "server"]) == 0
    assert _subcommand_token_index(["search", "the", "server", "crashed"]) == 0
    # Real subcommand positions (after global value/boolean flags) are found.
    assert _subcommand_token_index(["-p", "/x", "search", "q"]) == 2
    assert _subcommand_token_index(["-t", "cfg.yaml", "server"]) == 2
    assert _subcommand_token_index(["--no-dyn", "ls"]) == 1
    assert _subcommand_token_index(["--project-dir=/x", "search", "q"]) == 1
    assert _subcommand_token_index(["-p", "/x"]) is None
