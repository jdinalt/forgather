"""Docs vector index: chunking + build/load/query (with a fake embedder)."""

from __future__ import annotations

import numpy as np
import pytest

from forgather import docs_index


class FakeEmbedder:
    """Deterministic bag-of-vocab embedder — no model download."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.model_name = "fake"

    @property
    def dim(self):
        return len(self.vocab)

    def embed(self, texts):
        out = []
        for t in texts:
            low = t.lower()
            v = np.array([1.0 if w in low else 0.0 for w in self.vocab], dtype="float32")
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            out.append(v)
        return np.asarray(out, dtype="float32")


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "docs" / "api").mkdir(parents=True)
    (tmp_path / "docs" / "guides").mkdir(parents=True)
    (tmp_path / "docs" / "api" / "widget.md").write_text(
        "# Widget\n\nThe Widget frobnicates the flux capacitor.\n\n"
        "## Details\n\nMore on the capacitor here.\n"
    )
    (tmp_path / "docs" / "guides" / "intro.md").write_text(
        "# Intro\n\nGetting started with basics.\n"
    )
    return tmp_path


VOCAB = ["frobnicates", "capacitor", "basics", "widget", "intro", "details"]


# ---- chunking --------------------------------------------------------------


def test_chunk_markdown_heading_breadcrumb():
    text = "# Top\n\nintro body\n\n## Sub\n\nsub body\n"
    chunks = docs_index.chunk_markdown("p.md", text)
    headings = {c.heading: c.text for c in chunks}
    assert "Top" in headings and "intro body" in headings["Top"]
    # Nested heading carries the breadcrumb.
    assert "Top > Sub" in headings and "sub body" in headings["Top > Sub"]


def test_chunk_markdown_ignores_headings_in_code_fences():
    text = (
        "# Real Heading\n\nintro\n\n"
        "```python\n# this is a comment, not a heading\ndef foo():\n    pass\n```\n\n"
        "more body under Real Heading\n"
    )
    chunks = docs_index.chunk_markdown("p.md", text)
    headings = {c.heading for c in chunks}
    # The comment must NOT have become a heading/breadcrumb.
    assert headings == {"Real Heading"}
    assert not any("comment" in h for h in headings)
    # The fenced code stays intact within the section's chunk(s).
    joined = "\n".join(c.text for c in chunks)
    assert "def foo()" in joined and "# this is a comment" in joined


def test_chunk_markdown_splits_long_sections():
    body = "\n\n".join(f"paragraph number {i} " + "x" * 200 for i in range(10))
    chunks = docs_index.chunk_markdown("p.md", f"# H\n\n{body}\n", max_chars=400)
    assert len(chunks) > 1
    assert all(c.heading == "H" for c in chunks)
    assert all(c.text.strip() for c in chunks)  # no empty chunks


# ---- build / load / query --------------------------------------------------


def test_build_and_load_roundtrip(repo):
    emb = FakeEmbedder(VOCAB)
    report = docs_index.build_index(repo, embedder=emb)
    assert report.chunks >= 3 and report.pages == 2
    out = docs_index.index_dir(repo)
    assert (out / "meta.json").is_file() and (out / "vectors.npy").is_file()

    loaded = docs_index.load_index(repo)
    assert loaded is not None
    assert loaded.vectors.shape == (report.chunks, len(VOCAB))
    assert len(loaded.chunks) == report.chunks


def test_query_returns_relevant_chunk(repo):
    emb = FakeEmbedder(VOCAB)
    docs_index.build_index(repo, embedder=emb)
    loaded = docs_index.load_index(repo)
    qv = emb.embed(["capacitor"])[0]
    top = docs_index.query(loaded, qv, top_k=3)
    assert top  # non-empty
    best_idx, best_score = top[0]
    assert best_score > 0
    # The top hit comes from the widget page (which mentions the capacitor).
    assert loaded.chunks[best_idx]["rel"] == "api/widget.md"


def test_is_stale_detects_changes(repo):
    emb = FakeEmbedder(VOCAB)
    docs_index.build_index(repo, model_name="fake", embedder=emb)
    assert docs_index.is_stale(repo, model_name="fake") is False
    # A different model id invalidates the fingerprint.
    assert docs_index.is_stale(repo, model_name="other") is True
    # Editing a page invalidates it too.
    import os

    f = repo / "docs" / "guides" / "intro.md"
    st = f.stat()
    os.utime(f, (st.st_mtime + 100, st.st_mtime + 100))
    assert docs_index.is_stale(repo, model_name="fake") is True


def test_load_missing_index_is_none(tmp_path):
    assert docs_index.load_index(tmp_path) is None
