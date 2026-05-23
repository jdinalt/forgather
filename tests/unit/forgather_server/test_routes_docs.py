"""Tests for the built-variant fallback in routes/docs.py."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest
from forgather_server.routes import docs as docs_routes
from forgather_server.routes.docs import _maybe_built_variant, docs_root


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A repo-shaped tmp tree with docs/ and a fake repo_root indicator."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "api").mkdir()
    (tmp_path / "docs" / "api" / "page.md").write_text("# page\n::: foo.bar\n")
    return tmp_path


def _with_repo_root(repo_root: Path):
    """Patch forgather_repo_root() to point at ``repo_root``."""
    return patch(
        "forgather_server.routes.docs.sr.forgather_repo_root",
        return_value=str(repo_root),
    )


class TestMaybeBuiltVariant:
    def test_returns_none_when_no_cache(self, fake_repo):
        source = fake_repo / "docs" / "api" / "page.md"
        with _with_repo_root(fake_repo):
            assert _maybe_built_variant(source) is None

    def test_returns_built_when_present_and_fresh(self, fake_repo):
        source = fake_repo / "docs" / "api" / "page.md"
        built = fake_repo / "docs" / ".built" / "api" / "page.md"
        built.parent.mkdir(parents=True)
        built.write_text("# expanded\n")
        # Make built strictly newer than source.
        future = source.stat().st_mtime + 10
        import os

        os.utime(built, (future, future))

        with _with_repo_root(fake_repo):
            result = _maybe_built_variant(source)
        assert result == built

    def test_falls_back_when_source_is_newer(self, fake_repo):
        source = fake_repo / "docs" / "api" / "page.md"
        built = fake_repo / "docs" / ".built" / "api" / "page.md"
        built.parent.mkdir(parents=True)
        built.write_text("# expanded\n")
        # Touch source to make it strictly newer.
        time.sleep(0.02)
        source.touch()
        # And bump again to be safe across coarse filesystem timestamps.
        future = built.stat().st_mtime + 10
        import os

        os.utime(source, (future, future))

        with _with_repo_root(fake_repo):
            assert _maybe_built_variant(source) is None

    def test_returns_none_for_path_outside_docs(self, fake_repo):
        outside = fake_repo / "README.md"
        outside.write_text("hi")
        with _with_repo_root(fake_repo):
            assert _maybe_built_variant(outside) is None

    def test_does_not_recurse_into_built(self, fake_repo):
        # Asking for a path already inside .built/ returns None — guards
        # against infinite-loop scenarios if a frontend ever asks for
        # the built path directly.
        built = fake_repo / "docs" / ".built" / "api" / "page.md"
        built.parent.mkdir(parents=True)
        built.write_text("# expanded\n")
        with _with_repo_root(fake_repo):
            assert _maybe_built_variant(built) is None


@pytest.fixture
def reset_landing_override():
    """Ensure DOCS_LANDING_OVERRIDE doesn't leak between tests."""
    original = docs_routes.DOCS_LANDING_OVERRIDE
    yield
    docs_routes.DOCS_LANDING_OVERRIDE = original


class TestDocsRoot:
    def test_no_override_prefers_docs_readme(self, fake_repo, reset_landing_override):
        (fake_repo / "docs" / "README.md").write_text("# index\n")
        (fake_repo / "README.md").write_text("# root\n")
        docs_routes.DOCS_LANDING_OVERRIDE = None
        with _with_repo_root(fake_repo):
            resp = docs_root()
        assert resp.path == str(fake_repo / "docs" / "README.md")

    def test_no_override_falls_back_to_root_readme(
        self, fake_repo, reset_landing_override
    ):
        (fake_repo / "README.md").write_text("# root\n")
        docs_routes.DOCS_LANDING_OVERRIDE = None
        with _with_repo_root(fake_repo):
            resp = docs_root()
        assert resp.path == str(fake_repo / "README.md")

    def test_no_override_returns_null_when_nothing_exists(
        self, fake_repo, reset_landing_override
    ):
        docs_routes.DOCS_LANDING_OVERRIDE = None
        with _with_repo_root(fake_repo):
            resp = docs_root()
        assert resp.path is None

    def test_absolute_override_wins(self, fake_repo, reset_landing_override):
        target = fake_repo / "docs" / "guides" / "intro.md"
        target.parent.mkdir(parents=True)
        target.write_text("# intro\n")
        # docs/README.md exists too — the override should still win.
        (fake_repo / "docs" / "README.md").write_text("# index\n")
        docs_routes.DOCS_LANDING_OVERRIDE = str(target)
        with _with_repo_root(fake_repo):
            resp = docs_root()
        assert resp.path == str(target)

    def test_relative_override_resolves_against_repo(
        self, fake_repo, reset_landing_override
    ):
        target = fake_repo / "docs" / "guides" / "intro.md"
        target.parent.mkdir(parents=True)
        target.write_text("# intro\n")
        docs_routes.DOCS_LANDING_OVERRIDE = "docs/guides/intro.md"
        with _with_repo_root(fake_repo):
            resp = docs_root()
        assert resp.path == str(target.resolve())

    def test_missing_override_falls_back_to_default(
        self, fake_repo, reset_landing_override
    ):
        (fake_repo / "docs" / "README.md").write_text("# index\n")
        docs_routes.DOCS_LANDING_OVERRIDE = "docs/nowhere.md"
        with _with_repo_root(fake_repo):
            resp = docs_root()
        assert resp.path == str(fake_repo / "docs" / "README.md")
