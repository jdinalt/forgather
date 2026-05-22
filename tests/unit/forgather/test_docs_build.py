"""Tests for the docs_build library.

Covers the builder's incremental behavior and the mkdocstrings
directive's expansion against a real-but-tiny package on disk.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from forgather.docs_build import build


@pytest.fixture
def stub_repo(tmp_path: Path) -> Path:
    """A miniature repo: src/pkg/mod.py + docs/api/page.md with a directive."""
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text("")
    (src / "mod.py").write_text(textwrap.dedent('''\
            """A tiny module."""


            class Widget:
                """A widget.

                Parameters
                ----------
                size : int
                    How big it is.
                """

                def __init__(self, size: int = 1) -> None:
                    self.size = size

                def grow(self, amount: int) -> int:
                    """Increase the size by ``amount`` and return the new size."""
                    self.size += amount
                    return self.size
            '''))

    docs = tmp_path / "docs" / "api"
    docs.mkdir(parents=True)
    (docs / "page.md").write_text(
        "# API\n\nIntro.\n\n::: pkg.mod.Widget\n\nMore prose.\n"
    )
    # A second page with no directives — should be skipped entirely.
    (tmp_path / "docs" / "plain.md").write_text("# plain\n\nno directives here\n")
    return tmp_path


def _docs(repo: Path) -> Path:
    return repo / "docs"


class TestBuilder:
    def test_first_build_renders_directive(self, stub_repo):
        report = build(_docs(stub_repo), repo_root=stub_repo)
        assert len(report.built) == 1
        assert report.skipped_no_directives  # plain.md

        out = stub_repo / "docs" / ".built" / "api" / "page.md"
        text = out.read_text()
        # Directive expanded — class signature and method present.
        assert "class Widget(" in text
        assert "grow" in text
        assert "::: pkg.mod.Widget" not in text
        # Surrounding prose preserved.
        assert "Intro." in text
        assert "More prose." in text

    def test_no_directives_means_no_output_file(self, stub_repo):
        build(_docs(stub_repo), repo_root=stub_repo)
        plain_out = stub_repo / "docs" / ".built" / "plain.md"
        assert not plain_out.exists()

    def test_incremental_skips_when_unchanged(self, stub_repo):
        build(_docs(stub_repo), repo_root=stub_repo)
        report2 = build(_docs(stub_repo), repo_root=stub_repo)
        assert not report2.built
        assert len(report2.skipped_up_to_date) == 1

    def test_rebuilds_when_python_dep_changes(self, stub_repo):
        build(_docs(stub_repo), repo_root=stub_repo)
        # Touch the python source the directive resolved against.
        time.sleep(0.02)
        mod = stub_repo / "src" / "pkg" / "mod.py"
        future = mod.stat().st_mtime + 10
        os.utime(mod, (future, future))

        report = build(_docs(stub_repo), repo_root=stub_repo)
        assert len(report.built) == 1

    def test_clean_wipes_output(self, stub_repo):
        build(_docs(stub_repo), repo_root=stub_repo)
        out_root = stub_repo / "docs" / ".built"
        assert out_root.exists()

        report = build(_docs(stub_repo), repo_root=stub_repo, clean=True)
        assert report.cleaned
        assert len(report.built) == 1

    def test_check_only_does_not_write(self, stub_repo):
        report = build(_docs(stub_repo), repo_root=stub_repo, check_only=True)
        assert len(report.built) == 1
        assert not (stub_repo / "docs" / ".built").exists()

    def test_check_after_build_reports_clean(self, stub_repo):
        build(_docs(stub_repo), repo_root=stub_repo)
        report = build(_docs(stub_repo), repo_root=stub_repo, check_only=True)
        assert not report.built
        assert len(report.skipped_up_to_date) == 1

    def test_subset_restricts_walk(self, stub_repo):
        # Add a second directive page outside the subset; only the
        # subset path should be built.
        (stub_repo / "docs" / "api2").mkdir()
        (stub_repo / "docs" / "api2" / "other.md").write_text("::: pkg.mod.Widget\n")
        report = build(
            _docs(stub_repo),
            repo_root=stub_repo,
            subset=stub_repo / "docs" / "api",
        )
        names = sorted(p.name for p in report.built)
        assert names == ["page.md"]

    def test_unresolvable_symbol_produces_inline_error(self, stub_repo):
        (stub_repo / "docs" / "api" / "page.md").write_text(
            "# API\n\n::: pkg.does.not.exist\n"
        )
        report = build(_docs(stub_repo), repo_root=stub_repo)
        # No build error — the page is still produced, but with an
        # error block inline so the rest of docs/ continues to work.
        assert len(report.built) == 1
        assert not report.errors
        out = stub_repo / "docs" / ".built" / "api" / "page.md"
        text = out.read_text()
        assert "mkdocstrings expansion failed" in text

    def test_deps_sidecar_is_repo_relative(self, stub_repo):
        build(_docs(stub_repo), repo_root=stub_repo)
        deps_path = stub_repo / "docs" / ".built" / ".deps.json"
        deps = json.loads(deps_path.read_text())
        # Page deps should reference src/pkg/mod.py (relative to repo).
        page_deps = deps["api/page.md"]
        assert any(
            d.endswith("pkg/mod.py") and not Path(d).is_absolute() for d in page_deps
        ), page_deps

    def test_deps_sidecar_is_flushed_per_page(self, stub_repo):
        # Add a second directive page so the loop has more than one
        # iteration; the first page's deps must be visible on disk
        # before the second iteration begins (so a mid-loop crash
        # doesn't strand the first page with a stale/absent entry).
        (stub_repo / "docs" / "second.md").write_text("::: pkg.mod.Widget\n")
        build(_docs(stub_repo), repo_root=stub_repo)
        deps = json.loads((stub_repo / "docs" / ".built" / ".deps.json").read_text())
        # Both pages present in the deps map.
        assert set(deps.keys()) == {"api/page.md", "second.md"}

    def test_output_mtime_anchored_to_source_mtime(self, stub_repo):
        # The output's mtime should match the source mtime captured
        # before reading — that way an edit landing during the render
        # window leaves src.mtime > out.mtime and gets picked up next
        # build. We verify the invariant directly: after a build,
        # source and output mtimes are equal (modulo filesystem
        # resolution rounding).
        build(_docs(stub_repo), repo_root=stub_repo)
        src = stub_repo / "docs" / "api" / "page.md"
        out = stub_repo / "docs" / ".built" / "api" / "page.md"
        assert abs(src.stat().st_mtime - out.stat().st_mtime) < 0.01

    def test_signature_handles_positional_only(self, stub_repo, tmp_path):
        # def f(a, b, /, c, d) must render as exactly that, not as
        # def f(a, b, c, /, d).
        (stub_repo / "src" / "pkg" / "mod.py").write_text(
            "def slashy(a, b, /, c, d):\n"
            '    """Tail of slashy."""\n'
            "    return None\n"
        )
        (stub_repo / "docs" / "api" / "page.md").write_text("::: pkg.mod.slashy\n")
        build(_docs(stub_repo), repo_root=stub_repo)
        text = (stub_repo / "docs" / ".built" / "api" / "page.md").read_text()
        assert "def slashy(a, b, /, c, d)" in text

    def test_signature_with_trailing_positional_only(self, stub_repo):
        # def f(a, b, /) (all positional-only, no trailing kw params)
        # should still close the run with a trailing slash.
        (stub_repo / "src" / "pkg" / "mod.py").write_text(
            "def tail(a, b, /):\n" '    """."""\n' "    return None\n"
        )
        (stub_repo / "docs" / "api" / "page.md").write_text("::: pkg.mod.tail\n")
        build(_docs(stub_repo), repo_root=stub_repo)
        text = (stub_repo / "docs" / ".built" / "api" / "page.md").read_text()
        assert "def tail(a, b, /)" in text

    def test_python_fence_widens_when_body_contains_triple_backticks(self, stub_repo):
        # A default value containing a triple-backtick run would
        # otherwise close the wrapping fence prematurely. The renderer
        # widens the fence to one tick longer than the longest run.
        (stub_repo / "src" / "pkg" / "mod.py").write_text(
            'def code(x="```a```"):\n'
            '    """A function with backticks in its default."""\n'
            "    return None\n"
        )
        (stub_repo / "docs" / "api" / "page.md").write_text("::: pkg.mod.code\n")
        build(_docs(stub_repo), repo_root=stub_repo)
        text = (stub_repo / "docs" / ".built" / "api" / "page.md").read_text()
        # Fence is widened to 4 ticks so the inner ``` is just content.
        assert "````python\ndef code(" in text
        # Triple-backtick run survives inside the wider fence (griffe
        # may normalize the surrounding quotes, so we don't pin those).
        assert "```a```" in text

    def test_nested_4backtick_fence_in_source_is_respected(self, stub_repo):
        # A 4-backtick fenced block wrapping a 3-backtick example must
        # not be torn apart by the directive scanner — the inner ```
        # should not close the outer ```` block, and a directive on the
        # next line should still be invisible to the scanner.
        (stub_repo / "docs" / "api" / "page.md").write_text(
            "# API\n\n"
            "````markdown\n"
            "```python\n"
            "::: pkg.mod.Widget\n"
            "```\n"
            "````\n"
            "\n"
            "::: pkg.mod.Widget\n"
        )
        build(_docs(stub_repo), repo_root=stub_repo)
        text = (stub_repo / "docs" / ".built" / "api" / "page.md").read_text()
        # The directive inside the 4-backtick block is preserved
        # verbatim (not expanded); the bare one outside is expanded.
        assert "````markdown\n```python\n::: pkg.mod.Widget\n```\n````" in text
        assert "class Widget(" in text

    def test_directives_inside_fenced_code_block_are_ignored(self, stub_repo):
        (stub_repo / "docs" / "api" / "page.md").write_text(
            "# API\n\n"
            "```markdown\n"
            "::: pkg.mod.Widget\n"
            "```\n"
            "\nReal directive:\n\n::: pkg.mod.Widget\n"
        )
        report = build(_docs(stub_repo), repo_root=stub_repo)
        text = (stub_repo / "docs" / ".built" / "api" / "page.md").read_text()
        # The fenced occurrence is preserved verbatim, the real one is
        # expanded.
        assert "```markdown\n::: pkg.mod.Widget\n```" in text
        assert "class Widget(" in text
