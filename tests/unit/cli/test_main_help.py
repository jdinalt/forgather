"""Tests for the top-level command overview (shared by `--help` and the
interactive `commands` listing)."""

from forgather.cli.main import iter_command_summaries


def test_command_summaries_have_descriptions():
    rows = dict(iter_command_summaries())
    # A few well-known commands must be present with a non-empty, single-line
    # summary (the interactive `commands` listing relies on this).
    for name in ("train", "submit", "job", "ls"):
        assert name in rows, f"{name} missing from command overview"
        assert rows[name], f"{name} has an empty summary"
        assert "\n" not in rows[name], f"{name} summary is not single-line"
    # The removed `control` command must not appear.
    assert "control" not in rows
