"""Pre-render documentation directives for the lightweight webui docs viewer.

The webui's `/api/docs/file` endpoint serves raw markdown. Pages that
use `:::` mkdocstrings directives (or future directives) appear unrendered
because nothing expands them at read time.

This package walks `docs/` ahead of time, expands directives, and writes
the result to `docs/.built/<rel>.md`. The server route prefers the built
copy when it exists and is newer than the source.
"""

from .builder import BuildReport, build

__all__ = ["BuildReport", "build"]
