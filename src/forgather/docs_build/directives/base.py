"""Directive protocol.

A directive matches a single line in the source markdown and produces
a markdown block to substitute in. Directives may also report
filesystem paths whose mtime should invalidate the cached output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol


@dataclass
class DirectiveResult:
    """The output of expanding one directive line.

    Attributes
    ----------
    markdown
        Replacement text (may span multiple lines, no trailing newline).
    deps
        Absolute paths whose mtime invalidates this build product.
        Typically the source file(s) the directive resolved against.
    """

    markdown: str
    deps: list[Path] = field(default_factory=list)


class Directive(Protocol):
    """Protocol every directive implementation satisfies."""

    pattern: re.Pattern[str]
    """Regex that matches a directive line (anchored with ^ and \\Z)."""

    def expand(
        self, match: re.Match[str], *, context: "DirectiveContext"
    ) -> DirectiveResult:
        """Expand the matched directive into a markdown block."""
        ...


@dataclass
class DirectiveContext:
    """Build-time state passed to every directive invocation.

    Carries any per-build resources (loader handles, etc.) that
    directives need but shouldn't construct themselves.
    """

    repo_root: Path
    griffe_loader: Optional[object] = None
