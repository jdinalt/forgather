"""Directive registry.

A directive matches a line in a markdown source and expands it to a
markdown block, optionally reporting filesystem dependencies that
should invalidate the cached output when they change.

Adding a new directive is two steps: implement a `Directive` subclass
and register it in `ALL_DIRECTIVES`.
"""

from .base import Directive, DirectiveResult
from .mkdocstrings import MkdocstringsDirective

ALL_DIRECTIVES: tuple[Directive, ...] = (MkdocstringsDirective(),)

__all__ = ["Directive", "DirectiveResult", "MkdocstringsDirective", "ALL_DIRECTIVES"]
