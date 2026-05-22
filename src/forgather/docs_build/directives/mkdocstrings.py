"""Expand `::: dotted.path` into a markdown block via griffe.

This is a deliberately small subset of mkdocstrings' rendering — we
produce plain markdown that any markdown viewer can display, not the
themed HTML mkdocstrings would emit. Output covers the cases used in
this repo's `docs/api/*.md` pages: classes (signature, docstring,
public methods, attributes), functions, attributes/data, and
protocols.

The directive is resilient: if a symbol can't be resolved, the
output is a clear error block rather than a build failure, so a
single broken reference doesn't take down the docs build.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from textwrap import indent
from typing import Any, Iterable, Optional

from .base import Directive, DirectiveContext, DirectiveResult

log = logging.getLogger("forgather.docs_build.mkdocstrings")

# Matches a line like `::: forgather.project.Project` at the start of
# a line. The trailing capture is the dotted path; anything after
# whitespace is ignored (mkdocstrings supports trailing YAML-block
# options, but no doc in this repo uses them yet).
_PATTERN = re.compile(r"^:::\s+([A-Za-z_][\w.]*)\s*$")


class MkdocstringsDirective:
    """Resolve `::: dotted.path` via griffe and emit markdown."""

    pattern = _PATTERN

    def expand(
        self, match: re.Match[str], *, context: DirectiveContext
    ) -> DirectiveResult:
        path = match.group(1)
        loader = context.griffe_loader
        if loader is None:
            return DirectiveResult(
                markdown=_error_block(path, "griffe loader unavailable"),
            )
        try:
            obj = _resolve(loader, path)
        except Exception as exc:  # noqa: BLE001 — see module docstring
            log.warning("docs_build: failed to resolve %s: %s", path, exc)
            return DirectiveResult(
                markdown=_error_block(path, f"could not resolve symbol: {exc}"),
            )

        markdown = _render(obj)
        deps = _collect_deps(obj)
        return DirectiveResult(markdown=markdown, deps=deps)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _resolve(loader: Any, dotted: str) -> Any:
    """Return the griffe object identified by ``dotted``.

    Griffe can load packages and modules by name but doesn't directly
    accept ``pkg.module.Class.attr``-style references. We walk the
    dotted parts, loading the longest prefix that succeeds as a module
    and then descending into members.
    """
    parts = dotted.split(".")
    last_err: Optional[Exception] = None
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        try:
            root = loader.load(prefix)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
        obj = root
        for name in parts[i:]:
            members = getattr(obj, "members", None) or {}
            if name not in members:
                raise KeyError(
                    f"{prefix!r} has no member {name!r} (looking up {dotted!r})"
                )
            obj = members[name]
        return obj
    raise last_err or RuntimeError(f"failed to load {dotted!r}")


def _collect_deps(obj: Any) -> list[Path]:
    """Return the source files relevant to ``obj``.

    Includes the file of ``obj`` itself plus, for classes, the file of
    each member (members can be inherited from / live in a different
    module). Used by the builder to decide when a cached output is
    stale.
    """
    deps: set[Path] = set()
    fp = getattr(obj, "filepath", None)
    if isinstance(fp, Path):
        deps.add(fp)
    for member in getattr(obj, "members", {}).values():
        mfp = getattr(member, "filepath", None)
        if isinstance(mfp, Path):
            deps.add(mfp)
    return sorted(deps)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render(obj: Any) -> str:
    kind = type(obj).__name__
    if kind == "Class":
        return _render_class(obj)
    if kind == "Function":
        return _render_function(obj, heading_level=3)
    if kind in ("Attribute", "Module"):
        return _render_module_or_attribute(obj)
    # Unknown kind — fall back to docstring + canonical path.
    return _render_unknown(obj)


def _render_class(obj: Any) -> str:
    out: list[str] = []
    out.append(f"### `{obj.name}` {{#{_anchor(obj)}}}")
    out.append("")
    out.append(f"`{obj.canonical_path}`")
    out.append("")

    init = obj.members.get("__init__") if hasattr(obj, "members") else None
    if init is not None:
        sig = _signature(init, owner_name=obj.name, is_init=True)
        out.extend(_python_fence(sig))
        out.append("")

    out.append(_format_docstring(obj))
    out.append("")

    attrs = _public_attributes(obj)
    if attrs:
        out.append("**Attributes**")
        out.append("")
        for attr in attrs:
            out.append(_render_attribute_line(attr))
        out.append("")

    methods = _public_methods(obj)
    if methods:
        out.append("**Methods**")
        out.append("")
        for m in methods:
            out.append(_render_function(m, heading_level=4))
            out.append("")

    return _strip_trailing_blank(out)


def _render_function(obj: Any, *, heading_level: int) -> str:
    out: list[str] = []
    heading = "#" * heading_level
    out.append(f"{heading} `{obj.name}` {{#{_anchor(obj)}}}")
    out.append("")
    out.extend(_python_fence(_signature(obj, owner_name=obj.name, is_init=False)))
    out.append("")
    out.append(_format_docstring(obj))
    return _strip_trailing_blank(out)


def _render_module_or_attribute(obj: Any) -> str:
    out: list[str] = []
    out.append(f"### `{obj.name}` {{#{_anchor(obj)}}}")
    out.append("")
    out.append(f"`{obj.canonical_path}`")
    ann = getattr(obj, "annotation", None)
    if ann is not None:
        out.append("")
        out.append(f"Type: `{ann}`")
    out.append("")
    out.append(_format_docstring(obj))
    return _strip_trailing_blank(out)


def _render_unknown(obj: Any) -> str:
    out: list[str] = []
    out.append(f"### `{getattr(obj, 'name', '?')}` {{#{_anchor(obj)}}}")
    out.append("")
    out.append(f"`{getattr(obj, 'canonical_path', '?')}`")
    out.append("")
    out.append(_format_docstring(obj))
    return _strip_trailing_blank(out)


# ---------------------------------------------------------------------------
# Member filtering
# ---------------------------------------------------------------------------


def _public_methods(cls: Any) -> list[Any]:
    out: list[Any] = []
    for name, member in cls.members.items():
        if name == "__init__":
            continue
        if type(member).__name__ != "Function":
            continue
        if _is_public(name, member):
            out.append(member)
    return out


def _public_attributes(cls: Any) -> list[Any]:
    out: list[Any] = []
    for name, member in cls.members.items():
        if type(member).__name__ != "Attribute":
            continue
        if _is_public(name, member):
            out.append(member)
    return out


def _is_public(name: str, member: Any) -> bool:
    # Dunders other than __init__ aren't surfaced — too noisy for the
    # rendered docs and they rarely have user-facing docstrings.
    if name.startswith("_"):
        return False
    return True


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


def _signature(func: Any, *, owner_name: str, is_init: bool) -> str:
    parts: list[str] = []
    saw_pos_only = False
    pos_only_closed = False
    saw_kw_only_marker = False
    for p in func.parameters:
        if p.name == "self":
            continue
        kind = str(p.kind).rsplit(".", 1)[-1]

        # Close the positional-only run *before* appending the first
        # non-positional-only param so the slash lands in the right
        # spot — ``def f(a, b, /, c)`` not ``def f(a, b, c, /)``.
        if saw_pos_only and not pos_only_closed and kind != "positional_only":
            parts.append("/")
            pos_only_closed = True

        if kind == "positional_only":
            saw_pos_only = True
        if kind == "keyword_only" and not saw_kw_only_marker:
            parts.append("*")
            saw_kw_only_marker = True
        parts.append(_render_param(p))

    # Function ends with a positional-only tail (no later params).
    if saw_pos_only and not pos_only_closed:
        parts.append("/")

    head = f"class {owner_name}" if is_init else f"def {owner_name}"
    joined = ", ".join(parts)
    return f"{head}({joined})"


def _render_param(p: Any) -> str:
    kind = str(p.kind).rsplit(".", 1)[-1]
    prefix = ""
    if kind == "var_positional":
        prefix = "*"
    elif kind == "var_keyword":
        prefix = "**"
    name = f"{prefix}{p.name}"
    if p.annotation is not None:
        name += f": {p.annotation}"
    if p.default is not None and kind not in ("var_positional", "var_keyword"):
        name += f" = {p.default}"
    return name


# ---------------------------------------------------------------------------
# Docstring rendering
# ---------------------------------------------------------------------------


def _format_docstring(obj: Any) -> str:
    ds = getattr(obj, "docstring", None)
    if ds is None:
        return "_No documentation._"
    sections = list(ds.parsed)
    if not sections:
        return ds.value or "_No documentation._"
    rendered: list[str] = []
    for sec in sections:
        kind = str(sec.kind).rsplit(".", 1)[-1]
        block = _format_section(kind, sec.value)
        if block:
            rendered.append(block)
    return "\n\n".join(rendered).strip() or "_No documentation._"


def _format_section(kind: str, value: Any) -> str:
    if kind == "text":
        return str(value).strip()
    if kind == "parameters":
        return _format_param_list("Parameters", value)
    if kind == "other_parameters":
        return _format_param_list("Other Parameters", value)
    if kind == "returns":
        return _format_return_list("Returns", value)
    if kind == "yields":
        return _format_return_list("Yields", value)
    if kind == "raises":
        return _format_raise_list(value)
    if kind == "attributes":
        return _format_attr_list("Attributes", value)
    if kind == "examples":
        return _format_examples(value)
    if kind == "admonition":
        return _format_admonition(value)
    if kind in ("warns", "deprecated"):
        return _format_raise_list(value, header=kind.title())
    # Unknown section — best-effort string repr so content isn't lost.
    return f"**{kind.title()}**\n\n{value}"


def _format_param_list(header: str, params: Iterable[Any]) -> str:
    lines = [f"**{header}**", ""]
    for p in params:
        ann = f" ({p.annotation})" if p.annotation else ""
        lines.append(f"- `{p.name}`{ann} — {p.description.strip()}")
    return "\n".join(lines)


def _format_attr_list(header: str, attrs: Iterable[Any]) -> str:
    lines = [f"**{header}**", ""]
    for a in attrs:
        ann = f" ({a.annotation})" if a.annotation else ""
        lines.append(f"- `{a.name}`{ann} — {a.description.strip()}")
    return "\n".join(lines)


def _format_return_list(header: str, returns: Iterable[Any]) -> str:
    lines = [f"**{header}**", ""]
    for r in returns:
        ann = f"`{r.annotation}` — " if r.annotation else ""
        name = f"`{r.name}` " if getattr(r, "name", None) else ""
        lines.append(f"- {name}{ann}{(r.description or '').strip()}")
    return "\n".join(lines)


def _format_raise_list(raises: Iterable[Any], *, header: str = "Raises") -> str:
    lines = [f"**{header}**", ""]
    for r in raises:
        ann = f"`{r.annotation}` — " if r.annotation else ""
        lines.append(f"- {ann}{(r.description or '').strip()}")
    return "\n".join(lines)


def _format_examples(entries: Iterable[Any]) -> str:
    out: list[str] = ["**Examples**", ""]
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) != 2:
            out.append(str(entry))
            continue
        kind, value = entry
        kind_name = str(kind).rsplit(".", 1)[-1]
        if kind_name == "text":
            out.append(str(value).strip())
            out.append("")
        elif kind_name == "examples":
            out.extend(_python_fence(str(value).strip()))
            out.append("")
    return _strip_trailing_blank(out)


def _format_admonition(value: Any) -> str:
    annotation = (getattr(value, "annotation", None) or "note").title()
    description = getattr(value, "description", "") or getattr(value, "contents", "")
    body = indent(str(description).strip(), "> ")
    return f"> **{annotation}**\n>\n{body}"


def _render_attribute_line(attr: Any) -> str:
    ann = f" ({attr.annotation})" if getattr(attr, "annotation", None) else ""
    ds = getattr(attr, "docstring", None)
    desc = (ds.value.strip() if ds else "").splitlines()
    summary = desc[0] if desc else ""
    return f"- `{attr.name}`{ann} — {summary}".rstrip(" —")


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _anchor(obj: Any) -> str:
    """Stable anchor id from canonical_path so cross-page links resolve."""
    return getattr(obj, "canonical_path", obj.name).replace(".", "-").lower()


_BACKTICK_RUN_RE = re.compile(r"`{3,}")


def _python_fence(body: str) -> list[str]:
    """Wrap ``body`` in a `python`-tagged fence safe against inner backticks.

    A function default like ``foo("```bar```")`` would normally close
    a 3-backtick fence prematurely and break the rendering of every
    block that follows. We scan ``body`` for the longest run of
    backticks and emit a fence one tick longer (CommonMark allows
    nesting that way). Three backticks remain the default — the helper
    only widens when the body forces it.
    """
    longest = max((len(m.group(0)) for m in _BACKTICK_RUN_RE.finditer(body)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}python", body, fence]


def _strip_trailing_blank(lines: list[str]) -> str:
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _error_block(path: str, message: str) -> str:
    return (
        f"> **mkdocstrings expansion failed**\n"
        f">\n"
        f"> Could not render `::: {path}`. {message}"
    )
