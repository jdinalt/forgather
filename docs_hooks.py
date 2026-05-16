"""MkDocs hook: rewrite relative links on pages whose source is a symlink.

The problem
-----------
Some pages under ``docs/`` are symlinks to canonical files elsewhere
in the repo — e.g. ``docs/forgather-server.md`` -> ``../tools/forgather_server/README.md``.
Those canonical files contain relative markdown links that resolve
correctly when the file is read from its real location on disk
(in-app Docs viewer, GitHub source view), like::

    [walkthrough](../../docs/guides/forgather-server-walkthrough.md)

MkDocs's link resolver, on the other hand, computes relative paths
from each page's ``docs_dir``-relative location, not from the
realpath of the source file. So the same link resolves as if it
originated from ``docs/forgather-server.md``, ascending past
``docs_dir`` and producing a broken link.

The fix
-------
For every page whose source path is a symlink, we walk the markdown
and rewrite each relative inline link / image / reference link from
"interpreted relative to realpath" to "interpreted relative to the
docs_dir page location" — by resolving the original href against the
realpath's directory, then computing the new ``os.path.relpath`` from
the docs_dir page's directory. MkDocs then sees a link it can
resolve normally.

Caveats
-------
- Only inline ``[text](url)`` / ``![alt](src)`` and reference-style
  ``[ref]: url`` are rewritten. Autolinks like ``<https://...>`` are
  external and don't need rewriting.
- Absolute URLs (``http://...``, ``mailto:``, anchors, etc.) are left
  alone.
- Fenced code blocks are skipped (line-counted) so example markdown
  inside them isn't touched.
- We don't try to defend against malformed code spans on the same
  line as a real link; if a doc has ``` `[a](b)` `` *and* a real
  ``[c](d)`` link side by side, both will match the regex. That
  hasn't bitten us in practice; the affected docs use real links.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

log = logging.getLogger("mkdocs.hook.symlink_links")


# Built lazily on the first ``on_page_markdown`` call and reused for
# the rest of the build. Maps ``os.path.realpath(symlink)`` →
# ``docs_dir``-relative posix path. We use it to detect the case
# where a relative link's resolved realpath has its own alias inside
# docs/ — when it does, we prefer linking to the alias so the URL
# stays inside the rendered site rather than ascending out.
_SYMLINK_TARGETS: Optional[dict[str, str]] = None


_INLINE_LINK_RE = re.compile(
    r"""
    (!?)                                 # 1: image marker (optional '!')
    (\[(?:[^\[\]]|\[[^\[\]]*\])*\])      # 2: bracketed text (allows one level of nested brackets)
    \(                                   # opening paren
    [ \t]*
    (                                    # 3: href, no spaces, no closing paren
      <[^>]+>                              # angle-bracketed href
      |
      [^()\s]+                             # bare href
    )
    (                                    # 4: optional title (preserved verbatim)
      [ \t]+
      (?:"[^"]*"|'[^']*'|\([^)]*\))
    )?
    [ \t]*\)
    """,
    re.VERBOSE,
)

# Reference-style link definitions: at start of line, '[label]: url ["title"]'.
# Group 1 is everything up through ': ', group 2 is the url, group 3 is the
# optional trailing title. Indented (up to 3 spaces) is allowed by the
# CommonMark spec.
_REF_DEF_RE = re.compile(
    r"""
    ^(?P<prefix>[ ]{0,3}\[[^\]]+\]:[ \t]+)   # label and colon
    (?P<href><[^>]+>|\S+)                    # url (angle-bracketed or bare)
    (?P<title>[ \t]+.+)?                     # optional title
    [ \t]*$
    """,
    re.VERBOSE,
)


def _is_external(href: str) -> bool:
    return (
        href.startswith("http://")
        or href.startswith("https://")
        or href.startswith("//")
        or href.startswith("mailto:")
        or href.startswith("data:")
        or href.startswith("tel:")
        or href.startswith("ftp://")
        or href.startswith("ftps://")
    )


def _build_symlink_index(docs_dir: str) -> dict[str, str]:
    """Walk ``docs_dir`` once and index every symlink we see.

    Key: ``os.path.realpath`` of the symlink target. Value:
    ``docs_dir``-relative posix path of the symlink itself. Used to
    redirect rewritten links that would otherwise ascend out of
    ``docs_dir`` back into the in-tree alias.
    """
    out: dict[str, str] = {}
    for root, dirs, files in os.walk(docs_dir, followlinks=True):
        # Avoid descending into recursive symlink loops.
        dirs.sort()
        for name in files:
            full = os.path.join(root, name)
            if os.path.islink(full):
                real = os.path.realpath(full)
                rel = os.path.relpath(full, docs_dir).replace(os.sep, "/")
                # First hit wins so the result is deterministic if
                # multiple symlinks point at the same target.
                out.setdefault(real, rel)
    return out


def _docs_alias_for(real_target: str, docs_dir: str) -> Optional[str]:
    global _SYMLINK_TARGETS
    if _SYMLINK_TARGETS is None:
        _SYMLINK_TARGETS = _build_symlink_index(docs_dir)
    return _SYMLINK_TARGETS.get(real_target)


def _rewrite_href(
    href: str, real_dir: str, logical_dir: str, docs_dir: str
) -> Optional[str]:
    """Return a rewritten href, or None if it should be left alone.

    ``real_dir`` is the directory of the symlink target (where the
    source's author intended relative links to resolve from).
    ``logical_dir`` is the directory of the symlink itself (the
    docs_dir-relative location MkDocs uses for link resolution).
    """
    if not href:
        return None
    # Angle-bracketed hrefs: strip < >, restore after rewrite.
    angle = href.startswith("<") and href.endswith(">")
    inner = href[1:-1] if angle else href

    if inner.startswith("#"):
        return None
    if _is_external(inner):
        return None
    if inner.startswith("/"):
        # Site-absolute; leave alone — operator chose this intentionally.
        return None

    # Split fragment / query so we resolve only the path portion.
    fragment = ""
    if "#" in inner:
        i = inner.index("#")
        fragment = inner[i:]
        inner = inner[:i]
    query = ""
    if "?" in inner:
        i = inner.index("?")
        query = inner[i:]
        inner = inner[:i]

    if not inner:
        return None

    # Resolve against the realpath's directory — i.e. what the link
    # author intended.
    absolute = os.path.normpath(os.path.join(real_dir, inner))

    # If that absolute target is itself the realpath of one of the
    # docs/ symlinks, prefer linking to the in-tree alias so the URL
    # stays inside the rendered site rather than ascending out via
    # ../.. and landing on a file that isn't part of the MkDocs build.
    real_abs = os.path.realpath(absolute)
    alias = _docs_alias_for(real_abs, docs_dir)
    if alias is not None:
        target = os.path.join(docs_dir, alias)
    else:
        target = absolute

    # Compute a new relative path from the page's docs_dir location.
    try:
        new_rel = os.path.relpath(target, logical_dir)
    except ValueError:
        return None

    # On Windows os.path.relpath uses backslashes; normalize to forward
    # slashes for the markdown / URL world.
    new_rel = new_rel.replace(os.sep, "/")
    rewritten = f"{new_rel}{query}{fragment}"
    if angle:
        rewritten = f"<{rewritten}>"

    # Don't bother emitting a change if the rewrite is a no-op (saves
    # diff noise and avoids accidentally normalising a working link
    # into a different-but-equivalent shape).
    if rewritten == href:
        return None
    return rewritten


def _strip_fenced_code(markdown: str):
    """Yield (kind, text) chunks: 'prose' or 'code'.

    Track three- and four-backtick / tilde fences. Anything inside a
    fence is emitted verbatim and never rewritten. This is the
    cheapest way to avoid mangling example markdown inside code
    blocks without bringing in a real markdown parser.
    """
    fence_re = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")
    in_fence = False
    fence_marker = ""
    buf: list[str] = []
    kind = "prose"
    for line in markdown.splitlines(keepends=True):
        m = fence_re.match(line)
        if not in_fence and m:
            if buf:
                yield kind, "".join(buf)
                buf = []
            kind = "code"
            in_fence = True
            fence_marker = m.group(1)
            buf.append(line)
        elif in_fence and m and m.group(1).startswith(fence_marker[0]) and len(m.group(1)) >= len(fence_marker):
            # Closing fence must use the same character and be at
            # least as long as the opening fence.
            buf.append(line)
            yield kind, "".join(buf)
            buf = []
            kind = "prose"
            in_fence = False
            fence_marker = ""
        else:
            buf.append(line)
    if buf:
        yield kind, "".join(buf)


def on_page_markdown(markdown, page, config, files):
    """MkDocs event: rewrite relative links when the source is a symlink."""
    abs_src = page.file.abs_src_path
    if not abs_src:
        return markdown
    real_src = os.path.realpath(abs_src)
    if os.path.normpath(abs_src) == real_src:
        # Not a symlink — leave the page alone.
        return markdown

    docs_dir = os.path.realpath(config["docs_dir"])
    logical_dir = os.path.dirname(os.path.normpath(abs_src))
    real_dir = os.path.dirname(real_src)

    def fix_inline(match: re.Match[str]) -> str:
        bang, text, href, title = match.group(1), match.group(2), match.group(3), match.group(4) or ""
        new = _rewrite_href(href, real_dir, logical_dir, docs_dir)
        if new is None:
            return match.group(0)
        return f"{bang}{text}({new}{title})"

    def fix_ref(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        href = match.group("href")
        title = match.group("title") or ""
        new = _rewrite_href(href, real_dir, logical_dir, docs_dir)
        if new is None:
            return match.group(0)
        return f"{prefix}{new}{title}"

    out: list[str] = []
    rewrites = 0
    for kind, chunk in _strip_fenced_code(markdown):
        if kind == "code":
            out.append(chunk)
            continue
        before = chunk
        chunk = _INLINE_LINK_RE.sub(fix_inline, chunk)
        chunk = _REF_DEF_RE.sub(fix_ref, chunk)
        if chunk != before:
            rewrites += 1
        out.append(chunk)
    if rewrites:
        log.debug(
            "symlink_links: rewrote %d link chunk(s) on page %s "
            "(real source: %s)",
            rewrites,
            page.file.src_path,
            real_src,
        )
    return "".join(out)
