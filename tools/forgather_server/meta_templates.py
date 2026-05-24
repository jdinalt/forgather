"""Meta-template scaffolds: discover and render starter configs.

A *meta-template* is a (body, manifest) pair living under ``templatelib/meta/``
that the webui's New Config / New Template modal can render into a starting
point instead of an empty file. The body contains ``$VAR`` / ``${VAR}``
markers (Python ``string.Template`` syntax); the sidecar manifest declares
each variable so the UI can render a form.

Why ``string.Template`` and not Jinja for substitution: a Forgather config
*is* a Jinja template, so a Jinja-generating-Jinja design would force
escaping ``{{`` and ``{%`` in every meta-template. Picking a different
delimiter sidesteps that entirely. The rendered text is written to disk
as a normal config file; Forgather's preprocessor only sees it on next
load, same as a hand-edited config.

Layout::

    templatelib/meta/
        _category.yaml                 (optional)
        datasets/
            _category.yaml             (optional: display label for this group)
            huggingface/
                with_config.yaml       (the body)
                with_config.meta.yaml  (the manifest)

The directory path under ``templatelib/meta/`` is the hierarchy the webui
renders as a tree. A meta-template's ``id`` is its body path relative to
``templatelib/meta/``, minus the ``.yaml`` suffix, with ``/`` separators
(stable across platforms).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from string import Template
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

# Resolved relative to this file: forgather/tools/forgather_server/ -> repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
META_ROOT = os.path.join(_REPO_ROOT, "templatelib", "meta")


# ---------------------------------------------------------------------------
# Configurable search roots
#
# Discovery walks a list of root directories; each is scanned independently
# and the results are merged with **first-wins** semantics — same shape as
# Jinja's search path. The server CLI populates this at startup via
# ``configure_roots`` (``--meta-template-dir PATH`` is repeatable, and
# ``--no-default-meta-templates`` drops the bundled defaults). User-provided
# roots come first, so a scaffold whose id matches a bundled one overrides
# the default — exactly what you want when shipping a customised version
# of a built-in scaffold.
#
# Tests bypass all of this by passing ``meta_root=<tmp_path>`` directly to
# ``discover`` / ``get`` / ``render`` — that path is treated as the only
# root, ignoring whatever ``configure_roots`` has registered.

_extra_roots: List[str] = []
_use_default_root: bool = True


def configure_roots(
    extra_dirs: Iterable[str] = (),
    *,
    disable_default: bool = False,
) -> None:
    """Set the active meta-template search path.

    ``extra_dirs`` come first in scan order, so user-provided scaffolds
    take precedence over bundled defaults of the same id. The default
    ``templatelib/meta/`` directory is appended unless ``disable_default``
    is true. Paths must be absolute and existing; non-existent paths are
    quietly skipped (a typo in the CLI shouldn't kill the server, but the
    operator gets nothing from a missing directory either).

    Idempotent: calling again replaces the configuration. Pass no
    arguments to reset to "defaults only" (used by tests that monkeypatch
    individual call paths).
    """
    global _extra_roots, _use_default_root
    cleaned: List[str] = []
    for d in extra_dirs:
        ap = os.path.abspath(d)
        if os.path.isdir(ap) and ap not in cleaned:
            cleaned.append(ap)
    _extra_roots = cleaned
    _use_default_root = not disable_default


def _active_roots() -> List[str]:
    """Resolve the ordered list of roots to scan, dropping the default
    when disabled and skipping any non-existent extras."""
    roots: List[str] = list(_extra_roots)
    if _use_default_root and os.path.isdir(META_ROOT):
        roots.append(META_ROOT)
    return roots


@dataclass
class MetaField:
    """One form field declared by a meta-template's manifest.

    ``picker`` names a specialised input picker the webui should render
    alongside the text input — currently the only kind is ``"dataset"``
    (a popover listing the cluster's aggregated dataset inventory:
    HuggingFace cache entries and ``local/<name>`` registrations).
    Empty string ⇒ plain text input.
    """

    name: str
    label: str = ""
    description: str = ""
    placeholder: str = ""
    default: Optional[str] = None
    required: bool = False
    picker: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name


@dataclass
class MetaTemplate:
    """A single scaffold: body + declared fields + display metadata.

    ``summary`` is the short blurb shown next to the leaf in the picker
    tree (one line, kept terse). ``description`` is the full text shown
    in the detail panel when the leaf is selected. When ``summary`` is
    blank the picker falls back to ``description`` so older manifests
    that only set one field still render usefully.
    """

    id: str
    title: str
    summary: str
    description: str
    target_kind: str  # "config" | "template"
    fields: List[MetaField] = field(default_factory=list)
    body_path: str = ""  # absolute path to the body file


@dataclass
class MetaCategory:
    """One node of the meta-template tree (a directory under META_ROOT).

    A category may directly contain templates *and* further sub-categories;
    the webui renders both — common cases at the top, exotic ones nested
    below. ``summary`` / ``description`` follow the same convention as
    ``MetaTemplate``: summary for the tree row, description for any
    detail / tooltip surface.
    """

    name: str  # last path segment, e.g. "huggingface"
    title: str  # display label, falls back to title-cased name
    summary: str = ""
    description: str = ""
    templates: List[MetaTemplate] = field(default_factory=list)
    children: List["MetaCategory"] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery


def _title_case(segment: str) -> str:
    """Fallback display label when no _category.yaml is provided."""
    return segment.replace("_", " ").replace("-", " ").title()


def _load_yaml(path: str) -> Mapping[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def _is_manifest(filename: str) -> bool:
    return filename.endswith(".meta.yaml") and not filename.startswith("_")


def _scan_dir(abs_dir: str, rel_prefix: str) -> MetaCategory:
    """Recursively build a MetaCategory for ``abs_dir``.

    ``rel_prefix`` is the path from META_ROOT to ``abs_dir`` (forward slashes).
    Empty for the root directory.
    """
    name = os.path.basename(abs_dir) if rel_prefix else ""
    title = _title_case(name) if name else ""
    description = ""

    summary = ""
    cat_file = os.path.join(abs_dir, "_category.yaml")
    if os.path.isfile(cat_file):
        try:
            meta = _load_yaml(cat_file)
            title = str(meta.get("title", title))
            summary = str(meta.get("summary", ""))
            description = str(meta.get("description", ""))
        except Exception:
            # Malformed _category.yaml shouldn't break discovery — fall back
            # to the title-cased directory name.
            pass

    category = MetaCategory(
        name=name, title=title, summary=summary, description=description
    )

    try:
        entries = sorted(os.listdir(abs_dir))
    except OSError:
        return category

    for entry in entries:
        full = os.path.join(abs_dir, entry)
        if os.path.isdir(full):
            child_rel = f"{rel_prefix}/{entry}" if rel_prefix else entry
            child = _scan_dir(full, child_rel)
            # Skip empty categories so the picker doesn't show dead branches.
            if child.templates or child.children:
                category.children.append(child)
        elif os.path.isfile(full) and _is_manifest(entry):
            stem = entry[: -len(".meta.yaml")]
            if stem.startswith("_"):
                continue
            body = os.path.join(abs_dir, stem + ".yaml")
            if not os.path.isfile(body):
                continue
            try:
                manifest = _load_yaml(full)
            except Exception:
                # Skip unparseable manifests rather than failing the whole tree.
                continue
            mid = f"{rel_prefix}/{stem}" if rel_prefix else stem
            category.templates.append(
                MetaTemplate(
                    id=mid,
                    title=str(manifest.get("title", stem)),
                    summary=str(manifest.get("summary", "")),
                    description=str(manifest.get("description", "")),
                    target_kind=str(manifest.get("target_kind", "config")),
                    fields=[MetaField(**fd) for fd in manifest.get("fields", [])],
                    body_path=body,
                )
            )

    return category


def discover(meta_root: Optional[str] = None) -> List[MetaCategory]:
    """Top-level scan: return the merged tree of meta-templates.

    Without ``meta_root``, every configured search root is scanned in
    order (user-provided first, bundled default last) and the results
    are merged with **first-wins** semantics: a leaf id present in an
    earlier root shadows the same id in a later root; a category present
    in multiple roots has its children merged recursively, with the
    *first* root's ``_category.yaml`` providing the display label.

    The root directory itself is not surfaced as a category — its
    children are the top-level groups (Datasets, Models, Trainers, …).
    If a ``_category.yaml`` exists at the root it is ignored for now.

    When ``meta_root`` is given the configured search path is bypassed
    entirely and that directory is the sole root scanned. Used by tests
    to point at a fixture tree without disturbing global state.
    """
    if meta_root is not None:
        if not os.path.isdir(meta_root):
            return []
        return _scan_dir(meta_root, rel_prefix="").children
    merged: List[MetaCategory] = []
    for root in _active_roots():
        merged = _merge_categories(merged, _scan_dir(root, rel_prefix="").children)
    return merged


def _merge_categories(
    base: List[MetaCategory], incoming: List[MetaCategory]
) -> List[MetaCategory]:
    """Merge ``incoming`` into ``base`` with first-wins semantics.

    ``base`` represents the higher-priority root (or the accumulated
    union so far). ``incoming`` is appended underneath: new categories
    are added at the end; categories that already exist in ``base``
    receive any new sub-categories and any non-conflicting templates.
    """
    by_name = {c.name: c for c in base}
    for c in incoming:
        existing = by_name.get(c.name)
        if existing is None:
            base.append(c)
            by_name[c.name] = c
            continue
        # Category exists in base — merge children + templates first-wins.
        existing_template_ids = {t.id for t in existing.templates}
        for t in c.templates:
            if t.id not in existing_template_ids:
                existing.templates.append(t)
                existing_template_ids.add(t.id)
        existing.children = _merge_categories(existing.children, c.children)
    return base


def _walk_templates(categories: List[MetaCategory]) -> List[MetaTemplate]:
    out: List[MetaTemplate] = []
    for cat in categories:
        out.extend(cat.templates)
        out.extend(_walk_templates(cat.children))
    return out


def get(meta_id: str, meta_root: Optional[str] = None) -> MetaTemplate:
    """Look up a single meta-template by id, or raise KeyError.

    Honours the same first-wins semantics as ``discover``: when no
    explicit ``meta_root`` is given, scans each configured root in order
    and returns the first match. Walking each root individually (rather
    than calling ``discover`` and then linearly searching the merged
    tree) keeps render-time lookup independent of the merge cost.
    """
    if meta_root is not None:
        for mt in _walk_templates(_scan_dir(meta_root, rel_prefix="").children):
            if mt.id == meta_id:
                return mt
        raise KeyError(meta_id)
    for root in _active_roots():
        for mt in _walk_templates(_scan_dir(root, rel_prefix="").children):
            if mt.id == meta_id:
                return mt
    raise KeyError(meta_id)


# ---------------------------------------------------------------------------
# Rendering


class MissingFieldsError(ValueError):
    """Raised when render() is called without values for required fields."""

    def __init__(self, missing: List[str]):
        super().__init__(f"missing required fields: {', '.join(missing)}")
        self.missing = missing


def render(
    meta_id: str,
    values: Mapping[str, Any],
    *,
    meta_root: Optional[str] = None,
) -> str:
    """Substitute ``$VAR`` markers in the meta-template body and return text.

    Resolution order per field:
      1. Caller-supplied value (non-empty)
      2. Manifest ``default``
      3. If ``required``, collect as missing and raise
      4. Otherwise substitute the empty string

    Extra keys in ``values`` that aren't declared in the manifest are
    ignored — the manifest is the source of truth for which fields a
    meta-template accepts.

    A stray ``$VAR`` in the body whose name isn't declared in the manifest
    is treated as a bug in the meta-template and raises ``KeyError`` via
    ``string.Template.substitute``.
    """
    mt = get(meta_id, meta_root)

    resolved: Dict[str, str] = {}
    missing: List[str] = []
    for fd in mt.fields:
        raw = values.get(fd.name)
        if raw not in (None, ""):
            resolved[fd.name] = str(raw)
        elif fd.default is not None:
            resolved[fd.name] = fd.default
        elif fd.required:
            missing.append(fd.name)
        else:
            resolved[fd.name] = ""
    if missing:
        raise MissingFieldsError(missing)

    with open(mt.body_path) as f:
        body = f.read()
    return Template(body).substitute(resolved)
