"""Operations on Forgather configuration templates, wrapping existing APIs.

The server exposes these through HTTP endpoints; the pure-Python functions
here make it easier to test and reuse.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from forgather.cli.trefs import (
    render_template_hierarchy_dot,
    render_template_hierarchy_tree,
)
from forgather.cli.utils import get_config, get_env
from forgather.config import ConfigEnvironment
from forgather.latent import Latent
from forgather.meta_config import MetaConfig

from . import overrides_store


@dataclass
class LoadedEnv:
    meta: MetaConfig
    env: ConfigEnvironment
    config_path: str  # template-relative path used by the env


@dataclass
class TrefsNode:
    name: str
    path: str


@dataclass
class TrefsGraph:
    nodes: List[TrefsNode] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    root: str = ""


@dataclass
class ConfigMeta:
    """Human-facing metadata read from a config's ``meta`` block.

    ``config_name`` and ``config_description`` are the same fields
    ``forgather ls`` prints; ``parse_error`` is populated if preprocessing
    or materialization fails so the UI can show the filename alone.
    ``config_class`` lets callers tell training scripts apart from model /
    dataset / etc. configs so they can show only relevant actions
    (a model definition has no meaningful "Run" / "Clean Output" — those
    only apply to ``type.training_script*``).
    """

    name: Optional[str] = None
    description: Optional[str] = None
    config_class: Optional[str] = None
    parse_error: Optional[str] = None


@dataclass
class OutputDirInfo:
    """Where a config's training output would land + what's there now.

    ``output_dir`` is the per-model output directory (contains ``runs/``,
    checkpoints, etc). ``models_dir`` is the parent directory that holds
    every model produced by this project — deleting that wipes all models.

    ``nproc_per_node`` comes straight from ``config.meta``. It may be an
    integer (fixed worker count) or a torchrun keyword string such as
    ``"gpu"`` (one worker per CUDA-visible device), ``"cpu"``, or
    ``"auto"``. The submit UI uses it to seed a sensible default for
    ``requested_gpus`` and to warn when the two disagree.
    """

    output_dir: str
    models_dir: str
    output_dir_exists: bool
    models_dir_exists: bool
    output_dir_size_bytes: int = 0
    output_dir_entry_count: int = 0
    models_dir_size_bytes: int = 0
    models_dir_entry_count: int = 0
    # Kept as the raw value (int or str) so the UI can distinguish "fixed N
    # workers" from "auto-detect from CUDA_VISIBLE_DEVICES".
    nproc_per_node: Any = None


@dataclass
class DynamicArg:
    """One entry from a config's ``dynamic_args`` target.

    ``dest`` is the Python identifier used as the key when we pass dynamic
    args to the trainer (matches argparse's dest format — ``--model-project``
    becomes ``model_project``). ``cli_name`` is what the CLI would accept.
    ``choices`` is the optional argparse ``choices=`` list; when populated,
    the UI renders a dropdown instead of a free-text input.

    ``group`` is an optional colon-separated organizational path
    (e.g. ``"Trainer:LR-scaling"``) used by the webui to render a
    collapsible tree. Args without a group fall under an "Other" bucket
    if any sibling has a group, else the form stays flat. ``required``
    is enforced at action time (e.g. ``forgather train``, server enqueue
    of training jobs) — unset for ``pp`` so placeholder defaults still
    materialize.
    """

    dest: str
    cli_name: str
    type: str = "str"  # "int" | "str" | "float" | "bool" | "path"
    help: Optional[str] = None
    default: Any = None
    choices: Optional[List[Any]] = None
    group: Optional[str] = None
    required: bool = False
    # Inclusive numeric bounds. Only honoured for int / float types; the
    # webui ignores them on other types. Either may be unset independently.
    min: Optional[float] = None
    max: Optional[float] = None


def _merged_kwargs(project_dir: str, config_name: str, explicit: dict) -> dict:
    """Layer cached overrides under any explicit kwargs (explicit wins)."""
    base = overrides_store.get_overrides(project_dir, config_name)
    return {**base, **explicit}


def load_env(project_dir: str, config_name: str) -> LoadedEnv:
    meta = MetaConfig(project_dir)
    env = get_env(meta, project_dir)
    config_path = meta.config_path(config_name)
    return LoadedEnv(meta=meta, env=env, config_path=config_path)


def read_raw(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def render_pp(project_dir: str, config_name: str, **kwargs) -> str:
    loaded = load_env(project_dir, config_name)
    merged = _merged_kwargs(project_dir, config_name, kwargs)
    return str(loaded.env.preprocess(loaded.config_path, **merged))


@dataclass
class DebugTraceItem:
    """One template participating in a config's preprocess pass.

    ``name`` is the template name as Jinja2 sees it (relative to the search
    path). ``path`` is the absolute filesystem path of the source file (or
    ``""`` for synthetic / split-template fragments). ``raw`` is the
    pre-preprocess source as Jinja2's loader returned it; ``preprocessed`` is
    the same source after the LineStatementProcessor has rewritten the
    Forgather sugar (``--``, ``<<``, ``>>``, ``==``, ``=>``) into plain Jinja2.
    """

    name: str
    path: str
    raw: str
    preprocessed: str


def render_code(
    project_dir: str,
    config_name: str,
    target: Optional[str] = "main",
    **kwargs,
) -> str:
    """Generate Python source for *target* (or the entire config when ``None``).

    Mirrors what ``forgather code`` does on the CLI. ``target`` defaults to
    ``"main"`` to match the CLI; pass ``None`` to render every materialisable
    target in one document. Raises the same structured :class:`ConfigDiagnostic`
    subclasses as :meth:`ConfigEnvironment.render_code`, so the route can
    return the same JSON detail shape used by ``/api/config/pp``.
    """
    loaded = load_env(project_dir, config_name)
    merged = _merged_kwargs(project_dir, config_name, kwargs)
    return loaded.env.render_code(loaded.config_path, target=target, **merged)


def list_code_targets(project_dir: str, config_name: str) -> List[str]:
    """Return the top-level keys (materialisable targets) of the parsed config.

    Used by the **code** webui panel to populate the target list. The list
    matches what ``forgather targets`` prints. Raises the structured config
    diagnostics on failure, same as :func:`render_code`.
    """
    from collections.abc import Mapping as _Mapping

    loaded = load_env(project_dir, config_name)
    merged = _merged_kwargs(project_dir, config_name, {})
    config = loaded.env.load(loaded.config_path, **merged).config
    if not isinstance(config, _Mapping):
        return []
    return list(config.keys())


def render_pp_trace(
    project_dir: str, config_name: str, **kwargs
) -> List[DebugTraceItem]:
    """Return one DebugTraceItem per template that participated in the render.

    Drives :meth:`ConfigEnvironment.preprocess_with_trace` (which sets
    ``LineStatementProcessor.pp_capture`` for the duration of the call), then
    fetches the raw source + filesystem path of each template through Jinja2's
    standard loader API. Order matches load order.
    """
    loaded = load_env(project_dir, config_name)
    merged = _merged_kwargs(project_dir, config_name, kwargs)
    _, trace = loaded.env.preprocess_with_trace(loaded.config_path, **merged)

    pp_env = loaded.env.get_pp_environment()
    loader = pp_env.loader
    out: List[DebugTraceItem] = []
    seen: Set[str] = set()
    for name, preprocessed in trace:
        if name in seen:
            continue
        seen.add(name)
        raw = ""
        path = ""
        if loader is not None:
            try:
                raw, filename, _ = loader.get_source(pp_env, name)
                if filename:
                    path = os.path.abspath(filename)
            except Exception:
                # Inline / synthetic fragments may not resolve via the loader.
                pass
        out.append(
            DebugTraceItem(name=name, path=path, raw=raw, preprocessed=preprocessed)
        )
    return out


def render_trefs_dot(project_dir: str, config_name: str) -> str:
    loaded = load_env(project_dir, config_name)
    merged = _merged_kwargs(project_dir, config_name, {})
    return render_template_hierarchy_dot(loaded.env, loaded.config_path, **merged)


def render_trefs_tree(project_dir: str, config_name: str) -> str:
    loaded = load_env(project_dir, config_name)
    merged = _merged_kwargs(project_dir, config_name, {})
    return render_template_hierarchy_tree(loaded.env, loaded.config_path, **merged)


def render_trefs_json(project_dir: str, config_name: str) -> TrefsGraph:
    """Emit a structured template-dependency graph.

    Uses ``environment.get_template_dependencies`` which returns a
    ``(load_sequence, dependencies)`` pair: the load sequence gives every
    template's filesystem path, and ``dependencies`` maps parent -> set of
    child template names. Cached overrides are applied so dynamic-args-driven
    conditional includes (e.g. ``trainer_type`` selecting which trainer
    template gets included) show up in the graph.
    """
    loaded = load_env(project_dir, config_name)
    merged = _merged_kwargs(project_dir, config_name, {})
    load_sequence: List[Tuple[str, str]]
    dependencies: Dict[str, Set[str]]
    load_sequence, dependencies = loaded.env.get_template_dependencies(
        loaded.config_path, **merged
    )

    name_to_path = {name: path for name, path in load_sequence}
    names: Set[str] = set(name_to_path.keys())
    for parent, children in dependencies.items():
        names.add(parent)
        names.update(children)

    nodes = sorted(
        (
            TrefsNode(name=n, path=os.path.abspath(name_to_path.get(n, n)))
            for n in names
        ),
        key=lambda x: x.name,
    )
    edges: List[Tuple[str, str]] = []
    for parent, children in dependencies.items():
        for child in sorted(children):
            edges.append((parent, child))
    edges.sort()

    return TrefsGraph(nodes=nodes, edges=edges, root=loaded.config_path)


def load_config_meta(project_dir: str, config_name: str) -> ConfigMeta:
    """Materialize just the ``meta`` block of a config to pull out display name/description.

    Mirrors what ``forgather ls`` does per-config: preprocess + parse the
    config, then materialize ``config.meta`` to read ``config_name`` and
    ``config_description``. This is the expensive step (full Jinja2
    preprocessing + YAML parse), so callers should fetch lazily.
    """
    meta = ConfigMeta()
    try:
        loaded = load_env(project_dir, config_name)
        merged = _merged_kwargs(project_dir, config_name, {})
        config = get_config(loaded.meta, loaded.env, config_name, **merged)[0]
        config_meta = Latent.materialize(config.meta)
    except Exception as e:
        meta.parse_error = str(e)
        return meta

    if isinstance(config_meta, dict):
        meta.name = config_meta.get("config_name") or None
        meta.description = config_meta.get("config_description") or None
        meta.config_class = config_meta.get("config_class") or None
    return meta


def _dir_size_and_count(path: str) -> Tuple[int, int]:
    """Return ``(total_bytes, entry_count)`` for ``path`` — best-effort.

    Silent on per-entry errors (broken symlinks, permission-denied subdirs)
    because the only consumer is the Clean-Output UI which just wants to
    show a rough "how much will I delete" hint before asking for confirm.
    """
    total = 0
    count = 0
    for root, dirs, files in os.walk(path, followlinks=False, onerror=None):
        for name in files:
            count += 1
            try:
                total += os.stat(
                    os.path.join(root, name), follow_symlinks=False
                ).st_size
            except OSError:
                pass
        count += len(dirs)
    return total, count


def load_output_dir_info(project_dir: str, config_name: str) -> OutputDirInfo:
    """Resolve the config's default output_dir / models_dir and stat them.

    Goes through ``Project.config.meta`` — same path ``forgather tb`` uses —
    so this matches whatever the trainer will actually write to.
    """
    loaded = load_env(project_dir, config_name)
    try:
        merged = _merged_kwargs(project_dir, config_name, {})
        config = loaded.env.load(loaded.config_path, **merged).config
        config_meta = Latent.materialize(config.meta)
    except Exception as e:
        raise RuntimeError(f"failed to materialize config.meta: {e}") from e

    output_dir = os.path.abspath(config_meta["output_dir"])
    models_dir = os.path.abspath(config_meta["models_dir"])

    info = OutputDirInfo(
        output_dir=output_dir,
        models_dir=models_dir,
        output_dir_exists=os.path.isdir(output_dir),
        models_dir_exists=os.path.isdir(models_dir),
        nproc_per_node=config_meta.get("nproc_per_node"),
    )
    if info.output_dir_exists:
        info.output_dir_size_bytes, info.output_dir_entry_count = _dir_size_and_count(
            output_dir
        )
    if info.models_dir_exists:
        info.models_dir_size_bytes, info.models_dir_entry_count = _dir_size_and_count(
            models_dir
        )
    return info


def load_dynamic_args(project_dir: str, config_name: str) -> List[DynamicArg]:
    """Return the config's dynamic-args schema.

    Mirrors what ``forgather/cli/dynamic_args.py::parse_dynamic_args`` does
    internally, but without building an argparse parser — this is purely for
    rendering a form. We use ``Project("config")("dynamic_args")`` to
    materialize the list (same call the CLI makes), then normalise each
    entry to a ``DynamicArg`` with a single canonical CLI name and an
    argparse-style dest.
    """
    from forgather.project import Project

    try:
        proj = Project(config_name=config_name, project_dir=project_dir)
    except Exception:
        return []

    if "dynamic_args" not in proj.config:
        return []
    try:
        raw = proj("dynamic_args")
    except Exception:
        return []
    if not isinstance(raw, list):
        return []

    out: List[DynamicArg] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        names = entry.get("names")
        if isinstance(names, str):
            names = [names]
        elif not isinstance(names, list) or not names:
            continue

        # Prefer the long form for display; fall back to the first given.
        long_form = next((n for n in names if n.startswith("--")), names[0])
        dest = long_form.lstrip("-").replace("-", "_")
        type_str = entry.get("type", "str")
        if not isinstance(type_str, str):
            type_str = "str"
        default_value: Any = entry.get("default")

        # argparse store_true / store_false are presence-as-flag, not
        # value-bearing. Normalize them to type=bool with a *concrete*
        # default (False for store_true, True for store_false) so the
        # frontend can render a real checkbox instead of a tri-state
        # "(template default) / true / false" select. The action's default
        # wins over any explicit YAML ``default:`` to mirror what argparse
        # itself does — it ignores the YAML default for these actions.
        action = entry.get("action")
        if action == "store_true":
            type_str = "bool"
            default_value = False
        elif action == "store_false":
            type_str = "bool"
            default_value = True

        choices = entry.get("choices")
        if choices is not None and not isinstance(choices, list):
            choices = None

        group = entry.get("group")
        if group is not None and not isinstance(group, str):
            group = None
        required = bool(entry.get("required", False))

        # Numeric bounds only apply to numeric types; silently drop them
        # otherwise so a stray ``min`` on a string arg can't surface as
        # a confusing UI constraint.
        def _coerce_bound(v):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
            return None

        min_val = (
            _coerce_bound(entry.get("min")) if type_str in ("int", "float") else None
        )
        max_val = (
            _coerce_bound(entry.get("max")) if type_str in ("int", "float") else None
        )

        out.append(
            DynamicArg(
                dest=dest,
                cli_name=long_form,
                type=type_str,
                help=entry.get("help"),
                default=default_value,
                choices=choices,
                group=group,
                required=required,
                min=min_val,
                max=max_val,
            )
        )
    return out


def list_referenced_templates(project_dir: str, config_name: str):
    """Flat list of (level, name, abs_path) tuples consumed by the config."""
    loaded = load_env(project_dir, config_name)
    out = []
    for level, name, path in loaded.env.find_referenced_templates(loaded.config_path):
        out.append((level, name, os.path.abspath(path)))
    return out


@dataclass
class TemplatePaths:
    templates_dir: str
    configs_dir: str
    config_prefix: str


def project_template_paths(project_dir: str) -> TemplatePaths:
    """Resolve the project's templates dir and config-subdir paths so the
    UI can show a live "this is where the file will land" preview before
    the user submits a New Config / New Template prompt."""
    meta = MetaConfig(project_dir)
    if not meta.searchpath:
        raise RuntimeError("project has no template search path")
    base = os.path.abspath(meta.searchpath[0])
    return TemplatePaths(
        templates_dir=base,
        configs_dir=os.path.abspath(os.path.join(base, meta.config_prefix)),
        config_prefix=meta.config_prefix,
    )


def new_template_file(project_dir: str, kind: str, name: str) -> str:
    """Create an empty template / config file in ``project_dir``'s
    templates directory and return its absolute path.

    ``kind`` mirrors the CLI ``project new_config --type`` switch:
    - ``"config"``: under ``searchpath[0]/<config_prefix>/`` (the
      configs sub-tree, e.g. ``templates/configs/<name>``).
    - ``"template"``: under ``searchpath[0]/`` directly (e.g.
      ``templates/<name>``).

    A ``.yaml`` suffix is appended when ``name`` has none. Existing
    files are refused (no overwrite). Parent directories are created
    as needed. The file is written empty so the editor opens onto a
    blank canvas — same behavior as ``forgather project new_config``
    when no ``--copy-from`` is supplied (minus that command's default
    boilerplate, which would only get in the way for a from-scratch
    template the user is about to fill in).
    """
    if kind not in ("config", "template"):
        raise ValueError(f"unknown kind: {kind!r}")
    name = name.strip()
    if not name:
        raise ValueError("name is empty")
    if os.path.isabs(name):
        raise ValueError("name must be relative")
    # Reject obvious traversal. ``name`` is allowed to contain ``/`` so the
    # user can create nested files (``configs/foo/bar.yaml``), but no
    # ``..`` segments.
    parts = name.replace("\\", "/").split("/")
    if any(p in ("", "..", ".") for p in parts[:-1]) or parts[-1] in ("..", "."):
        raise ValueError("invalid name")
    if not os.path.splitext(name)[1]:
        name = name + ".yaml"

    meta = MetaConfig(project_dir)
    if not meta.searchpath:
        raise RuntimeError("project has no template search path")
    base = os.path.abspath(meta.searchpath[0])
    if kind == "config":
        target = os.path.abspath(os.path.join(base, meta.config_prefix, name))
    else:
        target = os.path.abspath(os.path.join(base, name))

    # Containment: target must remain under base (defends against a name
    # like ``configs/../../escape.yaml`` even though we filter ``..``
    # explicitly above — belt and suspenders).
    if os.path.commonpath([base, target]) != base:
        raise ValueError("name escapes the project's template directory")

    if os.path.exists(target):
        raise FileExistsError(target)

    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w") as f:
        f.write("")
    return target


@dataclass
class TemplateEntry:
    """One template file discovered on a project's search path."""

    name: str
    path: str  # absolute
    rel_path: str  # path made relative to the search-path root that owns it


@dataclass
class TemplateGroup:
    """Templates owned by one entry of a project's search path.

    ``category`` is the human-readable label the interactive CLI uses:
    "Workspace Templates", "Base Templates", "Project Templates", etc.
    ``search_path`` is the absolute search-root directory the group
    represents.
    """

    category: str
    search_path: str
    templates: List[TemplateEntry] = field(default_factory=list)


def _category_for_search_path(abs_path: str, project_dir: str) -> str:
    """Mirror of interactive.py::_get_search_path_categories logic.

    Kept as a pure helper so the same labels surface in the web UI as in the
    interactive CLI's `edit` selector.
    """
    if "forgather_workspace" in abs_path:
        return "Workspace Templates"
    if "templatelib/base" in abs_path:
        return "Base Templates"
    if "templatelib/examples" in abs_path:
        return "Example Templates"
    if "templatelib/" in abs_path:
        parts = abs_path.split("templatelib/")
        if len(parts) > 1:
            subdir = parts[1].split(os.sep)[0]
            return f"{subdir.replace('_', ' ').title()} Templates"
        return "Library Templates"
    if abs_path.endswith("templates"):
        parent = os.path.basename(os.path.dirname(abs_path))
        if parent == os.path.basename(os.path.abspath(project_dir)):
            return "Project Templates"
        return f"{parent.replace('_', ' ').title()} Templates"
    basename = os.path.basename(abs_path)
    return f"{basename.replace('_', ' ').title()} Templates"


def list_project_templates(project_dir: str) -> List[TemplateGroup]:
    """Enumerate every template on the project's search path, grouped by
    search-path entry. Mirrors the interactive CLI's `edit` selector.

    Templates that match more than one search path (e.g. when a project's
    own ``templates/`` and a library directory both contain a file with the
    same relative name) are attributed to the *first* matching search path
    in declaration order — same as how Jinja resolves them.
    """
    meta = MetaConfig(project_dir)
    search_paths: List[str] = [os.path.abspath(p) for p in meta.searchpath]

    groups: List[TemplateGroup] = []
    for sp in search_paths:
        groups.append(
            TemplateGroup(
                category=_category_for_search_path(sp, project_dir),
                search_path=sp,
                templates=[],
            )
        )

    seen_paths: set = set()
    for name, path in meta.find_templates():
        abs_path = os.path.abspath(path)
        if abs_path in seen_paths:
            continue
        seen_paths.add(abs_path)
        # Attribute to the first search-path entry that contains the file —
        # matches Jinja's first-match resolution.
        for grp in groups:
            try:
                rel = os.path.relpath(abs_path, grp.search_path)
            except ValueError:
                continue
            if rel.startswith(".."):
                continue
            grp.templates.append(TemplateEntry(name=name, path=abs_path, rel_path=rel))
            break

    # Sort templates inside each group by rel_path for stable display.
    for grp in groups:
        grp.templates.sort(key=lambda t: t.rel_path)
    # Drop empty groups so the UI doesn't have to filter them.
    groups = [g for g in groups if g.templates]

    # Prepend a synthetic "Meta" group with the project's meta.yaml so it
    # can be browsed / edited alongside templates. Done *after* attribution
    # so the project_dir search path doesn't sweep every project template
    # into this group (project_dir is on Jinja's searchpath but we only
    # surface meta.yaml here).
    meta_yaml = os.path.abspath(os.path.join(project_dir, "meta.yaml"))
    if os.path.isfile(meta_yaml):
        groups.insert(
            0,
            TemplateGroup(
                category="Meta",
                search_path=os.path.abspath(project_dir),
                templates=[
                    TemplateEntry(
                        name="meta.yaml",
                        path=meta_yaml,
                        rel_path="meta.yaml",
                    )
                ],
            ),
        )
    return groups
