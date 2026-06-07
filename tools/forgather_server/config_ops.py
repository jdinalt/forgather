"""Operations on Forgather configuration templates, wrapping existing APIs.

The server exposes these through HTTP endpoints; the pure-Python functions
here make it easier to test and reuse.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from forgather.cli.trefs import (
    render_template_hierarchy_dot,
    render_template_hierarchy_tree,
)
from forgather.cli.utils import get_config, get_env
from forgather.config import ConfigEnvironment
from forgather.latent import Latent
from forgather.meta_config import MetaConfig

from . import _atomic, meta_templates, overrides_store, paths


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


def render_graph_dot(
    project_dir: str,
    config_name: str,
    target: Optional[str] = None,
    include_values: bool = False,
    **kwargs,
) -> str:
    """Render the config's parsed node graph as Graphviz DOT.

    When *target* is given only that top-level key is rendered; when ``None``
    every top-level target appears in the same diagram, each with its own
    entry-point ellipse.

    When *include_values* is True, plain Python scalars and containers
    (str / int / float / bool / None / list / dict) also appear as nodes,
    with strings truncated to a sane length. The default skips them so the
    diagram is dominated by the Forgather node graph rather than its
    constants.
    """
    from collections.abc import Mapping as _Mapping
    from collections.abc import Sequence as _Sequence
    from typing import Any as _Any

    from forgather.latent import (
        CallableNode,
        FactoryNode,
        Node,
        PartialNode,
        SingletonNode,
        VarNode,
    )

    loaded = load_env(project_dir, config_name)
    merged = _merged_kwargs(project_dir, config_name, kwargs)
    config = loaded.env.load(loaded.config_path, **merged).config

    if isinstance(config, _Mapping):
        if target and target in config:
            roots: Dict[str, _Any] = {target: config[target]}
        else:
            roots = dict(config)
    else:
        roots = {"root": config}

    node_defs: List[str] = []
    edge_defs: List[str] = []
    entry_defs: List[str] = []
    node_by_key: Dict[Any, str] = {}
    emitted: Set[Any] = set()  # node keys already in node_defs
    traversed: Set[Any] = set()  # node keys whose children have been walked
    counter: List[int] = [0]

    def fresh_id() -> str:
        nid = f"n{counter[0]}"
        counter[0] += 1
        return nid

    def _key(node: Node) -> Any:
        # VarNode identities default to id(self), so two !var references to
        # the same variable end up as separate Python objects with distinct
        # identities. Dedupe them by variable name instead so a "hidden_size"
        # var shows up once with edges from every consumer.
        if isinstance(node, VarNode):
            return ("var", node.constructor)
        return ("node", node.identity)

    def _nid(node: Node) -> str:
        key = _key(node)
        if key not in node_by_key:
            node_by_key[key] = fresh_id()
        return node_by_key[key]

    def _short(constructor: str) -> str:
        rhs = constructor.rsplit(":", 1)[-1]
        parts = rhs.rsplit(".", 2)
        return ".".join(parts[-2:]) if len(parts) >= 2 else rhs

    def _esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    def _edge(src: str, dst: str, label: str) -> str:
        if label:
            return f'  {src} -> {dst} [label="{_esc(label)}"];'
        return f"  {src} -> {dst};"

    _TYPE_COLOR = {
        VarNode: "#aed6f1",
        SingletonNode: "#a9dfbf",
        FactoryNode: "#f9e79f",
        PartialNode: "#f1948a",
    }
    _TYPE_LABEL = {
        VarNode: "var",
        SingletonNode: "singleton",
        FactoryNode: "factory",
        PartialNode: "partial",
    }

    def emit_node(obj: Node) -> str:
        """Add obj to node_defs if not already done; always return its DOT id."""
        key = _key(obj)
        nid = _nid(obj)
        if key not in emitted:
            emitted.add(key)
            color = next(
                (c for t, c in _TYPE_COLOR.items() if isinstance(obj, t)), "#cccccc"
            )
            type_label = next(
                (lbl for t, lbl in _TYPE_LABEL.items() if isinstance(obj, t)), "node"
            )
            raw_label = (
                f"{type_label}\n{obj.constructor}"
                if isinstance(obj, VarNode)
                else f"{type_label}\n{_short(obj.constructor)}"
            )
            node_defs.append(
                f'  {nid} [label="{_esc(raw_label)}" fillcolor="{color}"];'
            )
        return nid

    def _format_scalar(value: Any, max_len: int = 60) -> str:
        """Short, human-readable label for a scalar value."""
        if isinstance(value, str):
            if len(value) > max_len:
                return repr(value[:max_len]) + "…"
            return repr(value)
        if value is None:
            return "None"
        return repr(value)

    def emit_value_node(label: str, shape: str, color: str) -> str:
        nid = fresh_id()
        node_defs.append(
            f'  {nid} [label="{_esc(label)}" shape={shape}'
            f' style="filled,rounded" fillcolor="{color}"];'
        )
        return nid

    def walk(
        obj: Any, parent_nid: Optional[str], edge_label: str, visited: Set[Any]
    ) -> None:
        if isinstance(obj, Node):
            key = _key(obj)
            nid = emit_node(obj)
            if parent_nid is not None:
                edge_defs.append(_edge(parent_nid, nid, edge_label))
            # Skip recursing if: already in the current traversal stack (cycle)
            # or already fully walked by a previous path (avoids duplicate edges).
            if key in visited or key in traversed:
                return
            visited = visited | {key}
            if isinstance(obj, CallableNode):
                for i, arg in enumerate(obj.args):
                    walk(arg, nid, f"arg{i}", visited)
                for kw, val in (obj.kwargs or {}).items():
                    walk(val, nid, kw, visited)
            traversed.add(key)
        elif isinstance(obj, _Mapping) and not isinstance(obj, str):
            if include_values:
                nid = emit_value_node(f"{{ ... }} ({len(obj)})", "box", "#d6eaf8")
                if parent_nid is not None:
                    edge_defs.append(_edge(parent_nid, nid, edge_label))
                for k, val in obj.items():
                    walk(val, nid, str(k), visited)
            else:
                for k, val in obj.items():
                    sub = f"{edge_label}.{k}" if edge_label else str(k)
                    walk(val, parent_nid, sub, visited)
        elif isinstance(obj, _Sequence) and not isinstance(obj, str):
            if include_values:
                nid = emit_value_node(f"[ ... ] ({len(obj)})", "box", "#d6eaf8")
                if parent_nid is not None:
                    edge_defs.append(_edge(parent_nid, nid, edge_label))
                for i, val in enumerate(obj):
                    walk(val, nid, f"[{i}]", visited)
            else:
                for i, val in enumerate(obj):
                    sub = f"{edge_label}[{i}]" if edge_label else f"[{i}]"
                    walk(val, parent_nid, sub, visited)
        elif include_values and parent_nid is not None:
            # Plain scalar leaf: int / float / bool / None / str.
            nid = emit_value_node(_format_scalar(obj), "note", "#fdf2e9")
            edge_defs.append(_edge(parent_nid, nid, edge_label))

    for tname, tobj in roots.items():
        entry_id = fresh_id()
        entry_defs.append(
            f'  {entry_id} [label="{_esc(tname)}" shape=ellipse '
            f'fillcolor="#d2b4de" style="filled"];'
        )
        walk(tobj, entry_id, "", set())

    lines = [
        "digraph {",
        "  rankdir=LR;",
        '  node [fontname="monospace" fontsize=10 shape=box'
        ' style="filled,rounded" margin="0.15,0.10"];',
        "  edge [fontsize=9 arrowsize=0.7];",
    ]
    lines.extend(entry_defs)
    lines.extend(node_defs)
    lines.extend(edge_defs)
    lines.append("}")
    return "\n".join(lines)


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


def resolve_new_template_target(
    project_dir: str,
    kind: str,
    name: str,
    *,
    meta_template: Optional[str] = None,
    values: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Pure: validate inputs and compute ``(target_path, content)``.

    Does **not** touch disk. Performs all the validation and rendering
    ``new_template_file`` used to do inline — name sanitation, search-path
    resolution, containment, the no-overwrite check, and (when
    ``meta_template`` is given) rendering the scaffold against ``values``.
    Raises the same errors (``ValueError`` / ``RuntimeError`` /
    ``FileExistsError``).

    Split out so callers can PREVIEW a new file (compute path + content)
    and gate the actual write — the agent's propose/commit flow relies on
    this separation; the write step is :func:`write_template_file`.

    ``kind`` mirrors the CLI ``project new_config --type`` switch:
    - ``"config"``: under ``searchpath[0]/<config_prefix>/`` (the
      configs sub-tree, e.g. ``templates/configs/<name>``).
    - ``"template"``: under ``searchpath[0]/`` directly.

    A ``.yaml`` suffix is appended when ``name`` has none. When
    ``meta_template`` is omitted the content is the empty string (a blank
    canvas), matching ``forgather project new_config`` with no
    ``--copy-from``.
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

    if meta_template:
        content = meta_templates.render(meta_template, values or {})
    else:
        content = ""

    return target, content


def write_template_file(target: str, content: str) -> str:
    """Atomically create a new file at ``target`` and return ``target``.

    The write step paired with :func:`resolve_new_template_target`. Uses
    the crash-atomic helper (tmp + fsync + rename) rather than a bare
    ``open(..., "w")`` so a freshly-created file is never visible
    half-written.

    Re-checks the no-overwrite and fs-root invariants **at write time**, not
    just in the resolver: for the agent's propose→approve flow there is a gap
    between previewing (resolve) and committing (write), during which the
    file could come to exist; refusing here keeps the no-clobber guarantee
    across the approval gate.
    """
    if not paths.is_path_in_fs_root(target):
        raise PermissionError(
            f"path is outside the configured filesystem roots: {target}"
        )
    if os.path.exists(target):
        raise FileExistsError(target)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    _atomic.atomic_write_text(Path(target), content)
    return target


class StaleEditError(RuntimeError):
    """Raised when an existing-file overwrite would clobber an external edit.

    Carries the on-disk and expected mtimes so callers can surface a
    keep-mine / reload prompt — the same optimistic-concurrency signal the
    ``PUT /api/template/source`` route returns as HTTP 409.
    """

    def __init__(self, path: str, current_mtime: float, expected_mtime: float):
        super().__init__(
            f"{path} changed on disk since it was read "
            f"(current_mtime={current_mtime}, expected_mtime={expected_mtime})"
        )
        self.path = path
        self.current_mtime = current_mtime
        self.expected_mtime = expected_mtime


def write_existing_file(
    path: str, content: str, *, expected_mtime: Optional[float] = None
) -> Dict[str, Any]:
    """Atomically overwrite an existing file, with fs-root + mtime guards.

    Mirrors the ``PUT /api/template/source`` route's logic
    (``_enforce_fs_root`` + optimistic ``expected_mtime`` check + atomic
    write) but raises plain exceptions instead of ``HTTPException`` so it is
    usable from the agent tool layer as well as routes. Returns
    ``{path, bytes_written, mtime}``.
    """
    if not os.path.isabs(path):
        raise ValueError("path must be absolute")
    if not paths.is_path_in_fs_root(path):
        raise PermissionError(
            f"path is outside the configured filesystem roots: {path}"
        )
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if not p.is_file():
        raise ValueError(f"not a regular file: {path}")
    if expected_mtime is not None:
        current = os.path.getmtime(path)
        # Tiny tolerance for float roundtripping (ns-precision filesystems).
        if current > expected_mtime + 1e-6:
            raise StaleEditError(path, current, expected_mtime)
    _atomic.atomic_write_text(p, content)
    try:
        new_mtime = os.path.getmtime(path)
    except OSError:
        new_mtime = 0.0
    return {
        "path": path,
        "bytes_written": len(content.encode("utf-8")),
        "mtime": new_mtime,
    }


def new_template_file(
    project_dir: str,
    kind: str,
    name: str,
    *,
    meta_template: Optional[str] = None,
    values: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a template / config file and return its absolute path.

    Thin wrapper over :func:`resolve_new_template_target` +
    :func:`write_template_file`, preserved for existing callers (the
    ``project new_config`` route). New code that needs a preview/commit
    split should call the two halves directly.
    """
    target, content = resolve_new_template_target(
        project_dir, kind, name, meta_template=meta_template, values=values
    )
    return write_template_file(target, content)


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
