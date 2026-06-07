"""Phase 0 read-only agent tools.

Each wraps an existing read path. None of these mutate state, so they all
register at ``risk="read"`` and run without an approval gate. They give the
agent enough to answer "what does this project / config do?", inspect
scheduler state, and find relevant documentation.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from .. import config_ops, discovery, paths, scheduler, search_roots
from .registry import READ, ToolRegistry, ToolSpec, UiDirective

log = logging.getLogger("forgather_server.agent.tools_readonly")


# ---- handlers --------------------------------------------------------------


def _list_workspaces(_args: Dict[str, Any]) -> Any:
    """Top level of the navigation tree: workspaces (no projects/configs)."""
    clusters = discovery.discover_projects()
    return [
        {
            "workspace_root": c.workspace_root,
            "name": c.name,
            "description": c.description,
            "parent_workspace_root": c.parent_workspace_root,
            "project_count": len(c.projects),
        }
        for c in clusters
    ]


def _project_summary(p) -> Dict[str, Any]:
    # Deliberately omit the per-project config list — fetch it with
    # list_configs so a broad listing stays small.
    return {
        "project_dir": p.project_dir,
        "name": p.name,
        "description": p.description,
        "default_config": p.default_config,
        "workspace_root": p.workspace_root,
        "config_count": len(p.configs),
        "parse_error": p.parse_error,
    }


def _list_projects(args: Dict[str, Any]) -> Any:
    """Project summaries (no config lists). Filter to one workspace with
    ``workspace_root``; omit it to summarize every project."""
    ws = args.get("workspace_root")
    out = []
    for c in discovery.discover_projects():
        if ws and c.workspace_root != ws:
            continue
        out.extend(_project_summary(p) for p in c.projects)
    return out


def _list_configs(args: Dict[str, Any]) -> Any:
    """The configs of one project (the leaf level of the nav tree)."""
    pi = discovery.load_project_info(args["project_dir"])
    return {
        "project_dir": pi.project_dir,
        "name": pi.name,
        "description": pi.description,
        "default_config": pi.default_config,
        "parse_error": pi.parse_error,
        "configs": [dataclasses.asdict(c) for c in pi.configs],
    }


def _inspect_config(args: Dict[str, Any]) -> Any:
    project_dir = args["project_dir"]
    config_name = args["config_name"]
    meta = config_ops.load_config_meta(project_dir, config_name)
    out: Dict[str, Any] = {"meta": dataclasses.asdict(meta)}
    # Dynamic args (config-specific CLI/form parameters) — empty list on
    # any failure, matching the function's own contract.
    try:
        out["dynamic_args"] = [dataclasses.asdict(a) for a in config_ops.load_dynamic_args(project_dir, config_name)]
    except Exception as e:  # defensive: never fail the whole inspect
        out["dynamic_args_error"] = str(e)
    try:
        out["code_targets"] = config_ops.list_code_targets(project_dir, config_name)
    except Exception as e:
        out["code_targets_error"] = str(e)
    return out


def _render_config_pp(args: Dict[str, Any]) -> Any:
    project_dir = args["project_dir"]
    config_name = args["config_name"]
    return config_ops.render_pp(project_dir, config_name)


def _render_config_code(args: Dict[str, Any]) -> Any:
    # forgather code: generated Python for a target ("main" by default; pass
    # null/"" for the whole config). A debug aid for understanding what a
    # config builds — NOT a normal pipeline stage. To merely check a config is
    # well-formed, use check_config (cheaper, and the right tool for it).
    target = args.get("target") or "main"
    return config_ops.render_code(args["project_dir"], args["config_name"], target=target)


def _check_config(args: Dict[str, Any]) -> Any:
    # forgather graph (validate-only): preprocess + YAML parse + build the node
    # graph, no materialization, no code. Returns {ok, targets} or {ok:false,
    # error}.
    return config_ops.check_config(
        args["project_dir"], args["config_name"], target=args.get("target") or None
    )


def _list_config_templates(args: Dict[str, Any]) -> Any:
    # forgather tlist: every template on the project's search path, grouped by
    # search-root — the set a config can `extends`/`include`.
    return [dataclasses.asdict(g) for g in config_ops.list_project_templates(args["project_dir"])]


def _config_template_refs(args: Dict[str, Any]) -> Any:
    # forgather trefs: the templates a config actually pulls in (inheritance
    # chain), as a readable tree.
    return config_ops.render_trefs_tree(args["project_dir"], args["config_name"])


def _known_projects_path(path: str) -> bool:
    """True if ``path`` is a known workspace root, project dir, or config file
    in the current discovery — i.e. something the Projects tree can reveal."""
    norm = path.rstrip("/")
    for cluster in discovery.discover_projects():
        if cluster.workspace_root and cluster.workspace_root.rstrip("/") == norm:
            return True
        for proj in cluster.projects:
            if proj.project_dir.rstrip("/") == norm:
                return True
            for cfg in proj.configs:
                if cfg.path.rstrip("/") == norm:
                    return True
    return False


def _reveal_in_ui(args: Dict[str, Any]) -> Any:
    # Steer the webui to expand to + highlight a path. Returns a UiDirective
    # the loop forwards to the client; no server-side side effect.
    path = args["path"]
    where = (args.get("where") or "projects").lower()
    if where not in ("projects", "files"):
        raise ValueError("where must be 'projects' or 'files'")
    if not os.path.isabs(path):
        raise ValueError("path must be absolute")
    if not paths.is_path_in_fs_root(path):
        raise PermissionError(
            f"path is outside the configured filesystem roots: {path}"
        )
    if not os.path.exists(path):
        raise ValueError(f"path does not exist: {path}")
    if where == "projects" and not _known_projects_path(path):
        raise ValueError(
            f"path is not a known workspace, project, or config: {path}. Use "
            "list_workspaces / list_projects / list_configs to get a valid "
            "target, or reveal it in the file explorer with where='files'."
        )
    label = "Projects tree" if where == "projects" else "file explorer"
    return UiDirective(
        action="reveal",
        payload={"path": path, "where": where},
        message=f"revealed {path} in the {label}.",
    )


def _read_file(args: Dict[str, Any]) -> Any:
    path = args["path"]
    if not os.path.isabs(path):
        raise ValueError("path must be absolute")
    if not paths.is_path_in_fs_root(path):
        raise PermissionError(f"path is outside the configured filesystem roots: {path}")
    return config_ops.read_raw(path)


# ---- filesystem browse / find ----------------------------------------------

# Directories never worth walking into for find/browse purposes.
_FS_SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ipynb_checkpoints",
}
_FIND_MAX_RESULTS = 100
_FIND_DIR_BUDGET = 50000  # cap directories walked, so a huge tree can't run away
_FIND_SCAN_BUDGET = 300000  # cap total entries examined (one huge flat dir too)


def _starting_roots() -> List[str]:
    """The places worth starting a browse/find from: the project search roots
    (where projects, tokenizers, datasets live) plus any configured fs-root
    sandbox dirs. Deduped, existing only."""
    out: List[str] = []
    for r in search_roots.list_roots():
        if r.exists and r.path not in out:
            out.append(r.path)
    for p in paths.fs_roots():
        s = str(p)
        if s not in out:
            out.append(s)
    return out


def _list_directory(args: Dict[str, Any]) -> Any:
    # No path -> hand back the starting roots so the agent knows where to look.
    path = (args.get("path") or "").strip()
    if not path:
        return {
            "roots": _starting_roots(),
            "note": "Pass one of these (or any absolute path within them) as "
            "'path' to list its contents; use find_files to search by name.",
        }
    if not os.path.isabs(path):
        raise ValueError("path must be absolute")
    if not paths.is_path_in_fs_root(path):
        raise PermissionError(f"path is outside the configured filesystem roots: {path}")
    p = Path(path)
    if not p.exists():
        raise ValueError(f"path does not exist: {path}")
    if not p.is_dir():
        raise ValueError(f"not a directory: {path}")
    entries: List[Dict[str, Any]] = []
    try:
        children = list(p.iterdir())
    except PermissionError as e:
        raise PermissionError(str(e))
    for child in sorted(children, key=lambda c: c.name.lower()):
        if child.name.startswith("."):
            continue
        try:
            is_dir = child.is_dir()
            resolved = child.resolve()
        except OSError:
            continue
        if not paths.is_path_in_fs_root(resolved):  # symlink escape guard
            continue
        e: Dict[str, Any] = {"name": child.name, "path": str(resolved), "is_dir": is_dir}
        if not is_dir:
            try:
                e["size"] = child.stat().st_size
            except OSError:
                pass
        entries.append(e)
    # Directories first, then files, each alphabetical.
    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
    return {"path": str(p.resolve()), "entries": entries}


def _find_files(args: Dict[str, Any]) -> Any:
    pattern = (args.get("pattern") or "").strip()
    if not pattern:
        raise ValueError("pattern is required (e.g. 'wikitext*' or 'tokenizer')")
    # A bare word (no glob metachars) is treated as a substring match — the
    # find-like behavior most callers expect.
    glob = pattern if any(ch in pattern for ch in "*?[") else f"*{pattern}*"
    glob = glob.lower()
    max_results = args.get("max_results")
    max_results = _FIND_MAX_RESULTS if max_results in (None, "") else int(max_results)
    max_results = max(1, min(max_results, 500))

    root = (args.get("root") or "").strip()
    if root:
        if not os.path.isabs(root):
            raise ValueError("root must be absolute")
        if not paths.is_path_in_fs_root(root):
            raise PermissionError(
                f"path is outside the configured filesystem roots: {root}"
            )
        search_dirs = [root]
    else:
        search_dirs = _starting_roots()

    matches: List[Dict[str, Any]] = []
    walked = 0
    examined = 0  # total entries fnmatch'd — bounds CPU on one huge flat dir
    truncated = False
    for base in search_dirs:
        for dirpath, dirnames, filenames in os.walk(base):
            # Prune noise + hidden dirs from the descent (in place).
            dirnames[:] = [
                d for d in dirnames if d not in _FS_SKIP_DIRS and not d.startswith(".")
            ]
            walked += 1
            if walked > _FIND_DIR_BUDGET:
                truncated = True
                break
            for name in dirnames + filenames:
                examined += 1
                if examined > _FIND_SCAN_BUDGET:
                    truncated = True
                    break
                if name.startswith("."):
                    continue
                if not fnmatch.fnmatch(name.lower(), glob):
                    continue
                full = os.path.join(dirpath, name)
                if not paths.is_path_in_fs_root(full):
                    continue
                matches.append({"path": full, "is_dir": os.path.isdir(full)})
                if len(matches) >= max_results:
                    truncated = True
                    break
            if truncated:
                break
        if truncated:
            break
    return {"pattern": pattern, "matches": matches, "truncated": truncated}


def _scheduler_status(_args: Dict[str, Any]) -> Any:
    state = scheduler.get_state()
    return {
        "enabled": state.enabled,
        "running_job_ids": sorted(state.running.keys()),
        "running_count": len(state.running),
        "tick_count": state.tick_count,
        "last_tick_at": state.last_tick_at,
    }


# ---- docs search -----------------------------------------------------------

_DOC_SUFFIXES = (".md", ".markdown")
_MAX_HITS = 8
_EXCERPT_RADIUS = 240


def _doc_roots() -> List[Path]:
    repo = Path(search_roots.forgather_repo_root())
    return [repo / "docs", repo / "CLAUDE.d", repo / "CLAUDE.md"]


def _iter_doc_files() -> List[Path]:
    files: List[Path] = []
    for root in _doc_roots():
        if root.is_file() and root.suffix.lower() in _DOC_SUFFIXES:
            files.append(root)
        elif root.is_dir():
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in _DOC_SUFFIXES:
                    files.append(p)
    return files


def _search_docs(args: Dict[str, Any]) -> Any:
    """Keyword search over docs/ + CLAUDE.d/ returning ranked excerpts.

    Deliberately simple (substring scoring, no embeddings): the model reads
    the excerpts and decides what's relevant / which file to point the user
    to. Embeddings can replace this later without changing the tool contract.
    """
    query = (args.get("query") or "").strip()
    if not query:
        raise ValueError("query is empty")
    terms = [t.lower() for t in query.split() if t]

    scored = []
    for path in _iter_doc_files():
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        low = text.lower()
        score = sum(low.count(t) for t in terms)
        if score == 0:
            continue
        # Build an excerpt around the first matching term.
        idx = min((low.find(t) for t in terms if low.find(t) >= 0), default=-1)
        if idx < 0:
            continue
        start = max(0, idx - _EXCERPT_RADIUS)
        end = min(len(text), idx + _EXCERPT_RADIUS)
        excerpt = text[start:end].strip()
        scored.append({"path": str(path), "score": score, "excerpt": excerpt})

    scored.sort(key=lambda h: h["score"], reverse=True)
    return {"query": query, "hits": scored[:_MAX_HITS]}


# ---- registration ----------------------------------------------------------


def register_all(reg: ToolRegistry) -> None:
    reg.register(
        ToolSpec(
            name="list_workspaces",
            description=(
                "List Forgather workspaces (top of the navigation tree) — name, "
                "description, and project_count, no projects/configs. Start here, "
                "then list_projects(workspace_root) and list_configs(project_dir). "
                "Mirrors the Projects sidebar; keeps listings small."
            ),
            json_schema={"type": "object", "properties": {}},
            handler=_list_workspaces,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="list_projects",
            description=(
                "List project summaries (project_dir, name, description, "
                "default_config, config_count) — NOT their config lists. Pass "
                "workspace_root to list one workspace's projects; omit it to "
                "summarize all. Use list_configs(project_dir) for a project's "
                "configs."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "workspace_root": {
                        "type": "string",
                        "description": "Optional: limit to this workspace (from list_workspaces).",
                    }
                },
            },
            handler=_list_projects,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="list_configs",
            description=(
                "List the configs of one project (name, path, is_default). Use "
                "the project_dir from list_projects, then inspect_config for "
                "details on a specific config."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string", "description": "Absolute project directory."}
                },
                "required": ["project_dir"],
            },
            handler=_list_configs,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="inspect_config",
            description=(
                "Inspect one config: its meta (name/description/class, or a "
                "parse_error), its dynamic-args schema (config-specific "
                "parameters), and its code targets."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string", "description": "Absolute project directory."},
                    "config_name": {"type": "string", "description": "Config file name within the project."},
                },
                "required": ["project_dir", "config_name"],
            },
            handler=_inspect_config,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="render_config_pp",
            description=(
                "Render a config's preprocessor output: the configuration text "
                "after template inheritance (stage 1 of the pipeline, Jinja2 "
                "only). Use to see exactly what templates resolve to. NOTE: this "
                "is a pure text render — it does NOT parse the YAML, validate the "
                "`!` tags, or check that the config compiles. For that, use "
                "check_config."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string"},
                    "config_name": {"type": "string"},
                },
                "required": ["project_dir", "config_name"],
            },
            handler=_render_config_pp,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="render_config_code",
            description=(
                "Translate a config into the equivalent stand-alone Python "
                "(forgather code) — the code that, if run, would build the same "
                "objects, with no Forgather dependency. This is a DEBUG / export "
                "tool, not a pipeline stage: use it to understand what a config "
                "actually builds, or to export a config as plain Python. Do NOT "
                "use it just to check whether a config is valid — that is what "
                "check_config is for (cheaper and purpose-built). target "
                "defaults to \"main\"."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string"},
                    "config_name": {"type": "string"},
                    "target": {"type": "string", "description": "Output target (default \"main\"; see list via inspect_config code_targets)."},
                },
                "required": ["project_dir", "config_name"],
            },
            handler=_render_config_code,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="check_config",
            description=(
                "Validate that a config compiles to a valid node graph "
                "(forgather graph, validate-only). Runs the real pipeline up "
                "to the graph stage: preprocess the templates, parse the YAML, "
                "and build the node graph (resolving the `!` tags). It does NOT "
                "construct any objects and does NOT generate code. This is the "
                "right, cheapest way to answer \"does this config compile?\" "
                "after you write or edit one. Returns {ok:true, targets:[...]} "
                "on success, or {ok:false, error, error_type} with a structured "
                "diagnostic on a malformed config."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string"},
                    "config_name": {"type": "string"},
                    "target": {"type": "string", "description": "Optional: also verify this target key exists in the graph."},
                },
                "required": ["project_dir", "config_name"],
            },
            handler=_check_config,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="reveal_in_ui",
            description=(
                "Reveal a workspace, project, or config to the user by "
                "expanding the UI to it and selecting it — useful after you "
                "locate something they asked for (e.g. \"show me a project "
                "that does X\"). where=\"projects\" (default) expands the "
                "Projects tree to the item; where=\"files\" expands the file "
                "explorer and highlights the path. Give an absolute path from "
                "list_workspaces / list_projects / list_configs (for "
                "\"projects\") or read_file results. This only navigates the "
                "UI; it changes nothing."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the workspace dir, project dir, or config file."},
                    "where": {"type": "string", "enum": ["projects", "files"], "description": "Which view to reveal in (default \"projects\")."},
                },
                "required": ["path"],
            },
            handler=_reveal_in_ui,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="list_config_templates",
            description=(
                "List every template on a project's search path (forgather "
                "tlist), grouped by search-root — the templates a config can "
                "`extends`/`include`. Use before writing a config to find the "
                "right base template to inherit from."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string", "description": "Absolute project directory."}
                },
                "required": ["project_dir"],
            },
            handler=_list_config_templates,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="config_template_refs",
            description=(
                "Show the templates a config actually pulls in — its inheritance "
                "chain (forgather trefs) — as a tree. Use to understand or debug "
                "where a config's values come from."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string"},
                    "config_name": {"type": "string"},
                },
                "required": ["project_dir", "config_name"],
            },
            handler=_config_template_refs,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="read_file",
            description=(
                "Read a file by absolute path (must be within the server's "
                "configured filesystem roots). Use for template/config source."
            ),
            json_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Absolute path."}},
                "required": ["path"],
            },
            handler=_read_file,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="list_directory",
            description=(
                "List the contents of a directory (names, whether each is a "
                "dir, file sizes). Use to walk the filesystem — e.g. to find a "
                "tokenizer under tokenizers/, a model output dir, or a data "
                "file — when it isn't a Forgather project/config (those use "
                "list_projects / list_configs). Call with no path to get the "
                "starting roots (the project search roots + any fs-root "
                "sandbox), then drill in. Paths must be within the configured "
                "filesystem roots."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute directory path; omit to list the starting roots."},
                },
            },
            handler=_list_directory,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="find_files",
            description=(
                "Find files and directories by name, recursively (like UNIX "
                "find). pattern is a glob (e.g. 'wikitext*', '*.yaml') or a "
                "bare word, which matches as a substring (e.g. 'tokenizer' "
                "finds anything containing 'tokenizer'). Searches under root if "
                "given, else across all starting roots (project search roots + "
                "fs-root sandbox). Matches directories too — a tokenizer is a "
                "directory. Results are capped (max_results, default 100; "
                "truncated=true means there were more). Use to locate a "
                "tokenizer, dataset, model, or config when you don't know its "
                "exact path."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob (e.g. 'wikitext*') or a bare word matched as a substring."},
                    "root": {"type": "string", "description": "Absolute dir to search under (default: all starting roots)."},
                    "max_results": {"type": "integer", "description": "Cap on matches (default 100, max 500)."},
                },
                "required": ["pattern"],
            },
            handler=_find_files,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="scheduler_status",
            description="Report the job scheduler's enabled flag and currently running jobs.",
            json_schema={"type": "object", "properties": {}},
            handler=_scheduler_status,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="search_docs",
            description=(
                "Search Forgather documentation (docs/, CLAUDE.d/, CLAUDE.md) "
                "for a query and return ranked excerpts with file paths. Use to "
                "answer questions about Forgather and to cite the right doc."
            ),
            json_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search terms."}},
                "required": ["query"],
            },
            handler=_search_docs,
            risk=READ,
        )
    )
