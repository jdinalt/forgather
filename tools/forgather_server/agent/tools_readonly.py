"""Phase 0 read-only agent tools.

Each wraps an existing read path. None of these mutate state, so they all
register at ``risk="read"`` and run without an approval gate. They give the
agent enough to answer "what does this project / config do?", inspect
scheduler state, and find relevant documentation.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from .. import config_ops, discovery, paths, scheduler, search_roots
from .registry import READ, ToolRegistry, ToolSpec

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
    # null/"" for the whole config). Raises the same structured diagnostics
    # the CLI does on a broken config — exactly what helps the agent debug.
    target = args.get("target") or "main"
    return config_ops.render_code(args["project_dir"], args["config_name"], target=target)


def _list_config_templates(args: Dict[str, Any]) -> Any:
    # forgather tlist: every template on the project's search path, grouped by
    # search-root — the set a config can `extends`/`include`.
    return [dataclasses.asdict(g) for g in config_ops.list_project_templates(args["project_dir"])]


def _config_template_refs(args: Dict[str, Any]) -> Any:
    # forgather trefs: the templates a config actually pulls in (inheritance
    # chain), as a readable tree.
    return config_ops.render_trefs_tree(args["project_dir"], args["config_name"])


def _read_file(args: Dict[str, Any]) -> Any:
    path = args["path"]
    if not os.path.isabs(path):
        raise ValueError("path must be absolute")
    if not paths.is_path_in_fs_root(path):
        raise PermissionError(f"path is outside the configured filesystem roots: {path}")
    return config_ops.read_raw(path)


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
                "Render a config's preprocessor output (the fully-materialized "
                "configuration after template inheritance). Use to understand "
                "exactly what a config resolves to."
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
                "Generate the Python a config materializes to (forgather code). "
                "The best validation that a config is well-formed: it raises a "
                "structured error pinpointing what's wrong, so use it to debug "
                "a config you wrote or edited. target defaults to \"main\"."
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
