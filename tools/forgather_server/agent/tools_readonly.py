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


def _list_projects(_args: Dict[str, Any]) -> Any:
    clusters = discovery.discover_projects()
    return [dataclasses.asdict(c) for c in clusters]


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
            name="list_projects",
            description=(
                "List all discovered Forgather workspaces and projects with "
                "their configs. Use this to find the project_dir and "
                "config_name values other tools need."
            ),
            json_schema={"type": "object", "properties": {}},
            handler=_list_projects,
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
