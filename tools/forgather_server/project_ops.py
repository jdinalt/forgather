"""Create workspaces and projects, wrapping the CLI's ws/project commands.

Factored out so the same logic backs the HTTP endpoints (routes/projects.py)
and the agent's authoring tools — the agent shouldn't have to hand-build
``meta.yaml`` / ``forgather_workspace/`` by guessing the layout. Functions
raise plain exceptions (``ValueError`` / ``FileExistsError`` /
``PermissionError`` / ``RuntimeError``); callers map them (the route to HTTP
status codes, the agent to tool errors).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from . import meta_templates, paths, search_roots


def _enforce_fs_root(path: str) -> None:
    if not paths.is_path_in_fs_root(path):
        raise PermissionError(f"path is outside the configured filesystem roots: {path}")


# ---- projects --------------------------------------------------------------


def resolve_new_project_target(
    workspace_dir: str, name: str, project_dir_name: Optional[str] = None
) -> Tuple[str, List[str]]:
    """Validate inputs and compute the project's target dir (no creation).

    Returns ``(target_dir, parts)``. Raises on bad name / traversal / a path
    outside the fs-root allowlist / an existing target.
    """
    if not (name or "").strip():
        raise ValueError("name is required")
    pdn = project_dir_name or name.replace(" ", "_").lower()
    if not pdn:
        raise ValueError("empty project_dir_name")
    if os.path.isabs(pdn):
        raise ValueError("project_dir_name must be relative")
    parts = pdn.replace("\\", "/").strip("/").split("/")
    if any(p in ("", "..", ".") for p in parts):
        raise ValueError("project_dir_name has invalid path segments")
    ws = os.path.abspath(workspace_dir)
    _enforce_fs_root(ws)
    target = os.path.abspath(os.path.join(ws, *parts))
    if os.path.commonpath([ws, target]) != ws:
        raise ValueError("project_dir_name escapes workspace_dir")
    if os.path.exists(target):
        raise FileExistsError(target)
    return target, parts


def create_project(
    *,
    workspace_dir: str,
    name: str,
    description: str,
    project_dir_name: Optional[str] = None,
    config_prefix: str = "configs",
    default_config: str = "default.yaml",
    copy_from: Optional[str] = None,
    meta_template: Optional[str] = None,
    values: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a project (dir + README + meta.yaml + default config) under
    ``workspace_dir``; return the created project directory.

    Mirrors ``forgather project create``. The default config is seeded from a
    ``meta_template`` scaffold, a ``copy_from`` file, or the built-in empty
    stub (the three are the New Project modal's "starting point"; copy_from
    and meta_template are mutually exclusive).
    """
    if not (description or "").strip():
        raise ValueError("description is required")
    target, _parts = resolve_new_project_target(workspace_dir, name, project_dir_name)
    if copy_from and meta_template:
        raise ValueError("copy_from and meta_template are mutually exclusive")
    if copy_from:
        _enforce_fs_root(copy_from)
        if not os.path.isfile(copy_from):
            raise ValueError(f"copy_from is not a file: {copy_from}")
    seed_text: Optional[str] = None
    if meta_template:
        # May raise MissingFieldsError (ValueError) / KeyError — propagate.
        seed_text = meta_templates.render(meta_template, values or {})

    from forgather.cli.project import project_create_cmd

    args = SimpleNamespace(
        project_dir=workspace_dir,
        project_dir_name=project_dir_name,
        name=name,
        description=description,
        config_prefix=config_prefix or "configs",
        default_config=default_config or "default.yaml",
        copy_from=copy_from,
        seed_text=seed_text,
    )
    rc = project_create_cmd(args)
    if rc not in (0, None):
        raise RuntimeError(f"project_create_cmd returned {rc}")
    return target


# ---- workspaces ------------------------------------------------------------


def resolve_new_workspace_target(
    parent_dir: str, name: str, workspace_dir_name: Optional[str] = None
) -> str:
    """Validate + compute the workspace's target dir (no creation)."""
    if not (name or "").strip():
        raise ValueError("name is required")
    wsname = (workspace_dir_name or name.replace(" ", "_").lower().replace(".", "")).strip("/")
    if not wsname or os.path.isabs(wsname):
        raise ValueError("invalid workspace_dir_name")
    parts = wsname.replace("\\", "/").split("/")
    if any(p in ("", "..", ".") for p in parts):
        raise ValueError("workspace_dir_name has invalid path segments")
    parent = os.path.abspath(parent_dir)
    _enforce_fs_root(parent)
    workspace_dir = os.path.abspath(os.path.join(parent, *parts))
    if os.path.commonpath([parent, workspace_dir]) != parent:
        raise ValueError("workspace_dir_name escapes parent_dir")
    if os.path.exists(workspace_dir):
        raise FileExistsError(workspace_dir)
    return workspace_dir


def create_workspace(
    *,
    parent_dir: str,
    name: str,
    description: str,
    workspace_dir_name: Optional[str] = None,
    forgather_dir: Optional[str] = None,
    libs: Optional[List[str]] = None,
    search_paths: Optional[List[str]] = None,
) -> str:
    """Create a workspace (dir + ``forgather_workspace/`` metadata) under
    ``parent_dir``; return the created workspace directory.

    Mirrors ``forgather ws create``. ``forgather_dir`` defaults to the
    Forgather repo root (where the base templates live) so the agent doesn't
    have to know it.
    """
    if not (description or "").strip():
        raise ValueError("description is required")
    parent = os.path.abspath(parent_dir)
    workspace_dir = resolve_new_workspace_target(parent_dir, name, workspace_dir_name)

    fdir = forgather_dir or search_roots.forgather_repo_root()
    if not fdir or not os.path.isdir(fdir):
        raise ValueError(f"forgather_dir is not a directory: {fdir}")

    from forgather.cli.workspace import ws_create_cmd

    args = SimpleNamespace(
        workspace_dir=workspace_dir,
        project_dir=parent,
        name=name,
        description=description,
        forgather_dir=fdir,
        lib=list(libs) if libs else None,
        search_path=list(search_paths) if search_paths else None,
    )
    rc = ws_create_cmd(args)
    if rc not in (0, None):
        raise RuntimeError(f"ws_create_cmd returned {rc}")
    return workspace_dir
