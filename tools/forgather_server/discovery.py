"""Project / workspace discovery by walking the configured search roots.

This module deliberately stays tolerant of broken configs: a single failing
``meta.yaml`` must not break the whole tree. Failures are surfaced as
``ProjectInfo`` entries with ``parse_error`` populated, matching the
``PARSE ERROR`` behavior of ``forgather ls -r``.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from forgather.meta_config import (
    PROJECT_META_NAME,
    WORKSPACE_CONFIG_DIR_NAME,
    MetaConfig,
)

from . import search_roots

WORKSPACE_METADATA_FILENAME = "workspace.yaml"

# Generic boilerplate that ships from the default `forgather ws create` template.
# When we see it, fall back to the directory basename for the workspace name so
# unrelated workspaces don't all show up as "Workspace Configuration".
_GENERIC_WS_TITLES = {"Workspace Configuration"}


@dataclass
class ConfigInfo:
    name: str
    path: str
    is_default: bool = False


@dataclass
class ProjectInfo:
    project_dir: str
    name: Optional[str] = None
    description: Optional[str] = None
    default_config: Optional[str] = None
    workspace_root: Optional[str] = None
    configs: List[ConfigInfo] = field(default_factory=list)
    parse_error: Optional[str] = None


@dataclass
class WorkspaceCluster:
    workspace_root: str  # "" for unaffiliated projects
    name: Optional[str] = None
    description: Optional[str] = None
    # Nearest enclosing workspace that also appeared in this discovery run.
    # None for top-level / orphaned workspaces. Used only for UI nesting —
    # Forgather's config resolution still attaches each project to its own
    # ``workspace_root`` and ignores enclosing workspaces.
    parent_workspace_root: Optional[str] = None
    projects: List[ProjectInfo] = field(default_factory=list)


def _read_workspace_readme(ws_root: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse forgather_workspace/README.md for title + first paragraph.

    Returns ``(title, description)`` — both may be ``None``. Title is the first
    ``# ...`` line; description is the first non-empty paragraph after the
    title. Falls back to ``(None, None)`` if the README is missing or has no
    heading.
    """
    readme = Path(ws_root) / WORKSPACE_CONFIG_DIR_NAME / "README.md"
    if not readme.is_file():
        return None, None
    try:
        text = readme.read_text()
    except OSError:
        return None, None

    lines = text.splitlines()
    title: Optional[str] = None
    title_idx = -1
    for i, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith("# "):
            title = line[2:].strip() or None
            title_idx = i
            break

    description: Optional[str] = None
    paragraph: List[str] = []
    for raw in lines[title_idx + 1 :]:
        line = raw.strip()
        if not line:
            if paragraph:
                break
            continue
        if line.startswith("#"):
            break
        paragraph.append(line)
    if paragraph:
        description = " ".join(paragraph)

    return title, description


def _read_workspace_yaml(ws_root: str) -> Tuple[Optional[str], Optional[str]]:
    """Read ``forgather_workspace/workspace.yaml`` if present.

    Plain YAML (not Jinja) with top-level ``name`` and ``description`` keys.
    Missing file, empty file, or parse errors return ``(None, None)`` so
    the caller can fall back to README parsing.
    """
    ws_yaml = Path(ws_root) / WORKSPACE_CONFIG_DIR_NAME / WORKSPACE_METADATA_FILENAME
    if not ws_yaml.is_file():
        return None, None
    try:
        data = yaml.safe_load(ws_yaml.read_text())
    except (OSError, yaml.YAMLError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    name = data.get("name")
    desc = data.get("description")
    return (
        (str(name).strip() or None) if name else None,
        (str(desc).strip() or None) if desc else None,
    )


def _workspace_display(ws_root: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a human-friendly ``(name, description)`` for a workspace root.

    Lookup order:
        1. ``forgather_workspace/workspace.yaml`` (structured, preferred)
        2. ``forgather_workspace/README.md`` title + first paragraph
        3. Directory basename fallback (for unnamed / generic-title workspaces)
    """
    if not ws_root:
        return "Unaffiliated", None

    name, description = _read_workspace_yaml(ws_root)
    if not name or not description:
        rd_title, rd_desc = _read_workspace_readme(ws_root)
        if not name:
            name = rd_title
        if not description:
            description = rd_desc

    if not name or name in _GENERIC_WS_TITLES:
        name = os.path.basename(os.path.normpath(ws_root)) or ws_root
    return name, description


def _iter_project_dirs(root: str):
    """Yield absolute directory paths that contain a meta.yaml, skipping hidden dirs
    and the workspace config directory itself."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden / workspace-config dirs in place so os.walk doesn't descend.
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and d != WORKSPACE_CONFIG_DIR_NAME
        ]
        if PROJECT_META_NAME in filenames:
            yield os.path.abspath(dirpath)


def _iter_workspace_dirs(root: str):
    """Yield absolute paths of directories that *contain* a
    ``forgather_workspace/`` subdirectory. The yielded path is the
    workspace root itself (the parent of ``forgather_workspace/``),
    matching ``MetaConfig.workspace_root``."""
    for dirpath, dirnames, _ in os.walk(root):
        # Stay out of hidden dirs but DO recurse into project dirs —
        # workspaces can be nested arbitrarily under search roots.
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if WORKSPACE_CONFIG_DIR_NAME in dirnames:
            yield os.path.abspath(dirpath)


def _load_project_info(project_dir: str) -> ProjectInfo:
    info = ProjectInfo(project_dir=project_dir)
    try:
        meta = MetaConfig(project_dir)
    except Exception as e:
        info.parse_error = str(e)
        return info

    info.name = meta.project_name
    info.description = meta.description
    info.workspace_root = meta.workspace_root

    try:
        default_cfg_name = meta.default_config()
    except Exception:
        default_cfg_name = None
    info.default_config = default_cfg_name

    try:
        for name, path in meta.find_templates(meta.config_prefix):
            info.configs.append(
                ConfigInfo(
                    name=name,
                    path=os.path.abspath(path),
                    is_default=(name == default_cfg_name),
                )
            )
    except Exception as e:
        info.parse_error = f"config enumeration failed: {e}"

    info.configs.sort(key=lambda c: c.name)
    return info


def discover_projects() -> List[WorkspaceCluster]:
    """Walk all configured search roots and return workspace-clustered projects.

    Two passes per root:
      1. ``_iter_workspace_dirs`` finds every workspace (anything with a
         ``forgather_workspace/`` subdir). These seed empty clusters so a
         freshly-created workspace shows up before it has any projects.
      2. ``_iter_project_dirs`` finds every project (``meta.yaml``) and
         attaches it to its enclosing workspace's cluster — or to the
         "Unaffiliated" cluster if it has no enclosing workspace.
    """
    seen_projects: set[str] = set()
    seen_workspaces: set[str] = set()
    projects: List[ProjectInfo] = []
    workspace_roots: List[str] = []

    for root in search_roots.list_roots():
        if not root.exists:
            continue
        for ws_dir in _iter_workspace_dirs(root.path):
            if ws_dir in seen_workspaces:
                continue
            seen_workspaces.add(ws_dir)
            workspace_roots.append(ws_dir)
        for project_dir in _iter_project_dirs(root.path):
            if project_dir in seen_projects:
                continue
            seen_projects.add(project_dir)
            projects.append(_load_project_info(project_dir))

    return _cluster_by_workspace(projects, workspace_roots)


def _find_enclosing_workspace(ws_root: str) -> Optional[str]:
    """Walk up from ``ws_root`` and return the first ancestor that contains a
    ``forgather_workspace/`` directory, or ``None`` if there isn't one.

    Purely filesystem-driven; the caller is responsible for discarding the
    result if that ancestor wasn't itself in the discovered set.
    """
    if not ws_root:
        return None
    current = os.path.abspath(ws_root)
    while True:
        parent = os.path.dirname(current)
        if parent == current:
            return None
        if os.path.isdir(os.path.join(parent, WORKSPACE_CONFIG_DIR_NAME)):
            return parent
        current = parent


def _cluster_by_workspace(
    projects: List[ProjectInfo],
    workspace_roots: Optional[List[str]] = None,
) -> List[WorkspaceCluster]:
    clusters: Dict[str, WorkspaceCluster] = {}
    # Seed clusters for every directly-discovered workspace so empty
    # workspaces (no projects yet) still show up in the tree. Projects
    # then attach to whichever workspace_root MetaConfig resolves them
    # to.
    for ws in workspace_roots or []:
        clusters.setdefault(ws, WorkspaceCluster(workspace_root=ws))
    for p in projects:
        key = p.workspace_root or ""
        clusters.setdefault(key, WorkspaceCluster(workspace_root=key)).projects.append(
            p
        )

    discovered = {c.workspace_root for c in clusters.values() if c.workspace_root}

    for c in clusters.values():
        c.projects.sort(key=lambda p: p.project_dir)
        c.name, c.description = _workspace_display(c.workspace_root)
        if c.workspace_root:
            enclosing = _find_enclosing_workspace(c.workspace_root)
            # Only nest under a parent that's actually in this response.
            # A workspace whose enclosing workspace wasn't discovered (e.g.
            # because it sits above the user's search roots) stays a top
            # level node so it doesn't get orphaned.
            if enclosing and enclosing in discovered:
                c.parent_workspace_root = enclosing

    return sorted(
        clusters.values(),
        key=lambda c: (c.workspace_root == "", c.workspace_root),
    )


def load_project_info(project_dir: str) -> ProjectInfo:
    """Load ProjectInfo for a specific project directory (used by detail endpoints)."""
    return _load_project_info(os.path.abspath(project_dir))
