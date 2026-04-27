"""Project & workspace discovery endpoints."""

import logging
import mimetypes
import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .. import config_ops, discovery
from .. import search_roots as sr

log = logging.getLogger("forgather_server.projects")
router = APIRouter(tags=["projects"])

_MAX_ASSET_BYTES = 50 * 1024 * 1024  # 50 MiB


class ConfigInfoModel(BaseModel):
    name: str
    path: str
    is_default: bool = False


class ProjectInfoModel(BaseModel):
    project_dir: str
    name: Optional[str] = None
    description: Optional[str] = None
    default_config: Optional[str] = None
    workspace_root: Optional[str] = None
    configs: List[ConfigInfoModel] = []
    parse_error: Optional[str] = None


class WorkspaceClusterModel(BaseModel):
    workspace_root: str
    name: Optional[str] = None
    description: Optional[str] = None
    parent_workspace_root: Optional[str] = None
    projects: List[ProjectInfoModel]


def _project_to_model(p: discovery.ProjectInfo) -> ProjectInfoModel:
    return ProjectInfoModel(
        project_dir=p.project_dir,
        name=p.name,
        description=p.description,
        default_config=p.default_config,
        workspace_root=p.workspace_root,
        configs=[
            ConfigInfoModel(name=c.name, path=c.path, is_default=c.is_default)
            for c in p.configs
        ],
        parse_error=p.parse_error,
    )


@router.get("/projects", response_model=List[WorkspaceClusterModel])
def list_projects():
    clusters = discovery.discover_projects()
    return [
        WorkspaceClusterModel(
            workspace_root=c.workspace_root,
            name=c.name,
            description=c.description,
            parent_workspace_root=c.parent_workspace_root,
            projects=[_project_to_model(p) for p in c.projects],
        )
        for c in clusters
    ]


@router.get("/project", response_model=ProjectInfoModel)
def get_project(project_dir: str):
    """Detailed info for a single project (full config list + errors)."""
    info = discovery.load_project_info(project_dir)
    if info.parse_error and not info.configs:
        raise HTTPException(status_code=404, detail=info.parse_error)
    return _project_to_model(info)


@router.get("/project/readme", response_class=PlainTextResponse)
def get_project_readme(project_dir: str):
    """Return the contents of README.md at the project root.

    Returns 404 with a clear detail if README.md does not exist.
    Only the canonical filename ``README.md`` is accepted — no case variants.
    """
    readme = Path(project_dir) / "README.md"
    if not readme.exists() or not readme.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"No README.md found in project: {project_dir}",
        )
    try:
        content = readme.read_bytes()
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
    )


@router.get("/project/asset")
def get_project_asset(project_dir: str, asset: str):
    """Serve a binary asset (image, etc.) relative to the project directory.

    Security guards:
    - ``project_dir`` must resolve to an existing directory.
    - ``asset`` must be a relative path (not absolute, not starting with ~).
    - Resolved target must remain inside the resolved project_dir.
    - Target must be a regular file (not a symlink, not a directory).
    - File size is limited to 50 MiB.
    """
    # Guard: asset must be relative
    if os.path.isabs(asset):
        raise HTTPException(
            status_code=400,
            detail="asset must be a relative path",
        )
    if asset.startswith("~"):
        raise HTTPException(
            status_code=400,
            detail="asset path may not start with ~",
        )

    # Resolve project_dir
    proj_path = Path(project_dir).resolve()
    if not proj_path.exists() or not proj_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"project_dir does not exist or is not a directory: {project_dir}",
        )

    # Resolve target and verify it stays inside project_dir
    target = (proj_path / asset).resolve()
    if proj_path not in target.parents and target != proj_path:
        raise HTTPException(
            status_code=403,
            detail="asset path resolves outside the project directory",
        )

    # Target must be a regular file (not a symlink to outside, not a dir)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset}")
    if target.is_symlink():
        # Symlink target was already resolved above; double-check containment
        # was verified, but refuse symlinks as an extra precaution.
        real = target.resolve()
        if proj_path not in real.parents and real != proj_path:
            raise HTTPException(
                status_code=403,
                detail="symlink resolves outside the project directory",
            )
    if not target.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"asset is not a regular file: {asset}",
        )

    # Size limit
    size = target.stat().st_size
    if size > _MAX_ASSET_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"asset too large ({size} bytes; limit is {_MAX_ASSET_BYTES})",
        )

    content_type, _ = mimetypes.guess_type(str(target))
    if not content_type:
        content_type = "application/octet-stream"

    try:
        data = target.read_bytes()
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(content=data, media_type=content_type)


class TemplateEntryModel(BaseModel):
    name: str
    path: str
    rel_path: str


class TemplateGroupModel(BaseModel):
    category: str
    search_path: str
    templates: List[TemplateEntryModel]


class NewTemplateRequest(BaseModel):
    project_dir: str
    kind: str  # "config" | "template"
    name: str


class NewTemplateResponse(BaseModel):
    path: str


class TemplatePathsModel(BaseModel):
    templates_dir: str
    configs_dir: str
    config_prefix: str


class NewProjectRequest(BaseModel):
    workspace_dir: str
    name: str
    description: str
    config_prefix: str = "configs"
    default_config: str = "default.yaml"
    project_dir_name: Optional[str] = None
    copy_from: Optional[str] = None


class NewProjectResponse(BaseModel):
    project_dir: str


class NewWorkspaceRequest(BaseModel):
    parent_dir: str  # MUST be a configured search root.
    name: str
    description: str
    workspace_dir_name: Optional[str] = None  # bare name; falls back to slugified name
    forgather_dir: str
    libs: List[str] = []
    search_paths: List[str] = []


class NewWorkspaceResponse(BaseModel):
    workspace_dir: str


@router.post("/workspace/new", response_model=NewWorkspaceResponse)
def new_workspace(req: NewWorkspaceRequest):
    """Create a new Forgather workspace.

    Wraps ``forgather ws create``: the workspace lives at
    ``<parent_dir>/<workspace_dir_name>/``, with a ``forgather_workspace/``
    subdir holding the standard ``base_directories.yaml`` /
    ``meta_defaults.yaml`` / ``workspace.yaml`` / ``README.md`` skeleton.
    The CLI's ``ws_create_cmd`` is invoked through a ``SimpleNamespace``
    so we don't duplicate the template-rendering logic.

    ``parent_dir`` MUST be one of the server's configured search roots —
    otherwise the resulting workspace can't be discovered. We don't try
    to *add* a search root automatically; that would be a side-effect the
    user didn't ask for. The caller can add the root explicitly first
    (via the existing ``POST /api/search-roots``) and then create the
    workspace.
    """
    from types import SimpleNamespace

    from forgather.cli.workspace import ws_create_cmd

    if not req.name.strip():
        raise HTTPException(status_code=400, detail="name is required")
    if not req.description.strip():
        raise HTTPException(status_code=400, detail="description is required")
    if not req.forgather_dir.strip():
        raise HTTPException(status_code=400, detail="forgather_dir is required")

    parent = os.path.abspath(req.parent_dir)
    if not os.path.isdir(parent):
        raise HTTPException(
            status_code=400, detail=f"parent_dir is not a directory: {parent}"
        )

    # Enforce: parent_dir must match a configured search root exactly.
    # Discovery walks each root recursively, but anchoring on an exact
    # root match keeps the contract simple — the user picks where to
    # plant the workspace, and that location is by construction visible
    # to the discovery walk.
    roots = {os.path.abspath(r.path) for r in sr.list_roots()}
    if parent not in roots:
        raise HTTPException(
            status_code=400,
            detail=(
                f"parent_dir is not a search root: {parent}. "
                f"Add it as a search root first (POST /api/search-roots), "
                f"or pick one of the existing roots."
            ),
        )

    # Slugify name -> dir if no explicit dir was given (matches the CLI:
    # spaces -> underscores, lowercased, dots stripped). Nested paths
    # (``a/b/c``) are allowed so the user can build out a directory
    # hierarchy in one go — ``ws_create_cmd`` calls ``os.makedirs`` on
    # the workspace dir, which already creates intermediate parents.
    if req.workspace_dir_name and req.workspace_dir_name.strip():
        ws_rel = req.workspace_dir_name.strip()
    else:
        ws_rel = req.name.replace(" ", "_").lower().replace(".", "")
    if not ws_rel:
        raise HTTPException(status_code=400, detail="empty workspace_dir_name")
    if os.path.isabs(ws_rel):
        raise HTTPException(
            status_code=400, detail="workspace_dir_name must be relative"
        )
    parts = ws_rel.replace("\\", "/").strip("/").split("/")
    if any(p in ("", "..", ".") for p in parts):
        raise HTTPException(
            status_code=400, detail="workspace_dir_name has invalid path segments"
        )

    workspace_dir = os.path.abspath(os.path.join(parent, *parts))
    # Containment: target must remain under parent (defends against any
    # weird normalization corner case past the textual filter above).
    if os.path.commonpath([parent, workspace_dir]) != parent:
        raise HTTPException(
            status_code=400, detail="workspace_dir_name escapes parent_dir"
        )
    if os.path.exists(workspace_dir):
        raise HTTPException(status_code=409, detail=f"already exists: {workspace_dir}")

    if not os.path.isdir(req.forgather_dir):
        raise HTTPException(
            status_code=400,
            detail=f"forgather_dir is not a directory: {req.forgather_dir}",
        )

    args = SimpleNamespace(
        # ws_create_cmd uses ``workspace_dir`` directly when provided,
        # so pass the resolved absolute path. ``project_dir`` is only
        # consulted when ``workspace_dir`` is empty (CLI defaulting),
        # which we've already handled above.
        workspace_dir=workspace_dir,
        project_dir=parent,
        name=req.name,
        description=req.description,
        forgather_dir=req.forgather_dir,
        lib=list(req.libs) if req.libs else None,
        search_path=list(req.search_paths) if req.search_paths else None,
    )
    try:
        rc = ws_create_cmd(args)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if rc not in (0, None):
        raise HTTPException(status_code=500, detail=f"ws_create_cmd returned {rc}")
    return NewWorkspaceResponse(workspace_dir=workspace_dir)


class InitWorkspaceRequest(BaseModel):
    """Body for ``POST /api/workspace/init-here``.

    Companion to ``/workspace/new`` for the Files-tree right-click flow:
    the user has already created (or wants to use) a specific
    directory and wants to populate it with workspace metadata. Unlike
    ``/workspace/new``, ``workspace_dir`` is NOT computed from
    ``parent + name`` — it's whatever directory the user clicked.
    """

    workspace_dir: str
    name: str
    description: str
    forgather_dir: str
    libs: List[str] = []
    search_paths: List[str] = []


@router.post("/workspace/init-here", response_model=NewWorkspaceResponse)
def init_workspace_here(req: InitWorkspaceRequest):
    """Initialize a workspace in an existing directory.

    Validates the directory exists, is at-or-under a configured search
    root (so the new workspace will be discoverable), and doesn't
    already contain a ``forgather_workspace/`` subdir. Then dispatches
    to ``ws_create_cmd`` with the ``init_existing`` flag, which skips
    the directory-must-not-exist check + ``mkdir`` step and only writes
    the four metadata files.
    """
    from types import SimpleNamespace

    from forgather.cli.workspace import ws_create_cmd

    if not req.name.strip():
        raise HTTPException(status_code=400, detail="name is required")
    if not req.description.strip():
        raise HTTPException(status_code=400, detail="description is required")
    if not req.forgather_dir.strip():
        raise HTTPException(status_code=400, detail="forgather_dir is required")

    workspace_dir = os.path.abspath(req.workspace_dir)
    if not os.path.isdir(workspace_dir):
        raise HTTPException(
            status_code=400,
            detail=f"workspace_dir is not a directory: {workspace_dir}",
        )
    if os.path.exists(os.path.join(workspace_dir, "forgather_workspace")):
        raise HTTPException(
            status_code=409,
            detail=(
                f"forgather_workspace/ already exists in {workspace_dir} — "
                f"this directory is already a workspace"
            ),
        )

    # Discoverability: workspace_dir must be at or under one of the
    # configured search roots, otherwise the discovery walk won't find
    # it. Same constraint as /workspace/new, just expressed via
    # path containment instead of exact equality (we allow nesting).
    roots = [os.path.abspath(r.path) for r in sr.list_roots()]
    enclosing = None
    for r in roots:
        if workspace_dir == r or workspace_dir.startswith(r.rstrip("/") + "/"):
            if enclosing is None or len(r) > len(enclosing):
                enclosing = r
    if enclosing is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"workspace_dir is not under any configured search root: "
                f"{workspace_dir}. Add an enclosing directory as a search "
                f"root first (POST /api/search-roots) so the new workspace "
                f"will be discoverable."
            ),
        )

    if not os.path.isdir(req.forgather_dir):
        raise HTTPException(
            status_code=400,
            detail=f"forgather_dir is not a directory: {req.forgather_dir}",
        )

    args = SimpleNamespace(
        workspace_dir=workspace_dir,
        project_dir=os.path.dirname(workspace_dir),  # unused when workspace_dir set
        name=req.name,
        description=req.description,
        forgather_dir=req.forgather_dir,
        lib=list(req.libs) if req.libs else None,
        search_path=list(req.search_paths) if req.search_paths else None,
        init_existing=True,
    )
    try:
        rc = ws_create_cmd(args)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if rc not in (0, None):
        raise HTTPException(status_code=500, detail=f"ws_create_cmd returned {rc}")
    return NewWorkspaceResponse(workspace_dir=workspace_dir)


@router.post("/workspace/new-project", response_model=NewProjectResponse)
def new_project(req: NewProjectRequest):
    """Create a new Forgather project inside ``workspace_dir``.

    Mirrors ``forgather project create``: writes README.md + meta.yaml at
    ``workspace_dir/<project_dir_name>/`` (with ``project_dir_name`` defaulted
    to ``name`` lowercased with spaces->underscores), then seeds the default
    config under the project's templates dir from either ``copy_from`` or
    the CLI's built-in empty stub.

    The workspace is identified solely by directory — the server doesn't
    re-validate it against the discovered workspace tree because the user
    can right-click any workspace they can see, and `discovery.py` treats
    "workspace" as anything with a `forgather_workspace/` sibling.
    """
    from types import SimpleNamespace

    from forgather.cli.project import project_create_cmd

    if not req.name.strip():
        raise HTTPException(status_code=400, detail="name is required")
    if not req.description.strip():
        raise HTTPException(status_code=400, detail="description is required")

    # Allow nested project_dir_name (``a/b/c``) so the user can place the
    # new project inside an existing subdirectory of the workspace —
    # ``project_create_cmd`` calls ``os.makedirs`` which already creates
    # intermediate parents. Reject only obvious traversal.
    project_dir_name = req.project_dir_name or req.name.replace(" ", "_").lower()
    if not project_dir_name:
        raise HTTPException(status_code=400, detail="empty project_dir_name")
    if os.path.isabs(project_dir_name):
        raise HTTPException(status_code=400, detail="project_dir_name must be relative")
    parts = project_dir_name.replace("\\", "/").strip("/").split("/")
    if any(p in ("", "..", ".") for p in parts):
        raise HTTPException(
            status_code=400, detail="project_dir_name has invalid path segments"
        )

    workspace_dir = os.path.abspath(req.workspace_dir)
    target_dir = os.path.abspath(os.path.join(workspace_dir, *parts))
    if os.path.commonpath([workspace_dir, target_dir]) != workspace_dir:
        raise HTTPException(
            status_code=400, detail="project_dir_name escapes workspace_dir"
        )
    if os.path.exists(target_dir):
        raise HTTPException(
            status_code=409,
            detail=f"already exists: {target_dir}",
        )
    if req.copy_from and not os.path.isfile(req.copy_from):
        raise HTTPException(
            status_code=400,
            detail=f"copy_from is not a file: {req.copy_from}",
        )

    args = SimpleNamespace(
        project_dir=req.workspace_dir,
        project_dir_name=req.project_dir_name,
        name=req.name,
        description=req.description,
        config_prefix=req.config_prefix or "configs",
        default_config=req.default_config or "default.yaml",
        copy_from=req.copy_from,
    )
    try:
        rc = project_create_cmd(args)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if rc not in (0, None):
        raise HTTPException(status_code=500, detail=f"project_create_cmd returned {rc}")
    return NewProjectResponse(project_dir=target_dir)


@router.get("/project/template-paths", response_model=TemplatePathsModel)
def get_project_template_paths(project_dir: str):
    """Resolved templates and configs directory for the project — used by
    the New Config / New Template modal to render a live target path."""
    try:
        paths = config_ops.project_template_paths(project_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return TemplatePathsModel(
        templates_dir=paths.templates_dir,
        configs_dir=paths.configs_dir,
        config_prefix=paths.config_prefix,
    )


@router.post("/project/new-template", response_model=NewTemplateResponse)
def new_project_template(req: NewTemplateRequest):
    """Create an empty template (or config) file in the project's
    templates directory. ``kind="config"`` lands under the configured
    ``config_prefix`` subdir; ``kind="template"`` lands at the templates
    root. Returns the absolute path so the caller can open it.
    """
    try:
        path = config_ops.new_template_file(req.project_dir, req.kind, req.name)
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=f"already exists: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return NewTemplateResponse(path=path)


@router.get("/project/templates", response_model=List[TemplateGroupModel])
def list_project_templates(project_dir: str):
    """Every template file discoverable on this project's search path,
    grouped by search-path entry. Mirrors the interactive CLI's `edit`
    selector — the web UI's `tlist` view consumes this directly.

    Templates resolvable through multiple search paths are attributed to
    the first match (Jinja first-match resolution order).
    """
    try:
        groups = config_ops.list_project_templates(project_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return [
        TemplateGroupModel(
            category=g.category,
            search_path=g.search_path,
            templates=[
                TemplateEntryModel(name=t.name, path=t.path, rel_path=t.rel_path)
                for t in g.templates
            ],
        )
        for g in groups
    ]
