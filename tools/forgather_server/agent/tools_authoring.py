"""Phase 1 authoring agent tools (propose/commit, behind the approval gate).

Each mutating tool computes a **preview** only — a before/after pair for a
diff — and returns a ``Proposal`` whose ``commit`` closure performs the
real, atomic write when (and only when) the user approves. The closure
captures the exact computed content at propose time, so what gets written
is exactly what was previewed; the model cannot alter it after the fact.

After a write lands, the commit result includes a parse check
(``load_config_meta``) so a config that now fails to preprocess is
surfaced back into the conversation rather than silently committed.

``list_meta_templates`` is a read tool included here so the model can
discover which scaffolds (and which fields) ``propose_new_config_from_template``
accepts.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, Dict, Optional

from .. import config_ops, meta_templates, paths, project_ops
from .registry import PROPOSE, READ, Proposal, ToolRegistry, ToolSpec

log = logging.getLogger("forgather_server.agent.tools_authoring")


def _enforce_readable_path(path: str) -> None:
    """Gate file reads the agent does at propose time to the fs-root allowlist.

    Without this, ``propose_edit_config`` could read any server-readable file
    (the read happens during preview, before approval, and its contents are
    streamed into the action card and the conversation). Mirrors the
    ``read_file`` tool's guard so the propose path can't bypass the chroot.
    """
    if not os.path.isabs(path):
        raise ValueError("path must be absolute")
    if not paths.is_path_in_fs_root(path):
        raise PermissionError(
            f"path is outside the configured filesystem roots: {path}"
        )


def _validate_after_write(project_dir: str, config_name: str) -> str:
    """Return a human-readable parse-check line for a freshly-written config."""
    try:
        meta = config_ops.load_config_meta(project_dir, config_name)
    except Exception as e:  # never let validation failure mask a successful write
        return f"(could not run post-write validation: {e})"
    if meta.parse_error:
        return f"WARNING: the config now has a parse error: {meta.parse_error}"
    return "post-write validation OK (config preprocesses cleanly)."


# ---- propose handlers ------------------------------------------------------


def _propose_edit_config(args: Dict[str, Any]) -> Proposal:
    project_dir = args["project_dir"]
    config_name = args["config_name"]
    path = args["path"]
    new_content = args["new_content"]

    _enforce_readable_path(path)  # fs-root gate before any read (see above)
    before = config_ops.read_raw(path)  # also validates the file exists/readable
    # Capture the mtime *here*, at propose time, as the optimistic-concurrency
    # baseline. The model cannot supply this reliably (no tool exposes an
    # mtime), so taking it from the model led to spurious StaleEditError on a
    # hallucinated value. Captured server-side it has real meaning: the commit
    # (on approval) fails only if the file actually changed between this read
    # and the user approving.
    try:
        expected_mtime: Optional[float] = os.path.getmtime(path)
    except OSError:
        expected_mtime = None

    def commit() -> str:
        info = config_ops.write_existing_file(
            path, new_content, expected_mtime=expected_mtime
        )
        check = _validate_after_write(project_dir, config_name)
        return (
            f"wrote {info['bytes_written']} bytes to {info['path']} "
            f"(mtime={info['mtime']}). {check}"
        )

    return Proposal(
        title=f"Edit {config_name}",
        summary=f"Overwrite {path}",
        path=path,
        before=before,
        after=new_content,
        commit=commit,
    )


def _propose_new_config(args: Dict[str, Any]) -> Proposal:
    project_dir = args["project_dir"]
    name = args["name"]
    kind = args.get("kind", "config")
    content = args.get("content", "")

    # Resolve (and validate name/containment/no-overwrite) without writing.
    target, _empty = config_ops.resolve_new_template_target(project_dir, kind, name)

    def commit() -> str:
        config_ops.write_template_file(target, content)
        check = _validate_after_write(project_dir, name)
        return f"created {target}. {check}"

    return Proposal(
        title=f"New {kind}: {name}",
        summary=f"Create {target}",
        path=target,
        before=None,
        after=content,
        reveal_kind="config",
        commit=commit,
    )


def _propose_new_config_from_template(args: Dict[str, Any]) -> Proposal:
    project_dir = args["project_dir"]
    name = args["name"]
    kind = args.get("kind", "config")
    meta_template = args["meta_template"]
    values = args.get("values") or {}

    # Pure render + path resolution (raises MissingFieldsError if a required
    # template field is absent, FileExistsError if the target exists).
    target, content = config_ops.resolve_new_template_target(
        project_dir, kind, name, meta_template=meta_template, values=values
    )

    def commit() -> str:
        config_ops.write_template_file(target, content)
        check = _validate_after_write(project_dir, name)
        return f"created {target} from meta-template {meta_template!r}. {check}"

    return Proposal(
        title=f"New {kind} from template: {name}",
        summary=f"Create {target} from meta-template {meta_template!r}",
        path=target,
        before=None,
        after=content,
        reveal_kind="config",
        commit=commit,
    )


def _propose_new_workspace(args: Dict[str, Any]) -> Proposal:
    parent_dir = args["parent_dir"]
    name = args["name"]
    description = args["description"]
    workspace_dir_name = args.get("workspace_dir_name")
    forgather_dir = args.get("forgather_dir")
    # Validate + compute the target up front (no creation) so the preview
    # shows the path and obvious errors surface before approval.
    target = project_ops.resolve_new_workspace_target(parent_dir, name, workspace_dir_name)

    def commit() -> str:
        created = project_ops.create_workspace(
            parent_dir=parent_dir,
            name=name,
            description=description,
            workspace_dir_name=workspace_dir_name,
            forgather_dir=forgather_dir,
        )
        return f"created workspace at {created}"

    return Proposal(
        title=f"New workspace: {name}",
        summary=f"Create workspace at {target}",
        path=target,
        extra={"name": name, "description": description, "parent_dir": parent_dir},
        reveal_kind="workspace",
        commit=commit,
    )


def _propose_new_project(args: Dict[str, Any]) -> Proposal:
    workspace_dir = args["workspace_dir"]
    name = args["name"]
    description = args["description"]
    project_dir_name = args.get("project_dir_name")
    config_prefix = args.get("config_prefix", "configs")
    default_config = args.get("default_config", "default.yaml")
    meta_template = args.get("meta_template")
    values = args.get("values")
    copy_from = args.get("copy_from")
    target, _ = project_ops.resolve_new_project_target(workspace_dir, name, project_dir_name)

    def commit() -> str:
        created = project_ops.create_project(
            workspace_dir=workspace_dir,
            name=name,
            description=description,
            project_dir_name=project_dir_name,
            config_prefix=config_prefix,
            default_config=default_config,
            meta_template=meta_template,
            values=values,
            copy_from=copy_from,
        )
        return f"created project at {created} (default config: {default_config})"

    # Show the chosen starting point in the preview.
    starting_point = (
        f"scaffold: {meta_template}"
        if meta_template
        else f"copy: {copy_from}"
        if copy_from
        else "empty stub"
    )
    return Proposal(
        title=f"New project: {name}",
        summary=f"Create project at {target}",
        path=target,
        extra={
            "name": name,
            "description": description,
            "config_prefix": config_prefix,
            "default_config": default_config,
            "starting_point": starting_point,
        },
        reveal_kind="project",
        commit=commit,
    )


def _list_meta_templates(_args: Dict[str, Any]) -> Any:
    return [dataclasses.asdict(c) for c in meta_templates.discover()]


# ---- registration ----------------------------------------------------------


def register_all(reg: ToolRegistry) -> None:
    reg.register(
        ToolSpec(
            name="list_meta_templates",
            description=(
                "List available meta-templates (config/workspace scaffolds) "
                "and their fields. Use before propose_new_config_from_template "
                "to learn the meta_template id and which values it accepts."
            ),
            json_schema={"type": "object", "properties": {}},
            handler=_list_meta_templates,
            risk=READ,
        )
    )
    reg.register(
        ToolSpec(
            name="propose_edit_config",
            description=(
                "Propose overwriting an existing config/template file with new "
                "content. Returns a diff for the user to approve; nothing is "
                "written until they do. Read the file first (read_file) to get "
                "its current content and base your edit on it (pass the FULL new "
                "content). Clobber protection is automatic — the tool records "
                "the file's state when it reads it and refuses the write if the "
                "file changed before you got approval."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string"},
                    "config_name": {"type": "string", "description": "For the post-write parse check."},
                    "path": {"type": "string", "description": "Absolute path of the file to overwrite."},
                    "new_content": {"type": "string", "description": "Full new file content."},
                },
                "required": ["project_dir", "config_name", "path", "new_content"],
            },
            handler=_propose_edit_config,
            risk=PROPOSE,
        )
    )
    reg.register(
        ToolSpec(
            name="propose_new_workspace",
            description=(
                "Propose creating a new Forgather workspace (a directory with "
                "forgather_workspace/ metadata) under parent_dir. Approval "
                "required before anything is written. Create a workspace when "
                "there isn't one to hold a new project."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "parent_dir": {"type": "string", "description": "Absolute directory to create the workspace under."},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "workspace_dir_name": {"type": "string", "description": "Directory name (default: slug of name)."},
                },
                "required": ["parent_dir", "name", "description"],
            },
            handler=_propose_new_workspace,
            risk=PROPOSE,
        )
    )
    reg.register(
        ToolSpec(
            name="propose_new_project",
            description=(
                "Propose creating a new Forgather project (directory + "
                "meta.yaml + a default config) inside an existing workspace. "
                "Scaffold a project before adding configs — "
                "propose_new_config/propose_edit_config require it to exist. "
                "The default config has three starting points (pick at most "
                "one of meta_template / copy_from): (1) a scaffold — set "
                "meta_template to an id from list_meta_templates (+ values); "
                "(2) copy an existing config — set copy_from to an absolute "
                "config path (find one via list_configs on a similar example "
                "project, so the new project starts close to a working "
                "example you then customize with propose_edit_config); "
                "(3) neither — an empty stub. Approval required."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "workspace_dir": {"type": "string", "description": "Absolute workspace directory to create the project in."},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "project_dir_name": {"type": "string", "description": "Project directory name (default: slug of name); may be nested a/b/c."},
                    "config_prefix": {"type": "string", "description": "Name of the configs sub-directory under the project's template root. Leave unset to use the default \"configs\" (configs then live at <project>/templatelib/configs/). This is a leaf name only, NOT a path: do not prefix it with \"templatelib/\" or the project name -- the template root is prepended automatically, so \"templatelib/configs\" would wrongly nest as templatelib/templatelib/configs/."},
                    "default_config": {"type": "string", "description": "Default config file name (default \"default.yaml\")."},
                    "meta_template": {"type": "string", "description": "Scaffold id from list_meta_templates to seed the default config (mutually exclusive with copy_from)."},
                    "values": {"type": "object", "description": "Field values for the meta-template scaffold."},
                    "copy_from": {"type": "string", "description": "Absolute path to an existing config to seed the default config from (mutually exclusive with meta_template). Get paths from list_configs."},
                },
                "required": ["workspace_dir", "name", "description"],
            },
            handler=_propose_new_project,
            risk=PROPOSE,
        )
    )
    reg.register(
        ToolSpec(
            name="propose_new_config",
            description=(
                "Propose creating a new config/template file with the given "
                "content. Refuses to overwrite an existing file. Approval "
                "required before the file is written."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string"},
                    "name": {"type": "string", "description": "Relative file name (\".yaml\" appended if omitted)."},
                    "kind": {"type": "string", "enum": ["config", "template"], "description": "Default \"config\"."},
                    "content": {"type": "string", "description": "Full file content (default empty)."},
                },
                "required": ["project_dir", "name"],
            },
            handler=_propose_new_config,
            risk=PROPOSE,
        )
    )
    reg.register(
        ToolSpec(
            name="propose_new_config_from_template",
            description=(
                "Propose creating a new config/template file scaffolded from a "
                "meta-template (see list_meta_templates for ids and fields). "
                "Approval required before the file is written."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string"},
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": ["config", "template"]},
                    "meta_template": {"type": "string", "description": "Meta-template id from list_meta_templates."},
                    "values": {"type": "object", "description": "Field values for the meta-template."},
                },
                "required": ["project_dir", "name", "meta_template"],
            },
            handler=_propose_new_config_from_template,
            risk=PROPOSE,
        )
    )
