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
from typing import Any, Dict, Optional

from .. import config_ops, meta_templates
from .registry import PROPOSE, READ, Proposal, ToolRegistry, ToolSpec

log = logging.getLogger("forgather_server.agent.tools_authoring")


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
    expected_mtime: Optional[float] = args.get("expected_mtime")

    before = config_ops.read_raw(path)  # also validates the file exists/readable

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
                "its current content and base your edit on it. Pass "
                "expected_mtime from a prior read to guard against clobbering "
                "an external edit."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string"},
                    "config_name": {"type": "string", "description": "For the post-write parse check."},
                    "path": {"type": "string", "description": "Absolute path of the file to overwrite."},
                    "new_content": {"type": "string", "description": "Full new file content."},
                    "expected_mtime": {"type": "number", "description": "Optional optimistic-concurrency guard."},
                },
                "required": ["project_dir", "config_name", "path", "new_content"],
            },
            handler=_propose_edit_config,
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
