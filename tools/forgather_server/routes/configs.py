"""Config inspection endpoints: raw source, preprocessed output, trefs graph."""

import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from forgather.preprocess import PreprocessError

from .. import _atomic, config_ops, overrides_store

router = APIRouter(tags=["configs"])


class TrefsNodeModel(BaseModel):
    name: str
    path: str


class TrefsEdgeModel(BaseModel):
    source: str
    target: str


class TrefsGraphModel(BaseModel):
    root: str
    nodes: List[TrefsNodeModel]
    edges: List[TrefsEdgeModel]


class ReferencedTemplate(BaseModel):
    level: int
    name: str
    path: str


class ConfigMetaModel(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config_class: Optional[str] = None
    parse_error: Optional[str] = None


class DebugTraceItemModel(BaseModel):
    name: str
    path: str
    raw: str
    preprocessed: str


class PreprocessErrorDetail(BaseModel):
    """Structured 400 body returned when Jinja2 preprocessing fails.

    Frontend renders ``template`` + ``lineno`` as a compiler-style header
    over a ``<pre>`` block containing ``message`` and ``source_context``.
    """

    kind: Literal["preprocess_error"] = "preprocess_error"
    template: Optional[str] = None
    lineno: Optional[int] = None
    message: str
    source_context: Optional[str] = None


def _preprocess_error_detail(exc: PreprocessError) -> Dict[str, Any]:
    return PreprocessErrorDetail(
        template=exc.template_name,
        lineno=exc.lineno,
        message=exc.message,
        source_context=exc.source_context,
    ).model_dump()


class OutputDirInfoModel(BaseModel):
    output_dir: str
    models_dir: str
    output_dir_exists: bool
    models_dir_exists: bool
    output_dir_size_bytes: int = 0
    output_dir_entry_count: int = 0
    models_dir_size_bytes: int = 0
    models_dir_entry_count: int = 0
    # Raw value (int or torchrun keyword "gpu"/"cpu"/"auto").
    nproc_per_node: Any = None


@router.get("/config/raw", response_class=PlainTextResponse)
def get_config_raw(path: str):
    """Raw contents of the config template file at the given absolute path."""
    try:
        return config_ops.read_raw(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Not found: {path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config/pp", response_class=PlainTextResponse)
def get_config_pp(
    project_dir: str,
    config: str,
):
    """Preprocessed (Jinja-rendered) YAML for a config in a project.

    Dynamic-args are not wired in this phase — the pp output uses default
    values only.
    """
    try:
        return config_ops.render_pp(project_dir, config)
    except PreprocessError as e:
        raise HTTPException(status_code=400, detail=_preprocess_error_detail(e))
    except Exception as e:
        detail = f"{e}\n\n{traceback.format_exc()}"
        raise HTTPException(status_code=400, detail=detail)


@router.get("/config/debug", response_model=List[DebugTraceItemModel])
def get_config_debug(project_dir: str, config: str):
    """Per-template preprocess trace for the **debug** webui panel.

    Returns one entry per template that participated in rendering ``config``,
    in load order, with both the raw template source (as seen by Jinja2) and
    the preprocessed source (after the LineStatementProcessor rewrite). The
    frontend uses this to render a three-column view (template list + raw +
    preprocessed) so users can step through the render pipeline.
    """
    try:
        items = config_ops.render_pp_trace(project_dir, config)
    except PreprocessError as e:
        raise HTTPException(status_code=400, detail=_preprocess_error_detail(e))
    except Exception as e:
        detail = f"{e}\n\n{traceback.format_exc()}"
        raise HTTPException(status_code=400, detail=detail)
    return [
        DebugTraceItemModel(
            name=item.name,
            path=item.path,
            raw=item.raw,
            preprocessed=item.preprocessed,
        )
        for item in items
    ]


@router.get("/config/trefs")
def get_config_trefs(
    project_dir: str,
    config: str,
    format: Literal["json", "dot", "tree"] = Query(default="json"),
):
    """Template reference hierarchy in the requested format.

    - ``json``: structured ``{nodes, edges}`` graph (default).
    - ``dot``: Graphviz DOT source (passthrough for frontend wasm renderer).
    - ``tree``: ASCII tree (for debugging / text clients).
    """
    try:
        if format == "dot":
            return PlainTextResponse(config_ops.render_trefs_dot(project_dir, config))
        if format == "tree":
            return PlainTextResponse(config_ops.render_trefs_tree(project_dir, config))
        graph = config_ops.render_trefs_json(project_dir, config)
        return TrefsGraphModel(
            root=graph.root,
            nodes=[TrefsNodeModel(name=n.name, path=n.path) for n in graph.nodes],
            edges=[TrefsEdgeModel(source=s, target=t) for s, t in graph.edges],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/config/templates", response_model=List[ReferencedTemplate])
def get_config_templates(project_dir: str, config: str):
    """Flat list of every template consumed by the config (depth-ordered)."""
    try:
        return [
            ReferencedTemplate(level=level, name=name, path=path)
            for level, name, path in config_ops.list_referenced_templates(
                project_dir, config
            )
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/config/output-dir", response_model=OutputDirInfoModel)
def get_config_output_dir(project_dir: str, config: str):
    """Resolve + stat the config's default output_dir and models_dir.

    Does a full materialize of ``config.meta`` — roughly the same cost as
    ``/api/config/meta``, so call it on demand (e.g. when the Clean Output
    modal opens), not in a list response.
    """
    try:
        info = config_ops.load_output_dir_info(project_dir, config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return OutputDirInfoModel(
        output_dir=info.output_dir,
        models_dir=info.models_dir,
        output_dir_exists=info.output_dir_exists,
        models_dir_exists=info.models_dir_exists,
        output_dir_size_bytes=info.output_dir_size_bytes,
        output_dir_entry_count=info.output_dir_entry_count,
        models_dir_size_bytes=info.models_dir_size_bytes,
        models_dir_entry_count=info.models_dir_entry_count,
        nproc_per_node=info.nproc_per_node,
    )


@router.get("/config/meta", response_model=ConfigMetaModel)
def get_config_meta(project_dir: str, config: str):
    """Human-facing name + description from the config's ``meta`` block.

    Expensive (full preprocess + YAML parse + materialize), so callers
    should fetch lazily when a project is expanded in the tree.
    """
    m = config_ops.load_config_meta(project_dir, config)
    return ConfigMetaModel(
        name=m.name,
        description=m.description,
        config_class=m.config_class,
        parse_error=m.parse_error,
    )


def _looks_binary(path: str, sample_size: int = 8192) -> bool:
    """Cheap binary-vs-text heuristic: scan the first ``sample_size`` bytes
    for a null byte. ``\\x00`` virtually never occurs in legitimate text
    encodings (UTF-8, UTF-16 with a BOM is a separate concern but rare in
    practice on this codebase) but is ubiquitous in binary formats like
    safetensors, pickle, parquet, executables, images, etc. Cheap enough
    to run on every read; false negatives (binary file with no null in
    the first 8 KiB) get caught by the UTF-8 decode below."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(sample_size)
    except OSError:
        return False
    return b"\x00" in chunk


@router.get("/template/source", response_class=PlainTextResponse)
def get_template_source(path: str):
    """Raw source of a file at ``path`` for the editor.

    Used by the frontend to display templates from the trefs tree, files
    clicked in the sidebar Files tree, and any other in-app preview /
    edit flow. Path must be absolute; discovery routes only emit
    absolute paths so callers should already have one.

    Files that look binary — null byte in the first 8 KiB or invalid
    UTF-8 — are refused with 415 (Unsupported Media Type) so the
    editor can surface a clear "this isn't a text file" message
    instead of streaming garbage into Monaco.
    """
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Not found: {path}")
    if _looks_binary(path):
        raise HTTPException(
            status_code=415,
            detail="file appears to be binary; not editable as text",
        )
    try:
        content = config_ops.read_raw(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Not found: {path}")
    except UnicodeDecodeError as e:
        # Survived the null-byte check but isn't valid UTF-8 — same outcome
        # as a binary refusal, just caught later.
        raise HTTPException(
            status_code=415,
            detail=f"file is not valid UTF-8: {e.reason} at byte {e.start}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Surface mtime as a response header so the editor can hold onto a
    # baseline and detect concurrent on-disk edits at save time. Float
    # seconds-since-epoch with 6-digit precision (microseconds) covers
    # everything Linux's stat reports.
    response = PlainTextResponse(content)
    try:
        response.headers["X-Mtime"] = f"{os.path.getmtime(path):.6f}"
    except OSError:
        pass
    return response


class PutTemplateSourceRequest(BaseModel):
    path: str
    content: str
    # Optional optimistic-concurrency guard. Frontend stamps this with
    # the mtime returned from the GET (via the X-Mtime header) and
    # passes it back on save. If the file's current mtime is newer
    # than ``expected_mtime`` the write is refused with 409 so the
    # caller can prompt the user before clobbering an external edit.
    # Pass null/omit to skip the check (force-overwrite).
    expected_mtime: Optional[float] = None


@router.put("/template/source")
def put_template_source(req: PutTemplateSourceRequest):
    """Save edited contents of a template file.

    Accepts an absolute path to an *existing* file (no create-new yet)
    and writes ``content`` atomically via the same tmp+fsync+rename
    helpers used for persistent server state. When
    ``expected_mtime`` is provided and the current on-disk mtime is
    newer, returns 409 with detail ``{message, current_mtime,
    expected_mtime}`` so the client can show the user a
    keep-mine / reload / cancel prompt instead of clobbering the
    external edit.

    Returns ``{path, bytes_written, mtime}`` so the caller can update
    its baseline mtime for the next save.
    """
    if not os.path.isabs(req.path):
        raise HTTPException(status_code=400, detail="path must be absolute")
    target = Path(req.path)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {req.path}")
    if not target.is_file():
        raise HTTPException(status_code=400, detail=f"Not a regular file: {req.path}")
    if req.expected_mtime is not None:
        try:
            current_mtime = os.path.getmtime(req.path)
        except OSError as e:
            raise HTTPException(status_code=500, detail=str(e))
        # Tiny tolerance covers float roundtripping; in practice we
        # only flag genuine disk updates, not equality jitter.
        if current_mtime > req.expected_mtime + 1e-3:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": ("file changed on disk since you opened it"),
                    "current_mtime": current_mtime,
                    "expected_mtime": req.expected_mtime,
                },
            )
    try:
        _atomic.atomic_write_text(target, req.content)
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        new_mtime = os.path.getmtime(req.path)
    except OSError:
        new_mtime = 0.0
    return {
        "path": str(target),
        "bytes_written": len(req.content.encode("utf-8")),
        "mtime": new_mtime,
    }


# ---------- Overrides cache ----------


class OverridesResponse(BaseModel):
    values: Dict[str, Any]
    updated_at: Optional[float] = None


class SetOverridesRequest(BaseModel):
    project_dir: str
    config: str
    values: Dict[str, Any]


@router.get("/config/overrides", response_model=OverridesResponse)
def get_overrides(project_dir: str, config: str):
    """Return the cached override values for a config.

    Returns ``{values: {}, updated_at: null}`` when no cache file exists.
    """
    payload = overrides_store.get_overrides_payload(project_dir, config)
    return OverridesResponse(
        values=payload["values"],
        updated_at=payload["updated_at"],
    )


@router.post("/config/overrides", response_model=OverridesResponse)
def set_overrides(req: SetOverridesRequest):
    """Persist override values for a config (upsert). Returns the new state."""
    payload = overrides_store.set_overrides(req.project_dir, req.config, req.values)
    return OverridesResponse(
        values=payload["values"],
        updated_at=payload["updated_at"],
    )


@router.delete("/config/overrides")
def delete_overrides(project_dir: str, config: str):
    """Clear the cached overrides for a config.

    Always returns 200 — clearing a non-existent cache is a no-op.
    """
    cleared = overrides_store.clear_overrides(project_dir, config)
    return {"cleared": cleared}
