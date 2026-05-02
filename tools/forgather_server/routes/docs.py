"""Docs endpoints: serve markdown/ipynb files plus their relative assets.

The Docs view in the webui browses arbitrary on-disk markdown and ipynb
documents. ``/api/docs/root`` resolves the default landing page (the
Forgather repo's top-level README), ``/api/docs/file`` serves a single
markdown or ipynb document, and ``/api/docs/asset`` serves any binary
referenced by relative path inside a doc.
"""

import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from .. import search_roots as sr

log = logging.getLogger("forgather_server.docs")
router = APIRouter(tags=["docs"])

_MAX_DOC_BYTES = 25 * 1024 * 1024  # 25 MiB
_MAX_ASSET_BYTES = 50 * 1024 * 1024  # 50 MiB


class DocsRootResponse(BaseModel):
    path: Optional[str] = None


class IpynbCell(BaseModel):
    cell_type: str  # "markdown" | "code" | "raw"
    source: str
    language: Optional[str] = None
    outputs: List[Dict[str, Any]] = []


class DocsFileResponse(BaseModel):
    path: str
    kind: str  # "markdown" | "ipynb"
    content: Optional[str] = None
    cells: Optional[List[IpynbCell]] = None


def _is_markdown(path: Path) -> bool:
    return path.suffix.lower() in (".md", ".markdown")


def _is_ipynb(path: Path) -> bool:
    return path.suffix.lower() == ".ipynb"


def _abs_resolve(path: str) -> Path:
    """Resolve ``path`` to an absolute filesystem path.

    The Docs view passes absolute paths (the frontend resolves any
    relative-to-doc references before calling the API). ``..`` and
    symlinks are followed by ``Path.resolve()``.
    """
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    p = Path(path)
    if not p.is_absolute():
        raise HTTPException(status_code=400, detail="path must be absolute")
    return p.resolve()


@router.get("/docs/root", response_model=DocsRootResponse)
def docs_root():
    """Default Docs landing page: the Forgather repo's top-level README.

    Returns ``{path: null}`` if the repo has no README — the frontend
    surfaces a friendly empty state in that case.
    """
    repo = Path(sr.forgather_repo_root())
    candidate = repo / "README.md"
    if candidate.is_file():
        return DocsRootResponse(path=str(candidate))
    return DocsRootResponse(path=None)


@router.get("/docs/file", response_model=DocsFileResponse)
def docs_file(path: str):
    """Read a markdown or ipynb document.

    For markdown, ``content`` carries the raw text. For ipynb, ``cells``
    carries a normalized cell list (markdown / code / raw) so the
    frontend doesn't have to know the notebook schema.
    """
    target = _abs_resolve(path)

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {path}")
    # GitHub-style affordance: when a link points at a directory, render
    # its README.md (case-insensitive) if one exists. Without this,
    # cross-references in tutorial docs that work on github.com surface
    # as plain 404s in the webui's Docs view. The error path stays
    # informative for directories that genuinely have no README.
    if target.is_dir():
        readme = None
        for name in ("README.md", "readme.md", "Readme.md"):
            candidate = target / name
            if candidate.is_file():
                readme = candidate
                break
        if readme is None:
            raise HTTPException(
                status_code=404,
                detail=f"No README.md in directory: {path}",
            )
        target = readme
    if not target.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {path}")

    size = target.stat().st_size
    if size > _MAX_DOC_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"document too large ({size} bytes; limit is {_MAX_DOC_BYTES})",
        )

    if _is_markdown(target):
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise HTTPException(
                status_code=415,
                detail=f"file is not valid UTF-8: {e.reason} at byte {e.start}",
            )
        except OSError as e:
            raise HTTPException(status_code=500, detail=str(e))
        return DocsFileResponse(path=str(target), kind="markdown", content=text)

    if _is_ipynb(target):
        try:
            raw = target.read_text(encoding="utf-8")
            nb = json.loads(raw)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            raise HTTPException(
                status_code=400, detail=f"failed to parse notebook: {e}"
            )

        language = (
            (nb.get("metadata") or {}).get("kernelspec", {}).get("language")
        ) or ((nb.get("metadata") or {}).get("language_info", {}).get("name"))

        normalized: List[IpynbCell] = []
        for cell in nb.get("cells", []) or []:
            ctype = cell.get("cell_type") or "raw"
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            cell_lang = language if ctype == "code" else None
            outputs: List[Dict[str, Any]] = []
            if ctype == "code":
                # Keep only fields the frontend renders: text/plain,
                # text/html, image/png, image/jpeg, stream text, errors.
                for out in cell.get("outputs", []) or []:
                    otype = out.get("output_type")
                    if otype == "stream":
                        text = out.get("text", "")
                        if isinstance(text, list):
                            text = "".join(text)
                        outputs.append(
                            {
                                "output_type": "stream",
                                "name": out.get("name") or "stdout",
                                "text": text,
                            }
                        )
                    elif otype in ("execute_result", "display_data"):
                        data = out.get("data") or {}
                        kept: Dict[str, Any] = {}
                        for mime in (
                            "text/html",
                            "image/png",
                            "image/jpeg",
                            "image/svg+xml",
                            "text/plain",
                        ):
                            if mime in data:
                                value = data[mime]
                                if isinstance(value, list):
                                    value = "".join(value)
                                kept[mime] = value
                        if kept:
                            outputs.append(
                                {
                                    "output_type": otype,
                                    "data": kept,
                                }
                            )
                    elif otype == "error":
                        outputs.append(
                            {
                                "output_type": "error",
                                "ename": out.get("ename") or "",
                                "evalue": out.get("evalue") or "",
                                "traceback": out.get("traceback") or [],
                            }
                        )
            normalized.append(
                IpynbCell(
                    cell_type=ctype,
                    source=src,
                    language=cell_lang,
                    outputs=outputs,
                )
            )
        return DocsFileResponse(path=str(target), kind="ipynb", cells=normalized)

    raise HTTPException(
        status_code=415,
        detail="unsupported docs file type (only .md / .markdown / .ipynb)",
    )


@router.get("/docs/asset")
def docs_asset(path: str):
    """Serve a binary asset (image, etc.) at an absolute path.

    Used by the Docs view to load images referenced by markdown or
    notebook cells. The frontend resolves the relative reference
    against the current doc's directory before calling this endpoint.
    """
    target = _abs_resolve(path)

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Asset not found: {path}")
    if not target.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {path}")

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
