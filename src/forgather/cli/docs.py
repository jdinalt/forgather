"""Run the docs-build step that pre-renders mkdocstrings directives.

The webui docs viewer serves raw markdown; ``forgather docs build``
expands ``:::`` directives ahead of time so the viewer can show
filled-in API pages without an in-process mkdocstrings dependency.
See ``src/forgather/docs_build/`` for the library.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def docs_cmd(args) -> int:
    action = getattr(args, "docs_action", None) or "build"
    if action == "build":
        return _build_cmd(args)
    if action == "clean":
        return _clean_cmd(args)
    if action == "index":
        return _index_cmd(args)
    print(f"unknown docs action: {action}", file=sys.stderr)
    return 2


def _index_cmd(args) -> int:
    """Build (or check) the docs vector-search index."""
    from forgather import docs_index

    repo_root, _ = _resolve_paths(args)
    model = getattr(args, "model", None) or docs_index.DEFAULT_MODEL

    if getattr(args, "check", False):
        stale = docs_index.is_stale(repo_root, model_name=model)
        print(f"docs index: {'stale — rebuild needed' if stale else 'up-to-date'}")
        return 1 if stale else 0

    report = docs_index.build_index(
        repo_root, model_name=model, clean=bool(getattr(args, "clean", False))
    )
    if not getattr(args, "quiet", False):
        if report.cleaned:
            print(f"cleaned: {report.out_dir}")
        print(
            f"docs index: {report.chunks} chunks from {report.pages} pages "
            f"-> {report.out_dir} (model={report.model_name})"
        )
    return 0


def _resolve_paths(args) -> tuple[Path, Path]:
    repo_root = (
        Path(args.repo_root).resolve() if args.repo_root else _autodetect_repo_root()
    )
    docs_dir = Path(args.docs_dir).resolve() if args.docs_dir else (repo_root / "docs")
    if not docs_dir.is_dir():
        raise SystemExit(f"docs directory does not exist: {docs_dir}")
    return repo_root, docs_dir


def _autodetect_repo_root() -> Path:
    """Locate the repo root the same way the rest of the CLI does.

    The forgather package sits at ``<repo>/src/forgather``; walking two
    levels up from the package file gets us there. This matches what
    ``mkdocs serve`` uses and keeps griffe's search path consistent.
    """
    import forgather

    pkg = Path(forgather.__file__).resolve().parent  # <repo>/src/forgather
    return pkg.parent.parent


def _build_cmd(args) -> int:
    from forgather.docs_build import build

    repo_root, docs_dir = _resolve_paths(args)
    subset = Path(args.path).resolve() if getattr(args, "path", None) else None

    report = build(
        docs_dir,
        repo_root=repo_root,
        clean=bool(getattr(args, "clean", False)),
        subset=subset,
        check_only=bool(getattr(args, "check", False)),
    )

    quiet = bool(getattr(args, "quiet", False))
    check = bool(getattr(args, "check", False))

    if not quiet:
        if report.cleaned:
            print(f"cleaned: {report.output_dir}")
        for src in report.built:
            label = "would-rebuild" if check else "built"
            print(f"  {label}: {src.relative_to(repo_root)}")
        for src in report.skipped_no_directives:
            print(f"  skip (no directives): {src.relative_to(repo_root)}")
        for src in report.skipped_up_to_date:
            print(f"  skip (up-to-date): {src.relative_to(repo_root)}")

    for src, err in report.errors:
        print(f"ERROR {src}: {err}", file=sys.stderr)

    summary = (
        f"docs build: {len(report.built)} {'would rebuild' if check else 'built'}, "
        f"{len(report.skipped_up_to_date)} up-to-date, "
        f"{len(report.skipped_no_directives)} no-directives, "
        f"{len(report.errors)} errors"
    )
    print(summary)

    if report.errors:
        return 1
    if check and report.built:
        return 1
    return 0


def _clean_cmd(args) -> int:
    _, docs_dir = _resolve_paths(args)
    target = docs_dir / ".built"
    if target.exists():
        shutil.rmtree(target)
        print(f"removed {target}")
    else:
        print(f"nothing to remove ({target} doesn't exist)")
    return 0
