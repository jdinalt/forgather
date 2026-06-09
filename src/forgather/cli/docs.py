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
    if action == "search":
        return _search_cmd(args)
    print(f"unknown docs action: {action}", file=sys.stderr)
    return 2


def _search_cmd(args) -> int:
    """Search the docs corpus (keyword / vector / hybrid).

    Same ranker the webui Docs box and the agent's ``search_docs`` tool use.
    Routing:

    - keyword: always in-process — instant, offline, full corpus (incl. agent
      docs by default).
    - vector / hybrid: queried on the running forgather server by default, so
      the embedding model stays warm across searches. ``--local`` forces an
      in-process search that loads the model locally (slow cold start) — handy
      for diagnostics, but not the default.
    """
    query = " ".join(args.query).strip()
    if not query:
        print("error: query is empty", file=sys.stderr)
        return 2
    mode = args.mode
    limit = max(1, int(getattr(args, "limit", 8) or 8))
    as_json = bool(getattr(args, "json", False))
    verbose = bool(getattr(args, "verbose", False))

    use_server = mode in ("vector", "hybrid") and not bool(getattr(args, "local", False))
    if use_server:
        return _search_via_server(args, query, mode, limit, as_json, verbose)
    return _search_local(args, query, mode, limit, as_json, verbose)


def _search_via_server(args, query, mode, limit, as_json, verbose) -> int:
    """Vector/hybrid search through the running server (warm shared model)."""
    import time

    from .server_client import AuthRequired, ServerClient, ServerUnreachable

    client = ServerClient.from_args(args)
    if verbose:
        print(f"# source: server {client.base}", file=sys.stderr)
    if getattr(args, "no_agent_docs", False):
        # The server endpoint already searches the user-facing corpus (no
        # CLAUDE.*), so the flag is a no-op here — say so rather than imply it
        # filtered anything. Use --local to filter agent docs in-process.
        print(
            "note: --no-agent-docs has no effect on the server path (it already "
            "searches the user-facing corpus); use --local to control it.",
            file=sys.stderr,
        )

    t0 = time.perf_counter()
    try:
        resp = client._get(
            "/docs/search",
            params={"q": query, "limit": min(limit, 50), "mode": mode},
        )
        payload = resp.json()
    except ServerUnreachable as e:
        print(f"error: {e}", file=sys.stderr)
        print(
            "hint: run the search in-process with --local (loads the embedding "
            "model locally; slower).",
            file=sys.stderr,
        )
        return 1
    except AuthRequired as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 — report, don't traceback
        print(f"error: server docs search failed: {e}", file=sys.stderr)
        print("hint: try --local for an in-process search.", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - t0

    ran_mode = payload.get("mode", "keyword")
    vector_available = payload.get("vector_available")
    hits = payload.get("hits", [])
    fell_back = mode in ("vector", "hybrid") and ran_mode == "keyword"
    diagnostics = {
        "source": "server",
        "server": client.base,
        "requested_mode": mode,
        "ran_mode": ran_mode,
        "vector_index_available": vector_available,
        "fell_back_to_keyword": fell_back,
        "elapsed_seconds": round(elapsed, 3),
        "hit_count": len(hits),
    }
    result = {"query": payload.get("query", query), "mode": ran_mode, "hits": hits}
    return _emit_search(result, diagnostics, as_json, verbose)


def _search_local(args, query, mode, limit, as_json, verbose) -> int:
    """In-process search: keyword always, or vector/hybrid under ``--local``.

    Reuses ``tools/forgather_server/docs_search.py`` directly (the server need
    not be running). For vector/hybrid this loads the embedding model in the
    CLI process — a slow cold start each invocation, which is why it's opt-in.
    """
    import logging

    import time

    repo_root = (
        Path(args.repo_root).resolve() if args.repo_root else _autodetect_repo_root()
    )
    tools_dir = repo_root / "tools"
    if not (tools_dir / "forgather_server" / "docs_search.py").is_file():
        print(
            f"docs search needs the forgather source tree at "
            f"{tools_dir / 'forgather_server'} (run from a checkout, or pass "
            f"--repo-root).",
            file=sys.stderr,
        )
        return 2
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))

    if verbose:
        # Surface docs_vector's "embed failed" warning (and model-load chatter)
        # so a silent vector→keyword fallback shows its cause.
        logging.basicConfig(level=logging.INFO, format="# %(name)s: %(message)s")

    # Imported after the sys.path insert; the empty package __init__ keeps this
    # cheap (no FastAPI), so the CLI stays fast for the common keyword path.
    from forgather_server import docs_search, docs_vector, search_roots  # noqa: E402

    # The backend derives the corpus root from search_roots, not from our
    # ``repo_root`` (which only located ``tools/`` for the import). Report THAT
    # root so the diagnostics can never claim to search a tree they didn't.
    searched_root = Path(search_roots.forgather_repo_root())
    include_agent_docs = not bool(getattr(args, "no_agent_docs", False))

    index_ok = docs_vector.index_available()
    if verbose:
        print(f"# source: local (in-process)  repo: {searched_root}", file=sys.stderr)
        print(
            f"# vector index: {'available' if index_ok else 'absent'}"
            + (f" ({_index_summary(searched_root)})" if index_ok else ""),
            file=sys.stderr,
        )

    t0 = time.perf_counter()
    try:
        result = docs_search.search(
            query, include_agent_docs=include_agent_docs, max_hits=limit, mode=mode
        )
    except ValueError as e:  # empty query (guarded upstream, but be safe)
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 — a diagnostic must report, not traceback
        print(f"error: docs search failed: {e}", file=sys.stderr)
        if mode in ("vector", "hybrid"):
            print(
                "hint: the vector index may be corrupt or incompatible — rebuild "
                "with `forgather docs index --clean`.",
                file=sys.stderr,
            )
        return 1
    elapsed = time.perf_counter() - t0
    ran_mode = result.get("mode", "keyword")
    fell_back = mode in ("vector", "hybrid") and ran_mode == "keyword"
    diagnostics = {
        "source": "local",
        "repo_root": str(searched_root),
        "requested_mode": mode,
        "ran_mode": ran_mode,
        "vector_index_available": index_ok,
        "fell_back_to_keyword": fell_back,
        "elapsed_seconds": round(elapsed, 3),
        "hit_count": len(result["hits"]),
    }
    return _emit_search(result, diagnostics, as_json, verbose)


def _emit_search(result, diagnostics, as_json, verbose) -> int:
    """Render a search result (shared by the local and server paths).

    Reads the fallback / availability / source signals straight off the
    ``diagnostics`` dict it is handed, so there is one source of truth.
    """
    import json as _json

    if as_json:
        print(_json.dumps({**result, "diagnostics": diagnostics}, indent=2))
        return 0

    fell_back = diagnostics["fell_back_to_keyword"]
    vector_available = diagnostics["vector_index_available"]
    server = diagnostics["source"] == "server"

    if verbose:
        print(
            f"# mode: requested={diagnostics['requested_mode']} "
            f"ran={diagnostics['ran_mode']}  {diagnostics['elapsed_seconds']:.2f}s  "
            f"{diagnostics['hit_count']} hits",
            file=sys.stderr,
        )
    if fell_back:
        if vector_available:
            cause = (
                "check the server logs"
                if server
                else "re-run with --verbose for the cause"
            )
            print(
                "warning: vector/hybrid requested but ran keyword — the index "
                f"exists, so the embedder was unavailable ({cause}).",
                file=sys.stderr,
            )
        else:
            print(
                "warning: vector/hybrid requested but no index found — ran "
                "keyword. Build the index with `forgather docs index`.",
                file=sys.stderr,
            )

    hits = result["hits"]
    if not hits:
        print("No matches.")
        return 0
    for i, h in enumerate(hits, 1):
        excerpt = " ".join((h.get("excerpt") or "").split())
        if len(excerpt) > 200:
            excerpt = excerpt[:200].rstrip() + "…"
        print(f"{i:>2}. {h['rel']}  (score {h['score']})")
        print(f"    {h['path']}")
        if excerpt:
            print(f"    {excerpt}")
    return 0


def _index_summary(repo_root: Path) -> str:
    """Short ``model=… chunks=…`` for the verbose vector-index line (best-effort)."""
    import json as _json

    from forgather import docs_index

    try:
        meta = _json.loads((docs_index.index_dir(repo_root) / "meta.json").read_text())
        return f"model={meta.get('model')} chunks={meta.get('count')}"
    except Exception:  # noqa: BLE001 — diagnostics only
        return "meta unavailable"


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
        # A running server re-reads the index on its next vector query (the
        # runtime cache is keyed on meta.json's mtime), so no restart is needed.
        print("a running forgather server picks this up automatically (no restart).")
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
