"""Argument parser for the `forgather docs` subcommand."""

import argparse
from argparse import RawTextHelpFormatter


def create_docs_parser(_global_args):
    parser = argparse.ArgumentParser(
        prog="forgather docs",
        description="Build / inspect the pre-rendered docs cache used by the webui docs viewer",
        formatter_class=RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="docs_action", required=False)

    build = sub.add_parser(
        "build",
        help="Expand directives (e.g. ::: mkdocstrings) and write to docs/.built/",
    )
    build.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (default: auto-detect via the forgather package).",
    )
    build.add_argument(
        "--docs-dir",
        default=None,
        help="Docs source directory (default: <repo-root>/docs).",
    )
    build.add_argument(
        "--clean",
        action="store_true",
        help="Remove docs/.built/ before building.",
    )
    build.add_argument(
        "--path",
        default=None,
        help="Restrict the build to a subtree (relative to docs-dir or absolute).",
    )
    build.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any page would be rebuilt; don't write anything.",
    )
    build.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Print only errors and the final summary line.",
    )

    clean = sub.add_parser("clean", help="Remove the docs/.built/ tree.")
    clean.add_argument("--docs-dir", default=None)
    clean.add_argument("--repo-root", default=None)

    search = sub.add_parser(
        "search",
        help="Search the docs corpus (keyword / vector / hybrid) — the same ranker the webui and agent use",
        formatter_class=RawTextHelpFormatter,
    )
    search.add_argument(
        "query",
        nargs="+",
        help="Search terms (joined into one query; quote to be safe).",
    )
    search.add_argument(
        "--mode",
        choices=("keyword", "vector", "hybrid"),
        default="keyword",
        help=(
            "Ranker (default: keyword).\n"
            "  keyword       substring term-frequency; fully offline.\n"
            "  vector/hybrid semantic; need a prebuilt index (forgather docs index)\n"
            "                + sentence-transformers, and fall back to keyword when\n"
            "                either is missing (the printed mode reflects what ran)."
        ),
    )
    search.add_argument(
        "--limit", type=int, default=8, help="Max hits to return (default: 8)."
    )
    search.add_argument(
        "--no-agent-docs",
        action="store_true",
        help="Exclude CLAUDE.md / CLAUDE.d/ (mirror the user-facing webui scope; by\n"
        "default the agent docs ARE included, matching the search_docs tool).",
    )
    search.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON ({query, mode, hits, diagnostics}) instead of text.",
    )
    search.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print diagnostics (index availability, model, timing, fallback\n"
        "reason, and any vector embed error) to stderr.",
    )
    search.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (default: auto-detect via the forgather package).",
    )

    index = sub.add_parser(
        "index",
        help="Build the vector-search index (docs/.built/.vector/) for hybrid docs search",
    )
    index.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (default: auto-detect via the forgather package).",
    )
    index.add_argument(
        "--docs-dir",
        default=None,
        help="Docs source directory (default: <repo-root>/docs).",
    )
    index.add_argument(
        "--model",
        default=None,
        help="sentence-transformers model id (default: all-MiniLM-L6-v2).",
    )
    index.add_argument(
        "--clean",
        action="store_true",
        help="Remove the existing index before building.",
    )
    index.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the index is missing or stale; don't write.",
    )
    index.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Print only the final summary line.",
    )

    return parser
