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
    add_search_args(search)

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


def add_search_args(parser):
    """Populate ``parser`` with the docs-search arguments.

    Shared by the ``forgather docs search`` subcommand and the top-level
    ``forgather search`` alias so they stay identical.
    """
    parser.add_argument(
        "query",
        nargs="+",
        help="Search terms (joined into one query).",
    )
    # Mode is a set of short, mutually-exclusive flags (default keyword). -h is
    # argparse's help, so hybrid is -H.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "-k", "--keyword", dest="mode", action="store_const", const="keyword",
        help="Keyword ranking (default): substring term-frequency, fully offline.",
    )
    mode.add_argument(
        "-v", "--vector", dest="mode", action="store_const", const="vector",
        help="Semantic vector ranking (needs a prebuilt index). Runs on the\n"
        "forgather server by default so the embedding model stays warm.",
    )
    mode.add_argument(
        "-H", "--hybrid", dest="mode", action="store_const", const="hybrid",
        help="Hybrid: vector + keyword fused (reciprocal-rank fusion). Like\n"
        "--vector, runs on the server by default.",
    )
    parser.set_defaults(mode="keyword")
    parser.add_argument(
        "--limit", type=int, default=8, help="Max hits to return (default: 8)."
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run the search in-process instead of on the server. For\n"
        "--vector/--hybrid this loads the embedding model locally (slow cold\n"
        "start) — useful for diagnostics; not the default.",
    )
    parser.add_argument(
        "--server",
        default=None,
        metavar="URL",
        help="forgather server base URL for --vector/--hybrid (default:\n"
        "$FORGATHER_SERVER_URL or the local default).",
    )
    parser.add_argument(
        "--no-agent-docs",
        action="store_true",
        help="Exclude CLAUDE.md / CLAUDE.d/ from a --local search (included by\n"
        "default, matching the search_docs tool). The server path always uses\n"
        "the user-facing corpus, which already excludes them.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON ({query, mode, hits, diagnostics}) instead of text.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print diagnostics (source, index availability, timing, fallback\n"
        "reason, vector embed errors) to stderr.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root for a --local search (default: auto-detect).",
    )
    return parser


def create_search_parser(_global_args):
    """Top-level ``forgather search`` — an alias for ``forgather docs search``."""
    parser = argparse.ArgumentParser(
        prog="forgather search",
        description="Search the Forgather docs (alias for `forgather docs search`)",
        formatter_class=RawTextHelpFormatter,
    )
    add_search_args(parser)
    return parser
