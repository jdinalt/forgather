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
