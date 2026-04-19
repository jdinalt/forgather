#!/usr/bin/env python3
"""Concatenate multiple trainer_logs.json files into a single continuous log.

Handles overlapping step ranges by using the last file's start step as the
cutoff point for the previous file. This is useful when training was interrupted
(e.g., power outage) and resumed from a checkpoint, producing overlapping logs.

Usage:
    python concat_training_logs.py log_dir1/trainer_logs.json log_dir2/trainer_logs.json [-o output.json]

    # Also accepts run directories directly (looks for trainer_logs.json inside):
    python concat_training_logs.py runs/log_2026-02-17T10-16-44 runs/log_2026-02-18T08-51-57

    # Output defaults to combined_logs.json in current directory
"""

import argparse
import json
import sys
from pathlib import Path


def load_log(path: Path) -> list[dict]:
    """Load a trainer_logs.json file.

    The file is a JSON array, one entry per line (with leading [ and trailing ]).
    Some entries are eval-only (no 'loss' key), which we keep as-is.

    Handles truncated files (e.g., from power outages) where the closing ]
    may be missing or the last line may be incomplete.
    """
    text = path.read_text()

    # Try standard parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try fixing: strip trailing whitespace, ensure closing bracket
    text = text.rstrip()
    if text.endswith(","):
        text = text[:-1]
    if not text.endswith("]"):
        text += "\n]"

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last line might be a partial JSON object; drop it
        lines = text.rsplit("\n", 2)
        text = lines[0].rstrip().rstrip(",") + "\n]"
        print(f"Warning: dropped truncated last entry from {path}", file=sys.stderr)
        return json.loads(text)


def concat_logs(log_files: list[Path]) -> list[dict]:
    """Concatenate log files, trimming overlap based on global_step.

    For each consecutive pair of files, entries from the earlier file whose
    global_step >= the first global_step of the later file are discarded.
    """
    if not log_files:
        return []

    all_logs = [(path, load_log(path)) for path in log_files]

    # Sort by the first global_step in each file
    all_logs.sort(key=lambda x: x[1][0]["global_step"] if x[1] else 0)

    result = []
    for i, (path, entries) in enumerate(all_logs):
        if not entries:
            print(f"Warning: {path} is empty, skipping", file=sys.stderr)
            continue

        first_step = entries[0]["global_step"]
        last_step = entries[-1]["global_step"]

        if i + 1 < len(all_logs):
            next_entries = all_logs[i + 1][1]
            if next_entries:
                cutoff_step = next_entries[0]["global_step"]
                kept = [e for e in entries if e["global_step"] < cutoff_step]
                trimmed = len(entries) - len(kept)
                print(
                    f"{path.name}: steps {first_step}..{last_step}, "
                    f"trimmed {trimmed} entries >= step {cutoff_step}",
                    file=sys.stderr,
                )
                result.extend(kept)
            else:
                result.extend(entries)
        else:
            print(
                f"{path.name}: steps {first_step}..{last_step}, "
                f"{len(entries)} entries (final file, kept all)",
                file=sys.stderr,
            )
            result.extend(entries)

    print(f"\nTotal: {len(result)} entries", file=sys.stderr)
    return result


def resolve_log_path(p: str) -> Path:
    """Resolve a path to a trainer_logs.json file.

    Accepts either a direct path to the JSON file, or a run directory
    containing trainer_logs.json.
    """
    path = Path(p)
    if path.is_dir():
        candidate = path / "trainer_logs.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No trainer_logs.json found in directory: {path}")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate overlapping trainer_logs.json files"
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help="Paths to trainer_logs.json files or run directories",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="combined_logs.json",
        help="Output file path (default: combined_logs.json)",
    )
    args = parser.parse_args()

    log_files = [resolve_log_path(p) for p in args.logs]
    combined = concat_logs(log_files)

    output = Path(args.output)
    with open(output, "w") as f:
        f.write("[\n")
        for i, entry in enumerate(combined):
            comma = "," if i < len(combined) - 1 else ""
            f.write(json.dumps(entry) + comma + "\n")
        f.write("]\n")

    print(f"Written to {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
