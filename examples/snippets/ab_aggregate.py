#!/usr/bin/env python3
"""Pool one or more ``ab_test.py`` JSON logs into combined statistics.

Reads ab_test logs (files and/or directories), groups them by
``(model_a, model_b, seed)`` -- only runs that judged the *same* generated pairs
are pooled together -- and for each group reports the per-participant breakdown,
the pooled win counts, a two-sided sign test, and a position-bias check.

    python ab_aggregate.py  results_dir/  [more_files_or_dirs ...]
"""

import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict


def two_sided_sign_test(wins_a, wins_b):
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def load_logs(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.json")))
        else:
            files.append(p)
    logs = []
    for f in files:
        try:
            d = json.load(open(f))
        except (OSError, ValueError) as e:
            print(f"  (skipping {f}: {e})")
            continue
        if (
            isinstance(d, dict)
            and d.get("meta", {}).get("model_a_name")
            and "results" in d
        ):
            logs.append((f, d))
    return logs


def main(args):
    logs = load_logs(args.paths)
    if not logs:
        print("No ab_test logs found.")
        return

    groups = defaultdict(list)
    for f, d in logs:
        m = d["meta"]
        groups[(m["model_a_name"], m["model_b_name"], m.get("seed"))].append((f, d))

    for (na, nb, seed), runs in sorted(groups.items()):
        print("=" * 72)
        print(f"{na}  vs  {nb}   (seed {seed})")
        print("-" * 72)
        total = Counter()
        left = Counter()
        for f, d in runs:
            results = [r for r in d["results"] if r["winner"] in ("a", "b")]
            c = Counter(r["winner"] for r in results)
            total += c
            for r in results:
                left[r["shown_left"]] += 1
            tot_p = c["a"] + c["b"]
            part = d["meta"].get("participant", "?")
            share = (
                f"{100 * c['a'] / tot_p:.0f}% / {100 * c['b'] / tot_p:.0f}%"
                if tot_p
                else "-"
            )
            print(
                f"  {part:16s} {na}:{c['a']:3d}  {nb}:{c['b']:3d}  ({share})  "
                f"[{os.path.basename(f)}]"
            )
        a, b = total["a"], total["b"]
        n = a + b
        print("-" * 72)
        if n:
            print(
                f"  POOLED ({len(runs)} run(s), {n} decisions):  "
                f"{na} {a} ({100 * a / n:.0f}%)   {nb} {b} ({100 * b / n:.0f}%)"
            )
            print(f"  two-sided sign test p = {two_sided_sign_test(a, b):.4f}")
            print(
                f"  position check (model shown on the left): "
                f"{na}={left['a']}, {nb}={left['b']}  (want ~balanced)"
            )
        else:
            print("  (no decided comparisons)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "paths",
        nargs="+",
        help="ab_test JSON files and/or directories containing them",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
