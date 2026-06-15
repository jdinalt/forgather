#!/usr/bin/env python3
"""Warm-start vs from-scratch: does a pretrained start close the async gap?

The central from-scratch finding (README §3.7) is that async DiLoCo — even
DN-stabilized and depth-tuned — does NOT reach the sync baseline from random
init, while the source papers warm-start their async runs from a pretrained
checkpoint. This script tests that directly: the same async arms, started from a
500M-token DDP checkpoint instead of scratch, judged against a *warm* sync
baseline (same start point — the fair comparison).

Two panels (the warm and scratch runs are deliberately NOT overlaid on one loss
axis — their y-ranges differ vastly, scratch ~9->2.85 vs warm ~3.2->2.83, which
would squash the warm detail; the cross-comparison is the *gap* panel, which
normalizes each group to its own baseline):
  * Left — the warm arms only, eval loss: baseline + async (+ DyLU) all hug each
    other near the warm baseline. (Warm curves start mid-descent, from the
    checkpoint.) Own y-range.
  * Right — the async gap to its *matched* sync baseline, scratch vs warm, per
    arm (deltas, so comparable): the collapse from +0.25..+0.80 (scratch) to
    ~+0.03..+0.06 (warm).

Reads the committed ``assets/curves.csv`` (analysis/harvest.py).

    python analysis/warm_compare.py
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")

# async arm key -> (scratch series, warm series, label). Only arms run in BOTH
# regimes appear in the gap comparison; the DyLU arms are warm-only (§3.7.3).
PAIRS = [
    ("async_dn4", "warm_async_dn4", "Async DN N=4"),
    ("async_dn8", "warm_async_dn8", "Async DN N=8"),
]
SCRATCH_BASE = "baseline"
WARM_BASE = "warm_baseline"


def load():
    data = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(ASSETS, "curves.csv")) as f:
        for row in csv.DictReader(f):
            data[row["series"]][row["metric"]].append(
                (int(row["step"]), float(row["value"]))
            )
    for s in data:
        for m in data[s]:
            data[s][m].sort()
    return data


def final_eval(data, s):
    ev = data.get(s, {}).get("eval_loss", [])
    return ev[-1][1] if ev else float("nan")


def main():
    data = load()
    sb = final_eval(data, SCRATCH_BASE)
    wb = final_eval(data, WARM_BASE)

    print(f"scratch baseline {sb:.4f} | warm baseline {wb:.4f}\n")
    print(f"{'arm':<22}{'scratch gap':>12}{'warm gap':>12}")
    rows = []
    for sk, wk, lbl in PAIRS:
        sg = final_eval(data, sk) - sb
        wg = final_eval(data, wk) - wb
        rows.append((lbl, sg, wg))
        print(f"{lbl:<22}{sg:>+12.4f}{wg:>+12.4f}")

    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: WARM arms only (own y-range — never overlaid with the scratch runs,
    # whose ~9->2.85 range would squash this). They all converge onto the warm
    # baseline.
    ax = axes[0]
    traj = [
        (WARM_BASE, "Warm baseline (sync)", "#000000", "-"),
        ("warm_async_dn4", "Warm async DN N=4", "#74c476", "-"),
        ("warm_async_dn8", "Warm async DN N=8", "#2ca25f", "-"),
    ]
    for s, lbl, c, ls in traj:
        ev = data.get(s, {}).get("eval_loss", [])
        if not ev:
            continue
        ax.plot(
            [x for x, _ in ev],
            [v for _, v in ev],
            color=c,
            ls=ls,
            lw=1.7,
            marker="o",
            ms=2.5,
            label=lbl,
        )
    ax.set_title("Warm-started arms — all reach the warm baseline")
    ax.set_xlabel("local step (∝ total tokens)")
    ax.set_ylabel("eval loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Right: gap-to-matched-baseline, scratch vs warm.
    ax = axes[1]
    labels = [r[0] for r in rows]
    x = range(len(rows))
    w = 0.38
    ax.bar(
        [i - w / 2 for i in x],
        [r[1] for r in rows],
        w,
        label="from scratch",
        color="#d9772b",
    )
    ax.bar(
        [i + w / 2 for i in x],
        [r[2] for r in rows],
        w,
        label="warm-started",
        color="#2ca25f",
    )
    ax.axhline(0, color="#000000", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("eval-loss gap to matched sync baseline")
    ax.set_title("Async gap collapses with a warm start")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "Warm-start closes the from-scratch async gap — small Llama (34.4M), 4 workers, 2B tokens",
        fontweight="bold",
    )
    fig.tight_layout()
    out = os.path.join(ASSETS, "warm_compare.png")
    fig.savefig(out, bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
