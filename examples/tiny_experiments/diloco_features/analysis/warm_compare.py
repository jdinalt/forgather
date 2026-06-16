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
  * Right — RELATIVE PERPLEXITY vs each group's *own* matched sync baseline
    (% worse, = (exp(eval - base) - 1)·100), scratch vs warm, per arm. This is the
    comparable cross-regime metric — raw losses are NOT comparable (the warm group
    trained 500M more tokens), but "% worse perplexity than your own baseline" is.
    The collapse: ~+123%/+28% (scratch) to ~+7% (warm).

Reads the committed ``assets/curves.csv`` (analysis/harvest.py).

    python analysis/warm_compare.py
"""

import csv
import math
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

    # The comparable quantity across regimes is RELATIVE PERPLEXITY vs each group's
    # OWN baseline: % worse perplexity = (exp(eval - base_eval) - 1) * 100. Raw
    # losses are not comparable (the warm group has 500M more tokens), but "how much
    # worse than your own sync baseline" is — and perplexity (the DiLoCo paper's
    # axis) is the natural unit for a log-scale loss.
    def rel_ppl_pct(arm, base_eval):
        return (math.exp(final_eval(data, arm) - base_eval) - 1.0) * 100.0

    print(f"scratch baseline {sb:.4f} | warm baseline {wb:.4f}\n")
    print(f"{'arm':<22}{'scratch %ppl':>14}{'warm %ppl':>12}")
    rows = []
    for sk, wk, lbl in PAIRS:
        sg = rel_ppl_pct(sk, sb)
        wg = rel_ppl_pct(wk, wb)
        rows.append((lbl, sg, wg))
        print(f"{lbl:<22}{sg:>+13.1f}%{wg:>+11.1f}%")

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

    # Right: RELATIVE PERPLEXITY vs each group's matched sync baseline (% worse),
    # scratch vs warm — the comparable cross-regime metric (raw losses aren't,
    # since the warm group trained 500M more tokens).
    ax = axes[1]
    labels = [r[0] for r in rows]
    x = range(len(rows))
    w = 0.38
    bars_s = ax.bar(
        [i - w / 2 for i in x],
        [r[1] for r in rows],
        w,
        label="from scratch",
        color="#d9772b",
    )
    bars_w = ax.bar(
        [i + w / 2 for i in x],
        [r[2] for r in rows],
        w,
        label="warm-started",
        color="#2ca25f",
    )
    for bars in (bars_s, bars_w):
        for b in bars:
            ax.annotate(
                f"+{b.get_height():.0f}%",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.axhline(0, color="#000000", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("perplexity vs matched sync baseline (% worse)")
    ax.set_title("Async perplexity gap collapses with a warm start")
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
