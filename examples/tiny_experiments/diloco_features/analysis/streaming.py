#!/usr/bin/env python3
"""Streaming DiLoCo: fragment count + assignment vs the sync baseline.

Streaming DiLoCo syncs the model in block-boundary fragments instead of all at
once, smoothing the per-round communication. Two knobs: the number of fragments
(grain) and how blocks are assigned to fragments (`--fragment-assignment`
strided vs sequential). The paper reports strided slightly better, with the gap
growing at finer grain / smaller fragments. This compares, against the sync
baseline:
  * strided N=2   vs  sequential N=2  — the assignment A/B at a coarse grain;
  * strided N=5                       — a finer grain (2 blocks/fragment) where
                                        the strided edge is expected to show.

Reads the committed ``assets/curves.csv`` (from analysis/harvest.py), series
``stream_str2`` / ``stream_seq2`` / ``stream_str5`` plus ``baseline``.

    python analysis/streaming.py
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")

# (curves.csv series key, label, color)
RUNS = [
    ("stream_str2", "Strided N=2", "#2ca25f"),
    ("stream_seq2", "Sequential N=2", "#d9772b"),
    ("stream_str5", "Strided N=5", "#1f6fb2"),
]


def _curves():
    data = defaultdict(lambda: defaultdict(list))  # series -> metric -> [(step, v)]
    with open(os.path.join(ASSETS, "curves.csv")) as f:
        for row in csv.DictReader(f):
            data[row["series"]][row["metric"]].append(
                (int(row["step"]), float(row["value"]))
            )
    for s in data:
        for m in data[s]:
            data[s][m].sort()
    return data


def main():
    data = _curves()
    base_ev = data.get("baseline", {}).get("eval_loss", [])
    present = [(s, lbl, c) for s, lbl, c in RUNS if data.get(s, {}).get("eval_loss")]
    if not present:
        print("No streaming data in curves.csv — run analysis/harvest.py first.")
        return

    base_fe = base_ev[-1][1] if base_ev else float("nan")
    print(f"{'arm':<18}{'final eval':>12}{'vs sync':>10}")
    print(f"{'sync baseline':<18}{base_fe:>12.4f}{'—':>10}")
    for s, lbl, _ in present:
        fe = data[s]["eval_loss"][-1][1]
        print(f"{lbl:<18}{fe:>12.4f}{fe - base_fe:>+10.4f}")
    fe = {s: data[s]["eval_loss"][-1][1] for s, _, _ in present}
    if "stream_str2" in fe and "stream_seq2" in fe:
        print(
            f"\nassignment A/B (sequential - strided, N=2): "
            f"{fe['stream_seq2'] - fe['stream_str2']:+.4f}  "
            f"(positive => strided better)"
        )

    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
    fig, (axc, axb) = plt.subplots(1, 2, figsize=(12, 4.4))

    # Left: eval-loss trajectories vs the sync baseline.
    if base_ev:
        axc.plot(
            [s for s, _ in base_ev],
            [v for _, v in base_ev],
            color="#000000",
            lw=2.0,
            label="sync baseline",
        )
    for s, lbl, color in present:
        ev = data[s]["eval_loss"]
        axc.plot(
            [x for x, _ in ev],
            [v for _, v in ev],
            color=color,
            lw=1.4,
            marker="o",
            ms=3,
            label=lbl,
        )
    axc.set_title("Eval loss — streaming fragmentation")
    axc.set_xlabel("local step (∝ total tokens)")
    axc.set_ylabel("eval loss")
    axc.legend(fontsize=8)
    axc.grid(alpha=0.25)

    # Right: final eval bar (the small steady cost + the assignment gap).
    labels = [lbl for _, lbl, _ in present]
    fes = [data[s]["eval_loss"][-1][1] for s, _, _ in present]
    colors = [c for _, _, c in present]
    axb.bar(range(len(labels)), fes, color=colors)
    if base_ev:
        axb.axhline(base_fe, color="#000000", lw=1.5, ls="--", label="sync baseline")
        axb.legend(fontsize=8)
    axb.set_xticks(range(len(labels)))
    axb.set_xticklabels(labels, rotation=15, ha="right")
    axb.set_title("Final eval loss")
    axb.set_ylabel("final eval loss")
    axb.grid(alpha=0.25, axis="y")
    lo = min(fes + ([base_fe] if base_ev else []))
    hi = max(fes + ([base_fe] if base_ev else []))
    pad = (hi - lo) * 0.2 or 0.01
    axb.set_ylim(lo - pad, hi + pad)

    fig.suptitle(
        "Streaming DiLoCo — fragment count + assignment "
        "(small Llama 34.4M, 10 blocks, 4 workers, H=100)",
        fontweight="bold",
    )
    fig.tight_layout()
    out = os.path.join(ASSETS, "streaming.png")
    fig.savefig(out, bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
