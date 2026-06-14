#!/usr/bin/env python3
"""Grace mechanism check (VALIDATE-only): does grace coalesce, and proceed alone?

Grace is **not a study arm** (README §3.5) — its payoff is wall-clock tail-reduction
in a heterogeneous / large-N pool, which a homogeneous loopback rig can't measure.
What we *can* check is that the mechanism works to the paper's spec on the `validate`
``v_grace`` run: a finished worker **coalesces** with a near-simultaneous finisher
(histogram mass at 2+) *and* **proceeds immediately** when none arrive within S
(mass at 1) — it should NOT swallow all k every round (that would be synchronous
DiLoCo in disguise).

This reads the grace-batch histogram from the live /status snapshot
``experiment.sh`` captured (``runs/<arm>/status.json``) and reports, per arm:
  * the batch-size histogram (count of outer steps that coalesced 1, 2, ... k
    submissions);
  * the mean grace batch size;
  * the ALL-K-COALESCED FRACTION (the guardrail) — fraction of grace batches that
    swept all k workers. A high value means S is too wide (no proceed-alone path
    exercised; sync in disguise) — lower S and re-run the validate probe.

    python analysis/grace_batches.py            # default: the validate run `v_grace`
    python analysis/grace_batches.py <arm> ...  # specific run(s)

Writes a bar chart to assets/grace_hist.png and the histogram to
assets/grace_hist.csv.
"""

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")
RUNS = os.path.join(HERE, "runs")

# all-k-coalesced fraction above this => S is too wide (the proceed-alone path is
# never exercised; sync in disguise) — lower S and re-run the validate probe.
ALL_K_GUARDRAIL = 0.5


def load_grace(arm):
    """Return (histogram {batch_size: count}, num_workers) from status.json."""
    path = os.path.join(RUNS, arm, "status.json")
    if not os.path.isfile(path):
        return {}, None
    try:
        with open(path, errors="replace") as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}, None
    # orchestrator-routed `status --json` wraps the snapshot under "status".
    if isinstance(d.get("status"), dict):
        d = d["status"]
    hist = {int(k): v for k, v in (d.get("grace_batch_histogram") or {}).items()}
    return hist, d.get("num_workers")


def main():
    arms = sys.argv[1:] or ["v_grace"]
    os.makedirs(ASSETS, exist_ok=True)

    results = []  # (arm, hist, nw)
    for arm in arms:
        hist, nw = load_grace(arm)
        if not hist:
            print(f"  skip {arm}: no grace histogram at runs/{arm}/status.json")
            continue
        results.append((arm, hist, nw))

    if not results:
        print("No grace data — run a `--grace-period` arm first (experiment.sh).")
        return

    # CSV (committable): arm, batch_size, count.
    with open(os.path.join(ASSETS, "grace_hist.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "batch_size", "count"])
        for arm, hist, _ in results:
            for k in sorted(hist):
                w.writerow([arm, k, hist[k]])

    print(f"{'arm':<22}{'mean batch':>12}{'all-k frac':>12}{'verdict':>12}")
    for arm, hist, nw in results:
        total = sum(hist.values())
        batches = sum(k * v for k, v in hist.items())
        mean = batches / total if total else 0.0
        all_k = hist.get(nw, 0) / total if (nw and total) else float("nan")
        verdict = "OK"
        if all_k == all_k and all_k > ALL_K_GUARDRAIL:
            verdict = "TOO WIDE"
        elif mean < 1.5:
            verdict = "no-op"
        print(f"{arm:<22}{mean:>12.2f}{all_k:>12.2%}{verdict:>12}")
        # per-arm histogram line
        dist = "  ".join(f"k={k}:{hist[k]}" for k in sorted(hist))
        print(f"  hist (nw={nw}): {dist}")

    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.8 / max(1, len(results))
    for i, (arm, hist, nw) in enumerate(results):
        ks = sorted(hist)
        xs = [k + (i - (len(results) - 1) / 2) * width for k in ks]
        ax.bar(xs, [hist[k] for k in ks], width=width, label=arm)
        if nw:
            ax.axvline(nw, color="#c44e52", ls="--", lw=1.0, alpha=0.6)
    ax.set_title(
        "Grace coalescing — outer steps by # submissions merged\n"
        "(dashed = all-k = synchronous-in-disguise)",
        fontweight="bold",
        fontsize=10,
    )
    ax.set_xlabel("submissions coalesced into one outer step (one DN tick)")
    ax.set_ylabel("count of outer steps")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out = os.path.join(ASSETS, "grace_hist.png")
    fig.savefig(out, bbox_inches="tight")
    print("\nwrote", out, "+ assets/grace_hist.csv")


if __name__ == "__main__":
    main()
