#!/usr/bin/env python3
"""Token efficiency: sync DiLoCo at 2 vs 4 workers.

In DDP, raising the global batch size (more data-parallel ranks) usually buys
*less* per token — token efficiency drops with batch size. DiLoCo averages
pseudo-gradients over its workers, so more workers ≈ a larger effective batch per
sync round. Does the same penalty hold? This overlays the synchronous baseline at
2 workers (``baseline_2w``) and 4 workers (``baseline``) on a **total-tokens** axis
(both run the same 520M tokens/worker, so 4 workers consume 2× the total tokens).

    python analysis/worker_scaling.py
"""

import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")
RUNS_DIR = os.path.join(HERE, "runs")

# Fallback tokens/step for one worker (from the historical baseline TrainOutput:
# 520,444,917 tok over its 16062-step run) — used only if status.json is absent.
FALLBACK_TOK_PER_STEP_WORKER = 520_444_917 / 16062

# (curves.csv series / run dir, num_workers, label, color)
RUNS = [
    ("baseline_2w", 2, "2 workers (sync)", "#1f6fb2"),
    ("baseline", 4, "4 workers (sync)", "#c44e52"),
]


def actual_total_tokens(arm):
    """Authoritative total tokens for an arm from its captured /status snapshot
    (aggregate_stats.total_tokens), or None if unavailable."""
    path = os.path.join(RUNS_DIR, arm, "status.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, errors="replace") as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return (d.get("aggregate_stats") or {}).get("total_tokens")


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


def _interp(curve, x):
    """Linear-interpolate value at total-tokens x on a [(tokens, val)] curve."""
    if not curve or x < curve[0][0] or x > curve[-1][0]:
        return float("nan")
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return curve[-1][1]


def main():
    data = load()
    series = {}  # series -> {metric: [(total_tokens, val)]}
    for s, nw, _, _ in RUNS:
        if s not in data:
            print(f"  missing series '{s}' in curves.csv")
            continue
        # Map per-worker step -> total tokens. Prefer the captured actual
        # total_tokens (token-budget runs don't match a fixed tok/step);
        # factor = total_tokens / max_step. Fall back to the historical constant.
        max_step = max((st for m in data[s] for st, _ in data[s][m]), default=0)
        tot = actual_total_tokens(s)
        if tot and max_step:
            factor = tot / max_step
        else:
            factor = FALLBACK_TOK_PER_STEP_WORKER * nw
        series[s] = {m: [(step * factor, v) for step, v in data[s][m]] for m in data[s]}

    if "baseline_2w" in series and "baseline" in series:
        ev2 = series["baseline_2w"].get("eval_loss", [])
        ev4 = series["baseline"].get("eval_loss", [])
        t2_final = ev2[-1][0]
        print("Token efficiency — sync DiLoCo, 2w vs 4w:")
        print(f"  2w final: eval {ev2[-1][1]:.4f} @ {t2_final/1e9:.2f}B total tokens")
        print(
            f"  4w at that same {t2_final/1e9:.2f}B total tokens: "
            f"eval {_interp(ev4, t2_final):.4f}"
        )
        print(
            f"  4w final: eval {ev4[-1][1]:.4f} @ {ev4[-1][0]/1e9:.2f}B "
            f"(2× the tokens)"
        )

    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, metric, title, mk in (
        (axes[0], "train_loss", "Train loss", False),
        (axes[1], "eval_loss", "Eval loss", True),
    ):
        for s, _, label, color in RUNS:
            d = series.get(s, {}).get(metric, [])
            if d:
                ax.plot(
                    [x / 1e9 for x, _ in d],
                    [v for _, v in d],
                    color=color,
                    lw=1.6,
                    marker=("o" if mk else None),
                    ms=3,
                    label=label,
                )
        ax.set_title(title)
        ax.set_xlabel("total tokens (B)")
        ax.set_ylabel("loss")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Token efficiency: sync DiLoCo at 2 vs 4 workers (same 520M tok/worker)",
        fontweight="bold",
    )
    fig.tight_layout()
    out = os.path.join(ASSETS, "worker_scaling.png")
    fig.savefig(out, bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
