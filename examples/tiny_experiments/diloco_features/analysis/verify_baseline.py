#!/usr/bin/env python3
"""Verify the gRPC+safetensors baseline reproduces the original 1B run.

The diloco_features baseline (sync DiLoCo, H=100, ~1B tokens, torch.optim.AdamW,
**gRPC + safetensors**) should match the diloco project's extended-sweep
``h100`` series (same model / seed / data / LR / H / budget / optimizer, run over
the historical **HTTP + pickle** transport). The bulk legs carry identical bf16
values regardless of codec, and gRPC vs HTTP is pure plumbing, so the loss
trajectories should overlay within run-to-run (CUDA/compile) nondeterminism — a
direct check that gRPC+safetensors are lossless.

Reads this project's harvested ``assets/curves.csv`` (baseline, per-worker step)
and the reference ``../diloco/assets/sweep_curves.csv`` (h100, total tokens),
puts both on a TOTAL-tokens axis, overlays them, and prints the final-metric gap.

    python analysis/verify_baseline.py
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")
REF_CSV = os.path.join(HERE, os.pardir, "diloco", "assets", "sweep_curves.csv")

# The baseline TrainOutput reported 520,444,917 tokens over its final step for one
# worker; with 2 workers that's the per-step total. We derive the per-worker step
# count from curves.csv itself (rather than hard-coding 16062) so the conversion
# follows the data if the budget changes.
BASELINE_TOKENS_PER_WORKER = 520_444_917
NUM_WORKERS = 2


def load_mine():
    """baseline {metric: [(total_tokens, value)]} from this project's curves.csv."""
    rows = defaultdict(list)  # metric -> [(step, value)]
    path = os.path.join(ASSETS, "curves.csv")
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["series"] != "baseline":
                continue
            rows[row["metric"]].append((int(row["step"]), float(row["value"])))
    max_step = max((s for m in rows for s, _ in rows[m]), default=0)
    if not max_step:
        return {}
    # step -> total tokens across all workers (curves are per-worker step).
    tok_per_step_total = BASELINE_TOKENS_PER_WORKER / max_step * NUM_WORKERS
    out = defaultdict(list)
    for m, pts in rows.items():
        out[m] = sorted((step * tok_per_step_total, v) for step, v in pts)
    return out


def load_ref():
    """h100 {metric: [(tokens, value)]} from the diloco sweep (already total tokens)."""
    out = defaultdict(list)
    with open(REF_CSV) as f:
        for row in csv.DictReader(f):
            if row["run"] != "h100":
                continue
            # sweep metric names are 'train'/'eval'; normalize to *_loss.
            m = {"train": "train_loss", "eval": "eval_loss"}.get(
                row["metric"], row["metric"]
            )
            out[m].append((int(row["tokens"]), float(row["value"])))
    for m in out:
        out[m].sort()
    return out


def final(series, metric):
    d = series.get(metric, [])
    return d[-1][1] if d else float("nan")


def best(series, metric):
    d = series.get(metric, [])
    return min((v for _, v in d), default=float("nan"))


def main():
    mine, ref = load_mine(), load_ref()
    if not mine:
        print("No baseline in assets/curves.csv — run analysis/harvest.py first.")
        return

    print("                         mine (gRPC+st)   h100 (HTTP+pickle)     delta")
    for label, m in (
        ("final train loss", "train_loss"),
        ("final eval loss", "eval_loss"),
    ):
        a, b = final(mine, m), final(ref, m)
        print(f"  {label:<22}{a:>12.4f}{b:>20.4f}{a-b:>+11.4f}")
    a, b = best(mine, "eval_loss"), best(ref, "eval_loss")
    print(f"  {'best eval loss':<22}{a:>12.4f}{b:>20.4f}{a-b:>+11.4f}")

    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, metric, title, mk in (
        (axes[0], "train_loss", "Train loss", False),
        (axes[1], "eval_loss", "Eval loss", True),
    ):
        for series, name, color in (
            (ref, "original h100 (HTTP+pickle)", "#999999"),
            (mine, "this baseline (gRPC+safetensors)", "#000000"),
        ):
            d = series.get(metric, [])
            if d:
                ax.plot(
                    [x / 1e9 for x, _ in d],
                    [v for _, v in d],
                    color=color,
                    lw=2.0 if "this" in name else 3.0,
                    alpha=1.0 if "this" in name else 0.6,
                    marker=("o" if mk and "this" in name else None),
                    ms=3,
                    label=name,
                )
        ax.set_title(title)
        ax.set_xlabel("total tokens (B)")
        ax.set_ylabel("loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Baseline verification: gRPC+safetensors vs original HTTP+pickle "
        "(DiLoCo 2w, H=100, ~1B, torch AdamW)",
        fontweight="bold",
    )
    fig.tight_layout()
    out = os.path.join(ASSETS, "baseline_vs_h100.png")
    fig.savefig(out, bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
