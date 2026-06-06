#!/usr/bin/env python3
"""Generate the MLP-ablation plots for the Attention-Only experiment README.

Reads the latest ``trainer_logs.json`` for each run under ``output_models/`` and
emits, into this ``assets/`` directory:

  - loss_comparison.png    train + eval loss vs. tokens, all four runs
  - loss_endgame.png       same, zoomed to the final stretch
  - final_loss_bar.png     best eval loss per run (sorted)
  - results.csv            final/best metrics per run

Color encodes the ablation (blue = with MLP, orange = attention-only); line
style encodes size (solid = small, dashed = 4M). Run from the project root:
``python assets/generate_plots.py``
"""

import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
OUTPUT_MODELS = os.path.join(PROJECT, "output_models")

BLUE, ORANGE, GREEN, PURPLE = "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"

# output-model dir -> (label, color, linestyle). Order controls legend / bars.
MODELS = [
    ("mlp_small", "Attn + MLP (small, ~24M)", BLUE, "-"),
    ("attn_only_matched_small", "Attention-only, wide match (960x11)", GREEN, "-"),
    ("attn_only_deep_small", "Attention-only, deep match (512x38)", PURPLE, "-"),
    ("attn_only_small", "Attention-only (small, ~8M)", ORANGE, "-"),
    ("mlp_4m", "Attn + MLP (4M, ~4.6M)", BLUE, "--"),
    ("attn_only_4m", "Attention-only (4M, ~2.8M)", ORANGE, "--"),
]

ZOOM_TOKENS_M = 150.0


def load_latest(model_dir):
    runs = sorted(
        glob.glob(os.path.join(OUTPUT_MODELS, model_dir, "runs", "*", "")),
        key=os.path.getmtime,
    )
    for run in reversed(runs):
        path = os.path.join(run, "trainer_logs.json")
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                # A run still in progress may have a partially-written log;
                # fall back to an older complete run for this model.
                continue
    return None


def series(records, key):
    """X = tokens (millions), Y = metric. Eval records lack ``total_tokens``,
    so map their ``global_step`` onto the token axis via the train records."""
    step_to_tok = {
        r["global_step"]: r["total_tokens"]
        for r in records
        if "total_tokens" in r and "global_step" in r
    }
    steps = sorted(step_to_tok)

    def tokens_at(step):
        if step in step_to_tok:
            return step_to_tok[step]
        if not steps:
            return None
        return step_to_tok[min(steps, key=lambda s: abs(s - step))]

    xs, ys = [], []
    for r in records:
        if key not in r:
            continue
        tok = (
            r["total_tokens"]
            if "total_tokens" in r
            else tokens_at(r.get("global_step", -1))
        )
        if tok is None:
            continue
        xs.append(tok / 1e6)
        ys.append(r[key])
    return xs, ys


def main():
    data = {}
    for mdir, label, color, ls in MODELS:
        recs = load_latest(mdir)
        if recs is None:
            print(f"warning: no logs for {mdir}, skipping")
            continue
        data[mdir] = (label, recs, color, ls)

    # --- Figure 1: train + eval loss vs tokens -----------------------------
    fig, (ax_tr, ax_ev) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for mdir, (label, recs, color, ls) in data.items():
        tx, ty = series(recs, "loss")
        ax_tr.plot(tx, ty, ls, color=color, lw=1.7, label=label, alpha=0.9)
        ex, ey = series(recs, "eval_loss")
        ax_ev.plot(ex, ey, ls, color=color, lw=1.7, label=label, alpha=0.9)

    ax_tr.set_title("Training loss")
    ax_ev.set_title("Eval loss")
    for ax in (ax_tr, ax_ev):
        ax.set_xlabel("Tokens (millions)")
        ax.grid(True, alpha=0.3)
    ax_tr.set_ylabel("Cross-entropy loss")
    ax_ev.legend(fontsize=9, loc="upper right")
    fig.suptitle(
        "Single-head ALiBi on TinyStories — MLP ablation (250M tokens)",
        fontsize=12,
    )
    fig.tight_layout()
    out1 = os.path.join(HERE, "loss_comparison.png")
    fig.savefig(out1, dpi=120)
    print("wrote", out1)

    # --- Figure 1b: end-game zoom ------------------------------------------
    def tail(xs, ys):
        pairs = [(x, y) for x, y in zip(xs, ys) if x >= ZOOM_TOKENS_M]
        return ([x for x, _ in pairs], [y for _, y in pairs])

    figz, (zx_tr, zx_ev) = plt.subplots(1, 2, figsize=(13, 5.2))
    for mdir, (label, recs, color, ls) in data.items():
        tx, ty = tail(*series(recs, "loss"))
        zx_tr.plot(tx, ty, ls, color=color, lw=1.5, label=label, alpha=0.9)
        ex, ey = tail(*series(recs, "eval_loss"))
        zx_ev.plot(
            ex, ey, ls, color=color, lw=1.6, marker="o", ms=3, label=label, alpha=0.9
        )

    zx_tr.set_title(f"Training loss — end-game (>= {ZOOM_TOKENS_M:.0f}M tokens)")
    zx_ev.set_title(f"Eval loss — end-game (>= {ZOOM_TOKENS_M:.0f}M tokens)")
    for ax in (zx_tr, zx_ev):
        ax.set_xlabel("Tokens (millions)")
        ax.grid(True, alpha=0.3)
    zx_tr.set_ylabel("Cross-entropy loss")
    zx_ev.legend(fontsize=9, loc="upper right")
    figz.suptitle(
        "MLP ablation — end-game detail (y-axis auto-scaled to the tail)",
        fontsize=12,
    )
    figz.tight_layout()
    out1b = os.path.join(HERE, "loss_endgame.png")
    figz.savefig(out1b, dpi=120)
    print("wrote", out1b)

    # --- Figure 2: best eval loss bar chart --------------------------------
    bars = []
    for mdir, (label, recs, color, ls) in data.items():
        evals = [r["eval_loss"] for r in recs if "eval_loss" in r]
        if evals:
            bars.append((label, min(evals), color))
    bars.sort(key=lambda b: b[1])

    fig2, ax = plt.subplots(figsize=(8, 4))
    ypos = range(len(bars))
    ax.barh(
        list(ypos),
        [b[1] for b in bars],
        color=[b[2] for b in bars],
        edgecolor="black",
        alpha=0.85,
    )
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([b[0] for b in bars])
    ax.invert_yaxis()
    ax.set_xlabel("Best eval loss (lower is better)")
    ax.set_title("Best eval loss")
    lo = min(b[1] for b in bars)
    hi = max(b[1] for b in bars)
    span = (hi - lo) or 1.0
    ax.set_xlim(lo - 0.20 * span, hi + 0.30 * span)
    for i, b in enumerate(bars):
        ax.text(b[1] + 0.02 * span, i, f"{b[1]:.3f}", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig2.tight_layout()
    out2 = os.path.join(HERE, "final_loss_bar.png")
    fig2.savefig(out2, dpi=120)
    print("wrote", out2)

    # --- results.csv -------------------------------------------------------
    out3 = os.path.join(HERE, "results.csv")
    with open(out3, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "model",
                "final_train_loss",
                "final_eval_loss",
                "best_eval_loss",
                "avg_mfu_pct",
                "total_tokens_m",
            ]
        )
        for mdir, label, color, ls in MODELS:
            if mdir not in data:
                continue
            _, recs, _, _ = data[mdir]
            trains = [r for r in recs if "loss" in r]
            evals = [r["eval_loss"] for r in recs if "eval_loss" in r]
            mfus = [r["mfu"] for r in trains if r.get("mfu")]
            w.writerow(
                [
                    mdir,
                    f"{trains[-1]['loss']:.4f}" if trains else "",
                    f"{evals[-1]:.4f}" if evals else "",
                    f"{min(evals):.4f}" if evals else "",
                    f"{100 * sum(mfus) / len(mfus):.2f}" if mfus else "",
                    f"{trains[-1]['total_tokens'] / 1e6:.1f}" if trains else "",
                ]
            )
    print("wrote", out3)


if __name__ == "__main__":
    main()
