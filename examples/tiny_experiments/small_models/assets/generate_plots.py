#!/usr/bin/env python3
"""Generate the architecture-comparison plots for the Small Models README.

Reads the latest ``trainer_logs.json`` for each model under ``output_models/``
and emits, into this ``assets/`` directory:

  - loss_comparison.png    train + eval loss vs. tokens, all architectures
  - loss_endgame.png       same, zoomed to the final stretch (>=600M tokens)
  - final_loss_bar.png     best eval loss per architecture (sorted)
  - results.csv            final/best metrics per model

Run from the project root:  python assets/generate_plots.py
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

# output-model dir -> display label. Order controls legend / bar order.
MODELS = [
    ("small_causal", "Causal (vanilla)"),
    ("small_llama", "Llama"),
    ("small_llama_canon", "Llama + Canon"),
    ("small_deepone", "DeepOne (ALiBi)"),
    ("small_qwen3", "Qwen3"),
    ("small_mistral", "Mistral"),
    ("small_gemma3", "Gemma-3"),
]


def load_latest(model_dir):
    runs = sorted(
        glob.glob(os.path.join(OUTPUT_MODELS, model_dir, "runs", "*", "")),
        key=os.path.getmtime,
    )
    for run in reversed(runs):
        path = os.path.join(run, "trainer_logs.json")
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
    return None


def series(records, key):
    """X = tokens (millions), Y = metric.

    Train records carry ``total_tokens``; eval records carry only
    ``global_step`` + ``eval_loss``, so map eval steps onto the token axis
    using the train records' (step -> total_tokens) relationship.
    """
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
    cmap = plt.get_cmap("tab10")
    data = {}
    for i, (mdir, label) in enumerate(MODELS):
        recs = load_latest(mdir)
        if recs is None:
            print(f"warning: no logs for {mdir}, skipping")
            continue
        data[mdir] = (label, recs, cmap(i % 10))

    # --- Figure 1: train + eval loss vs tokens -----------------------------
    fig, (ax_tr, ax_ev) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for mdir, (label, recs, color) in data.items():
        tx, ty = series(recs, "loss")
        ax_tr.plot(tx, ty, "-", color=color, lw=1.7, label=label, alpha=0.9)
        ex, ey = series(recs, "eval_loss")
        ax_ev.plot(ex, ey, "-", color=color, lw=1.7, label=label, alpha=0.9)

    ax_tr.set_title("Training loss")
    ax_ev.set_title("Eval loss")
    for ax in (ax_tr, ax_ev):
        ax.set_xlabel("Tokens (millions)")
        ax.grid(True, alpha=0.3)
    ax_tr.set_ylabel("Cross-entropy loss")
    ax_ev.legend(fontsize=9, loc="upper right")
    fig.suptitle(
        "Small Models (~30M params) on Fineweb-Edu, WSD schedule, 1B tokens",
        fontsize=12,
    )
    fig.tight_layout()
    out1 = os.path.join(HERE, "loss_comparison.png")
    fig.savefig(out1, dpi=120)
    print("wrote", out1)

    # --- Figure 1b: end-game zoom (>= ZOOM_TOKENS_M) ------------------------
    # At full scale the architectures pile on top of each other; zooming the
    # tail (and letting matplotlib auto-scale y to it) makes the final ordering
    # and the WSD anneal legible.
    ZOOM_TOKENS_M = 600.0

    def tail(xs, ys):
        pairs = [(x, y) for x, y in zip(xs, ys) if x >= ZOOM_TOKENS_M]
        return ([x for x, _ in pairs], [y for _, y in pairs])

    figz, (zx_tr, zx_ev) = plt.subplots(1, 2, figsize=(13, 5.2))
    for mdir, (label, recs, color) in data.items():
        tx, ty = tail(*series(recs, "loss"))
        zx_tr.plot(tx, ty, "-", color=color, lw=1.4, label=label, alpha=0.9)
        ex, ey = tail(*series(recs, "eval_loss"))
        zx_ev.plot(
            ex, ey, "-", color=color, lw=1.6, marker="o", ms=3, label=label, alpha=0.9
        )

    zx_tr.set_title(f"Training loss — end-game (>= {ZOOM_TOKENS_M:.0f}M tokens)")
    zx_ev.set_title(f"Eval loss — end-game (>= {ZOOM_TOKENS_M:.0f}M tokens)")
    for ax in (zx_tr, zx_ev):
        ax.set_xlabel("Tokens (millions)")
        ax.grid(True, alpha=0.3)
    zx_tr.set_ylabel("Cross-entropy loss")
    zx_ev.legend(fontsize=9, loc="upper right")
    figz.suptitle(
        "Small Models — end-game detail (y-axis auto-scaled to the tail)",
        fontsize=12,
    )
    figz.tight_layout()
    out1b = os.path.join(HERE, "loss_endgame.png")
    figz.savefig(out1b, dpi=120)
    print("wrote", out1b)

    # --- Figure 2: best eval loss bar chart --------------------------------
    bars = []
    for mdir, (label, recs, color) in data.items():
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
    ax.set_title("Best eval loss by architecture")
    # Zoom the loss axis to the spread of the values (a 0-based axis hides the
    # differences). Start just below the best and leave headroom for labels.
    lo = min(b[1] for b in bars)
    hi = max(b[1] for b in bars)
    span = hi - lo
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
        for mdir, label in MODELS:
            if mdir not in data:
                continue
            _, recs, _ = data[mdir]
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
