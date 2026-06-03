#!/usr/bin/env python3
"""Generate the architecture-comparison plots for the Tiny Models README.

Reads the latest ``trainer_logs.json`` for each model under ``output_models/``
and emits two figures into this ``assets/`` directory:

  - loss_comparison.png    train + eval loss vs. tokens, all architectures
  - final_loss_bar.png     best eval loss per architecture (sorted)

Forgather-native implementations are drawn solid; the HuggingFace reference
implementations are drawn dashed so the two families are easy to tell apart.

Run from the project root:  python assets/generate_plots.py
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
OUTPUT_MODELS = os.path.join(PROJECT, "output_models")

# Display label + family for each output-model directory. Order controls the
# legend / bar order. "hf" models are the HuggingFace reference implementations.
MODELS = [
    ("tiny_causal", "Causal (vanilla)", "fg"),
    ("tiny_fg_llama", "Llama", "fg"),
    ("tiny_deepone", "DeepOne", "fg"),
    ("tiny_fg_mistral", "Mistral", "fg"),
    ("tiny_fg_qwen3", "Qwen3", "fg"),
    ("tiny_llama_canon", "Llama + Canon", "fg"),
    ("tiny_singlehead", "Single-Head ALiBi", "fg"),
    ("tiny_hf_llama", "HF Llama", "hf"),
    ("tiny_hf_gpt2", "HF GPT-2", "hf"),
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

    Train records carry ``total_tokens`` directly. Eval records carry only
    ``global_step`` + ``eval_loss``, so map their step onto the token axis using
    the train records' (step -> total_tokens) relationship.
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
        # nearest known step (eval often fires a few steps off a log step)
        if not steps:
            return None
        nearest = min(steps, key=lambda s: abs(s - step))
        return step_to_tok[nearest]

    xs, ys = [], []
    for r in records:
        if key not in r:
            continue
        if "total_tokens" in r:
            tok = r["total_tokens"]
        else:
            tok = tokens_at(r.get("global_step", -1))
        if tok is None:
            continue
        xs.append(tok / 1e6)
        ys.append(r[key])
    return xs, ys


def main():
    cmap = plt.get_cmap("tab10")
    data = {}
    for i, (mdir, label, family) in enumerate(MODELS):
        recs = load_latest(mdir)
        if recs is None:
            print(f"warning: no logs for {mdir}, skipping")
            continue
        data[mdir] = (label, family, recs, cmap(i % 10))

    # --- Figure 1: train + eval loss vs tokens -----------------------------
    fig, (ax_tr, ax_ev) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for mdir, (label, family, recs, color) in data.items():
        ls = "--" if family == "hf" else "-"
        lw = 1.6 if family == "hf" else 1.8
        tx, ty = series(recs, "loss")
        ax_tr.plot(tx, ty, ls, color=color, lw=lw, label=label, alpha=0.9)
        ex, ey = series(recs, "eval_loss")
        ax_ev.plot(ex, ey, ls, color=color, lw=lw, label=label, alpha=0.9)

    ax_tr.set_title("Training loss")
    ax_ev.set_title("Eval loss")
    for ax in (ax_tr, ax_ev):
        ax.set_xlabel("Tokens (millions)")
        ax.grid(True, alpha=0.3)
    ax_tr.set_ylabel("Cross-entropy loss")
    ax_ev.legend(fontsize=8, ncol=1, loc="upper right")
    fig.suptitle(
        "Tiny Models (~4M params) on TinyStories — solid = Forgather, dashed = HF reference",
        fontsize=12,
    )
    fig.tight_layout()
    out1 = os.path.join(HERE, "loss_comparison.png")
    fig.savefig(out1, dpi=120)
    print("wrote", out1)

    # --- Figure 2: best eval loss bar chart --------------------------------
    bars = []
    for mdir, (label, family, recs, color) in data.items():
        evals = [r["eval_loss"] for r in recs if "eval_loss" in r]
        if evals:
            bars.append((label, family, min(evals), color))
    bars.sort(key=lambda b: b[2])

    fig2, ax = plt.subplots(figsize=(8, 4.5))
    ypos = range(len(bars))
    ax.barh(
        list(ypos),
        [b[2] for b in bars],
        color=[b[3] for b in bars],
        hatch=["//" if b[1] == "hf" else "" for b in bars],
        edgecolor="black",
        alpha=0.85,
    )
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([b[0] for b in bars])
    ax.invert_yaxis()
    ax.set_xlabel("Best eval loss (lower is better)")
    ax.set_title("Best eval loss by architecture (hatched = HF reference)")
    for i, b in enumerate(bars):
        ax.text(b[2] + 0.01, i, f"{b[2]:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, max(b[2] for b in bars) * 1.15)
    ax.grid(True, axis="x", alpha=0.3)
    fig2.tight_layout()
    out2 = os.path.join(HERE, "final_loss_bar.png")
    fig2.savefig(out2, dpi=120)
    print("wrote", out2)

    # --- results.csv: final/best metrics per model -------------------------
    import csv

    out3 = os.path.join(HERE, "results.csv")
    with open(out3, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "model",
                "family",
                "final_train_loss",
                "final_eval_loss",
                "best_eval_loss",
                "avg_mfu_pct",
                "total_tokens_m",
            ]
        )
        for mdir, label, family in MODELS:
            if mdir not in data:
                continue
            _, _, recs, _ = data[mdir]
            trains = [r for r in recs if "loss" in r]
            evals = [r["eval_loss"] for r in recs if "eval_loss" in r]
            mfus = [r["mfu"] for r in trains if r.get("mfu")]
            w.writerow(
                [
                    mdir,
                    family,
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
