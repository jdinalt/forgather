#!/usr/bin/env python3
"""Generate summary plots for the optimizer comparison experiment.

Reads trainer_logs.json from each training run and produces:
  - plots/amp_eval_loss_bar.png    -- Horizontal bar chart of AMP eval losses
  - plots/grad8_eval_loss_bar.png  -- Horizontal bar chart of grad8 eval losses
  - plots/memory_speed.png         -- Dual panel: peak memory + throughput
  - plots/bf16_impact.png          -- Grouped bars: AMP vs bf16 by optimizer

Usage:
    python generate_plots.py              # Generate all plots
    python generate_plots.py --dpi 300    # Higher resolution
    python generate_plots.py --show       # Display instead of saving

For loss curve plots, use generate_all_plots.sh which calls forgather logs plot.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Run definitions
# --------------------------------------------------------------------------- #

RUNS_DIR = Path(__file__).parent / "output_models" / "default" / "runs"
PLOTS_DIR = Path(__file__).parent / "plots"

# Map display name -> run directory name.
# Names ending with * are marked as in-progress in the charts.

AMP_RUNS = {
    "AdamW": "adamw_2026-03-15T01-42-11",
    "FG Adam": "fg_adam_2026-03-15T02-10-37",
    "FG Adafactor": "fg_adafactor_2026-03-15T01-45-49",
    "HF Adafactor": "hf_adafactor_2026-03-15T06-38-26",
    "Apollo": "apollo_2026-03-15T06-38-10",
    "Apollo PCA": "apollo_r64_pca_2026-03-15T06-40-49",
    "SinkGD": "sinkgd_2026-03-15T02-09-56",
    "Muon": "muon_2026-03-15T08-26-54",
    "SGD": "fg_sgd_2026-03-15T04-17-19",
    "Nesterov SGD": "nesterov_sgd_2026-03-15T06-39-57",
}

GRAD8_RUNS = {
    "AdamW": "adamw-8_2026-03-15T05-51-51",
    "Adafactor": "fg_adafactor-8_2026-03-15T05-59-36",
    "SinkGD": "sinkgd-8_2026-03-15T06-09-45",
    "Muon": "muon-8_2026-03-15T08-26-58",
}

BF16_RUNS = {
    "AdamW (bf16)": "adamw_bf16_2026-03-15T02-26-42",
    "FG Adam (bf16)": "fg_adam_bf16_2026-03-15T03-01-10",
    "Adafactor (bf16)": "fg_adafactor_bf16_2026-03-15T03-01-46",
}

COLORS = {
    "AdamW": "#2196F3",
    "FG Adam": "#1976D2",
    "FG Adafactor": "#4CAF50",
    "HF Adafactor": "#66BB6A",
    "Apollo": "#FF9800",
    "Apollo PCA": "#FFB74D",
    "SinkGD": "#9C27B0",
    "Muon": "#F44336",
    "SGD": "#795548",
    "Nesterov SGD": "#8D6E63",
    "Adafactor": "#4CAF50",
    "AdamW (bf16)": "#2196F3",
    "FG Adam (bf16)": "#1976D2",
    "Adafactor (bf16)": "#4CAF50",
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_records(run_name: str) -> list[dict]:
    """Load trainer_logs.json, handling truncated files from in-progress runs."""
    log_path = RUNS_DIR / run_name / "trainer_logs.json"
    if not log_path.exists():
        print(f"WARNING: {log_path} not found, skipping", file=sys.stderr)
        return []
    text = log_path.read_text().strip()
    if text.endswith(","):
        text = text[:-1]
    if not text.endswith("]"):
        text += "]"
    return json.loads(text)


def load_metrics(run_name: str) -> dict:
    """Extract summary metrics from a training run."""
    records = load_records(run_name)
    if not records:
        return {
            "best_eval": None,
            "peak_mem_gb": 0,
            "avg_tok_per_sec": 0,
            "avg_mfu": 0,
            "last_step": 0,
        }

    eval_losses = [
        (r["global_step"], r["eval_loss"]) for r in records if "eval_loss" in r
    ]
    best_eval = min(eval_losses, key=lambda x: x[1])[1] if eval_losses else None

    def _record_peak(r) -> float:
        # peak_mem_allocated is a per-rank list in current logs; old logs
        # used a scalar. Accept both.
        v = r.get("peak_mem_allocated", 0)
        if isinstance(v, list):
            return max(v) if v else 0
        return v or 0

    peak_mem: float = max((_record_peak(r) for r in records), default=0)
    tok_vals = [r["tok_per_sec"] for r in records if "tok_per_sec" in r]
    mfu_vals = [r["mfu"] for r in records if "mfu" in r]

    return {
        "best_eval": best_eval,
        "peak_mem_gb": peak_mem / (1024**3),
        "avg_tok_per_sec": np.mean(tok_vals) if tok_vals else 0,
        "avg_mfu": np.mean(mfu_vals) if mfu_vals else 0,
        "last_step": records[-1].get("global_step", 0),
    }


def color_for(name: str) -> str:
    return COLORS.get(name, "#607D8B")


def sorted_by_value(names, values):
    """Return (names, values) sorted by values ascending."""
    idx = np.argsort(values)
    return [names[i] for i in idx], [values[i] for i in idx]


# --------------------------------------------------------------------------- #
# Plot functions
# --------------------------------------------------------------------------- #


def plot_amp_eval_bar(metrics: dict, dpi: int) -> None:
    """Horizontal bar chart of AMP eval losses."""
    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(metrics.keys())
    vals = [metrics[n]["best_eval"] for n in names]

    names_s, vals_s = sorted_by_value(names, vals)
    colors_s = [color_for(n) for n in names_s]

    bars = ax.barh(names_s, vals_s, color=colors_s, edgecolor="white", height=0.6)
    ax.set_xlabel("Best Eval Loss", fontsize=12)
    ax.set_title(
        "AMP: Best Evaluation Loss by Optimizer (batch=32, ~560M tokens)", fontsize=13
    )
    ax.set_xlim(2.65, 3.1)

    for bar, val in zip(bars, vals_s):
        ax.text(
            val + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=10,
        )
    for i, name in enumerate(names_s):
        if "*" in name:
            ax.text(
                2.66,
                i,
                "(in progress)",
                va="center",
                fontsize=8,
                color="red",
                fontstyle="italic",
            )

    baseline = metrics.get("AdamW", {}).get("best_eval")
    if baseline:
        ax.axvline(
            x=baseline,
            color="#2196F3",
            linestyle="--",
            alpha=0.5,
            label="AdamW baseline",
        )
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "amp_eval_loss_bar.png", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_grad8_eval_bar(metrics: dict, dpi: int) -> None:
    """Horizontal bar chart of grad8 eval losses."""
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(metrics.keys())
    vals = [metrics[n]["best_eval"] for n in names]

    names_s, vals_s = sorted_by_value(names, vals)
    colors_s = [color_for(n) for n in names_s]

    bars = ax.barh(names_s, vals_s, color=colors_s, edgecolor="white", height=0.5)
    ax.set_xlabel("Best Eval Loss", fontsize=11)
    ax.set_title(
        "Gradient Accumulation 8x: Best Eval Loss (eff. batch=256)", fontsize=12
    )
    # Set xlim dynamically based on data
    min_val = min(vals_s) - 0.02
    max_val = max(vals_s) + 0.04
    ax.set_xlim(min_val, max_val)

    for bar, val in zip(bars, vals_s):
        ax.text(
            val + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=10,
        )
    for i, name in enumerate(names_s):
        if "*" in name:
            ax.text(
                min_val + 0.005,
                i,
                "(in progress)",
                va="center",
                fontsize=8,
                color="red",
                fontstyle="italic",
            )

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "grad8_eval_loss_bar.png", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_memory_speed(amp_metrics: dict, bf16_metrics: dict, dpi: int) -> None:
    """Dual panel: peak memory (left) and throughput (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    all_metrics = {**amp_metrics, **bf16_metrics}
    all_names = list(amp_metrics.keys()) + list(bf16_metrics.keys())

    # --- Memory panel ---
    mem_names = [n for n in all_names if all_metrics[n]["peak_mem_gb"] > 0]
    mem_vals = [all_metrics[n]["peak_mem_gb"] for n in mem_names]
    mem_names_s, mem_vals_s = sorted_by_value(mem_names, mem_vals)
    mem_colors = [color_for(n) for n in mem_names_s]

    ax1.barh(mem_names_s, mem_vals_s, color=mem_colors, edgecolor="white", height=0.6)
    for i, (name, val) in enumerate(zip(mem_names_s, mem_vals_s)):
        ax1.text(val + 0.02, i, f"{val:.2f} GB", va="center", fontsize=9)
    ax1.set_xlabel("Peak Memory Allocated (GB)", fontsize=11)
    ax1.set_title("Peak GPU Memory", fontsize=12)
    ax1.set_xlim(0, 4.5)

    # --- Speed panel ---
    speed_names = [
        n for n in list(amp_metrics.keys()) if amp_metrics[n]["avg_tok_per_sec"] > 0
    ]
    speed_vals = [amp_metrics[n]["avg_tok_per_sec"] / 1000 for n in speed_names]
    speed_names_s, speed_vals_s = sorted_by_value(speed_names, speed_vals)
    speed_names_s.reverse()
    speed_vals_s.reverse()
    speed_colors = [color_for(n) for n in speed_names_s]

    ax2.barh(
        speed_names_s, speed_vals_s, color=speed_colors, edgecolor="white", height=0.6
    )
    for i, (name, val) in enumerate(zip(speed_names_s, speed_vals_s)):
        ax2.text(val + 1, i, f"{val:.0f}K tok/s", va="center", fontsize=9)
    ax2.set_xlabel("Throughput (K tokens/sec)", fontsize=11)
    ax2.set_title("Training Throughput (AMP, batch=32)", fontsize=12)
    ax2.set_xlim(0, 320)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "memory_speed.png", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_bf16_impact(amp_metrics: dict, bf16_metrics: dict, dpi: int) -> None:
    """Grouped bar chart: AMP vs bf16 for each optimizer."""
    pairs = [
        ("AdamW", "AdamW (bf16)", "AdamW"),
        ("FG Adam", "FG Adam (bf16)", "FG Adam"),
        ("FG Adafactor", "Adafactor (bf16)", "Adafactor"),
    ]

    x = np.arange(len(pairs))
    width = 0.35

    amp_vals = [amp_metrics[p[0]]["best_eval"] for p in pairs]
    bf16_vals = [bf16_metrics[p[1]]["best_eval"] for p in pairs]
    pair_labels = [p[2] for p in pairs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(
        x - width / 2,
        amp_vals,
        width,
        label="AMP (fp32 weights)",
        color="#2196F3",
        edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2,
        bf16_vals,
        width,
        label="Pure bf16",
        color="#FF9800",
        edgecolor="white",
    )

    ax.set_ylabel("Best Eval Loss", fontsize=11)
    ax.set_title("AMP vs Pure bfloat16: Impact on Final Loss", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(2.6, 3.05)

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for i, (a, b) in enumerate(zip(amp_vals, bf16_vals)):
        delta = b - a
        ax.annotate(
            f"+{delta:.4f}",
            xy=(i, max(a, b) + 0.025),
            ha="center",
            fontsize=9,
            color="red",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "bf16_impact.png", dpi=dpi, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary plots for the optimizer comparison experiment."
    )
    parser.add_argument("--dpi", type=int, default=150, help="Plot DPI (default: 150)")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively instead of saving",
    )
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    PLOTS_DIR.mkdir(exist_ok=True)

    print("Loading training metrics...")
    amp_metrics = {k: load_metrics(v) for k, v in AMP_RUNS.items()}
    grad8_metrics = {k: load_metrics(v) for k, v in GRAD8_RUNS.items()}
    bf16_metrics = {k: load_metrics(v) for k, v in BF16_RUNS.items()}

    # Drop entries with no eval data
    amp_metrics = {k: v for k, v in amp_metrics.items() if v["best_eval"] is not None}
    grad8_metrics = {
        k: v for k, v in grad8_metrics.items() if v["best_eval"] is not None
    }
    bf16_metrics = {k: v for k, v in bf16_metrics.items() if v["best_eval"] is not None}

    print("Generating amp_eval_loss_bar.png...")
    plot_amp_eval_bar(amp_metrics, args.dpi)

    print("Generating grad8_eval_loss_bar.png...")
    plot_grad8_eval_bar(grad8_metrics, args.dpi)

    print("Generating memory_speed.png...")
    plot_memory_speed(amp_metrics, bf16_metrics, args.dpi)

    print("Generating bf16_impact.png...")
    plot_bf16_impact(amp_metrics, bf16_metrics, args.dpi)

    print(f"Done. Plots saved to {PLOTS_DIR}/")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
