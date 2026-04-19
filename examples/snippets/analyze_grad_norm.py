#!/usr/bin/env python3
"""Analyze grad_norm trends from combined training logs.

Computes 10K-step bucket statistics, fits various curves, and generates
diagnostic plots.

Usage:
    python analyze_grad_norm.py <combined_logs.json> [-o output_dir]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def load_grad_norms(path: Path) -> tuple[np.ndarray, np.ndarray]:
    logs = json.loads(path.read_text())
    entries = [(e["global_step"], e["grad_norm"]) for e in logs if "grad_norm" in e]
    steps = np.array([s for s, _ in entries])
    gnorms = np.array([g for _, g in entries])
    return steps, gnorms


def bucket_stats(
    steps: np.ndarray, values: np.ndarray, bucket_size: int = 10000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (centers, means, stds, mins, maxs) for each bucket."""
    max_step = int(steps[-1])
    centers, means, stds, mins, maxs, counts = [], [], [], [], [], []
    for start in range(0, max_step + 1, bucket_size):
        end = start + bucket_size
        mask = (steps >= start) & (steps < end)
        if mask.sum() < 10:  # skip tiny buckets
            continue
        vals = values[mask]
        centers.append(start + bucket_size / 2)
        means.append(vals.mean())
        stds.append(vals.std())
        mins.append(vals.min())
        maxs.append(vals.max())
        counts.append(mask.sum())
    return tuple(np.array(a) for a in (centers, means, stds, mins, maxs))


# --- Fit functions ---


def linear(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def exponential(x, a, b):
    return a * np.exp(b * x)


def power_law(x, a, b):
    return a * np.power(x, b)


def logarithmic(x, a, b):
    return a * np.log(x) + b


def sqrt_func(x, a, b):
    return a * np.sqrt(x) + b


MODELS = {
    "Linear": (linear, lambda x, y: [1e-6, y[0]]),
    "Quadratic": (quadratic, lambda x, y: [1e-11, 1e-6, y[0]]),
    "Exponential": (exponential, lambda x, y: [y[0], 1e-6]),
    "Power law": (power_law, lambda x, y: [0.01, 0.3]),
    "Logarithmic": (logarithmic, lambda x, y: [0.1, 0]),
    "Sqrt": (sqrt_func, lambda x, y: [1e-3, y[0]]),
}


def fit_all(x, y, skip_first=False):
    """Fit all models, return dict of {name: (popt, r2, y_pred)}."""
    if skip_first:
        x, y = x[1:], y[1:]
    results = {}
    ss_tot = np.sum((y - y.mean()) ** 2)
    for name, (func, p0_fn) in MODELS.items():
        try:
            popt, _ = curve_fit(func, x, y, p0=p0_fn(x, y), maxfev=50000)
            y_pred = func(x, *popt)
            r2 = 1 - np.sum((y - y_pred) ** 2) / ss_tot
            results[name] = (popt, r2, y_pred, x)
        except Exception as e:
            results[name] = (None, None, None, x)
    return results


def print_fit_results(results: dict, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    ranked = sorted(results.items(), key=lambda kv: kv[1][1] or -1, reverse=True)
    for name, (popt, r2, _, _) in ranked:
        if r2 is None:
            print(f"  {name:14s}  FAILED")
            continue
        params = ", ".join(f"{p:.4e}" for p in popt)
        marker = " <-- best" if name == ranked[0][0] else ""
        print(f"  {name:14s}  R2={r2:.6f}  params=({params}){marker}")


def plot_analysis(
    steps, gnorms, centers, means, stds, mins, maxs, fit_mean, fit_std, output_dir: Path
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Grad-Norm Analysis", fontsize=14, fontweight="bold")

    # --- Top left: raw grad_norm with bucket stats ---
    ax = axes[0, 0]
    ax.scatter(steps, gnorms, s=0.3, alpha=0.15, color="steelblue", rasterized=True)
    ax.plot(
        centers,
        means,
        "o-",
        color="red",
        linewidth=2,
        markersize=4,
        label="Bucket mean",
    )
    ax.fill_between(
        centers,
        means - stds,
        means + stds,
        alpha=0.25,
        color="red",
        label="Mean +/- 1 std",
    )
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Grad Norm")
    ax.set_title("Raw grad_norm with 10K-step bucket statistics")
    ax.legend(loc="upper left")
    ax.set_ylim(0, np.percentile(gnorms, 99.5) * 1.2)
    ax.grid(True, alpha=0.3)

    # --- Top right: best fits on mean ---
    ax = axes[0, 1]
    ax.plot(centers, means, "ko", markersize=6, label="Bucket means", zorder=5)
    ranked = sorted(fit_mean.items(), key=lambda kv: kv[1][1] or -1, reverse=True)
    colors = ["red", "blue", "green", "orange", "purple", "brown"]
    for i, (name, (popt, r2, y_pred, xf)) in enumerate(ranked[:4]):
        if r2 is None:
            continue
        ax.plot(
            xf,
            y_pred,
            "-",
            color=colors[i],
            linewidth=1.5,
            label=f"{name} (R2={r2:.4f})",
            alpha=0.8,
        )
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Mean Grad Norm")
    ax.set_title("Curve fits: grad_norm mean")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Bottom left: std with fits ---
    ax = axes[1, 0]
    ax.plot(centers, stds, "ko", markersize=6, label="Bucket std", zorder=5)
    ranked_s = sorted(fit_std.items(), key=lambda kv: kv[1][1] or -1, reverse=True)
    for i, (name, (popt, r2, y_pred, xf)) in enumerate(ranked_s[:3]):
        if r2 is None:
            continue
        ax.plot(
            xf,
            y_pred,
            "-",
            color=colors[i],
            linewidth=1.5,
            label=f"{name} (R2={r2:.4f})",
            alpha=0.8,
        )
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Std of Grad Norm")
    ax.set_title("Curve fits: grad_norm std (first bucket excluded)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Bottom right: min/max envelope ---
    ax = axes[1, 1]
    ax.fill_between(
        centers, mins, maxs, alpha=0.2, color="steelblue", label="Min-Max range"
    )
    ax.fill_between(
        centers,
        means - stds,
        means + stds,
        alpha=0.4,
        color="red",
        label="Mean +/- 1 std",
    )
    ax.plot(centers, means, "r-", linewidth=2, label="Mean")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Grad Norm")
    ax.set_title("Grad-norm envelope (10K buckets)")
    ax.legend(loc="upper left")
    ax.set_ylim(0, np.percentile(maxs, 90) * 1.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "grad_norm_analysis.png"
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze grad_norm trends")
    parser.add_argument("logs", help="Path to combined_logs.json")
    parser.add_argument(
        "-o", "--output-dir", default=".", help="Output directory for plots"
    )
    args = parser.parse_args()

    log_path = Path(args.logs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps, gnorms = load_grad_norms(log_path)
    print(f"Loaded {len(steps)} entries, steps {steps[0]}..{steps[-1]}")

    centers, means, stds, mins, maxs = bucket_stats(steps, gnorms)

    # Print bucket table
    print(f"\n{'Bucket':>14s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
    print("-" * 55)
    for i in range(len(centers)):
        lo = int(centers[i] - 5000)
        hi = int(centers[i] + 5000)
        print(
            f"{lo:>6d}-{hi:>6d}  {means[i]:>8.4f}  {stds[i]:>8.4f}  {mins[i]:>8.4f}  {maxs[i]:>8.4f}"
        )

    # Fit mean
    fit_mean = fit_all(centers, means)
    print_fit_results(fit_mean, "Grad-norm MEAN fits")

    # Fit std (skip first bucket -- warmup distortion)
    fit_std = fit_all(centers, stds, skip_first=True)
    print_fit_results(fit_std, "Grad-norm STD fits (first bucket excluded)")

    # Best fit details
    best_name = max(fit_mean, key=lambda k: fit_mean[k][1] or -1)
    popt, r2, _, _ = fit_mean[best_name]
    print(f"\nBest fit for mean: {best_name} (R2={r2:.6f})")
    func, _ = MODELS[best_name]
    # Print predictions at key points
    print(f"\n  Extrapolation using {best_name} fit:")
    for s in [50000, 100000, 150000, 200000, 250000, 300000, 400000, 500000]:
        pred = func(s, *popt)
        print(f"    step {s:>7d}: predicted grad_norm mean = {pred:.4f}")

    plot_analysis(
        steps, gnorms, centers, means, stds, mins, maxs, fit_mean, fit_std, output_dir
    )


if __name__ == "__main__":
    main()
