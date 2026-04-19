#!/usr/bin/env python3
"""Compare multiple training runs side-by-side.

Plots key metrics (loss, grad_norm, eval_loss, learning_rate) for 2-3 runs
aligned by global_step. Includes rolling statistics for grad_norm.

Usage:
    python compare_runs.py --baseline combined_pre_lr_change.json \
        --runs "LR change" log_2026-02-19T03-51-12 \
               "Optimizer reset" log_2026-02-19T07-54-27 \
        [-o output.png] [--window 500] [--tail STEPS]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_log(path: Path) -> list[dict]:
    text = path.read_text().rstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if text.endswith(","):
            text = text[:-1]
        if not text.endswith("]"):
            text += "\n]"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            lines = text.rsplit("\n", 2)
            text = lines[0].rstrip().rstrip(",") + "\n]"
            return json.loads(text)


def resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_dir():
        return path / "trainer_logs.json"
    return path


def extract(logs, key):
    """Extract (step, value) arrays for entries containing key."""
    entries = [(e["global_step"], e[key]) for e in logs if key in e]
    if not entries:
        return np.array([]), np.array([])
    return np.array([s for s, _ in entries]), np.array([v for _, v in entries])


def rolling_stats(steps, values, window):
    """Compute rolling mean and std over a step-count window."""
    if len(values) == 0:
        return steps, values, values
    means, stdevs = [], []
    for i in range(len(values)):
        lo = steps[i] - window
        mask = (steps >= lo) & (steps <= steps[i])
        w = values[mask]
        means.append(w.mean())
        stdevs.append(w.std())
    return steps, np.array(means), np.array(stdevs)


def main():
    parser = argparse.ArgumentParser(description="Compare training runs")
    parser.add_argument("--baseline", required=True, help="Baseline log (combined)")
    parser.add_argument(
        "--runs",
        nargs="+",
        action="append",
        default=[],
        help="Pairs of: label path [label path ...]",
    )
    parser.add_argument("-o", "--output", default="run_comparison.png")
    parser.add_argument(
        "--window",
        type=int,
        default=3000,
        help="Rolling window in steps for grad_norm stats",
    )
    parser.add_argument(
        "--tail", type=int, default=None, help="Only show last N steps of baseline"
    )
    args = parser.parse_args()

    # Flatten --runs
    run_args = []
    for group in args.runs:
        run_args.extend(group)
    if len(run_args) % 2 != 0:
        parser.error("--runs requires pairs of: label path")
    runs = []
    for i in range(0, len(run_args), 2):
        label, path = run_args[i], resolve_path(run_args[i + 1])
        runs.append((label, load_log(path)))

    baseline = load_log(resolve_path(args.baseline))

    # Determine step range from the comparison runs
    all_min_steps = []
    all_max_steps = []
    for label, logs in runs:
        steps, _ = extract(logs, "loss")
        if len(steps) > 0:
            all_min_steps.append(steps[0])
            all_max_steps.append(steps[-1])

    if not all_min_steps:
        print("No comparison runs with data")
        return

    # Show baseline context: from some steps before the fork point
    fork_step = min(all_min_steps)
    max_step = max(all_max_steps)
    context_steps = args.tail or 50000  # show 50K steps of baseline context

    metrics = [
        ("loss", "Training Loss", False),
        ("grad_norm", "Grad Norm", False),
        ("eval_loss", "Validation Loss", True),
        ("learning_rate", "Learning Rate", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Run Comparison from Checkpoint", fontsize=14, fontweight="bold")

    colors_runs = ["tab:red", "tab:green", "tab:purple"]

    for idx, (key, title, scatter) in enumerate(metrics):
        ax = axes[idx // 2][idx % 2]

        # Baseline
        bs, bv = extract(baseline, key)
        if len(bs) > 0:
            mask = bs >= (fork_step - context_steps)
            bs_ctx, bv_ctx = bs[mask], bv[mask]
            if key == "grad_norm":
                # Rolling mean for baseline
                _, rm, rs = rolling_stats(bs_ctx, bv_ctx, args.window)
                ax.plot(
                    bs_ctx,
                    rm,
                    "-",
                    color="tab:blue",
                    linewidth=1.5,
                    label=f"Baseline (rolling mean, w={args.window})",
                    alpha=0.9,
                )
                ax.fill_between(bs_ctx, rm - rs, rm + rs, alpha=0.15, color="tab:blue")
                ax.scatter(
                    bs_ctx, bv_ctx, s=0.3, alpha=0.1, color="tab:blue", rasterized=True
                )
            elif scatter:
                ax.scatter(
                    bs_ctx,
                    bv_ctx,
                    s=8,
                    color="tab:blue",
                    alpha=0.6,
                    label="Baseline",
                    zorder=2,
                )
            else:
                ax.plot(
                    bs_ctx,
                    bv_ctx,
                    "-",
                    color="tab:blue",
                    linewidth=0.5,
                    alpha=0.3,
                    rasterized=True,
                )
                # Rolling mean
                if len(bs_ctx) > 20:
                    _, rm, _ = rolling_stats(bs_ctx, bv_ctx, args.window)
                    ax.plot(
                        bs_ctx,
                        rm,
                        "-",
                        color="tab:blue",
                        linewidth=1.5,
                        label=f"Baseline (rolling mean)",
                        alpha=0.9,
                    )

        # Comparison runs
        for i, (label, logs) in enumerate(runs):
            color = colors_runs[i % len(colors_runs)]
            rs, rv = extract(logs, key)
            if len(rs) == 0:
                continue

            if key == "grad_norm":
                _, rm, rstd = rolling_stats(rs, rv, args.window)
                ax.plot(
                    rs,
                    rm,
                    "-",
                    color=color,
                    linewidth=1.5,
                    label=f"{label} (rolling mean)",
                    alpha=0.9,
                )
                ax.fill_between(rs, rm - rstd, rm + rstd, alpha=0.15, color=color)
                ax.scatter(rs, rv, s=0.5, alpha=0.15, color=color, rasterized=True)
            elif scatter:
                ax.scatter(rs, rv, s=12, color=color, alpha=0.7, label=label, zorder=3)
            else:
                ax.plot(
                    rs, rv, "-", color=color, linewidth=0.5, alpha=0.3, rasterized=True
                )
                if len(rs) > 20:
                    _, rm, _ = rolling_stats(rs, rv, args.window)
                    ax.plot(
                        rs,
                        rm,
                        "-",
                        color=color,
                        linewidth=1.5,
                        label=f"{label} (rolling mean)",
                        alpha=0.9,
                    )
                else:
                    ax.plot(
                        rs,
                        rv,
                        "o-",
                        color=color,
                        linewidth=1.5,
                        markersize=2,
                        label=label,
                        alpha=0.9,
                    )

        # Fork point marker
        ax.axvline(fork_step, color="black", linestyle="--", alpha=0.4, linewidth=1)

        # Clip grad_norm y-axis to 99.5th percentile to avoid spike compression
        if key == "grad_norm":
            all_vals = []
            if len(bs) > 0:
                all_vals.append(bv[bs >= (fork_step - context_steps)])
            for _, rlogs in runs:
                _, rv2 = extract(rlogs, key)
                if len(rv2) > 0:
                    all_vals.append(rv2)
            if all_vals:
                combined = np.concatenate(all_vals)
                p995 = np.percentile(combined, 99.5)
                ax.set_ylim(0, p995 * 1.15)

        ax.set_title(title)
        ax.set_xlabel("Global Step")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K")
        )

    plt.tight_layout()
    output = Path(args.output)
    fig.savefig(output, dpi=150)
    print(f"Saved to {output}")
    plt.close()

    # Print summary stats for each run in the post-fork region
    print("\n" + "=" * 70)
    print(f"  Summary statistics (post-fork, step >= {fork_step})")
    print("=" * 70)

    for label, logs in [("Baseline (tail)", baseline)] + runs:
        steps_l, loss_l = extract(logs, "loss")
        steps_g, gnorm_g = extract(logs, "grad_norm")
        steps_e, eval_l = extract(logs, "eval_loss")

        mask_l = steps_l >= fork_step
        mask_g = steps_g >= fork_step
        mask_e = steps_e >= fork_step

        print(f"\n  {label}:")
        if mask_l.sum() > 0:
            print(
                f"    Loss:      mean={loss_l[mask_l].mean():.4f}  std={loss_l[mask_l].std():.4f}  "
                f"range=[{loss_l[mask_l].min():.4f}, {loss_l[mask_l].max():.4f}]  n={mask_l.sum()}"
            )
        if mask_g.sum() > 0:
            print(
                f"    Grad norm: mean={gnorm_g[mask_g].mean():.4f}  std={gnorm_g[mask_g].std():.4f}  "
                f"range=[{gnorm_g[mask_g].min():.4f}, {gnorm_g[mask_g].max():.4f}]  n={mask_g.sum()}"
            )
        if mask_e.sum() > 0:
            print(
                f"    Eval loss: mean={eval_l[mask_e].mean():.4f}  std={eval_l[mask_e].std():.4f}  "
                f"range=[{eval_l[mask_e].min():.4f}, {eval_l[mask_e].max():.4f}]  n={mask_e.sum()}"
            )


if __name__ == "__main__":
    main()
