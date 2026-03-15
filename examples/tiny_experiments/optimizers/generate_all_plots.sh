#!/usr/bin/env bash
# Generate all plots for the optimizer comparison experiment.
#
# Produces:
#   plots/amp_loss_curves.png      -- Loss curves for all AMP runs (forgather)
#   plots/grad8_loss_curves.png    -- Loss curves for grad8 runs (forgather)
#   plots/bf16_comparison.png      -- Loss curves for AMP vs bf16 (forgather)
#   plots/amp_eval_loss_bar.png    -- Bar chart: AMP eval losses (python)
#   plots/grad8_eval_loss_bar.png  -- Bar chart: grad8 eval losses (python)
#   plots/memory_speed.png         -- Memory and throughput comparison (python)
#   plots/bf16_impact.png          -- AMP vs bf16 impact chart (python)
#
# Usage:
#   ./generate_all_plots.sh          # from the project directory
#   ./generate_all_plots.sh --dpi 300  # pass extra args to the python script

set -euo pipefail
cd "$(dirname "$0")"

RUNS=output_models/default/runs
mkdir -p plots

# ---- Loss curves via forgather logs plot ---------------------------------- #

echo "Generating AMP loss curves..."
forgather logs plot --loss-curves \
    --compare \
        "$RUNS/adamw_2026-03-15T01-42-11" \
        "$RUNS/fg_adam_2026-03-15T02-10-37" \
        "$RUNS/fg_adafactor_2026-03-15T01-45-49" \
        "$RUNS/hf_adafactor_2026-03-15T06-38-26" \
        "$RUNS/apollo_2026-03-15T06-38-10" \
        "$RUNS/apollo_r64_pca_2026-03-15T06-40-49" \
        "$RUNS/sinkgd_2026-03-15T02-09-56" \
        "$RUNS/muon_2026-03-15T08-26-54" \
        "$RUNS/fg_sgd_2026-03-15T04-17-19" \
        "$RUNS/nesterov_sgd_2026-03-15T06-39-57" \
    --labels \
        "AdamW" "FG Adam" "FG Adafactor" "HF Adafactor" \
        "Apollo" "Apollo PCA" "SinkGD" "Muon" \
        "SGD" "Nesterov SGD" \
    --title "AMP: Optimizer Loss Curves (batch=32)" \
    --smooth 50 \
    -o plots/amp_loss_curves.png

echo "Generating grad8 loss curves..."
forgather logs plot --loss-curves \
    --compare \
        "$RUNS/adamw-8_2026-03-15T05-51-51" \
        "$RUNS/fg_adafactor-8_2026-03-15T05-59-36" \
        "$RUNS/sinkgd-8_2026-03-15T06-09-45" \
        "$RUNS/muon-8_2026-03-15T08-26-58" \
    --labels \
        "AdamW (ga=8)" "Adafactor (ga=8)" "SinkGD (ga=8)" "Muon (ga=8)" \
    --title "Gradient Accumulation 8x: Optimizer Loss Curves (eff. batch=256)" \
    --smooth 10 \
    -o plots/grad8_loss_curves.png

echo "Generating bf16 comparison loss curves..."
forgather logs plot --loss-curves \
    --compare \
        "$RUNS/adamw_2026-03-15T01-42-11" \
        "$RUNS/adamw_bf16_2026-03-15T02-26-42" \
        "$RUNS/fg_adam_2026-03-15T02-10-37" \
        "$RUNS/fg_adam_bf16_2026-03-15T03-01-10" \
        "$RUNS/fg_adafactor_2026-03-15T01-45-49" \
        "$RUNS/fg_adafactor_bf16_2026-03-15T03-01-46" \
    --labels \
        "AdamW (AMP)" "AdamW (bf16)" \
        "FG Adam (AMP)" "FG Adam (bf16)" \
        "Adafactor (AMP)" "Adafactor (bf16)" \
    --title "AMP vs Pure bfloat16: Impact of Stochastic Rounding" \
    --smooth 50 \
    -o plots/bf16_comparison.png

# ---- Summary charts via python -------------------------------------------- #

echo "Generating summary bar charts..."
python3 generate_plots.py "$@"

echo "All plots generated in plots/"
