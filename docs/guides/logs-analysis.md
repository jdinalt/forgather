# Training Log Analysis

Forgather provides powerful tools for analyzing and visualizing training logs through the `forgather logs` command. The logger automatically records training metrics to `trainer_logs.json` files in each training run directory.

## Log Format

Training logs are stored as JSON arrays with the following structure:

```json
[
  {"timestamp": 1769495119.472016, "global_step": 100, "epoch": 0.015, "loss": 6.171, "grad_norm": 1.022, "learning_rate": 0.000282},
  {"timestamp": 1769495123.907347, "global_step": 200, "epoch": 0.060, "loss": 3.771, "grad_norm": 0.593, "learning_rate": 0.000564},
  {"timestamp": 1769495136.678786, "global_step": 500, "epoch": 0.151, "eval_loss": 2.314},
  ...
  {"timestamp": 1769495257.024434, "global_step": 3312, "epoch": 1.0, "train_runtime": 141.48, "train_samples": 211904}
]
```

Each record contains:
- **Training records**: timestamp, global_step, epoch, loss, grad_norm, max_grad_norm, learning_rate
- **Evaluation records**: timestamp, global_step, epoch, eval_loss
- **Final summary**: train_runtime, train_samples, train_samples_per_second, train_steps_per_second, effective_batch_size

## Available Commands

### List Training Logs

List all available training logs in the current project:

```bash
forgather logs list
```

Output:
```
Found 4 training log(s):

1. default_model/iterable_2026-01-27T06-25-07
   Path: output_models/default_model/runs/iterable_2026-01-27T06-25-07/trainer_logs.json
   Modified: 2026-01-27 06:27:37

2. default_model/sharded_fast_2026-01-26T11-09-33
   Path: output_models/default_model/runs/sharded_fast_2026-01-26T11-09-33/trainer_logs.json
   Modified: 2026-01-26 11:11:27
```

### Generate Summary Statistics

Generate summary statistics from a training log:

```bash
# Auto-detect latest log in project
forgather logs summary

# Specify log file explicitly
forgather logs summary path/to/trainer_logs.json

# Or use run directory
forgather logs summary path/to/run_dir
```

**Output formats:**

```bash
# Text (default)
forgather logs summary --format text

# JSON
forgather logs summary --format json

# Markdown
forgather logs summary --format md

# One-line (compact summary)
forgather logs summary --format one-line
```

**Process all logs:**

```bash
# Summarize all logs in project (one-line format recommended)
forgather logs summary --all --format one-line

# All logs in text format
forgather logs summary --all --format text

# All logs as JSON array
forgather logs summary --all --format json
```

**Save to file:**

```bash
forgather logs summary --output summary.txt
forgather logs summary --format json --output summary.json
```

**Example output (text format):**

```
Training Run Summary
============================================================
Run: iterable_2026-01-27T06-25-07
Duration: 141.48s
Total Steps: 3300
Final Epoch: 0.9964

Metrics:
  Final Loss: 1.4422
  Best Loss: 1.4422 (step 3300)
  Average Loss: 1.9881
  Final Eval Loss: 1.3666 (step 3000)
  Best Eval Loss: 1.3666 (step 3000)

Training Speed:
  Samples/sec: 1497.78
  Steps/sec: 23.40
  Effective Batch Size: 64

Gradient Statistics:
  Average Grad Norm: 0.4614
  Max Grad Norm: 1.0218 (step 100)

Learning Rate:
  Initial: 0.000282
  Final: 0.001400
```

### Generate Plots

Generate plots from training logs.

**Default behavior**: Saves plot to `tmp/` directory (gitignored, easy to cleanup).

**Basic usage:**

```bash
# Save to tmp/ directory (default)
forgather logs plot
# Creates: tmp/training_plot.png

# Open plot in editor after generation
forgather logs plot -e
# Creates tmp/training_plot.png and opens in VS Code (works on remote SSH)

# Loss curves plot
forgather logs plot --loss-curves
# Creates: tmp/loss_curves.png

# Custom output location
forgather logs plot --output figures/training.png

# Specify output format (default: png)
forgather logs plot --output plot --format svg  # Creates: plot.svg
forgather logs plot --output plot --format pdf  # Creates: plot.pdf

# With config name prefix
forgather -t my_config.yaml logs plot
# Creates: tmp/my_config_training_plot.png
```

**X-axis options:**

```bash
# Plot by global step (default)
forgather logs plot --x-axis step

# Plot by epoch
forgather logs plot --x-axis epoch

# Plot by time (minutes)
forgather logs plot --x-axis time
```

**Select metrics to plot:**

```bash
# Plot specific metrics
forgather logs plot --metrics "loss,eval_loss,learning_rate"

# Plot gradient statistics
forgather logs plot --metrics "grad_norm,max_grad_norm"
```

**Smoothing:**

```bash
# Apply moving average smoothing
forgather logs plot --smooth 10

# Smoothing with window size 20
forgather logs plot --metrics "loss" --smooth 20
```

**Loss curves plot:**

Generate a specialized loss curves plot with learning rate on secondary axis:

```bash
forgather logs plot --loss-curves

# With smoothing
forgather logs plot --loss-curves --smooth 5
```

**Other options:**

```bash
# Use log scale for y-axis
forgather logs plot --log-scale

# Plot by time instead of steps
forgather logs plot --x-axis time

# Plot gradient norm (own layout; cannot combine with --loss-curves/--metrics)
forgather logs plot --grad-norm --smooth 10

# Plot losses as perplexity (exp(loss)); composes with --log-scale
forgather logs plot --loss-curves --perplexity --log-scale

# Clip the plot window in data space (respects --x-axis units)
forgather logs plot --x-max 36448 --metrics eval_loss

# Override y-axis bounds; takes precedence over --ignore-outliers
forgather logs plot --y-min 2.2 --y-max 3.5
```

**Outlier-aware auto-scaling**

By default, `forgather logs plot` computes the y-axis window for loss-like
metrics (`loss`, `eval_loss`, `train_loss`, `grad_norm`, `max_grad_norm`) from
the 5th/95th percentiles of the plotted series instead of the raw min/max.
This keeps the huge early-training values from squashing the tail of the
curve, similar to TensorBoard's "Ignore outliers in chart scaling" option.
Use `--no-ignore-outliers` to restore the previous full-range behaviour, or
`--y-min` / `--y-max` to pin the bounds explicitly. Learning-rate subplots
and secondary axes are never auto-clipped.

**Plot mode and scaling options:**

| Flag | Description |
|------|-------------|
| `--loss-curves` | Dual-axis train/eval loss with learning rate on a secondary axis. |
| `--grad-norm` | Gradient-norm plot; mutually exclusive with `--loss-curves` / `--metrics`. |
| `--ignore-outliers` / `--no-ignore-outliers` | Enable (default) or disable percentile-based y-axis auto-scaling. |
| `--perplexity` | Plot loss-like metrics as `exp(loss)` with relabelled axes. |
| `--x-min` / `--x-max` | Clip plotted data and x-axis to the given domain (in `--x-axis` units). |
| `--y-min` / `--y-max` | Override y-axis bounds; takes precedence over `--ignore-outliers`. |

### Compare Multiple Runs

Compare metrics across multiple training runs:

```bash
# Compare two runs
forgather logs plot --compare run1/trainer_logs.json run2/trainer_logs.json

# Compare with loss curves plot
forgather logs plot --compare run1/trainer_logs.json run2/trainer_logs.json --loss-curves

# Save comparison
forgather logs plot --compare run1/trainer_logs.json run2/trainer_logs.json \
    --output comparison.png --no-show
```

## Usage Examples

### Quick Training Analysis

After training, quickly check how it went:

```bash
cd my_project
forgather logs summary
```

### Compare All Runs at a Glance

View all training runs in a compact table:

```bash
forgather logs summary --all --format one-line
```

Output:
```
Run Name                         | Steps       | Time         | Loss     | Eval     | Throughput
------------------------------------------------------------------------------------------------
iterable_2026-01-27T06-25-07     | steps=3300  | time=02:21  | loss=1.4422 | eval=1.3666 | samp/s=1497.8
sharded_fast_2026-01-26T11-09-   | steps=3300  | time=01:47  | loss=1.4302 | eval=1.3100 | samp/s=1968.0
sharded-iterable_2026-01-26T10   | steps=3300  | time=01:47  | loss=1.4304 | eval=1.3153 | samp/s=1977.3
```

### Compare Optimizer Experiments

Compare training runs with different optimizers:

```bash
cd examples/tiny_experiments/optimizers
forgather logs plot --compare \
    output_models/tiny_causal/runs/adamw_*/trainer_logs.json \
    output_models/tiny_causal/runs/apollo_*/trainer_logs.json \
    --loss-curves --smooth 5 --output optimizer_comparison.png --no-show
```

### Analyze Loss Convergence

Plot smoothed loss curves to see convergence behavior:

```bash
forgather logs plot --loss-curves --smooth 10 --x-axis epoch
```

### Export for Further Analysis

Export summary statistics as JSON for programmatic analysis:

```bash
forgather logs summary --format json --output results.json
```

### Generate Training Report

Create a markdown report with summary and plots:

```bash
# Generate summary
forgather logs summary --format md --output report.md

# Generate plots
forgather logs plot --loss-curves --output loss_plot.png --no-show
forgather logs plot --metrics "grad_norm" --output grad_plot.png --no-show
```

## Per-Parameter Diagnostic Logging

Forgather provides callbacks for logging per-parameter weight norms, spectral norms, and gradient norms to JSON files. These are useful for diagnosing training instability, such as the issues described in [arxiv 2510.04212](https://arxiv.org/abs/2510.04212).

Diagnostic logs are written on each **evaluation step** to keep overhead low. They are saved alongside `trainer_logs.json` in the run directory.

### Enabling Diagnostic Callbacks

Add the callbacks to your trainer:

```python
from forgather.ml.trainer.callbacks import ParameterNormLogger, GradNormLogger

callbacks = [
    # Per-parameter L2 norms and spectral norms
    ParameterNormLogger(
        log_norms=True,            # Log per-parameter L2 norms
        log_spectral_norms=True,   # Log per-parameter spectral norms
        power_iter_steps=10,       # Power iteration steps (accuracy vs speed)
    ),
    # Per-parameter gradient norms
    GradNormLogger(),
    # ... your other callbacks
]

trainer = Trainer(model=model, args=args, callbacks=callbacks, ...)
trainer.train()
```

Both callbacks support checkpoint resume via the Stateful protocol. On resume, log files are truncated to the checkpoint step and appending continues.

### ParameterNormLogger

Writes `parameter_norms.json` containing per-parameter L2 norms and/or spectral norms:

```json
[
  {"timestamp": 1700000000.0, "global_step": 500, "epoch": 0.5,
   "norms": {"model.layers.0.attention.q.weight": 1.23, "model.layers.0.attention.k.weight": 1.18, ...},
   "spectral_norms": {"model.layers.0.attention.q.weight": 0.98, "model.layers.0.attention.k.weight": 0.91, ...}},
  ...
]
```

Spectral norms are estimated via power iteration, which is significantly faster than full SVD for large weight matrices. Direction vectors are cached across evaluation steps for warm-starting, improving accuracy over time.

### GradNormLogger

Writes `gradient_norms.json` containing per-parameter gradient L2 norms. Gradients are captured after clipping but before the optimizer step (via the `on_pre_optimizer_step` hook):

```json
[
  {"timestamp": 1700000000.0, "global_step": 500, "epoch": 0.5,
   "grad_norms": {"model.layers.0.attention.q.weight": 0.012, "model.layers.0.attention.k.weight": 0.045, ...}},
  ...
]
```

When `fuse_optim_with_backward` is enabled, gradients are consumed during the backward pass and this callback disables itself with a warning.

### Diagnostic Log Location

Diagnostic logs are saved to the same directory as `trainer_logs.json`:

```
output_models/MODEL_NAME/runs/RUN_NAME/
    trainer_logs.json        # Standard training metrics
    parameter_norms.json     # Per-parameter weight/spectral norms
    gradient_norms.json      # Per-parameter gradient norms
```

## Per-Parameter Heatmap Plots

The `forgather plot heatmap` command generates grid heatmaps from diagnostic log files, with parameter FQN labels on the y-axis and training steps on the x-axis. Each cell is color-coded by the metric value.

### Basic Usage

```bash
# Auto-detect latest diagnostic log in project
forgather plot heatmap

# Specify a log file
forgather plot heatmap path/to/parameter_norms.json

# Open in editor after generation
forgather plot heatmap -e
```

The metric type is auto-detected from the file content (`norms`, `spectral_norms`, or `grad_norms`). To override:

```bash
# Plot spectral norms from a file that contains both norms and spectral_norms
forgather plot heatmap path/to/parameter_norms.json --metric spectral_norm
```

### Filtering Parameters

Use `--filter` / `-f` with a regex pattern to show only matching parameter names:

```bash
# Only attention parameters
forgather plot heatmap parameter_norms.json -f "attention"

# Only query and key matrices
forgather plot heatmap parameter_norms.json -f "(q_proj|k_proj)"

# Only a specific layer
forgather plot heatmap parameter_norms.json -f "layers\.2\."

# Only MLP parameters across all layers
forgather plot heatmap gradient_norms.json -f "mlp"
```

### Controlling Step Density

For long training runs, use `--step-stride` to reduce the number of plotted steps:

```bash
# Plot every 5th evaluation step
forgather plot heatmap parameter_norms.json --step-stride 5
```

### Color Scale Options

```bash
# Log scale (useful when values span multiple orders of magnitude)
forgather plot heatmap parameter_norms.json --log-scale

# Manual color range
forgather plot heatmap parameter_norms.json --vmin 0.5 --vmax 2.0
```

### Output Options

```bash
# Custom output path
forgather plot heatmap parameter_norms.json -o figures/spectral_heatmap.png

# SVG format
forgather plot heatmap parameter_norms.json --format svg

# Custom figure size (width height in inches)
forgather plot heatmap parameter_norms.json --figsize 20 30

# Custom title
forgather plot heatmap parameter_norms.json --title "Spectral Norms: BF16 Run"
```

### Typical Workflow

```bash
# 1. Train with diagnostic callbacks enabled
forgather -t config.yaml train

# 2. List available logs
forgather logs list

# 3. Check overall spectral norms
forgather plot heatmap output_models/my_model/runs/latest/parameter_norms.json --metric spectral_norm -e

# 4. Zoom in on attention layers
forgather plot heatmap output_models/my_model/runs/latest/parameter_norms.json \
    --metric spectral_norm -f "attention" -e

# 5. Check gradient norms for the same layers
forgather plot heatmap output_models/my_model/runs/latest/gradient_norms.json \
    -f "attention" -e
```

## Programmatic API

You can also use the analysis tools programmatically in Python:

```python
from forgather.ml.analysis import TrainingLog, compute_summary_statistics, plot_training_metrics

# Load log
log = TrainingLog.from_file("path/to/trainer_logs.json")

# Get summary statistics
summary = compute_summary_statistics(log)
print(f"Best loss: {summary['best_loss']} at step {summary['best_loss_step']}")

# Generate plots
from forgather.ml.analysis.plotting import plot_loss_curves

fig = plot_loss_curves([log], smooth_window=10, output_path="loss.png")
```

### Heatmap API

```python
from forgather.ml.analysis import plot_parameter_heatmap

# Plot spectral norms for attention parameters only
fig = plot_parameter_heatmap(
    "path/to/parameter_norms.json",
    metric="spectral_norm",
    filter_pattern="attention",
    step_stride=2,
    log_scale=True,
    output_path="spectral_heatmap.png",
)
```

## Log Location

Training logs are automatically saved to:
```
output_models/MODEL_NAME/runs/RUN_NAME/trainer_logs.json
```

For example:
```
output_models/tiny_llama/runs/log_2026-01-27T06-25-07/trainer_logs.json
```

## Dependencies

The plotting functionality requires matplotlib:

```bash
pip install matplotlib
```

For additional analysis capabilities, pandas is recommended but optional:

```bash
pip install pandas
```
