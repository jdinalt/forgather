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
```

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
