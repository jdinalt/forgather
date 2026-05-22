# Analysis

Tools for parsing, summarizing, and visualizing training logs produced by Forgather's JSON logger. For CLI usage, plots, and the full `forgather logs` command reference, see [Log Analysis](../guides/logs-analysis.md).

## Quick Example

```python
from forgather.ml.analysis import TrainingLog, compute_summary_statistics

log = TrainingLog.from_file("output_models/my_model/runs/run_id/trainer_logs.json")
summary = compute_summary_statistics(log)
print(f"Best loss: {summary['best_loss']} at step {summary['best_loss_step']}")
```

---

### `TrainingLog` {#forgather-ml-analysis-log_parser-traininglog}

`forgather.ml.analysis.log_parser.TrainingLog`

Container for a parsed Forgather training log.

Holds all JSON records emitted by Forgather's JSON logger
(``trainer_logs.json``) together with metadata inferred from the
file-system path.  Typically created via :meth:`from_file` or
:meth:`from_run_dir` rather than constructed directly.

**Parameters**

- `log_path` (Path) — Absolute path to the ``trainer_logs.json`` file.
- `records` (list of dict) — Raw JSON records as loaded from the log file.  Each record is a
dictionary that may contain keys such as ``global_step``, ``loss``,
``eval_loss``, ``learning_rate``, ``grad_norm``, ``epoch``,
``timestamp``, and ``train_runtime``.
- `run_name` (str) — Human-readable name of the training run, usually the timestamped
directory name under ``runs/``.  Inferred from *log_path* when not
provided.
- `model_name` (str) — Name of the model, usually the directory immediately above ``runs/``
in the output path.  Inferred from *log_path* when not provided.
- `label` (str) — Explicit display label used when plotting.  When set, this takes
priority over *model_name* and *run_name*.

**Examples**

```python
>>> from forgather.ml.analysis import TrainingLog
>>> log = TrainingLog.from_file("output_models/my_model/runs/run_001/trainer_logs.json")
>>> train_records = log.get_training_records()
>>> losses = log.get_metric_values("loss", train_records)
```

---

### `compute_summary_statistics` {#forgather-ml-analysis-metrics-compute_summary_statistics}

`forgather.ml.analysis.metrics.compute_summary_statistics`

Compute summary statistics from a training log.

Aggregates training-step records, evaluation records, and the final
summary record into a flat dictionary of key metrics.  Keys are only
present when the underlying data exists; callers should use
``summary.get(key)`` rather than direct indexing.

**Parameters**

- `log` (TrainingLog) — Parsed training log to summarise.

**Returns**

- `dict` — Dictionary with a subset of the following keys, depending on what
data is available in *log*:

``run_name`` : str or None
    Name of the training run.
``log_path`` : str
    String representation of the log file path.
``total_steps`` : int
    Global step number of the last training record.
``final_epoch`` : float
    Epoch number at the last training step.
``final_loss`` : float
    Training loss at the last recorded step.
``avg_loss`` : float
    Mean training loss over all recorded steps.
``min_loss`` : float
    Minimum training loss observed during the run.
``best_loss`` : float
    Training loss at the step where it was lowest (same as
    ``min_loss`` but paired with ``best_loss_step``).
``best_loss_step`` : int
    Global step at which ``best_loss`` was achieved.
``avg_grad_norm`` : float
    Mean gradient norm over all training steps that recorded it.
``max_grad_norm_value`` : float
    Peak gradient norm observed during training.
``max_grad_norm_step`` : int
    Global step at which ``max_grad_norm_value`` was observed.
``initial_lr`` : float
    Learning rate at the first training step.
``final_lr`` : float
    Learning rate at the last training step.
``final_eval_loss`` : float
    Evaluation loss from the most recent evaluation checkpoint.
``best_eval_loss`` : float
    Lowest evaluation loss observed.
``best_eval_loss_step`` : int
    Global step at which ``best_eval_loss`` was achieved.
``train_runtime`` : float
    Total training wall-clock time in seconds.
``train_samples`` : int
    Total number of training samples processed.
``train_samples_per_second`` : float
    Average throughput in samples per second.
``train_steps_per_second`` : float
    Average throughput in optimizer steps per second.
``effective_batch_size`` : int
    Effective batch size (local batch x gradient accumulation x
    world size).

**Examples**

```python
>>> from forgather.ml.analysis import TrainingLog, compute_summary_statistics
>>> log = TrainingLog.from_file("output_models/my_model/runs/run_001/trainer_logs.json")
>>> summary = compute_summary_statistics(log)
>>> print(f"Best loss: {summary['best_loss']:.4f} at step {summary['best_loss_step']}")
```

---

### `plot_training_metrics` {#forgather-ml-analysis-plotting-plot_training_metrics}

`forgather.ml.analysis.plotting.plot_training_metrics`

Plot one or more training metrics from one or more training logs.

Creates a grid of subplots (up to two columns) with one panel per metric.
When multiple logs are supplied each run is drawn in a distinct colour with
a legend entry.  For loss-like metrics (``loss``, ``eval_loss``,
``grad_norm``) the y-axis is automatically clipped to the 5th–95th
percentile window to suppress early-training outliers; pass
``ignore_outliers=False`` to disable this.

**Parameters**

- `logs` (list of TrainingLog) — One or more parsed training logs to plot.
- `metrics` (list of str) — Metric keys to plot.  Each element must be a key present in at least
some log records (e.g. ``'loss'``, ``'eval_loss'``,
``'learning_rate'``, ``'grad_norm'``).  Default is
``['loss', 'eval_loss', 'learning_rate']``.
- `x_axis` ((step, epoch, time)) — X-axis variable.  ``'step'`` uses ``global_step``, ``'epoch'`` uses
``epoch``, and ``'time'`` converts timestamps to elapsed minutes.
Default is ``'step'``.
- `smooth_window` (int) — When greater than 1, draws the raw series at low opacity and overlays
a centred moving-average with the given window size.  Default is
``None`` (no smoothing).
- `log_scale` (bool) — Use a logarithmic y-axis.  Outlier-aware auto-scaling is suppressed
on log axes.  Default is ``False``.
- `output_path` (str or Path) — If provided, the figure is saved to this path at 300 dpi.  Parent
directories are created automatically.
- `figsize` (tuple of int) — ``(width, height)`` in inches passed to ``plt.subplots``.  Default is
``(12, 8)``.
- `show` (bool) — Call ``plt.show()`` after rendering.  Default is ``False``.
- `title` (str) — Figure-level suptitle.  When ``None`` no title is added.
- `ignore_outliers` (bool) — Apply percentile-based y-axis clipping for loss-like metrics.
Default is ``True``.
- `perplexity` (bool) — Convert loss values to perplexity (``exp(loss)``) for ``loss``,
``train_loss``, and ``eval_loss`` metrics.  Default is ``False``.
- `x_min` (float) — Clip data and set the left x-axis limit to this value.
- `x_max` (float) — Clip data and set the right x-axis limit to this value.
- `y_min` (float) — Override the bottom y-axis limit.  Takes priority over auto-scaling.
- `y_max` (float) — Override the top y-axis limit.  Takes priority over auto-scaling.

**Returns**

- `matplotlib.figure.Figure` — The rendered figure.  The caller is responsible for closing it when
no longer needed (``plt.close(fig)``).

**Examples**

```python
>>> from forgather.ml.analysis import TrainingLog
>>> from forgather.ml.analysis.plotting import plot_training_metrics
>>> log = TrainingLog.from_file("output_models/my_model/runs/run_001/trainer_logs.json")
>>> fig = plot_training_metrics([log], metrics=["loss", "eval_loss"], smooth_window=20)
>>> fig.savefig("training.png", dpi=150)
```
