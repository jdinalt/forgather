# Analysis

Tools for parsing, summarizing, and visualizing training logs produced by Forgather's JSON logger.

## Quick Example

```python
from forgather.ml.analysis import TrainingLog, compute_summary_statistics

log = TrainingLog.from_file("output_models/my_model/runs/run_id/trainer_logs.json")
summary = compute_summary_statistics(log)
print(f"Best loss: {summary['best_loss']} at step {summary['best_loss_step']}")
```

---

::: forgather.ml.analysis.TrainingLog

---

::: forgather.ml.analysis.compute_summary_statistics

---

::: forgather.ml.analysis.plot_training_metrics
