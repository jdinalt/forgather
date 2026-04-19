# Trainer Callbacks

Callbacks extend trainer behaviour at well-defined lifecycle events (step start/end,
epoch start/end, evaluation, checkpoint save/load, etc.) without modifying trainer
source. Pass a list of callback instances to any trainer's `callbacks` argument.

**Related documentation:**

- [Trainer Control](../trainers/trainer-control.md) — saving, stopping, and aborting running jobs
- [Checkpointing](../checkpointing/README.md) — stateful callbacks and checkpoint resume
- [Divergence Detection](../checkpointing/divergence_detection.md) — detecting and recovering from training instability
- [Log Analysis](../guides/logs-analysis.md) — working with logs produced by `JsonLogger`

---

## Base Class

::: forgather.ml.trainer.trainer_types.TrainerCallback

---

## Built-in Callbacks

Default callbacks included automatically by all trainers.

::: forgather.ml.trainer.callbacks.DefaultMetrics

---

::: forgather.ml.trainer.callbacks.ProgressCallback

---

::: forgather.ml.trainer.callbacks.InfoCallback

---

## Job Control

::: forgather.ml.trainer.callbacks.TrainerControlCallback

---

## Divergence Detection

::: forgather.ml.trainer.callbacks.DivergenceDetector

---

## Logging

::: forgather.ml.trainer.callbacks.JsonLogger

---

::: forgather.ml.trainer.callbacks.TBLogger

---

::: forgather.ml.trainer.callbacks.GradNormLogger

---

::: forgather.ml.trainer.callbacks.ParameterNormLogger

---

::: forgather.ml.trainer.callbacks.WeightNormLogger

---

::: forgather.ml.trainer.callbacks.PeakMemory

---

## Text Generation

::: forgather.ml.trainer.callbacks.TextgenCallback

---

## Advanced

::: forgather.ml.trainer.callbacks.ProfilerCallback

---

::: forgather.ml.trainer.callbacks.DiLoCoCallback

---

::: forgather.ml.trainer.callbacks.ResumableSummaryWriter
