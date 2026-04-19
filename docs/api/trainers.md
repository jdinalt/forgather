# Trainers

Forgather provides a hierarchy of trainer classes for single-GPU through multi-node distributed training.

| Trainer | Use case |
|---------|----------|
| `BaseTrainer` | Single-GPU baseline |
| `AccelTrainer` | Multi-GPU via HuggingFace Accelerate (DDP, FSDP) |
| `PipelineTrainer` | Pipeline parallelism across GPUs with limited interconnect |

For a complete reference of every training argument and constructor parameter across all trainers, see [Trainer Options Reference](../trainers/trainer_options.md).

---

::: forgather.ml.trainer.base_trainer.BaseTrainer

---

::: forgather.ml.trainer.base_trainer.BaseTrainingArguments

---

::: forgather.ml.trainer.accelerate.accel_trainer.AccelTrainer

---

::: forgather.ml.trainer.pipeline.pipeline_trainer.PipelineTrainer
