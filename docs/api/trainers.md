# Trainers

Forgather provides a hierarchy of trainer classes for single-GPU through multi-node distributed training.

| Trainer | Use case |
|---------|----------|
| `Trainer` | Single-GPU, the fast path for small-model experiments |
| `DDPTrainer` | Multi-GPU DistributedDataParallel, with optional PostLocalSGD |
| `FSDP2Trainer` | FSDP-2 sharded data parallel with CPU offload support |
| `PipelineTrainer` | Pipeline parallelism for bandwidth-limited environments |

For a complete reference of every training argument and constructor parameter across all trainers, see also:

- [Trainer Options Reference](../trainers/trainer_options.md)
- [DiLoCo Architecture & Maintainer Guide](../trainers/diloco-architecture.md)

---

## Core Types

Shared types and protocols used across all trainers.

::: forgather.ml.distributed.DistributedEnvironment

---

::: forgather.ml.trainer.trainer_types.MinimalTrainingArguments

---

::: forgather.ml.trainer.trainer_types.TrainerState

---

::: forgather.ml.trainer.trainer_types.TrainerControl

---

::: forgather.ml.trainer.trainer_types.TrainOutput

---

## Base Classes

Abstract base from which all concrete trainers derive. Implement these three
methods to build a custom trainer: `_prepare`, `_train_loop`, `_eval_loop`.

::: forgather.ml.trainer.base_trainer.BaseTrainer

---

::: forgather.ml.trainer.base_trainer.BaseTrainingArguments

---

## Single-GPU Trainer

::: forgather.ml.trainer.trainer.Trainer

---

::: forgather.ml.trainer.trainer.TrainingArguments

---

## Distributed Data Parallel (DDP) Trainer

::: forgather.ml.trainer.ddp.ddp_trainer.DDPTrainer

---

::: forgather.ml.trainer.ddp.ddp_trainer.DDPTrainingArguments

## Fully Sharded Distributed Data Parallel (FSDP2) Trainer

---

::: forgather.ml.trainer.fsdp2.fsdp2_trainer.FSDP2Trainer

---

::: forgather.ml.trainer.fsdp2.fsdp2_trainer.FSDP2Arguments

---

## Pipeline Parallel Trainer

::: forgather.ml.trainer.pipeline.pipeline_trainer.PipelineTrainer

---

::: forgather.ml.trainer.pipeline.pipeline_trainer.PipelineTrainingArguments
