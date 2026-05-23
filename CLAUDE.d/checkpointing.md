# Checkpointing

User guide: `docs/checkpointing/user_guide.md`.
Architecture: `docs/checkpointing/distributed_checkpoint_abstraction.md`.
Custom-trainer migration: `docs/checkpointing/migration_guide.md`.

The system saves **all** state automatically — model, optimizer,
scheduler, dataset position, RNG, training progress — and coordinates
across multi-GPU / multi-node runs via explicit sharing patterns
(`GLOBAL`, `PER_RANK`, `REPLICATED`, `PER_GROUP`, `PER_NODE`).

## Enable

```python
args = TrainingArguments(
    output_dir="output_models/my_model",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
)
trainer = Trainer(model=model, args=args, ...)
trainer.train()
```

## Resume

```python
args = TrainingArguments(
    output_dir="output_models/my_model",
    resume_from_checkpoint=True,  # auto-finds latest
    max_steps=5000,
)
```

To skip restoring a component (e.g., changing datasets), delete the
file from the checkpoint dir before resuming — load warns and continues.
Model weights cannot be skipped.

## Output layout

```
output_models/my_model/checkpoint-1000/
├── model.safetensors
├── optimizer_state.pt
├── scheduler_state.pt
├── dataset_state.pt
├── rng_state.pt
├── trainer_state.pt
└── checkpoint_manifest.json   # all metadata for debugging
```

`checkpoint_manifest.json` lists every component, its sharing pattern,
the ranks that contributed, and sizes. Use it as the first stop when a
checkpoint behaves oddly.

## Patterns

- **DDP**: model/optimizer REPLICATED, dataset GLOBAL, RNG PER_RANK.
  Use `DDPTrainer` + `DDPTrainingArguments(dispatch_batches=True)`,
  launch with `torchrun --nproc_per_node=N`.
- **Pipeline parallel**: model/optimizer PER_RANK, dataset GLOBAL.
  Use `PipelineTrainer` with a `model_splitter`.

## Common pitfalls

| Symptom | Likely cause |
|---|---|
| Hangs during save | Distributed barrier deadlock — ensure every rank reaches save. Fixed in built-in trainers. |
| Different results after resume | RNG or dataset state file was deleted — both are required for exact reproducibility. |
| `Validation failed for component 'optimizer'` | Known issue with `AccelerateOptimizer` wrapper; validation auto-disabled for it. |
