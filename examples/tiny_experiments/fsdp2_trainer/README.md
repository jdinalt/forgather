# FSDP2 Trainer Integration Test

Minimal integration test project for the FSDP2 (`fully_shard`) trainer.
Parallels the `ddp_trainer` example but exercises the FSDP2-specific
code paths: layer-wise `fully_shard` wrapping, sharded DTensor
`state_dict` save/load, and `CPUOffloadPolicy`.

**Trainer implementation**: `src/forgather/ml/trainer/fsdp2/fsdp2_trainer.py`

## Configurations

| Config | Purpose |
|--------|---------|
| `2gpu.yaml` | Default 2-GPU smoke test — verifies init, forward/backward, and gradient reduction |
| `checkpoint_train.yaml` | Trains 500 steps saving per-rank sharded checkpoints every 200 steps |
| `checkpoint_resume.yaml` | Resumes from the sharded checkpoint and continues to 1000 steps |
| `cpu_offload.yaml` | Enables `fsdp2.cpu_offload=True` (`CPUOffloadPolicy`) |

## Usage

```bash
# 2-GPU smoke test
forgather -t 2gpu.yaml train

# Sharded checkpoint round-trip
forgather -t checkpoint_train.yaml train
forgather -t checkpoint_resume.yaml train

# CPU offload
forgather -t cpu_offload.yaml train

# View logs
forgather logs summary --all --format one-line
forgather logs plot --loss-curves
```

## Notes

- **Sharded checkpoints**: the model and optimizer are saved with
  `SharingPattern.PER_RANK` — each rank writes its own file with
  `get_model_state_dict(model)` / `get_optimizer_state_dict(model, optim)`.
  The resulting checkpoints are tied to the world size they were saved
  at; resuming at a different world size is not supported in this first
  cut and would require routing through `torch.distributed.checkpoint`.
- **Layer-wise sharding**: the default
  `fsdp2.transformer_layers_path="model.layer_stack.layers"` matches
  the standard Forgather transformer (`modelsrc/transformer/`). Override
  in a child config if your model uses a different block-list path.
- The `tiny.yaml` base project uses a ~4M-parameter model on Tiny
  Stories — small enough that FSDP2 offers no real memory savings; this
  project exists solely to validate that the code paths work.
