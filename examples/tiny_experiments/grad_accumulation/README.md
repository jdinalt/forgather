# Gradient Accumulation

Tests that gradient accumulation produces equivalent results to larger batch
sizes, across different trainer implementations (standard, pipeline parallel,
Accelerate DDP).

## Configurations

| Config | Description |
|--------|-------------|
| `control.yaml` | Baseline: default batch size, no accumulation |
| `batch_10.yaml` | 10x batch size (320), no accumulation |
| `accum_10.yaml` | Default batch size with 10 gradient accumulation steps |
| `pp_control.yaml` | Pipeline parallel baseline (GPipe, 4 microbatches) |
| `pp_accum_10.yaml` | Pipeline parallel with 10 gradient accumulation steps |
| `accum_accel_ddp_2x5.yaml` | Accelerate DDP: 2 processes, 5 accumulation steps |

The `batch_10` and `accum_10` configs should produce similar training curves,
validating that gradient accumulation correctly approximates larger batch training.

## Usage

```bash
# Run baseline and accumulation variant, then compare loss curves
forgather -t control.yaml train
forgather -t accum_10.yaml train
forgather logs plot --compare output_models/*/runs/*/trainer_logs.json --loss-curves
```
