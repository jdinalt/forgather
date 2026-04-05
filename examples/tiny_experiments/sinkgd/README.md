# SinkGD

Experiments with the Sinkhorn GD optimizer, a rounding-aware optimizer that
applies doubly-stochastic normalization to weight matrices. This project
provides a hyperparameter sweep across alpha, learning rate, weight decay,
gradient accumulation, momentum, and normalization modes.

## Configurations

**Baselines:**

| Config | Description |
|--------|-------------|
| `adam_baseline.yaml` | AdamW baseline for comparison |
| `paper.yaml` | SinkGD with the paper's original multi-optimizer setup |
| `experimental.yaml` | Default SinkGD configuration |

**Alpha sweep** (controls normalization strength):

| Config | Description |
|--------|-------------|
| `alpha-025.yaml` | alpha = 0.025 |
| `alpha-10.yaml` | alpha = 0.1 |

**Learning rate sweep:**

| Config | Description |
|--------|-------------|
| `lr-01.yaml` | lr = 0.01 |
| `lr-03.yaml` | lr = 0.03 |
| `lr-04.yaml` | lr = 0.04 |

**Weight decay sweep:**

| Config | Description |
|--------|-------------|
| `wd-001.yaml` | weight_decay = 0.001 |
| `wd-005.yaml` | weight_decay = 0.005 |
| `wd-01.yaml` | weight_decay = 0.01 |

**Gradient accumulation sweep:**

| Config | Description |
|--------|-------------|
| `ga-2.yaml` | 2 accumulation steps (32K tokens/step) |
| `ga-4.yaml` | 4 accumulation steps (64K tokens/step) |
| `ga-8.yaml` | 8 accumulation steps (128K tokens/step) |

**Normalization modes:**

| Config | Description |
|--------|-------------|
| `colonly.yaml` | Column-only normalization for embedding layers |
| `factored.yaml` | Adafactor-style factored normalization |
| `sparse.yaml` | Sparse normalization for embedding/LM-head |

**Momentum:**

| Config | Description |
|--------|-------------|
| `momentum.yaml` | SGD momentum = 0.9 |
| `nesterov.yaml` | Nesterov momentum = 0.9 |
| `short-warmup.yaml` | ~5% warmup instead of 10% |

## Usage

```bash
# Run Adam baseline and SinkGD default
forgather -t adam_baseline.yaml train
forgather -t experimental.yaml train

# Run a sweep
for cfg in lr-01.yaml lr-03.yaml lr-04.yaml; do
    forgather -t $cfg train
done
```
