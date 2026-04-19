# Tiny Experiment

A minimal project that uses the default template library presets to train a
small (~4M parameter) causal language model. Primarily used to validate that
the base project templates work correctly.

## Configurations

| Config | Base template | Description |
|--------|---------------|-------------|
| `default.yaml` | `projects/tiny.yaml` | Default tiny project settings |
| `v2.yaml` | `projects/tinyv2.yaml` | Updated tiny project template |
| `v2_packed.yaml` | `projects/tinyv2_packed.yaml` | Updated template with sequence packing |
| `small.yaml` | `projects/small.yaml` | Slightly larger model and dataset |

## Usage

```bash
forgather ls
forgather -t default.yaml train
```
