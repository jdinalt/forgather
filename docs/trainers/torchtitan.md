# Torch Titan Integration

Forgather integrates with [Torch Titan](https://github.com/pytorch/torchtitan/), PyTorch's large-scale training framework. The integration comes in two flavors:

**Native Titan** uses Forgather purely for configuration management. The training loop runs unmodified Torch Titan code; Forgather only supplies the YAML configuration and uses its template inheritance system to derive and override Titan configs. This requires no custom Titan code.

**Forgather Titan** (`tiny_titan`) constructs a `TrainSpec` from the Forgather configuration using dependency injection. This allows custom optimizers, LR schedulers, datasets, and other training assets to be swapped in via configuration without touching Python.

## Basic usage

```bash
# List available configurations
forgather ls

# Show details of preprocessed configuration
forgather [-t CONFIG_NAME] pp

# Train (launches via torchrun with the correct settings)
forgather [-t CONFIG_NAME] train

# Start Tensorboard (on another terminal)
# Use "--bind_all" to expose on all interfaces, not just localhost
forgather tb [-- --bind_all]
```

## Example projects

See [`examples/torchtitan/`](../examples/torchtitan/README.md) for working configurations:

- **llama3** — Native Titan: reproduces the official Torch Titan Llama3 base configs via Forgather, demonstrating how template inheritance simplifies managing Titan YAML variants.
- **tiny_titan** — Forgather Titan: a native Torch Titan trainer with dependency injection for training assets; includes an FSDP config for a 117M parameter Llama3 model.
- **test_parallelisms** — Compares DDP, tensor parallel, and pipeline parallel strategies against a single-GPU control with matched effective batch size.
