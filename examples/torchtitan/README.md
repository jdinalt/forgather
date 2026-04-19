# Torch Titan

Forgather integration with Torch Titan

## Projects

- **[llama3/](./llama3/README.md)** - Reproduces Torch Titan's Llama3 base configurations via Forgather, with a few derived configs demonstrating template-based config management.
- **[tiny_titan/](./tiny_titan/README.md)** - First Forgather-native Torch Titan trainer using dependency injection for training assets; includes an FSDP config for a 117M Llama3 model.
- **[test_parallelisms/](./test_parallelisms/README.md)** - Compares various parallelism strategies (DDP, tensor parallel, pipeline parallel, etc.) against a single-GPU control with matched effective batch size.


For an overview of the Native Titan vs Forgather Titan approaches and general usage, see **[docs/trainers/torchtitan.md](../../docs/trainers/torchtitan.md)**.
