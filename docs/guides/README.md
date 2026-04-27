# Guides

Practical guides for common tasks and tools.

- **[Interactive CLI](interactive-cli.md)** - Interactive shell with tab completion, editor integration, and multi-file editing
- **[Forgather Server from the CLI](server-cli.md)** - Talk to a running forgather-server from the terminal: `--enqueue` for training/eval/tb/inference/convert/finalize/mkdocs, plus `forgather sched` / `job` / `gpu` for queue, log, and GPU control
- **[Model Conversion](model-conversion.md)** - Bidirectional HuggingFace / Forgather model conversion
- **[Creating a Model Project](creating-a-model-project.md)** - Define a custom model architecture from scratch
- **[Creating a Dataset Project](creating-a-dataset-project.md)** - Load, pack, and interleave HuggingFace datasets
- **[Finalize Model](finalize-model.md)** - Build a clean handoff directory after pre-training: source code + tokenizer + chat template + generation_config + a single preserved checkpoint
- **[Add-Tokens Config](add-tokens-config.md)** - YAML format for `--add-tokens`: how to wire ChatML / new EOS / pad tokens onto an existing tokenizer
- **[Evaluating Models](evaluating-models.md)** - Loss/perplexity evaluation via `forgather eval` with named dataset configs
- **[EOS Tokens and `generate()` Stopping Criteria](eos-and-generate-stopping.md)** - Theory of operation: how HF's `generate()` resolves stopping across the multiple files that carry EOS information
- **[Debugging Configuration Errors](debugging.md)** - Systematic troubleshooting workflow and common error patterns