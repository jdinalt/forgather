# Forgather Template Library

This directory contains the built-in template library — the reusable building blocks that Forgather projects inherit from. Templates here define trainers, models, datasets, callbacks, tokenizers, and complete project scaffolds. Project-level configs extend these templates rather than duplicating their logic.

## Directory Structure

- **[base/](./base/)** - Abstract base types and lowest-level building blocks (trainer, model, dataset, tokenizer, callback, and training-script type definitions). These are the root of the template inheritance hierarchy and are not used directly.
- **[examples/](./examples/)** - Concrete, reusable component templates: standard transformer models (Llama, DeepOne, GPT-2), dataset helpers, tokenizer configurations, LM callbacks, and the `lm_training_project.yaml` and `finetune_v2.yaml` project scaffolds that most training projects inherit from.
- **[finetune/](./finetune/)** - Fine-tuning project base template (`base_finetune_proj.yaml`); extends the examples scaffold with fine-tuning-specific defaults for model paths, dataset directories, and LR configuration.
- **[tiny_experiments/](./tiny_experiments/)** - Lightweight project templates and matching tiny model definitions (tiny/small Llama and GPT-2) for fast iteration and ablations on the TinyStories or Fineweb-Edu datasets.
- **[torchtitan/](./torchtitan/)** - Project templates for running training through Torch Titan (`fg_titan.yaml` for Forgather-managed Titan jobs, `native_titan.yaml` for native Titan configs).
