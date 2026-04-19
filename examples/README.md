# Forgather Examples

This directory contains example projects demonstrating various Forgather capabilities and patterns.

For a curated list of the best-documented examples organized by what you
might want to do with them (pretrain, fine-tune, optimize memory, swap
optimizers, long-context training, etc.), see the **Featured Examples**
section of the [top-level README](../README.md).

## Directory Structure

- **[tutorials/](./tutorials/README.md)** - Educational projects with step-by-step learning materials (start here if you're new)
- **[pretrain/](./pretrain/README.md)** - Pretrain models from scratch
- **[finetune/](./finetune/README.md)** - Fine-tune pretrained models
- **[tiny_experiments/](./tiny_experiments/README.md)** - Small-scale ablations and experiments for testing specific features
- **[base_lm_project/](./base_lm_project/README.md)** - Bare harness that drives the raw `projects/lm_training_project.yaml` template (useful for inspecting the base template itself)
- **[datasets/](./datasets/README.md)** - Dataset projects referenced by the training recipes above
- **[models/](./models/README.md)** - Custom model-definition projects (Llama, Mistral, Qwen3, Gemma-3, DeepOne, etc.)
- **[tokenizers/](./tokenizers/README.md)** - Custom tokenizer configuration examples
- **[evaluation/](./evaluation/README.md)** - Named eval configs consumed by `forgather eval test`
- **[trainer_control/](./trainer_control/README.md)** - Minimal worked example of the trainer-control protocol
- **[torchtitan/](./torchtitan/README.md)** - Configure Torch Titan via Forgather (proof-of-concept)
- **[snippets/](./snippets/README.md)** - Small utility fragments and snippets
