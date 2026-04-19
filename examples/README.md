# Forgather Examples

This directory contains example projects demonstrating various Forgather capabilities and patterns.

For a curated list of the best-documented examples organized by what you
might want to do with them (pretrain, fine-tune, optimize memory, swap
optimizers, long-context training, etc.), see the **Featured Examples**
section of the [top-level README](../README.md).

## Directory Structure

- **tutorials/** - Educational projects with step-by-step learning materials (start here if you're new)
- **pretrain/** - Pretrain models from scratch
- **finetune/** - Fine-tune pretrained models
- **tiny_experiments/** - Small-scale ablations and experiments for testing specific features
- **base_lm_project/** - Bare harness that drives the raw `projects/lm_training_project.yaml` template (useful for inspecting the base template itself)
- **datasets/** - Dataset projects referenced by the training recipes above
- **models/** - Custom model-definition projects (Llama, Mistral, Qwen3, Gemma-3, DeepOne, etc.)
- **tokenizers/** - Custom tokenizer configuration examples
- **evaluation/** - Named eval configs consumed by `forgather eval test`
- **evaluate/** - Additional evaluation harness
- **trainer_control/** - Minimal worked example of the trainer-control protocol
- **torchtitan/** - Configure Torch Titan via Forgather (proof-of-concept)
- **snippets/** - Small utility fragments and snippets
