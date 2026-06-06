# Snippets

A place for short code examples and debugging tools.

## Contents

- **debug_checkpoint_state/** - Tools for debugging InterleavedDataset checkpoint issues
  - `dump_checkpoint_state.py` - Dump and analyze dataset checkpoint state
  - `analyze_checkpoint_probabilities.py` - Analyze soft_sequential probability calculations
  - See `debug_checkpoint_state/README.md` for detailed usage

- **debug_datasets.py** - Debug dataset loading and iteration
- **fast_hf_loader.py** - Fast loading example for HuggingFace datasets
- **log_analysis_example.py** - Example of using the training log analysis API
- **log_comparison_example.sh** - Script for comparing multiple training runs
- **prompt_test.py** - Testing prompts and model outputs
- **ab_test.py** - Interactive blind A/B subjective comparison of two models over a
  prompt set (shuffled, randomized pairs, JSON log for pooling participants). See
  the [attention_only experiment](../tiny_experiments/attention_only) for the
  motivating use case.