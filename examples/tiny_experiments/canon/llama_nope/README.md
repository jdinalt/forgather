# Llama NoPE

A Llama model definition with positional encoding disabled
(`rel_positional_encoder: null`). This is a model sub-project of the Canon
experiments, serving as a control for comparing Llama without RoPE against
Canon variants that also lack RoPE.

## Configurations

| Config | Description |
|--------|-------------|
| `nope_4M.yaml` | 4M parameter Llama without positional encoding |

This sub-project is consumed by `train_llama_nope.yaml` in the parent Canon
project. See the [Canon README](../README.md) for the full experimental context.
