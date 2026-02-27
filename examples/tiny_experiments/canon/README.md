# Canon Layer Experiments

Experiments with Canon layers from "Physics of Language Models: Part 4.1,
Architecture Design and the Magic of Canon Layers" (Allen-Zhu, 2025).

Canon layers are small depthwise causal 1D convolutions (kernel_size=4) inserted
at four positions in each transformer block (A: pre-attention, B: on QKV, C:
pre-FFN, D: on gate/up). They provide local token mixing with minimal overhead.

## Questions

1. How much does RoPE contribute when Canon layers provide local token mixing?
2. Does Canon compensate for the absence of positional encoding?

## Configurations

Model configs (in this project):
- `baseline.yaml` -- LlamaCanon 4M, identical to `examples/models/llama_canon` 4M
- `nope.yaml` -- LlamaCanon 4M without RoPE (NoPE)

Model configs (in `examples/tiny_experiments/llama_nope/`):
- `nope_4M.yaml` -- Plain Llama 4M without RoPE (no Canon layers)

Training configs:
- `train_baseline.yaml` -- Train the Canon baseline
- `train_nope.yaml` -- Train the Canon NoPE variant
- `train_llama_nope.yaml` -- Train the plain Llama NoPE variant

## Results

All models trained for 1 epoch on TinyStories (abridged), batch_size=32,
max_seq_len=512, AdamW lr=1e-3 with InfiniteLR scheduler.

| Model | Eval Loss | Tok/s | Memory | Notes |
|-------|-----------|-------|--------|-------|
| Llama 4M | 1.3028 | 287K | 1.50 GiB | Reference (from tiny_models) |
| **Canon Baseline** (RoPE) | **1.2256** | 227K | 2.12 GiB | Canon + RoPE |
| Canon NoPE | 1.2379 | 229K | 2.11 GiB | Canon, no RoPE |
| Llama NoPE | 1.3667 | 348K | 1.49 GiB | No Canon, no RoPE |

## Observations

- Canon layers improve eval loss by 5.9% over plain Llama (1.2256 vs 1.3028).
- Removing RoPE from Canon costs only 1.0% eval loss (1.2379 vs 1.2256).
- Removing RoPE from plain Llama costs 4.9% eval loss (1.3667 vs 1.3028).
- Canon NoPE still outperforms Llama+RoPE by 5.0% (1.2379 vs 1.3028).
- Canon layers largely compensate for the lack of positional encoding:
  the RoPE penalty with Canon is only 1.0% vs 4.9% without Canon.

## Project Structure

```
examples/tiny_experiments/canon/        # Main experiment project
  meta.yaml                             # Extends llama_canon templates
  templates/
    project.yaml                        # Training base (extends projects/tiny.yaml)
    configs/
      baseline.yaml                     # Model: Canon 4M with RoPE
      nope.yaml                         # Model: Canon 4M without RoPE
      train_baseline.yaml               # Training config for baseline
      train_nope.yaml                   # Training config for NoPE
      train_llama_nope.yaml             # Training config for Llama NoPE

examples/tiny_experiments/llama_nope/   # Companion model project
  meta.yaml                             # Extends plain llama templates
  templates/configs/
    nope_4M.yaml                        # Model: Llama 4M without RoPE
```
