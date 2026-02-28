# Canon Layer Experiments

Experiments with Canon layers from "Physics of Language Models: Part 4.1,
Architecture Design and the Magic of Canon Layers" (Allen-Zhu, 2025).

Canon layers are small depthwise causal 1D convolutions (kernel_size=4) inserted
at four positions in each transformer block (A: pre-attention, B: on QKV, C:
pre-FFN, D: on gate/up). They provide local token mixing with minimal overhead.

## Questions

1. How much does RoPE contribute when Canon layers provide local token mixing?
2. Does Canon compensate for the absence of positional encoding?
3. Which subset of Canon positions (A, B, C, D) gives the best quality/speed tradeoff?
4. Can Canon-A alone replace RoPE while maintaining quality?

## Results

All models are 4M parameter, trained for 1 epoch on TinyStories (abridged),
batch_size=32, max_seq_len=512, AdamW lr=1e-3 with InfiniteLR scheduler.

### Positional Encoding Ablation

| Model | Eval Loss | Tok/s | Memory | Notes |
|-------|-----------|-------|--------|-------|
| Llama 4M | 1.3011 | 291K | 1.50 GiB | Reference |
| Canon ABCD (RoPE) | 1.2256 | 227K | 2.12 GiB | Full Canon baseline |
| Canon ABCD (NoPE) | 1.2379 | 229K | 2.11 GiB | Canon, no RoPE |
| Canon-A (NoPE) | 1.3120 | 323K | 1.54 GiB | Minimal Canon, no RoPE |
| Llama NoPE | 1.3667 | 348K | 1.49 GiB | No Canon, no RoPE |

Removing RoPE from full Canon costs 1.0% eval loss. Removing RoPE from Llama
costs 5.0%. Canon layers largely compensate for the lack of positional encoding.

Canon-A NoPE is particularly interesting: at 323K tok/s it is 11% faster than
Llama+RoPE, with only 0.8% worse eval loss (1.3120 vs 1.3011). The single
pre-attention convolution recovers most of the positional encoding benefit that
RoPE provides, at a fraction of the cost. At larger model sizes, where RoPE's
complex-valued computation and memory footprint become significant, this tradeoff
becomes even more favorable.

### Canon Position Ablation

| Model | Positions | Eval Loss | Tok/s | Memory | Overhead vs Llama |
|-------|-----------|-----------|-------|--------|-------------------|
| Llama 4M | -- | 1.3011 | 291K | 1.50 GiB | -- |
| **Canon-A** | A | 1.2643 | **316K** | **1.56 GiB** | **+9% faster**, +4% mem |
| Canon-A NoPE | A (no RoPE) | 1.3120 | **323K** | **1.54 GiB** | **+11% faster**, +3% mem |
| Canon-AB | A, B | 1.2601 | 262K | 1.73 GiB | -10% time, +15% mem |
| Canon-AC | A, C | 1.2420 | 296K | 1.62 GiB | +2% faster, +8% mem |
| **Canon-B** | B | **1.2149** | 278K | 1.67 GiB | -4% time, +11% mem |
| Canon-BD | B, D | 1.2557 | 250K | 2.01 GiB | -14% time, +34% mem |
| Canon-ABCD | A, B, C, D | 1.2256 | 227K | 2.12 GiB | -22% time, +41% mem |

### Analysis

**Best quality**: Canon-B alone achieves the lowest eval loss (1.2149),
even beating full ABCD (1.2256). This is unexpected and suggests that at
this model size, the QKV convolution is the single most impactful position.

**Fastest with meaningful gains**: Canon-A alone runs at 316K tok/s -- 9%
*faster* than plain Llama (291K) despite adding a convolution per layer. The
speedup likely comes from the Triton kernel being faster than the standard
module overhead it replaces in the code path. Canon-A achieves 2.8% better
eval loss (1.2643 vs 1.3011) with essentially zero cost.

**Canon-A and induction heads**: Canon-A sits before attention, mixing
adjacent token representations. This is exactly the pattern needed to
facilitate induction head formation: the first attention head can match
copied-forward keys from the convolution, enabling better sequence copying.
The consistent improvement from Canon-A alone supports this hypothesis.

**Canon-A as RoPE replacement**: Canon-A NoPE (1.3120) nearly matches
Llama+RoPE (1.3011) -- just 0.8% worse -- while running 11% faster. The
pre-attention convolution provides enough local positional signal that
explicit positional encoding becomes nearly redundant. This is promising for
larger models where RoPE's O(d_head) complex computation per head is costly.

**Best quality/speed tradeoff**: Canon-AC runs at 296K tok/s (2% faster
than Llama) while achieving 4.5% better eval loss (1.2420 vs 1.3011).

**Canon-B surprise**: Canon-B at 278K tok/s delivers the best eval loss
(1.2149) at moderate overhead. This suggests the QKV convolution provides
complementary benefits beyond what pre-attention mixing achieves.

**Combining A+B hurts**: Canon-AB (1.2601) is worse than B alone (1.2149)
and barely better than A alone (1.2643). The pre-attention convolution (A)
and QKV convolution (B) may interfere -- A mixes tokens before attention,
then B mixes the already-mixed representations again, potentially blurring
the signal. Canon-B works best in isolation at this scale.

**Canon-D adds overhead without benefit**: Comparing B-only (1.2149) vs BD
(1.2557), adding Canon-D actually hurts quality while costing 10% throughput.
The feedforward gate/up convolutions may interfere at this model size.

## Configurations

Model configs (in `models/` sub-project):
- `baseline.yaml` -- Canon ABCD 4M (full Canon, identical to llama_canon 4M)
- `nope.yaml` -- Canon ABCD 4M without RoPE
- `canon_a.yaml` -- Canon-A: only pre-attention convolution
- `canon_a_nope.yaml` -- Canon-A without RoPE
- `canon_ab.yaml` -- Canon-AB: pre-attention (A) and attn QKV (B)
- `canon_ac.yaml` -- Canon-AC: pre-attention (A) and pre-FFN (C)
- `canon_b.yaml` -- Canon-B: only attention QKV convolution
- `canon_bd.yaml` -- Canon-BD: attention QKV (B) and FFN gate/up (D)

Model configs (in `examples/tiny_experiments/llama_nope/`):
- `nope_4M.yaml` -- Plain Llama 4M without RoPE

Training configs (prefix `train_`):
- `train_baseline.yaml`, `train_nope.yaml`, `train_llama_nope.yaml`
- `train_canon_a.yaml`, `train_canon_a_nope.yaml`
- `train_canon_ab.yaml`, `train_canon_ac.yaml`
- `train_canon_b.yaml`, `train_canon_bd.yaml`

## Project Structure

```
examples/tiny_experiments/canon/        # Main experiment project
  meta.yaml                             # Training project metadata
  templates/
    project.yaml                        # Training base (extends projects/tiny.yaml)
    configs/
      train_*.yaml                      # Training configs

  models/                               # Model definitions sub-project
    meta.yaml                           # Adds llama_canon templates to search path
    templates/configs/
      baseline.yaml                     # Model: Canon ABCD (extends llama_canon 4M)
      nope.yaml                         # Model: Canon ABCD, no RoPE
      canon_a.yaml                      # Model: Canon A only
      canon_a_nope.yaml                 # Model: Canon A only, no RoPE
      canon_ab.yaml                     # Model: Canon A+B
      canon_ac.yaml                     # Model: Canon A+C
      canon_b.yaml                      # Model: Canon B only
      canon_bd.yaml                     # Model: Canon B+D

examples/tiny_experiments/llama_nope/   # Companion model project
  meta.yaml                             # Adds llama templates to search path
  templates/configs/
    nope_4M.yaml                        # Model: Llama 4M without RoPE
```
