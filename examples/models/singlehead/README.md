# Singlehead

A minimal, single-file custom model: a transformer whose attention layers each
have exactly **one** attention head. It is a teaching example on two fronts — how
to ship a stand-alone Forgather model with its own source code, and a structural
fact about attention itself.

## The point: MHA is a sum of rank-reduced heads

Multi-head attention splits `d_model` into `h` heads of width `d_head =
d_model/h`. Each head projects the hidden state *down* to `d_head` for its query,
key, and value, then projects the attended values back *up* to `d_model` through
the output matrix. In the QK/OV-circuit view of Elhage et al. (*A Mathematical
Framework for Transformer Circuits*, 2021), a head is therefore a **low-rank**
(rank-`d_head`) factorization of two `d_model × d_model` interactions:

- the **QK circuit** `Wq^T Wk` — which positions attend to which; and
- the **OV circuit** `Wo Wv` — what an attended token writes to the residual stream.

The down- and up-projections *are* that low-rank factorization. With a single
**full-rank** head you no longer need them — the interaction can stay at full
`d_model` rank — and each circuit collapses to one matrix:

- `QK`: a single `d_model × d_model` matrix (the query and key projections
  merged), so attention scores are simply `x @ QK @ xᵀ · scale`;
- `OV`: just the value matrix `V`, with **no output projection** — nothing was
  down-projected, so there is nothing to project back up.

So a Singlehead layer carries two weight matrices (`QK`, `V`) instead of four,
and attention is a single bilinear form. It makes concrete that a head's
"query", "key", "value", and "output" matrices are intermediate bookkeeping —
the computation that matters is the two circuits.

```python
attention_scores = x @ QK @ x.transpose(-2, -1) * scale
```

## Positional encoding

The model uses **ALiBi** (Press et al. 2021): a linear, head-specific bias added
to the attention scores rather than to the embeddings. Because the position
signal lives in the scores and never mixes into the values, semantic and
positional relevance stay fully segregated — which keeps the QK circuit clean to
read. Here there is a single (optionally trainable) slope per layer.

## Normalization / MLP

Pre-LN with RMSNorm, and a ReLU-gated GLU feedforward.

## Inference

Single-head attention here is implemented eager-only and **without** a KV cache:

```bash
# Example inference server settings for this model
forgather inf server -m output_models/tiny_singlehead/ -c --attn-implementation eager --disable-kv-cache
```

## References

- Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits* (QK / OV circuits, low-rank heads) — <https://transformer-circuits.pub/2021/framework/index.html>
- Press et al. 2021, *Train Short, Test Long: Attention with Linear Biases (ALiBi)* — [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
