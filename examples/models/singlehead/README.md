# Singlehead

A minimal, single-file custom model: a transformer whose attention layers each
have exactly **one** attention head. It is a teaching example on two fronts — how
to ship a stand-alone Forgather model with its own source code, and a structural
fact about attention itself.

It grew out of an experiment built around Elhage et al.'s
[*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html)
(2021). The original version was **attention-only** — no MLP — exactly the
object that paper studies, where a one- or two-layer attention-only transformer
can be read directly off its weights. This project keeps that variant
(`attention_only.yaml`) alongside a standard attention-plus-MLP version, and the
[`attention_only`](../../tiny_experiments/attention_only) Tiny Experiment trains
both to see what the MLP is worth.

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

## Positional encoding: why ALiBi

The choice of **ALiBi** (Press et al. 2021) is deliberate and central to the
analysis goal. ALiBi adds a linear, distance-based bias directly to the attention
*scores*; it never touches the query, key, or value content. So it **completely
removes the need for the QK and V matrices to carry positional information** —
those weights are free to encode purely *semantic* relationships.

Contrast RoPE or sinusoidal embeddings: there, position is mixed into the
queries/keys (or the token embeddings), so the same QK circuit has to handle
*both* "what is this token about" and "where is it" at once. That entanglement is
exactly what makes attention patterns hard to interpret. By segregating the two,
ALiBi should — in theory — make the learned QK circuit far easier to analyze.
Here there is a single (optionally trainable) slope per layer.

## The two variants: attention-only vs. attention + MLP

The model ships in two forms, selected by the `use_mlp` flag in the config:

- **Attention-only** (`attention_only.yaml`, `small_attention_only.yaml`) — each
  block is just `x = x + attention(norm(x))`, no MLP. This is the paper-faithful
  attention-only transformer (`AttentionOnlyLayer`).
- **Attention + MLP** (`4M.yaml`, `small.yaml`) — the standard pre-LN block with a
  ReLU-gated GLU feedforward after attention (`PreLNLayer`).

Both use pre-LN with RMSNorm and tied embeddings. The
[`attention_only` Tiny Experiment](../../tiny_experiments/attention_only) trains
the two head-to-head on TinyStories.

## Inference

Single-head attention here is implemented eager-only and **without** a KV cache:

```bash
# Example inference server settings for this model
forgather inf server -m output_models/tiny_singlehead/ -c --attn-implementation eager --disable-kv-cache
```

## References

- Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits* (QK / OV circuits, low-rank heads) — <https://transformer-circuits.pub/2021/framework/index.html>
- Press et al. 2021, *Train Short, Test Long: Attention with Linear Biases (ALiBi)* — [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
