# Singlehead

A simple ALiBi transformer with a single attention head in each layer

This project primarily serves as an example of a stand-alone custom model with its own source code.

## Attention

Each attention layer only has a single attention head. A consequence of this is that we don't need to project the hidden states down to the head dimension and back to the hidden dimension. The result is that we only need two matrices:

- QK : Combines the function of the Query and Key matrix into a single matrix
- V : Value, which serves the usual function.

There is no "output" matrix, as we did not down-project the value.

This configuration greatly simplifies the attention computation:

```python
attention_scores = x @ QK @ x.transpose(-2, -1) * scale
```

Unlike absolute or RoPE positional encoding, ALiBi positional encoding ensures that semantic and positional relevance
are fully segregated, making for easier analysis of the model.

## Inference

This model does not support the KV Cache and only supports "eager" attention.

```bash
# Example inference server settings for this model
forgather inf server -m output_models/tiny_singlehead/ -c --attn-implementation eager --disable-kv-cache
```