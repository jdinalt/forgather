# Causal LM

A vanilla transformer

This model is a decoder-only transformer model, roughly based on "Attention is All You Need."

- decoder-only (causal)
- post-layer-norm
- Absolute Sinusoidal positional embeddings
- Multi-head attention
- MPL Feedforward
- ReLU activations
- Layer Norm
- Embeddings init is scaled by 1/sqrt(d_model) and input embeddings are scaled by sqrt(d_model)

It supports the HF Attention interface and is compatible with vLLM