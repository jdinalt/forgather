## Tiny Models

A collection of tiny models to train on the Tiny Stories dataset with the tiny_stories_2k tokenizer.

This allows for direct comparison of model architectures.

### Models
- Tiny Causal -- A basic decoder-only Transformer, roughly based upon "Attention is All You Need" 
- Tiny Forgather Llama -- Forgather's Llama implementation
- Tiny HF Llama -- The Huggingface Llama implementation
- Tiny HF GPT2 -- A Huggingface GPT2 model
- Tiny Deepone -- This is a post-layer-norm model, using Deepnet initialization and ALiBI relative positional encoding
- Tiny Forgather Qwen3 -- A Forgather Qwen3 model
- Tiny Mistral -- A tiny Mistral model. This is essential the same as Llama, but uses sliding-window attention