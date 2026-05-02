# Models

A collection of model definitions.

## Models

- **[causal_lm](./causal_lm/README.md)** - A vanilla decoder-only transformer, loosely based on "Attention is All You Need."
- **[llama](./llama/README.md)** - Llama models in various sizes.
- **[llama_canon](./llama_canon/README.md)** - Llama extended with Canon layers (depthwise causal 1D convolutions) from *Physics of Language Models: Part 4.1*.
- **[mistral](./mistral/README.md)** - Mistral with sliding-window attention support.
- **[qwen3](./qwen3/README.md)** - Qwen3 architecture from the Qwen3 model family.
- **[gemma3](./gemma3/README.md)** - Google Gemma-3 text model with HuggingFace ↔ Forgather round-trip conversion support.
- **[deepone](./deepone/README.md)** - A large Deepnet transformer with ALiBi positional encoding.
- **[singlehead](./singlehead/README.md)** - A minimal ALiBi transformer with a single attention head per layer; primarily a standalone custom-model example.


For the full `forgather model` command reference — constructing, testing, checkpoint handling, and using models with the HuggingFace and Forgather APIs — see **[docs/guides/model-cli.md](../../guides/model-cli.md)**.

