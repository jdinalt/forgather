# Qwen3

[Qwen3 Family](https://huggingface.co/collections/Qwen/qwen3)

## Architecture

Qwen3 is a transformer-based causal language model that follows the Llama architecture with one key distinction: **RMSNorm is applied to the Query and Key projections** before attention computation. This Query/Key normalization helps stabilize training and improve model performance.

Additional architectural features:
- **Attention Mechanism**: Multi-head attention with Grouped Query Attention (GQA) support
- **Activation Function**: SiLU (Swish) activation in feedforward layers
- **Position Embeddings**: Rotary Position Embeddings (RoPE) with configurable theta (1M base)
- **Normalization**: RMS LayerNorm for pre-normalization and Q/K normalization
- **Attention Biases**: No biases on attention projections (Q/K/V/O)
- **Extended Context**: Supports up to 32K+ token context windows depending on model variant

### Document Packing Support

Qwen3 models do not define a BOS (Beginning-of-Sequence) token in their default tokenizer configuration. This can be problematic when using document packing, where multiple documents are packed into a single training sequence with the BOS token serving as the document delimiter (combined with attention masking to prevent cross-document attention).

**Workaround**: The Forgather model converter supports adding a BOS token during conversion using the `--add-tokens` flag. This allows you to add named special tokens (bos_token, eos_token, pad_token, unk_token) with customizable initialization strategies.

Example:
```bash
# Create token configuration
cat > qwen3_bos.yaml << EOF
bos_token: "<|begin_of_text|>"
EOF

# Convert with BOS token addition
forgather convert --add-tokens qwen3_bos.yaml ~/models/qwen3-8b ~/models/qwen3-8b-with-bos
```

For complete documentation on token addition and initialization strategies, see the [Model Conversion Utility documentation](../../../tools/convert_model/README.md#vocabulary-extension).

### Limitations

- **MoE Models Not Supported**: The current Qwen3 converter does not support Mixture-of-Experts (MoE) variants of Qwen3 models. Only dense decoder-only models are supported.
- **Qwen3-VL Not Supported**: Vision-language variants are not currently supported. Only text-only causal language models are supported.


## Tested Models

- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base)