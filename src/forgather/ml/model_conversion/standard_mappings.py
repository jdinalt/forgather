"""Standard configuration field mappings for transformer models.

This module defines common configuration field mappings that are shared across
most transformer model types (Llama, Mistral, Qwen, etc.). Model-specific converters
can extend these standard mappings with their own unique fields.
"""

from typing import Dict


def reverse_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    """Reverse a configuration mapping (swap keys and values).

    Args:
        mapping: Dictionary mapping field names from source to destination

    Returns:
        Dictionary with keys and values swapped
    """
    return {v: k for k, v in mapping.items()}


# Base mapping: Forgather field names -> HuggingFace field names
# Most fields map to themselves; only a few differ between the two formats
_BASE_FORGATHER_TO_HF = {
    # Standard model architecture fields
    "bos_token_id": "bos_token_id",
    "eos_token_id": "eos_token_id",
    "pad_token_id": "pad_token_id",
    "initializer_range": "initializer_range",
    "head_dim": "head_dim",
    "attention_dropout": "attention_dropout",
    "use_cache": "use_cache",
    "vocab_size": "vocab_size",
    "hidden_size": "hidden_size",
    "intermediate_size": "intermediate_size",
    "num_hidden_layers": "num_hidden_layers",
    "num_attention_heads": "num_attention_heads",
    "num_key_value_heads": "num_key_value_heads",
    "rms_norm_eps": "rms_norm_eps",
    "attention_dropout": "attention_dropout",
    "rope_parameters": "rope_parameters",
    "tie_word_embeddings": "tie_word_embeddings",
    "max_position_embeddings": "max_position_embeddings",
    "sliding_window": "sliding_window",
}

# Standard mappings for bidirectional conversion
STANDARD_FORGATHER_TO_HF = _BASE_FORGATHER_TO_HF.copy()

STANDARD_HF_TO_FORGATHER = reverse_mapping(_BASE_FORGATHER_TO_HF)

__all__ = [
    "STANDARD_FORGATHER_TO_HF",
    "STANDARD_HF_TO_FORGATHER",
    "reverse_mapping",
]

# HuggingFace Llama to Forgather Dynamic Llama parameter name mappings
# Format: List of (pattern, replacement, [children]) tuples for recursive regex substitution
LLAMA_HF_TO_FORGATHER = [
    (r"lm_head\.", r"lm_head.", []),
    (
        r"model\.",
        r"causal_lm.",
        [
            (r"embed_tokens\.", r"input_encoder.embedding.", []),
            (r"norm\.", r"layer_stack.layer_norm.", []),
            (
                r"layers\.(\d+)\.",
                r"layer_stack.layers.\1.",
                [
                    (
                        r"self_attn\.",
                        r"attention.",
                        [
                            (r"q_proj\.", r"query_linear.", []),
                            (r"k_proj\.", r"key_linear.", []),
                            (r"v_proj\.", r"value_linear.", []),
                            (r"o_proj\.", r"output_linear.", []),
                        ],
                    ),
                    (r"mlp\.", r"feedforward.", []),
                    (r"input_layernorm\.", r"norm1.", []),
                    (r"post_attention_layernorm\.", r"norm2.", []),
                ],
            ),
        ],
    ),
]

# Forgather Dynamic Llama to HuggingFace Llama parameter name mappings
LLAMA_FORGATHER_TO_HF = [
    (r"lm_head\.", r"lm_head.", []),
    (
        r"causal_lm\.",
        r"model.",
        [
            (r"input_encoder\.embedding\.", r"embed_tokens.", []),
            (r"layer_stack\.layer_norm\.", r"norm.", []),
            (
                r"layer_stack\.layers\.(\d+)\.",
                r"layers.\1.",
                [
                    (
                        r"attention\.",
                        r"self_attn.",
                        [
                            (r"query_linear\.", r"q_proj.", []),
                            (r"key_linear\.", r"k_proj.", []),
                            (r"value_linear\.", r"v_proj.", []),
                            (r"output_linear\.", r"o_proj.", []),
                        ],
                    ),
                    (r"feedforward\.", r"mlp.", []),
                    (r"norm1\.", r"input_layernorm.", []),
                    (r"norm2\.", r"post_attention_layernorm.", []),
                ],
            ),
        ],
    ),
]
