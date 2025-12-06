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
    "rope_theta": "rope_theta",
    "rope_scaling": "rope_scaling",
    "tie_word_embeddings": "tie_word_embeddings",
    "max_position_embeddings": "max_position_embeddings",
}

# Standard mappings for bidirectional conversion
STANDARD_FORGATHER_TO_HF = _BASE_FORGATHER_TO_HF.copy()

STANDARD_HF_TO_FORGATHER = reverse_mapping(_BASE_FORGATHER_TO_HF)

__all__ = [
    "STANDARD_FORGATHER_TO_HF",
    "STANDARD_HF_TO_FORGATHER",
    "reverse_mapping",
]
