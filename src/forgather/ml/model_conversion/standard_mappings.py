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
    "vocab_size": "vocab_size",
    "hidden_size": "hidden_size",
    "dim_feedforward": "intermediate_size",  # Forgather uses dim_feedforward, HF uses intermediate_size
    "num_hidden_layers": "num_hidden_layers",
    "num_attention_heads": "num_attention_heads",
    "num_kv_heads": "num_key_value_heads",  # Forgather uses num_kv_heads, HF uses num_key_value_heads
    "d_head": "head_dim",  # Forgather uses d_head, HF uses head_dim
    # Normalization and regularization
    "rms_norm_eps": "rms_norm_eps",
    "attention_dropout": "attention_dropout",
    # Positional encoding
    "rope_theta": "rope_theta",
    "rope_scaling": "rope_scaling",
    # Model configuration
    "tie_word_embeddings": "tie_word_embeddings",
}

# Additional mappings that only appear in one direction
_HF_ONLY_FIELDS = {
    "max_position_embeddings": "max_model_length",  # HF -> Forgather only
}

# Standard mappings for bidirectional conversion
STANDARD_FORGATHER_TO_HF = _BASE_FORGATHER_TO_HF.copy()

STANDARD_HF_TO_FORGATHER = {
    **reverse_mapping(_BASE_FORGATHER_TO_HF),
    **_HF_ONLY_FIELDS,
}

__all__ = [
    "STANDARD_FORGATHER_TO_HF",
    "STANDARD_HF_TO_FORGATHER",
    "reverse_mapping",
]
