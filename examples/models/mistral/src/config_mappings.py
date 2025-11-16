"""Configuration field mappings between HuggingFace Mistral and Forgather formats."""

# Mapping from Forgather config fields to HuggingFace Mistral config fields
# Used when converting from Forgather to HuggingFace
FORGATHER_TO_HF = {
    "vocab_size": "vocab_size",
    "hidden_size": "hidden_size",
    "dim_feedforward": "intermediate_size",
    "num_hidden_layers": "num_hidden_layers",
    "num_attention_heads": "num_attention_heads",
    "num_kv_heads": "num_key_value_heads",
    "d_head": "head_dim",
    "rms_norm_eps": "rms_norm_eps",
    "rope_theta": "rope_theta",
}

# Mapping from HuggingFace Mistral config fields to Forgather config fields
# Used when converting from HuggingFace to Forgather
HF_TO_FORGATHER = {
    "vocab_size": "vocab_size",
    "hidden_size": "hidden_size",
    "intermediate_size": "dim_feedforward",
    "num_hidden_layers": "num_hidden_layers",
    "num_attention_heads": "num_attention_heads",
    "num_key_value_heads": "num_kv_heads",
    "head_dim": "d_head",
    "rms_norm_eps": "rms_norm_eps",
    "rope_theta": "rope_theta",
    "max_position_embeddings": "max_model_length",
    "sliding_window": "sliding_window",  # Mistral-specific
}
