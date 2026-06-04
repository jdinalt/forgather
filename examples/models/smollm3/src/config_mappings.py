"""Configuration field mappings between HuggingFace SmolLM3 and Forgather formats.

SmolLM3 uses the standard Llama-family fields (hidden_size, num_hidden_layers,
rope_parameters, tie_word_embeddings, ...) plus one model-specific scalar that
controls the NoPE schedule: ``no_rope_layer_interval``. The Forgather template
regenerates the explicit ``no_rope_layers`` list from that interval, so only the
interval needs to be carried across the conversion boundary.
"""

from forgather.ml.model_conversion import (
    STANDARD_FORGATHER_TO_HF,
    STANDARD_HF_TO_FORGATHER,
)

FORGATHER_TO_HF = STANDARD_FORGATHER_TO_HF.copy()
FORGATHER_TO_HF["no_rope_layer_interval"] = "no_rope_layer_interval"

HF_TO_FORGATHER = STANDARD_HF_TO_FORGATHER.copy()
HF_TO_FORGATHER["no_rope_layer_interval"] = "no_rope_layer_interval"
