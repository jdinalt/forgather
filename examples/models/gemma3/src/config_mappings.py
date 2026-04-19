"""Configuration field mappings between HuggingFace Gemma-3 and Forgather formats."""

from forgather.ml.model_conversion import (
    STANDARD_FORGATHER_TO_HF,
    STANDARD_HF_TO_FORGATHER,
)

# Gemma-3 uses the standard field set plus several model-specific fields.
# ``rope_parameters`` is already in the standard mapping; for Gemma-3 it is a
# nested dict keyed by layer type (``full_attention`` / ``sliding_attention``),
# which ``GemmaDualRotaryEmbedding`` consumes directly.
_GEMMA_EXTRA = {
    "layer_types": "layer_types",
    "query_pre_attn_scalar": "query_pre_attn_scalar",
    "hidden_activation": "hidden_activation",
    "final_logit_softcapping": "final_logit_softcapping",
}

FORGATHER_TO_HF = STANDARD_FORGATHER_TO_HF.copy()
FORGATHER_TO_HF.update(_GEMMA_EXTRA)

HF_TO_FORGATHER = STANDARD_HF_TO_FORGATHER.copy()
HF_TO_FORGATHER.update({v: k for k, v in _GEMMA_EXTRA.items()})
