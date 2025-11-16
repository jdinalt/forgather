"""Configuration field mappings between HuggingFace Mistral and Forgather formats."""

from forgather.ml.model_conversion import (
    STANDARD_FORGATHER_TO_HF,
    STANDARD_HF_TO_FORGATHER,
)

# Mistral extends standard mappings with sliding_window support
FORGATHER_TO_HF = {
    **STANDARD_FORGATHER_TO_HF,
    "sliding_window": "sliding_window",  # Mistral-specific
}

HF_TO_FORGATHER = {
    **STANDARD_HF_TO_FORGATHER,
    "sliding_window": "sliding_window",  # Mistral-specific
}
