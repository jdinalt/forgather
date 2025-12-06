"""Configuration field mappings between HuggingFace Mistral and Forgather formats."""

from forgather.ml.model_conversion import (
    STANDARD_FORGATHER_TO_HF,
    STANDARD_HF_TO_FORGATHER,
)

# Llama uses all standard mappings with no model-specific fields
FORGATHER_TO_HF = STANDARD_FORGATHER_TO_HF.copy()

HF_TO_FORGATHER = STANDARD_HF_TO_FORGATHER.copy()
