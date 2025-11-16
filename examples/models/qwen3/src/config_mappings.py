"""Configuration field mappings between HuggingFace Qwen3 and Forgather formats."""

from forgather.ml.model_conversion import (
    STANDARD_FORGATHER_TO_HF,
    STANDARD_HF_TO_FORGATHER,
)

# Qwen3 uses all standard mappings with no model-specific fields
FORGATHER_TO_HF = STANDARD_FORGATHER_TO_HF.copy()

HF_TO_FORGATHER = STANDARD_HF_TO_FORGATHER.copy()
