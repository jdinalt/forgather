"""Parameter name mappings between HuggingFace Llama and Forgather formats."""

from forgather.ml.model_conversion.standard_mappings import (
    LLAMA_FORGATHER_TO_HF,
    LLAMA_HF_TO_FORGATHER,
)

# HuggingFace Llama to Forgather Dynamic Llama parameter name mappings
HF_TO_FORGATHER = LLAMA_HF_TO_FORGATHER

# Forgather Dynamic Llama to HuggingFace Llama parameter name mappings
FORGATHER_TO_HF = LLAMA_FORGATHER_TO_HF
