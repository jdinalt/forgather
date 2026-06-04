"""Parameter name mappings between HuggingFace SmolLM3 and Forgather formats.

SmolLM3's parameter names are identical to Llama's (NoPE removes rotary
application from some layers but adds no parameters), so we reuse the shared
Llama weight-name mappings verbatim.
"""

from forgather.ml.model_conversion.standard_mappings import (
    LLAMA_FORGATHER_TO_HF,
    LLAMA_HF_TO_FORGATHER,
)

HF_TO_FORGATHER = LLAMA_HF_TO_FORGATHER

FORGATHER_TO_HF = LLAMA_FORGATHER_TO_HF
