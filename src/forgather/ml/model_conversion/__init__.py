"""Model conversion framework for Forgather.

This package provides an extensible framework for converting between
different model formats (e.g., HuggingFace <-> Forgather).
"""

from .base import ModelConverter
from .registry import (
    register_converter,
    get_converter,
    list_converters,
    detect_model_type_from_hf,
    detect_model_type_from_forgather,
)
from .hf_converter import HFConverter
from .standard_mappings import (
    STANDARD_FORGATHER_TO_HF,
    STANDARD_HF_TO_FORGATHER,
    reverse_mapping,
)

__all__ = [
    "ModelConverter",
    "HFConverter",
    "register_converter",
    "get_converter",
    "list_converters",
    "detect_model_type_from_hf",
    "detect_model_type_from_forgather",
    "STANDARD_FORGATHER_TO_HF",
    "STANDARD_HF_TO_FORGATHER",
    "reverse_mapping",
]
