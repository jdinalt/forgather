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

__all__ = [
    "ModelConverter",
    "HFConverter",
    "register_converter",
    "get_converter",
    "list_converters",
    "detect_model_type_from_hf",
    "detect_model_type_from_forgather",
]
