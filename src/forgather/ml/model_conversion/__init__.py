"""Model conversion framework for Forgather.

This package provides an extensible framework for converting between
different model formats (e.g., HuggingFace <-> Forgather).
"""

from .base import ModelConverter
from .registry import (
    register_converter,
    get_converter,
    list_converters,
    detect_model_type,
    detect_model_type_from_hf,
    detect_model_type_from_forgather,
    discover_and_register_converters,
)
from .hf_converter import HFConverter
from .standard_mappings import (
    STANDARD_FORGATHER_TO_HF,
    STANDARD_HF_TO_FORGATHER,
    reverse_mapping,
)
from .vllm_validator import (
    validate_tp_plan,
    validate_pp_plan,
    validate_vllm_plans,
    print_model_structure,
)

# Also expose discovery module for advanced use
from . import discovery

__all__ = [
    "ModelConverter",
    "HFConverter",
    "register_converter",
    "get_converter",
    "list_converters",
    "detect_model_type",
    "detect_model_type_from_hf",
    "detect_model_type_from_forgather",
    "discover_and_register_converters",
    "discovery",
    "STANDARD_FORGATHER_TO_HF",
    "STANDARD_HF_TO_FORGATHER",
    "reverse_mapping",
    "validate_tp_plan",
    "validate_pp_plan",
    "validate_vllm_plans",
    "print_model_structure",
]
