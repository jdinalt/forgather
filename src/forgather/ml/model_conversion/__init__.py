"""Model conversion framework for Forgather.

This package provides an extensible framework for converting between
different model formats (e.g., HuggingFace <-> Forgather).
"""

# Also expose discovery module for advanced use
from . import discovery
from .base import ForgatherOnlyConverter, ModelConverter, VersionMigration
from .hf_converter import HFConverter
from .registry import (
    compose_migration_chain,
    detect_model_type,
    detect_model_type_from_forgather,
    detect_model_type_from_hf,
    discover_and_register_converters,
    get_converter,
    list_converters,
    parse_arch_version,
    register_converter,
)
from .standard_mappings import (
    STANDARD_FORGATHER_TO_HF,
    STANDARD_HF_TO_FORGATHER,
    reverse_mapping,
)

__all__ = [
    "ModelConverter",
    "ForgatherOnlyConverter",
    "VersionMigration",
    "HFConverter",
    "register_converter",
    "get_converter",
    "list_converters",
    "compose_migration_chain",
    "parse_arch_version",
    "detect_model_type",
    "detect_model_type_from_hf",
    "detect_model_type_from_forgather",
    "discover_and_register_converters",
    "discovery",
    "STANDARD_FORGATHER_TO_HF",
    "STANDARD_HF_TO_FORGATHER",
    "reverse_mapping",
]
