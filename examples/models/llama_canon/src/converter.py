"""Llama Canon model converter (Forgather-only).

Llama Canon is a Llama variant that adds Triton-fused Canon layers
inside the transformer block. The state_dict layout differs from
stock Llama (extra Canon parameters), so the standard ``llama``
converter cannot migrate Canon models — they get their own arch
identifier and converter here.

There is no HuggingFace counterpart for Canon; ``forgather convert``
is unsupported. ``forgather update`` works through the migration
chain registered here.
"""

import os
from typing import Any, Dict, override

from forgather import MetaConfig
from forgather.ml.model_conversion import ForgatherOnlyConverter, register_converter


@register_converter("llama_canon")
class LlamaCanonConverter(ForgatherOnlyConverter):
    """Forgather-only converter for Llama Canon models."""

    arch = "llama_canon"
    arch_version = "1"
    forgather_migrations: dict = {}

    def __init__(self):
        super().__init__(model_type="llama_canon")

    @override
    def get_project_info(self) -> Dict[str, Any]:
        return dict(
            project_dir=MetaConfig.find_project_dir(os.path.abspath(__file__)),
            config_name="",
        )
