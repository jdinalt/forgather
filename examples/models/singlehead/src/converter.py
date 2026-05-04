"""Singlehead model converter (Forgather-only).

Singlehead is a research variant that replaces multi-head attention
with a single ALiBi attention head per layer. There is no HuggingFace
counterpart — ``forgather convert`` is unsupported. ``forgather update``
works through the migration chain registered here.
"""

import os
from typing import Any, Dict, override

from forgather import MetaConfig
from forgather.ml.model_conversion import ForgatherOnlyConverter, register_converter


@register_converter("singlehead")
class SingleheadConverter(ForgatherOnlyConverter):
    """Forgather-only converter for Singlehead models."""

    arch = "singlehead"
    arch_version = "1"
    forgather_migrations: dict = {}

    def __init__(self):
        super().__init__(model_type="singlehead")

    @override
    def get_project_info(self) -> Dict[str, Any]:
        return dict(
            project_dir=MetaConfig.find_project_dir(os.path.abspath(__file__)),
            config_name="",
        )
