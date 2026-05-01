"""Dynamic causal transformer converter (Forgather-only).

The vanilla post-LN causal transformer baseline (sinusoidal absolute
positional encoding, ReLU feedforward) defined in
``templatelib/examples/models/transformers/dynamic_causal_transformer.yaml``.

There is no HuggingFace counterpart — ``forgather convert`` is
unsupported. ``forgather update`` works through the migration chain
registered here.
"""

import os
from typing import Any, Dict, override

from forgather import MetaConfig
from forgather.ml.model_conversion import ForgatherOnlyConverter, register_converter


@register_converter("dynamic_causal_transformer")
class DynamicCausalTransformerConverter(ForgatherOnlyConverter):
    """Forgather-only converter for the baseline dynamic causal transformer."""

    arch = "dynamic_causal_transformer"
    arch_version = "1"
    forgather_migrations: dict = {}

    def __init__(self):
        super().__init__(model_type="dynamic_causal_transformer")

    @override
    def get_project_info(self) -> Dict[str, Any]:
        return dict(
            project_dir=MetaConfig.find_project_dir(os.path.abspath(__file__)),
            config_name="",
        )
