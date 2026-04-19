"""Gemma-3 model converter for HuggingFace <-> Forgather conversion."""

import os
from typing import Any, Dict, List, Tuple, override

from transformers.models.gemma3 import Gemma3ForCausalLM, Gemma3TextConfig

from forgather import MetaConfig
from forgather.ml.model_conversion import HFConverter, register_converter

from . import config_mappings, hf_mappings


@register_converter("gemma3_text")
class Gemma3Converter(HFConverter):
    """Converter for Google Gemma-3 (text) models.

    Registered under the HuggingFace model type string ``gemma3_text`` so that
    ``forgather convert`` can auto-detect it from a Gemma-3 config.json.
    """

    def __init__(self):
        super().__init__(model_type="gemma3_text")

    @override
    def get_hf_config_class(self):
        return Gemma3TextConfig

    @override
    def get_hf_model_class(self):
        return Gemma3ForCausalLM

    @override
    def get_parameter_mappings(self, direction: str) -> List[Tuple]:
        if direction == "to_forgather":
            return hf_mappings.HF_TO_FORGATHER
        elif direction == "from_forgather":
            return hf_mappings.FORGATHER_TO_HF
        else:
            raise ValueError(
                f"Invalid direction: {direction}. "
                "Must be 'to_forgather' or 'from_forgather'"
            )

    @override
    def get_config_field_mapping(self, direction: str) -> Dict[str, str]:
        if direction == "to_forgather":
            return config_mappings.HF_TO_FORGATHER
        elif direction == "from_forgather":
            return config_mappings.FORGATHER_TO_HF
        else:
            raise ValueError(
                f"Invalid direction: {direction}. "
                "Must be 'to_forgather' or 'from_forgather'"
            )

    @override
    def validate_source_config(self, config: Any, direction: str) -> None:
        if direction == "to_forgather":
            assert (
                config.model_type == "gemma3_text"
            ), f"Expected model_type 'gemma3_text', got '{config.model_type}'"
            hidden_activation = getattr(config, "hidden_activation", None)
            assert hidden_activation == "gelu_pytorch_tanh", (
                f"Expected hidden_activation 'gelu_pytorch_tanh', "
                f"got '{hidden_activation}'"
            )

    @override
    def get_project_info(self) -> dict[str, Any]:
        return dict(
            project_dir=MetaConfig.find_project_dir(os.path.abspath(__file__)),
            config_name="",
        )

    def create_hf_config(
        self, src_config: Any, max_length: int = None
    ) -> Gemma3TextConfig:
        """Create a HuggingFace Gemma3TextConfig from a Forgather config."""
        hf_config = super().create_hf_config(src_config, max_length)
        hf_config.model_type = "gemma3_text"
        return hf_config
