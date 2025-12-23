"""Mistral model converter for HuggingFace <-> Forgather conversion."""

import os
from typing import Any, Dict, List, Optional, Tuple, override

from transformers.models.mistral import MistralConfig, MistralForCausalLM

from forgather import MetaConfig
from forgather.ml.model_conversion import HFConverter, register_converter

from . import config_mappings, hf_mappings


@register_converter("mistral")
class MistralConverter(HFConverter):
    """Converter for Mistral models between HuggingFace and Forgather formats."""

    def __init__(self):
        """Initialize Mistral converter."""
        super().__init__(model_type="mistral")

    @override
    def get_hf_config_class(self):
        """Get HuggingFace Mistral config class."""
        return MistralConfig

    @override
    def get_hf_model_class(self):
        """Get HuggingFace Mistral model class."""
        return MistralForCausalLM

    @override
    def get_parameter_mappings(self, direction: str) -> List[Tuple]:
        """Get parameter name mapping rules for Mistral models.

        Args:
            direction: Either "to_forgather" or "from_forgather"

        Returns:
            List of tuples representing recursive regex substitution patterns
        """
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
        """Get configuration field mappings for Mistral models.

        Args:
            direction: Either "to_forgather" or "from_forgather"

        Returns:
            Dictionary mapping field names from source to destination format
        """
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
        """Validate source Mistral model configuration.

        Args:
            config: Source model configuration
            direction: Either "to_forgather" or "from_forgather"

        Raises:
            AssertionError if configuration is invalid
        """
        if direction == "to_forgather":
            # Validating HuggingFace Mistral config
            assert (
                config.model_type == "mistral"
            ), f"Expected model_type 'mistral', got '{config.model_type}'"
            assert (
                config.hidden_act == "silu"
            ), f"Expected hidden_act 'silu', got '{config.hidden_act}'"
            # Mistral models have mlp_bias and attention_bias hardcoded to False

    @override
    def get_project_info(
        self,
    ) -> dict[str, Any]:
        return dict(
            project_dir=MetaConfig.find_project_dir(os.path.abspath(__file__)),
            config_name="",
        )

    @override
    def create_project_config(
        self, src_config: Any, max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        config = super().create_project_config(src_config, max_length)

        # vLLM raises an exception if sliding_window is None
        if "sliding_window" in config and config["sliding_window"] is None:
            del config["sliding_window"]
        return config
