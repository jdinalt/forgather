"""Mistral model converter for HuggingFace <-> Forgather conversion."""

import os
from typing import List, Tuple, Dict, Any, Optional
from transformers.models.mistral import MistralConfig, MistralForCausalLM

from forgather.ml.model_conversion import HFConverter, register_converter
from forgather import MetaConfig
from . import hf_mappings, config_mappings


@register_converter("mistral")
class MistralConverter(HFConverter):
    """Converter for Mistral models between HuggingFace and Forgather formats."""

    def __init__(self):
        """Initialize Mistral converter."""
        # Find Forgather root directory
        forgather_root = MetaConfig.find_workspace_dir(os.path.abspath(__file__))
        model_project_dir = os.path.join(forgather_root, "examples/models/mistral")

        super().__init__(model_type="mistral", model_project_dir=model_project_dir)

    def get_hf_config_class(self):
        """Get HuggingFace Mistral config class."""
        return MistralConfig

    def get_hf_model_class(self):
        """Get HuggingFace Mistral model class."""
        return MistralForCausalLM

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

    def create_project_config(
        self, src_config: Any, max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create Forgather Project configuration from HuggingFace Mistral config.

        Args:
            src_config: HuggingFace Mistral configuration
            max_length: Optional max sequence length override

        Returns:
            Dictionary of parameters to pass to Project() constructor
        """
        # Determine max model length
        max_model_length = src_config.max_position_embeddings
        if max_length:
            max_model_length = max_length

        # Handle None -> 'null' for YAML config
        sliding_window = getattr(src_config, "sliding_window", None)
        if sliding_window is None:
            sliding_window = "null"

        return {
            "attention_dropout": getattr(src_config, "attention_dropout", 0.0),
            "max_model_length": max_model_length,
            "hidden_size": src_config.hidden_size,
            "num_attention_heads": src_config.num_attention_heads,
            "num_kv_heads": src_config.num_key_value_heads,
            "d_head": src_config.hidden_size // src_config.num_attention_heads,
            "num_hidden_layers": src_config.num_hidden_layers,
            "dim_feedforward": src_config.intermediate_size,
            "rope_theta": src_config.rope_theta,
            "tie_word_embeddings": getattr(src_config, "tie_word_embeddings", False),
            "sliding_window": sliding_window,
            "rms_norm_eps": src_config.rms_norm_eps,
        }

    def create_hf_config(
        self, src_config: Any, max_length: int = None
    ) -> MistralConfig:
        """Create HuggingFace Mistral config from Forgather config.

        Args:
            src_config: Forgather model configuration
            max_length: Optional max sequence length override

        Returns:
            MistralConfig instance
        """
        # Get base config from parent class
        hf_config = super().create_hf_config(src_config, max_length)

        # Add Mistral-specific fields
        # Default sliding window is 4096 for Mistral
        hf_config.sliding_window = getattr(src_config, "sliding_window", 4096)

        return hf_config
