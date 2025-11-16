"""Llama model converter for HuggingFace <-> Forgather conversion."""

import os
from typing import List, Tuple, Dict, Any
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

from forgather.ml.model_conversion import HFConverter, register_converter
from forgather import MetaConfig
from . import hf_mappings, config_mappings


@register_converter("llama")
class LlamaConverter(HFConverter):
    """Converter for Llama models between HuggingFace and Forgather formats."""

    def __init__(self):
        """Initialize Llama converter."""
        # Find Forgather root directory
        forgather_root = MetaConfig.find_workspace_dir(os.path.abspath(__file__))
        model_project_dir = os.path.join(forgather_root, "examples/models/llama")

        super().__init__(model_type="llama", model_project_dir=model_project_dir)

    def get_hf_config_class(self):
        """Get HuggingFace Llama config class."""
        return LlamaConfig

    def get_hf_model_class(self):
        """Get HuggingFace Llama model class."""
        return LlamaForCausalLM

    def get_parameter_mappings(self, direction: str) -> List[Tuple]:
        """Get parameter name mapping rules for Llama models.

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
        """Get configuration field mappings for Llama models.

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
        """Validate source Llama model configuration.

        Args:
            config: Source model configuration
            direction: Either "to_forgather" or "from_forgather"

        Raises:
            AssertionError if configuration is invalid
        """
        if direction == "to_forgather":
            # Validating HuggingFace Llama config
            assert (
                config.model_type == "llama"
            ), f"Expected model_type 'llama', got '{config.model_type}'"
            assert (
                config.hidden_act == "silu"
            ), f"Expected hidden_act 'silu', got '{config.hidden_act}'"
            assert config.mlp_bias == False, "mlp_bias must be False"
            assert config.attention_bias == False, "attention_bias must be False"

            # Validate rope_scaling if present
            if config.rope_scaling is not None:
                rope_type = config.rope_scaling.get(
                    "rope_type", config.rope_scaling.get("type")
                )
                assert (
                    rope_type == "llama3"
                ), f"Unsupported rope_scaling type: {rope_type}. Only 'llama3' is supported."

    def create_hf_config(self, src_config: Any, max_length: int = None) -> LlamaConfig:
        """Create HuggingFace Llama config from Forgather config.

        Args:
            src_config: Forgather model configuration
            max_length: Optional max sequence length override

        Returns:
            LlamaConfig instance
        """
        # Get base config from parent class
        hf_config = super().create_hf_config(src_config, max_length)

        # Add Llama-specific fields
        hf_config.mlp_bias = False
        hf_config.attention_bias = False

        # Preserve rope_scaling if present
        if hasattr(src_config, "rope_scaling") and src_config.rope_scaling is not None:
            hf_config.rope_scaling = src_config.rope_scaling

        # Preserve tie_word_embeddings if present
        if hasattr(src_config, "tie_word_embeddings"):
            hf_config.tie_word_embeddings = src_config.tie_word_embeddings

        return hf_config
