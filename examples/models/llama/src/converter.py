"""Llama model converter for HuggingFace <-> Forgather conversion."""

import os
from typing import List, Tuple, Dict, Any, Optional
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

from forgather.ml.model_conversion import HFConverter, register_converter
from forgather import MetaConfig
from . import hf_mappings, config_mappings


@register_converter("llama")
class LlamaConverter(HFConverter):
    """Converter for Llama models between HuggingFace and Forgather formats."""

    def __init__(self):
        """Initialize Llama converter."""
        super().__init__(model_type="llama")

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

    def get_project_info(
        self,
    ) -> dict[str, Any]:
        return dict(
            project_dir=MetaConfig.find_project_dir(os.path.abspath(__file__)),
            config_name="",
        )
    
    def create_project_config(
        self, src_config: Any, max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create Forgather Project configuration from HuggingFace Llama config.

        Args:
            src_config: HuggingFace Llama configuration
            max_length: Optional max sequence length override

        Returns:
            Dictionary of parameters to pass to Project() constructor
        """
        # Determine max model length
        max_model_length = src_config.max_position_embeddings
        if max_length:
            max_model_length = max_length

        # Handle None -> 'null' for YAML config
        rope_scaling = getattr(src_config, "rope_scaling", None)
        if rope_scaling is None:
            rope_scaling = "null"

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
            "rope_scaling": rope_scaling,
            "tie_word_embeddings": getattr(src_config, "tie_word_embeddings", False),
            "rms_norm_eps": src_config.rms_norm_eps,
        }

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
