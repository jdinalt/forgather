"""SmolLM3 model converter for HuggingFace <-> Forgather conversion."""

import os
from typing import Any, Dict, List, Tuple, override

from transformers.models.smollm3 import SmolLM3Config, SmolLM3ForCausalLM

from forgather import MetaConfig
from forgather.ml.model_conversion import HFConverter, register_converter

from . import config_mappings, hf_mappings


@register_converter("smollm3")
class SmolLM3Converter(HFConverter):
    """Converter for SmolLM3 models between HuggingFace and Forgather formats.

    SmolLM3 is a Llama-family architecture (GQA, SiLU GLU MLP, RMSNorm, tied
    embeddings) whose only structural difference is NoPE: every
    ``no_rope_layer_interval``-th layer omits rotary position embeddings.
    Weight names match Llama exactly, so the Llama parameter mappings are
    reused; the NoPE schedule is carried by the scalar ``no_rope_layer_interval``
    config field and regenerated into ``no_rope_layers`` by the template.
    """

    arch = "smollm3"
    arch_version = "1"
    forgather_migrations: dict = {}

    def __init__(self):
        """Initialize SmolLM3 converter."""
        super().__init__(model_type="smollm3")

    @override
    def get_hf_config_class(self):
        """Get HuggingFace SmolLM3 config class."""
        return SmolLM3Config

    @override
    def get_hf_model_class(self):
        """Get HuggingFace SmolLM3 model class."""
        return SmolLM3ForCausalLM

    @override
    def get_parameter_mappings(self, direction: str) -> List[Tuple]:
        """Get parameter name mapping rules for SmolLM3 models.

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
        """Get configuration field mappings for SmolLM3 models.

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
        """Validate source SmolLM3 model configuration.

        Args:
            config: Source model configuration
            direction: Either "to_forgather" or "from_forgather"

        Raises:
            AssertionError if configuration is invalid
        """
        if direction == "to_forgather":
            # Validating HuggingFace SmolLM3 config
            assert (
                config.model_type == "smollm3"
            ), f"Expected model_type 'smollm3', got '{config.model_type}'"
            assert (
                config.hidden_act == "silu"
            ), f"Expected hidden_act 'silu', got '{config.hidden_act}'"
            assert config.mlp_bias == False, "mlp_bias must be False"
            assert config.attention_bias == False, "attention_bias must be False"

    @override
    def get_project_info(
        self,
    ) -> dict[str, Any]:
        return dict(
            project_dir=MetaConfig.find_project_dir(os.path.abspath(__file__)),
            config_name="",
        )

    @override
    def create_hf_config(
        self, src_config: Any, max_length: int = None
    ) -> SmolLM3Config:
        """Create HuggingFace SmolLM3 config from Forgather config.

        Args:
            src_config: Forgather model configuration
            max_length: Optional max sequence length override

        Returns:
            SmolLM3Config instance
        """
        # Get base config from parent class
        hf_config = super().create_hf_config(src_config, max_length)

        # SmolLM3-specific fields (match the validated source-side invariants)
        hf_config.mlp_bias = False
        hf_config.attention_bias = False

        return hf_config
