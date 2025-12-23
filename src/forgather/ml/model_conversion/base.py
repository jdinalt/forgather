"""Abstract base class for model converters."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch


class ModelConverter(ABC):
    """Abstract base class for model format converters.

    Subclasses should implement model-specific conversion logic for
    transforming models between different formats (e.g., HuggingFace, Forgather).
    """

    def __init__(self, model_type: str):
        """Initialize converter.

        Args:
            model_type: String identifier for the model type (e.g., "llama", "mistral")
        """
        self.model_type = model_type

    @abstractmethod
    def get_parameter_mappings(self, direction: str) -> List[Tuple]:
        """Get parameter name mapping rules for the specified direction.

        Args:
            direction: Either "to_forgather" or "from_forgather"

        Returns:
            List of tuples representing recursive regex substitution patterns.
            Format: [(pattern, replacement, [children]), ...]
        """
        pass

    @abstractmethod
    def get_config_field_mapping(self, direction: str) -> Dict[str, str]:
        """Get configuration field mappings for the specified direction.

        Args:
            direction: Either "to_forgather" or "from_forgather"

        Returns:
            Dictionary mapping field names from source to destination format.
        """
        pass

    def transform_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        direction: str,
        src_config: Any,
        dst_config: Any,
    ) -> Dict[str, torch.Tensor]:
        """Apply model-specific transformations to state dict.

        This method can be overridden to apply custom weight transformations
        beyond simple name remapping (e.g., Q/K permutation for RoPE).

        Args:
            state_dict: State dictionary to transform
            direction: Either "to_forgather" or "from_forgather"
            src_config: Source model configuration
            dst_config: Destination model configuration

        Returns:
            Transformed state dictionary
        """
        # Default: no transformations beyond remapping
        return state_dict

    def validate_source_config(self, config: Any, direction: str) -> None:
        """Validate source model configuration.

        This method can be overridden to perform model-specific validation
        of the source model configuration before conversion.

        Args:
            config: Source model configuration
            direction: Either "to_forgather" or "from_forgather"

        Raises:
            AssertionError or ValueError if configuration is invalid
        """
        # Default: no validation
        pass

    @abstractmethod
    def convert_to_forgather(
        self,
        src_model_path: str,
        dst_model_path: str,
        dtype: Optional[str] = None,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Convert model from external format to Forgather format.

        Args:
            src_model_path: Path to source model directory
            dst_model_path: Path to destination model directory
            dtype: Optional dtype for output model
            max_length: Optional max sequence length override
            **kwargs: Additional conversion options
        """
        pass

    @abstractmethod
    def convert_from_forgather(
        self,
        src_model_path: str,
        dst_model_path: str,
        dtype: Optional[str] = None,
        max_length: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Convert model from Forgather format to external format.

        Args:
            src_model_path: Path to source Forgather model directory
            dst_model_path: Path to destination model directory
            dtype: Optional dtype for output model
            max_length: Optional max sequence length override
            checkpoint_path: Optional specific checkpoint to load
            **kwargs: Additional conversion options
        """
        pass
