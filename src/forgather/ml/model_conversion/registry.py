"""Registry for model converters."""

from typing import Dict, Type, List, Optional
from .base import ModelConverter


# Global registry of model converters
_CONVERTER_REGISTRY: Dict[str, Type[ModelConverter]] = {}


def register_converter(model_type: str):
    """Decorator to register a model converter.

    Args:
        model_type: String identifier for the model type (e.g., "llama", "mistral")

    Example:
        @register_converter("llama")
        class LlamaConverter(HFConverter):
            ...
    """

    def decorator(cls: Type[ModelConverter]):
        if model_type in _CONVERTER_REGISTRY:
            raise ValueError(
                f"Converter for model type '{model_type}' already registered"
            )
        _CONVERTER_REGISTRY[model_type] = cls
        return cls

    return decorator


def get_converter(model_type: str) -> Type[ModelConverter]:
    """Get converter class for the specified model type.

    Args:
        model_type: String identifier for the model type

    Returns:
        Converter class for the model type

    Raises:
        ValueError: If no converter is registered for the model type
    """
    if model_type not in _CONVERTER_REGISTRY:
        raise ValueError(
            f"No converter registered for model type '{model_type}'. "
            f"Available types: {list(_CONVERTER_REGISTRY.keys())}"
        )
    return _CONVERTER_REGISTRY[model_type]


def list_converters() -> List[str]:
    """List all registered model types.

    Returns:
        List of registered model type strings
    """
    return list(_CONVERTER_REGISTRY.keys())


def detect_model_type_from_hf(model_path: str) -> str:
    """Detect model type from HuggingFace model directory.

    Args:
        model_path: Path to HuggingFace model directory

    Returns:
        Model type string (e.g., "llama", "mistral")

    Raises:
        ValueError: If model type cannot be detected or is unsupported
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path)
    model_type = config.model_type

    if model_type not in _CONVERTER_REGISTRY:
        raise ValueError(
            f"Detected model type '{model_type}' is not supported. "
            f"Supported types: {list(_CONVERTER_REGISTRY.keys())}"
        )

    return model_type


def detect_model_type(model_path: str) -> Optional[tuple[str, str]]:
    """Detect model source and type from model directory.

    Args:
        model_path: Path to model directory (HF or Forgather)

    Returns:
        Tuple of (source, model_type) where:
        - source: "forgather" if FG model, "huggingface" if HF model
        - model_type: HF model type string (e.g., "llama", "mistral", "qwen3")
        Returns None if detection fails.

    Note:
        Forgather models are identified by the presence of 'hf_model_type' field,
        which is stored during HF->FG conversion. HuggingFace models have 'model_type'
        but not 'hf_model_type'.
    """
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Check for Forgather model (has hf_model_type metadata)
        if hasattr(config, "hf_model_type"):
            return ("forgather", config.hf_model_type)

        # Check for HuggingFace model (has model_type but not hf_model_type)
        if hasattr(config, "model_type"):
            return ("huggingface", config.model_type)

        # Fallback: Check for forgather-specific metadata
        if hasattr(config, "forgather_model_type"):
            return ("forgather", config.forgather_model_type)
    except Exception:
        pass

    return None


def detect_model_type_from_forgather(model_path: str) -> Optional[str]:
    """Detect model type from Forgather model directory.

    DEPRECATED: Use detect_model_type() instead, which returns both source and type.

    Args:
        model_path: Path to Forgather model directory

    Returns:
        Model type string if detectable, None otherwise
    """
    result = detect_model_type(model_path)
    if result and result[0] == "forgather":
        return result[1]
    return None
