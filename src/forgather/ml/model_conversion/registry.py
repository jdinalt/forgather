"""Registry for model converters."""

from typing import Dict, List, Optional, Tuple, Type, Union

from packaging.version import InvalidVersion, Version

from .base import ModelConverter, VersionMigration

# Import discovery functions for convenience
# Actual implementation is in discovery.py to avoid circular imports
_discovery_functions = None


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


def parse_arch_version(value: Union[str, int, float, Version]) -> Version:
    """Parse an arch version into a :class:`packaging.version.Version`.

    Accepts:
      - ``Version`` instances (returned as-is).
      - PEP 440 strings (e.g. ``"1.0"``, ``"2.3.1"``).
      - Bare integers (legacy: pre-PEP-440 saved configs stamped a plain
        int such as ``1``); coerced to the equivalent ``Version("1")``.
      - Floats (defensive: in case YAML/JSON loaders inferred a number);
        coerced via ``str(value)``.

    Raises:
        ValueError: when ``value`` is neither a known type nor a valid
            PEP 440 version string.
    """
    if isinstance(value, Version):
        return value
    if isinstance(value, bool):
        # bool is a subclass of int — reject before the int branch so
        # ``True`` / ``False`` don't silently become ``Version("1")`` /
        # ``Version("0")``.
        raise ValueError(f"arch_version must not be a bool; got {value!r}")
    if isinstance(value, (int, float)):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError(
            f"arch_version must be str / int / Version, got {type(value).__name__}"
        )
    try:
        return Version(value)
    except InvalidVersion as e:
        raise ValueError(
            f"arch_version {value!r} is not a valid PEP 440 version: {e}"
        ) from e


def compose_migration_chain(
    converter: ModelConverter,
    from_version: Union[str, int, Version],
    to_version: Union[str, int, Version],
) -> List[Tuple[int, VersionMigration]]:
    """Resolve the ordered list of in-Forgather migrations from version
    ``from_version`` to ``to_version``.

    Versions follow PEP 440. Only the **major** component drives
    migrations — minor and patch bumps within a major are considered
    backwards-compatible and require no explicit migration entry.

    Returns a list of ``(source_major, VersionMigration)`` pairs to
    apply in order. When the source and target share the same major
    component an empty list is returned, regardless of how the minor /
    patch components compare. This is what makes ``forgather update``
    a no-op (apart from re-stamping the target version) for compatible
    upgrades.

    Raises:
        ValueError: when ``to_version`` is older than ``from_version``
            (downgrades are not supported), or when the converter has
            no migration registered for some intermediate major.
    """
    from_v = parse_arch_version(from_version)
    to_v = parse_arch_version(to_version)
    if to_v < from_v:
        raise ValueError(
            f"Cannot migrate {converter.arch or converter.model_type} "
            f"backwards: from_version={from_v} > to_version={to_v}. "
            "Forgather only supports forward schema migrations."
        )

    chain: List[Tuple[int, VersionMigration]] = []
    missing: List[int] = []
    # Walk majors from the source to (but not including) the target.
    # Each migration step bridges major M -> M+1; same-major upgrades
    # produce an empty chain.
    for major in range(from_v.major, to_v.major):
        step = converter.forgather_migrations.get(major)
        if step is None:
            missing.append(major)
        else:
            chain.append((major, step))
    if missing:
        arch_label = converter.arch or converter.model_type
        formatted = ", ".join(f"{m}.x->{m + 1}.0" for m in missing)
        raise ValueError(
            f"Converter for arch '{arch_label}' is missing forgather_migrations "
            f"entries for major step(s) {formatted}. Add migrations to bridge "
            f"version {from_v} to {to_v}, or pass --to-version to stop at a "
            f"version the converter can reach."
        )
    return chain


def discover_and_register_converters(
    custom_paths: Optional[List[str]] = None, forgather_root: Optional[str] = None
) -> None:
    """Discover and register model converters from standard and custom locations.

    This is a convenience function that wraps the discovery module functions.
    It will:
    1. Discover builtin converters from examples/models/*/src/converter.py
    2. Discover converters from custom paths if provided

    Args:
        custom_paths: Optional list of directory paths to search for converters
        forgather_root: Optional path to Forgather root directory

    Note:
        Converters are registered automatically when their modules are imported,
        so this function only needs to be called once at startup.
    """
    global _discovery_functions

    # Lazy import to avoid circular dependencies
    if _discovery_functions is None:
        from . import discovery as _discovery_functions

    if custom_paths:
        _discovery_functions.discover_from_paths(custom_paths, forgather_root)
    else:
        _discovery_functions.discover_builtin_converters(forgather_root)
