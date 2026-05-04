"""Abstract base class for model converters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


@dataclass
class VersionMigration:
    """One step in an in-Forgather schema migration chain.

    A migration moves a saved Forgather model from arch_version V to V+1.
    Renames are applied to both the config dict and the state dict;
    structural reshapes go through ``transform_state_dict``.

    Attributes:
        description: Human-readable summary, surfaced in the audit log.
        migrate_config: Translates the (already partially migrated) config
            dict into the next-version dict. Field renames, default
            backfills, removals all happen here.
        param_subs: Recursive regex substitution list, in the format
            understood by ``forgather.ml.remap_params.remap_state_dict``.
            Used to rewrite parameter FQNs.
        transform_state_dict: Optional weight-level transform applied
            after ``param_subs``. Receives the (already-renamed) state
            dict and the (already-migrated) config dict, returns a new
            state dict. Use for shape changes, head-dim reshapes,
            permutations, etc.
    """

    description: str
    migrate_config: Callable[[Dict[str, Any]], Dict[str, Any]] = lambda cfg: cfg
    param_subs: Tuple = field(default_factory=tuple)
    transform_state_dict: Optional[
        Callable[[Dict[str, "torch.Tensor"], Dict[str, Any]], Dict[str, "torch.Tensor"]]
    ] = None


class ModelConverter(ABC):
    """Abstract base class for model format converters.

    Subclasses should implement model-specific conversion logic for
    transforming models between different formats (e.g., HuggingFace, Forgather).

    Forgather schema versioning
    ---------------------------
    Subclasses also declare an in-Forgather arch identity and a chain of
    version migrations that ``forgather update`` uses to bring an older
    saved model up to the current source layout:

        arch:
            Stable string identifier matching the converter registry key
            (e.g. ``"llama"``). Stamped into newly saved configs as
            ``forgather_arch``.
        arch_version:
            PEP 440 version string (e.g. ``"1.0"``, ``"2.3.1"``). The
            current Forgather schema version of this arch's model code.
            Bump the *major* component each time a non-backwards-
            compatible change to parameter FQNs or config fields lands;
            minor / patch bumps signal compatible changes that
            ``forgather update`` can carry across without an explicit
            migration entry.
        forgather_migrations:
            ``{source_major: VersionMigration}``. Each entry migrates
            *any* version with major component ``source_major`` to the
            next major. Within a major, all minor / patch versions are
            considered compatible — no migration required, since the
            generated code accepts the saved schema unchanged.
            Multi-major updates are composed by walking the chain over
            majors; missing entries surface a clear error to the user.
    """

    arch: str = ""
    arch_version: str = "1"
    forgather_migrations: Dict[int, VersionMigration] = {}

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


class ForgatherOnlyConverter(ModelConverter):
    """Base class for archs that have no HuggingFace equivalent.

    Forgather-only archs (e.g. ``deepone``, ``llama_canon``,
    ``singlehead``, ``dynamic_causal_transformer``) support
    ``forgather update`` (in-Forgather schema migrations) but not
    ``forgather convert`` (HF<->FG round-trip). Subclasses declare the
    minimum needed by the update path:

      - ``arch`` -- string registry key (must match the
        ``forgather_arch`` value stamped into saved configs).
      - ``arch_version`` -- PEP 440 string.
      - ``forgather_migrations`` -- ``{source_major: VersionMigration}``.
      - ``get_project_info()`` -- returns
        ``{"project_dir": ..., "config_name": ...}`` so the update
        tool can regenerate model code from current sources.

    The HF-side abstract methods are stubbed here:

      - ``get_parameter_mappings`` / ``get_config_field_mapping``
        return identity / empty defaults that the update path can
        still consult; ``get_config_field_mapping("from_forgather")``
        in particular returns an identity over the standard
        Forgather config fields so saved hyperparameters flow through
        ``Project()`` during code regen.
      - ``convert_to_forgather`` / ``convert_from_forgather`` raise
        ``NotImplementedError`` with a clear message.

    Subclasses with arch-specific config fields beyond the standard
    set (e.g. ``alpha``, ``trainable_alibi``) should override
    ``get_config_field_mapping`` to extend the identity map.
    """

    def get_parameter_mappings(self, direction: str) -> List[Tuple]:
        # No HF<->FG remap; in-FG updates carry their renames via
        # ``forgather_migrations[i].param_subs`` instead.
        return []

    def get_config_field_mapping(self, direction: str) -> Dict[str, str]:
        # Identity over the standard FG config fields. The update
        # tool reads these to know which saved-config fields to
        # forward into the regenerated project as kwargs.
        from .standard_mappings import STANDARD_FORGATHER_TO_HF

        return {k: k for k in STANDARD_FORGATHER_TO_HF.keys()}

    @abstractmethod
    def get_project_info(self) -> Dict[str, Any]:
        """Path to the model project + config the update tool drives."""
        pass

    def convert_to_forgather(
        self,
        src_model_path: str,
        dst_model_path: str,
        dtype: Optional[str] = None,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: arch {self.arch!r} has no HuggingFace "
            "equivalent; HF<->FG conversion is not supported. Use "
            "`forgather update` for in-Forgather schema migrations."
        )

    def convert_from_forgather(
        self,
        src_model_path: str,
        dst_model_path: str,
        dtype: Optional[str] = None,
        max_length: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: arch {self.arch!r} has no HuggingFace "
            "equivalent; HF<->FG conversion is not supported."
        )
