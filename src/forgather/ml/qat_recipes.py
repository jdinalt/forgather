"""Shared QAT recipe registry for the trainer (prepare) and finalize (convert).

The same recipe string is supplied at training time via ``qat_recipe`` (which
inserts ``FakeQuantizedLinear`` modules) and at finalize time via
``--qat-convert`` (which swaps them for real low-bit quantized linear ops).
Both call sites resolve the string through :func:`recipe_to_base_config`.
"""

from __future__ import annotations


QAT_RECIPES: tuple[str, ...] = (
    "int8-dynamic-act-int4-weight",
    "int4-weight-only",
    "float8-dynamic-act-float8-weight",
    "float8-dynamic-act-int4-weight",
)


def recipe_to_base_config(recipe: str):
    """Map a Forgather QAT recipe string to a torchao base config instance.

    The returned object is the ``base_config`` argument for
    ``torchao.quantization.qat.QATConfig(base_config, step=...)``. It must be
    the *same* config (same parameters) for both the prepare and convert
    phases.
    """
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Float8DynamicActivationInt4WeightConfig,
        Int4WeightOnlyConfig,
        Int8DynamicActivationInt4WeightConfig,
    )

    if recipe == "int8-dynamic-act-int4-weight":
        return Int8DynamicActivationInt4WeightConfig(group_size=32)
    if recipe == "int4-weight-only":
        return Int4WeightOnlyConfig(group_size=128)
    if recipe == "float8-dynamic-act-float8-weight":
        return Float8DynamicActivationFloat8WeightConfig()
    if recipe == "float8-dynamic-act-int4-weight":
        return Float8DynamicActivationInt4WeightConfig()
    raise ValueError(
        f"Unknown QAT recipe: {recipe!r}. Valid recipes: {QAT_RECIPES}"
    )
