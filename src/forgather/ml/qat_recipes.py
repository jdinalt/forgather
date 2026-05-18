"""Shared QAT recipe registry for the trainer (prepare) and finalize (convert).

The same recipe string is supplied at training time via ``qat_recipe`` (which
inserts ``FakeQuantizedLinear`` modules) and at finalize time via
``--quantize`` (which swaps them for real low-bit quantized linear ops).
Both call sites resolve the string through :func:`recipe_to_base_config`.
``--quantize`` also accepts a plain bf16 source, in which case the same
pipeline performs post-training quantization (PTQ).
"""

from __future__ import annotations


# Source of truth for QAT recipe names. Consumed by:
#   - BaseTrainingArguments validator (src/forgather/ml/trainer/base_trainer.py)
#   - the Forgather finalize CLI (src/forgather/cli/finalize.py)
#   - finalize_model.py's --quantize (tools/finalize_model/finalize_model.py)
#   - the lm_training_project.yaml template's --qat-recipe `choices:` list,
#     rendered from this tuple via the `qat_recipes` Jinja global
#   - tools/forgather_server/webui/src/components/FinalizeModal.tsx
#     (TSX duplicate — keep the four strings here and there in sync)
QAT_RECIPES: tuple[str, ...] = (
    "int8-dynamic-act-int4-weight",
    "int4-weight-only",
    "float8-dynamic-act-float8-weight",
)


def recipe_to_base_config(recipe: str):
    """Map a Forgather QAT recipe string to a torchao base config instance.

    The returned object is the ``base_config`` argument for
    ``torchao.quantization.qat.QATConfig(base_config, step=...)``. It must be
    the *same* config (same parameters) for both the prepare and convert
    phases.
    """
    import torch
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Int4WeightOnlyConfig,
        Int8DynamicActivationIntxWeightConfig,
    )
    from torchao.quantization.granularity import PerGroup

    if recipe == "int8-dynamic-act-int4-weight":
        # Replaces the deprecated Int8DynamicActivationInt4WeightConfig
        # (see pytorch/ao#2752). Same semantics: int8 per-token dynamic
        # activations, int4 per-group symmetric weights.
        return Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(group_size=32),
        )
    if recipe == "int4-weight-only":
        return Int4WeightOnlyConfig(group_size=128)
    if recipe == "float8-dynamic-act-float8-weight":
        return Float8DynamicActivationFloat8WeightConfig()
    # `float8-dynamic-act-int4-weight` is intentionally not exposed in v1:
    # torchao's Float8DynamicActivationInt4WeightConfig requires the
    # `preshuffled` int4 packing format, which is Hopper-only (SM90+,
    # FBGEMM). When we add capability-gated recipe exposure, re-introduce
    # it behind a runtime check.
    raise ValueError(
        f"Unknown QAT recipe: {recipe!r}. Valid recipes: {QAT_RECIPES}"
    )
