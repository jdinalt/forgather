"""Unit tests for forgather.ml.quantization_detect.

Covers both detection paths (config.json hint, state_dict scan) and the
install helper. Tests use real torchao on tiny synthetic models so we
don't have to ship pickle fixtures; per-recipe cost is well under a
second on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from forgather.ml.qat_recipes import recipe_to_base_config
from forgather.ml.quantization_detect import (
    detect_torchao_quantization,
    install_torchao_quantization,
)


# Build a model -> install one recipe -> harvest its state_dict.
# Cached at module scope to avoid running quantize_ once per test.
_FIXTURES: dict[str, dict] = {}


def _state_dict_for(recipe: str) -> dict:
    if recipe in _FIXTURES:
        return _FIXTURES[recipe]
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256, bias=False),
        torch.nn.Linear(256, 128, bias=False),
    )
    install_torchao_quantization(model, recipe_to_base_config(recipe))
    _FIXTURES[recipe] = model.state_dict()
    return _FIXTURES[recipe]


@pytest.mark.parametrize(
    "recipe",
    [
        "int8-dynamic-act-int4-weight",
        "int4-weight-only",
        # float8-* skipped: requires SM >= 8.9 to actually convert.
    ],
)
def test_detect_from_state_dict(recipe: str):
    """State_dict scan recovers a base config matching the recipe."""
    sd = _state_dict_for(recipe)
    base = detect_torchao_quantization(state_dict=sd)
    assert base is not None, f"failed to detect {recipe}"
    expected_type = type(recipe_to_base_config(recipe))
    assert type(base) is expected_type, (
        f"expected {expected_type.__name__}, got {type(base).__name__}"
    )


@pytest.mark.parametrize(
    "recipe",
    [
        "int8-dynamic-act-int4-weight",
        "int4-weight-only",
        "float8-dynamic-act-float8-weight",
    ],
)
def test_detect_from_config_json(tmp_path: Path, recipe: str):
    """config.json with a torchao quantization_config block resolves to the right base config.

    Includes float8: TorchAoConfig.from_dict doesn't need hardware to
    deserialize — only the quantize_ install step requires SM >= 8.9.
    """
    from transformers import TorchAoConfig

    block = TorchAoConfig(quant_type=recipe_to_base_config(recipe)).to_dict()
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["dummy"], "quantization_config": block})
    )
    base = detect_torchao_quantization(model_dir=str(tmp_path))
    assert base is not None
    assert type(base) is type(recipe_to_base_config(recipe))


def test_detect_returns_none_on_bf16(tmp_path: Path):
    """Plain config.json without a quantization_config block returns None."""
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["dummy"]}))
    assert detect_torchao_quantization(model_dir=str(tmp_path)) is None


def test_detect_returns_none_on_missing_config(tmp_path: Path):
    """No config.json anywhere → None (don't crash)."""
    assert detect_torchao_quantization(model_dir=str(tmp_path)) is None


def test_detect_walks_up_from_checkpoint_subdir(tmp_path: Path):
    """When passed <root>/checkpoints/checkpoint-N, look for config.json at <root>."""
    from transformers import TorchAoConfig

    recipe = "int4-weight-only"
    block = TorchAoConfig(quant_type=recipe_to_base_config(recipe)).to_dict()
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["dummy"], "quantization_config": block})
    )
    ckpt = tmp_path / "checkpoints" / "checkpoint-100"
    ckpt.mkdir(parents=True)
    base = detect_torchao_quantization(model_dir=str(ckpt))
    assert base is not None
    assert type(base) is type(recipe_to_base_config(recipe))


def test_detect_from_plain_state_dict_returns_none():
    """state_dict of plain bf16 tensors → no detection."""
    sd = {"a.weight": torch.randn(4, 4), "b.weight": torch.randn(8, 8)}
    assert detect_torchao_quantization(state_dict=sd) is None


def test_detect_unknown_subclass_raises(monkeypatch):
    """A torchao-namespaced tensor subclass we don't know about → ValueError."""
    import torch

    # Synthesize a subclass that lives in the torchao namespace but isn't on
    # our reverse-lookup table.
    class _UnknownTorchaoTensor(torch.Tensor):
        pass

    _UnknownTorchaoTensor.__module__ = "torchao.fake.module"
    fake = _UnknownTorchaoTensor()
    with pytest.raises(ValueError, match="Re-finalize the model"):
        detect_torchao_quantization(state_dict={"x.weight": fake})


def test_install_swaps_linear_for_quantized():
    """After install, the model's Linear modules carry torchao quantized weights."""
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128, bias=False),
        torch.nn.Linear(128, 64, bias=False),
    )
    install_torchao_quantization(
        model, recipe_to_base_config("int8-dynamic-act-int4-weight")
    )
    # Either the module class or the weight class becomes a torchao subclass —
    # depending on the recipe, torchao may swap the module or just wrap the
    # weight tensor. Verify *something* downstream is torchao-flavored.
    has_torchao_artifact = False
    for m in model.modules():
        if type(m).__module__.startswith("torchao"):
            has_torchao_artifact = True
            break
        if isinstance(m, torch.nn.Linear) and type(m.weight).__module__.startswith(
            "torchao"
        ):
            has_torchao_artifact = True
            break
    assert has_torchao_artifact, "install did not produce a torchao tensor or module"
