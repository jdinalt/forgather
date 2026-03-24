"""Tests for FP8 training integration via torchao.

Tests are organized into:
- TestFP8TrainingArgs: Validation of fp8_recipe / fp8_dim_alignment fields (no GPU)
- TestFP8ModuleConversion: Module swap and filtering logic (GPU, SM >= 8.9)
- TestFP8TrainingLoop: End-to-end training steps (GPU, SM >= 8.9)
"""

import pytest
import torch
import torch.nn as nn

from forgather.ml.trainer import TrainingArguments
from forgather.ml.trainer.base_trainer import BaseTrainingArguments

requires_fp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="FP8 requires CUDA SM >= 8.9",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SmallMLP(nn.Module):
    """MLP with dims divisible by 16 for FP8 compatibility."""

    def __init__(self, d_in=64, d_hidden=128, d_out=64):
        super().__init__()
        self.up = nn.Linear(d_in, d_hidden)
        self.act = nn.ReLU()
        self.down = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.down(self.act(self.up(x)))


class MixedDimMLP(nn.Module):
    """MLP with one unaligned layer (15) and one aligned layer (64)."""

    def __init__(self):
        super().__init__()
        self.aligned = nn.Linear(64, 128)
        self.unaligned = nn.Linear(15, 32)

    def forward(self, x):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# TestFP8TrainingArgs -- no GPU needed
# ---------------------------------------------------------------------------


class TestFP8TrainingArgs:
    def test_default_none(self):
        args = BaseTrainingArguments(output_dir="/tmp/test")
        assert args.fp8_recipe is None

    def test_dim_alignment_default(self):
        args = BaseTrainingArguments(output_dir="/tmp/test")
        assert args.fp8_dim_alignment == 16

    def test_valid_recipes(self):
        for recipe in ("tensorwise", "rowwise", "rowwise_with_gw_hp"):
            args = BaseTrainingArguments(output_dir="/tmp/test", fp8_recipe=recipe)
            assert args.fp8_recipe == recipe

    def test_invalid_recipe_raises(self):
        with pytest.raises(ValueError, match="fp8_recipe must be one of"):
            BaseTrainingArguments(output_dir="/tmp/test", fp8_recipe="invalid")

    def test_fp8_with_bf16_amp_accepted(self):
        args = BaseTrainingArguments(
            output_dir="/tmp/test", fp8_recipe="rowwise", mixed_precision="bf16"
        )
        assert args.fp8_recipe == "rowwise"
        assert args.mixed_precision == "bf16"


# ---------------------------------------------------------------------------
# TestFP8ModuleConversion -- requires GPU with FP8 support
# ---------------------------------------------------------------------------


@requires_fp8
class TestFP8ModuleConversion:
    def _make_trainer_args(self, recipe="tensorwise", dim_alignment=16, **kwargs):
        return TrainingArguments(
            output_dir="/tmp/test_fp8",
            fp8_recipe=recipe,
            fp8_dim_alignment=dim_alignment,
            **kwargs,
        )

    def _apply(self, model, recipe="tensorwise", dim_alignment=16):
        """Apply FP8 conversion using the Trainer method directly."""
        from forgather.ml.trainer.trainer import Trainer

        args = self._make_trainer_args(recipe=recipe, dim_alignment=dim_alignment)
        # Create a minimal Trainer instance just to use _apply_fp8_training
        trainer = Trainer.__new__(Trainer)
        trainer.args = args
        return trainer._apply_fp8_training(model)

    def test_conversion_basic(self):
        from torchao.float8.float8_linear import Float8Linear

        model = SmallMLP().cuda().bfloat16()
        model = self._apply(model)
        fp8_count = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
        assert fp8_count == 2  # up and down

    def test_filter_skips_unaligned(self):
        from torchao.float8.float8_linear import Float8Linear

        model = MixedDimMLP().cuda().bfloat16()
        model = self._apply(model)
        fp8_count = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
        # Float8Linear is a subclass of nn.Linear, so count only plain nn.Linear
        plain_linear_count = sum(1 for m in model.modules() if type(m) is nn.Linear)
        assert fp8_count == 1  # only aligned (64->128)
        assert plain_linear_count == 1  # unaligned stays as nn.Linear

    def test_filter_disabled(self):
        from torchao.float8.float8_linear import Float8Linear

        model = MixedDimMLP().cuda().bfloat16()
        model = self._apply(model, dim_alignment=0)
        fp8_count = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
        assert fp8_count == 2  # both converted

    def test_state_dict_compatible(self):
        model = SmallMLP().cuda().bfloat16()
        original_keys = set(model.state_dict().keys())
        original_shapes = {k: v.shape for k, v in model.state_dict().items()}

        model = self._apply(model)
        converted_keys = set(model.state_dict().keys())
        converted_shapes = {k: v.shape for k, v in model.state_dict().items()}

        assert original_keys == converted_keys
        assert original_shapes == converted_shapes

    def test_checkpoint_roundtrip(self):
        """Save Float8Linear state_dict, load into plain Linear."""
        model_fp8 = SmallMLP().cuda().bfloat16()
        model_fp8 = self._apply(model_fp8)
        sd = model_fp8.state_dict()

        model_plain = SmallMLP().cuda().bfloat16()
        model_plain.load_state_dict(sd)

        for key in sd:
            assert torch.equal(sd[key], model_plain.state_dict()[key])

    def test_all_recipes(self):
        """Verify all three recipes can convert successfully."""
        from torchao.float8.float8_linear import Float8Linear

        for recipe in ("tensorwise", "rowwise", "rowwise_with_gw_hp"):
            model = SmallMLP().cuda().bfloat16()
            model = self._apply(model, recipe=recipe)
            fp8_count = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
            assert fp8_count == 2, f"Recipe {recipe} failed to convert"


# ---------------------------------------------------------------------------
# TestFP8TrainingLoop -- requires GPU with FP8 support
# ---------------------------------------------------------------------------


@requires_fp8
class TestFP8TrainingLoop:
    def _make_fp8_model(self, recipe="tensorwise"):
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        model = SmallMLP(d_in=64, d_hidden=128, d_out=64).cuda().bfloat16()
        config = Float8LinearConfig.from_recipe_name(recipe)
        convert_to_float8_training(model, config=config)
        return model

    def test_forward_backward(self):
        model = self._make_fp8_model()
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        y = model(x)
        loss = y.sum()
        loss.backward()

        assert y.shape == (32, 64)
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"

    def test_training_steps(self):
        model = self._make_fp8_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Verify no NaN
        for i, loss_val in enumerate(losses):
            assert not torch.isnan(torch.tensor(loss_val)), f"NaN loss at step {i}"

    def test_with_bf16_autocast(self):
        model = self._make_fp8_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for _ in range(3):
            optimizer.zero_grad()
            x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x).sum()
            loss.backward()
            optimizer.step()

    def test_with_torch_compile(self):
        model = self._make_fp8_model()
        model = torch.compile(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for _ in range(3):
            optimizer.zero_grad()
            x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()

    def test_all_recipes_train(self):
        """Verify all three recipes can complete a training step."""
        for recipe in ("tensorwise", "rowwise", "rowwise_with_gw_hp"):
            model = self._make_fp8_model(recipe=recipe)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            optimizer.zero_grad()
            x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
