#!/usr/bin/env python3
"""
Unit tests for the forgather ML optim module components.

Tests cover:
- opt_utils: make_regex_optimizer_groups, make_grouped_optimizer
- rounding_utils: fp32_to_bf16_stochastic_round
- infinite_lr_scheduler: InfiniteLRScheduler (warmup, cooldown, constant, annealing)
- sequential_lr_factory: sequential_lr_factory
- subspace_proj: OnlinePCAProjector, RandProjector
- adamw: AdamW optimizer
- sgd: SGD optimizer
"""

import math
from functools import partial
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, StepLR

from forgather.ml.optim.opt_utils import (
    make_grouped_optimizer,
    make_regex_optimizer_groups,
)
from forgather.ml.optim.rounding_utils import fp32_to_bf16_stochastic_round
from forgather.ml.optim.infinite_lr_scheduler import InfiniteLRScheduler
from forgather.ml.optim.sequential_lr_factory import sequential_lr_factory
from forgather.ml.optim.subspace_proj import (
    OnlinePCAProjector,
    RandProjector,
    SubspaceProjector,
)
from forgather.ml.optim.adamw import AdamW
from forgather.ml.optim.sgd import SGD


# ---------------------------------------------------------------------------
# Helper: simple multi-layer model for optimizer grouping tests
# ---------------------------------------------------------------------------

class TwoLayerModel(nn.Module):
    """A simple model with distinct weight and bias parameters for grouping tests."""

    def __init__(self, in_features=8, hidden_features=16, out_features=4):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


# ===========================================================================
# Tests for opt_utils.py
# ===========================================================================

class TestMakeRegexOptimizerGroups:
    """Tests for make_regex_optimizer_groups."""

    def _make_model(self):
        return TwoLayerModel()

    def test_basic_weight_bias_grouping(self):
        """Parameters are correctly split into weight and bias groups."""
        model = self._make_model()
        group_map = [
            (r"weight", "weight_group"),
            (r"bias", "bias_group"),
        ]
        group_config = {
            "weight_group": {"lr": 1e-3, "weight_decay": 0.01},
            "bias_group": {"lr": 1e-4, "weight_decay": 0.0},
        }

        param_groups = make_regex_optimizer_groups(
            model.named_parameters(), group_map, group_config
        )

        assert len(param_groups) == 2

        # Weight group
        weight_group = param_groups[0]
        assert weight_group["lr"] == 1e-3
        assert weight_group["weight_decay"] == 0.01
        weight_params: list[Any] = weight_group["params"]  # type: ignore[assignment]
        weight_names = [name for name, _ in weight_params]
        assert all("weight" in n for n in weight_names)
        assert len(weight_names) == 2  # linear1.weight, linear2.weight

        # Bias group
        bias_group = param_groups[1]
        assert bias_group["lr"] == 1e-4
        assert bias_group["weight_decay"] == 0.0
        bias_params: list[Any] = bias_group["params"]  # type: ignore[assignment]
        bias_names = [name for name, _ in bias_params]
        assert all("bias" in n for n in bias_names)
        assert len(bias_names) == 2  # linear1.bias, linear2.bias

    def test_first_match_wins(self):
        """A parameter matches the first regex that applies."""
        model = self._make_model()
        # "linear1.weight" contains both "linear1" and "weight". If "linear1"
        # comes first in group_map it should capture both weight and bias of linear1.
        group_map = [
            (r"linear1", "first_layer"),
            (r".*", "rest"),
        ]
        group_config = {
            "first_layer": {"lr": 1e-2},
            "rest": {"lr": 1e-3},
        }

        param_groups = make_regex_optimizer_groups(
            model.named_parameters(), group_map, group_config
        )

        first_params: list[Any] = param_groups[0]["params"]  # type: ignore[assignment]
        rest_params: list[Any] = param_groups[1]["params"]  # type: ignore[assignment]
        first_names = [n for n, _ in first_params]
        rest_names = [n for n, _ in rest_params]

        assert len(first_names) == 2  # linear1.weight, linear1.bias
        assert all("linear1" in n for n in first_names)
        assert len(rest_names) == 2  # linear2.weight, linear2.bias
        assert all("linear2" in n for n in rest_names)

    def test_unmatched_parameters_omitted(self):
        """Parameters that match no regex are excluded from all groups."""
        model = self._make_model()
        # Only match linear1 parameters
        group_map = [
            (r"linear1", "group_a"),
        ]
        group_config = {
            "group_a": {"lr": 1e-3},
        }

        param_groups = make_regex_optimizer_groups(
            model.named_parameters(), group_map, group_config
        )

        assert len(param_groups) == 1
        group_params: list[Any] = param_groups[0]["params"]  # type: ignore[assignment]
        names = [n for n, _ in group_params]
        assert len(names) == 2  # linear1.weight, linear1.bias
        assert all("linear1" in n for n in names)

    def test_empty_model(self):
        """An empty parameter list produces groups with empty param lists."""
        group_map = [
            (r"weight", "w"),
        ]
        group_config = {
            "w": {"lr": 0.1},
        }

        param_groups = make_regex_optimizer_groups([], group_map, group_config)
        assert len(param_groups) == 1
        empty_params: list[Any] = param_groups[0]["params"]  # type: ignore[assignment]
        assert len(empty_params) == 0

    def test_debug_flag_does_not_crash(self):
        """Setting debug=True should not raise."""
        model = self._make_model()
        group_map = [
            (r".*", "all"),
        ]
        group_config = {
            "all": {"lr": 0.1},
        }

        # Should not raise
        make_regex_optimizer_groups(
            model.named_parameters(), group_map, group_config, debug=True
        )

    def test_group_config_hyperparams_forwarded(self):
        """All hyperparameters from group_config appear in the returned dicts."""
        model = self._make_model()
        group_map = [
            (r".*", "all"),
        ]
        group_config = {
            "all": {"lr": 0.1, "weight_decay": 0.05, "betas": (0.9, 0.98), "eps": 1e-8},
        }

        param_groups = make_regex_optimizer_groups(
            model.named_parameters(), group_map, group_config
        )

        pg = param_groups[0]
        assert pg["lr"] == 0.1
        assert pg["weight_decay"] == 0.05
        assert pg["betas"] == (0.9, 0.98)
        assert pg["eps"] == 1e-8


class TestMakeGroupedOptimizer:
    """Tests for make_grouped_optimizer."""

    def test_creates_optimizer_with_groups(self):
        """make_grouped_optimizer returns a working optimizer instance."""
        model = TwoLayerModel()
        group_map = [
            (r"weight", "weights"),
            (r"bias", "biases"),
        ]
        group_config = {
            "weights": {"lr": 1e-3, "weight_decay": 0.01},
            "biases": {"lr": 1e-4, "weight_decay": 0.0},
        }

        optimizer = make_grouped_optimizer(
            model.named_parameters(),
            opt_ctor=torch.optim.SGD,
            group_map=group_map,
            group_config=group_config,
        )

        assert isinstance(optimizer, torch.optim.SGD)
        assert len(optimizer.param_groups) == 2

    def test_opt_kwargs_forwarded(self):
        """Extra keyword arguments are forwarded to the optimizer constructor."""
        model = TwoLayerModel()
        group_map = [
            (r".*", "all"),
        ]
        group_config = {
            "all": {"lr": 1e-3},
        }

        optimizer = make_grouped_optimizer(
            model.named_parameters(),
            opt_ctor=torch.optim.SGD,
            group_map=group_map,
            group_config=group_config,
            opt_kwargs={"momentum": 0.9},
        )

        assert isinstance(optimizer, torch.optim.SGD)
        # Momentum should be set in the param group defaults
        assert optimizer.param_groups[0]["momentum"] == 0.9

    def test_opt_args_forwarded(self):
        """Positional arguments beyond param_groups are forwarded."""
        model = TwoLayerModel()
        group_map = [(r".*", "all")]
        group_config = {"all": {}}

        # torch.optim.SGD requires lr; pass it as a positional arg via opt_args
        optimizer = make_grouped_optimizer(
            model.named_parameters(),
            opt_ctor=torch.optim.SGD,
            group_map=group_map,
            group_config=group_config,
            opt_args=[0.01],  # lr as positional argument
        )
        assert isinstance(optimizer, torch.optim.SGD)


# ===========================================================================
# Tests for rounding_utils.py
# ===========================================================================

class TestFp32ToBf16StochasticRound:
    """Tests for fp32_to_bf16_stochastic_round."""

    def test_output_dtype_is_bfloat16(self):
        """Returned tensor must be bfloat16."""
        x = torch.randn(64, dtype=torch.float32)
        result = fp32_to_bf16_stochastic_round(x)
        assert result.dtype == torch.bfloat16

    def test_output_shape_matches_input(self):
        """Output tensor shape must match the input."""
        for shape in [(10,), (4, 8), (2, 3, 5)]:
            x = torch.randn(shape, dtype=torch.float32)
            result = fp32_to_bf16_stochastic_round(x)
            assert result.shape == x.shape, f"Shape mismatch for input shape {shape}"

    def test_close_to_input_values(self):
        """Stochastic rounding should produce values within bf16 precision of input."""
        torch.manual_seed(0)
        x = torch.randn(1000, dtype=torch.float32)
        result = fp32_to_bf16_stochastic_round(x)

        # Compare in float32 space
        diff = (x - result.float()).abs()
        # bf16 has 7-8 bit mantissa, so max rounding error per value is bounded
        # by the ULP (unit in the last place) of bfloat16, roughly |x| * 2^-7
        # For standard normal values most are < 4, so max error ~ 0.03
        assert diff.max().item() < 0.1, (
            f"Maximum rounding error {diff.max().item()} exceeds tolerance"
        )

    def test_determinism_with_seeded_generator(self):
        """Using the same generator seed should produce identical results."""
        x = torch.randn(256, dtype=torch.float32)

        gen1 = torch.Generator()
        gen1.manual_seed(42)
        result1 = fp32_to_bf16_stochastic_round(x, generator=gen1)

        gen2 = torch.Generator()
        gen2.manual_seed(42)
        result2 = fp32_to_bf16_stochastic_round(x, generator=gen2)

        assert torch.equal(result1, result2), "Results should be identical with same seed"

    def test_different_seeds_differ(self):
        """Different generator seeds should (very likely) produce different results."""
        x = torch.randn(1024, dtype=torch.float32)

        gen1 = torch.Generator()
        gen1.manual_seed(0)
        result1 = fp32_to_bf16_stochastic_round(x, generator=gen1)

        gen2 = torch.Generator()
        gen2.manual_seed(999)
        result2 = fp32_to_bf16_stochastic_round(x, generator=gen2)

        # Not all values will differ (many fp32 values round exactly to bf16)
        # but with 1024 elements some should differ.
        assert not torch.equal(result1, result2), (
            "Results with different seeds should differ for a large enough tensor"
        )

    def test_zero_input(self):
        """Zero should round to zero."""
        x = torch.zeros(16, dtype=torch.float32)
        result = fp32_to_bf16_stochastic_round(x)
        assert torch.equal(result, torch.zeros(16, dtype=torch.bfloat16))

    def test_exact_bf16_values_unchanged(self):
        """Values that are exactly representable in bf16 should remain unchanged."""
        # Powers of two and small integers are exact in bf16
        x = torch.tensor([1.0, 2.0, -4.0, 0.5, 0.25], dtype=torch.float32)
        result = fp32_to_bf16_stochastic_round(x)
        expected = x.bfloat16()
        assert torch.equal(result, expected), (
            f"Exact bf16 values should be unchanged: got {result}, expected {expected}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_tensor(self):
        """Stochastic rounding should work on CUDA tensors."""
        x = torch.randn(128, dtype=torch.float32, device="cuda")
        gen = torch.Generator(device="cuda")
        gen.manual_seed(42)
        result = fp32_to_bf16_stochastic_round(x, generator=gen)
        assert result.dtype == torch.bfloat16
        assert result.device.type == "cuda"
        assert result.shape == x.shape

    def test_unbiased_rounding_mean(self):
        """Stochastic rounding should be unbiased: mean of many rounds should converge."""
        torch.manual_seed(123)
        # Use a value that is NOT exactly representable in bf16
        # 1.001953125 is exact in bf16; pick something between two bf16 values.
        # bf16 near 1.0 has step size 2^-7 = 0.0078125
        # 1.0 + 0.003 is between bf16 values 1.0 and 1.0078125
        val = 1.003
        x = torch.full((1,), val, dtype=torch.float32)

        results = []
        for _ in range(5000):
            r = fp32_to_bf16_stochastic_round(x)
            results.append(r.float().item())

        mean_rounded = sum(results) / len(results)
        # Mean should be close to the original value (unbiased)
        assert abs(mean_rounded - val) < 0.005, (
            f"Mean of stochastic rounding {mean_rounded} too far from true value {val}"
        )


# ===========================================================================
# Tests for infinite_lr_scheduler.py
# ===========================================================================

class TestInfiniteLRScheduler:
    """Tests for InfiniteLRScheduler with 4 phases: warmup, cooldown, constant, annealing."""

    def _make_optimizer(self, lr=1.0):
        """Create a simple optimizer with a single parameter group."""
        param = nn.Parameter(torch.randn(4))
        return torch.optim.SGD([param], lr=lr)

    # --- warmup phase ---

    def test_warmup_starts_at_zero(self):
        """At step 0, LR should be 0 during warmup phase."""
        opt = self._make_optimizer(lr=1.0)
        sched = InfiniteLRScheduler(opt, warmup_steps=10, cooldown_steps=0, constant_lr=0.5)

        # After __init__, last_epoch=0 and get_lr has been called once
        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(0.0), f"Expected LR=0 at warmup start, got {lr}"

    def test_warmup_linear_ramp(self):
        """LR should increase linearly during warmup."""
        opt = self._make_optimizer(lr=1.0)
        warmup_steps = 10
        sched = InfiniteLRScheduler(
            opt, warmup_steps=warmup_steps, cooldown_steps=0, constant_lr=0.5
        )

        lrs = [sched.get_last_lr()[0]]
        for _ in range(warmup_steps):
            sched.step()
            lrs.append(sched.get_last_lr()[0])

        # LR at step i = base_lr * i / warmup_steps
        for i in range(warmup_steps + 1):
            expected = 1.0 * i / warmup_steps
            assert lrs[i] == pytest.approx(expected, abs=1e-7), (
                f"Step {i}: expected LR={expected}, got {lrs[i]}"
            )

    def test_warmup_end_matches_base_lr(self):
        """At the last warmup step, LR should equal base_lr."""
        opt = self._make_optimizer(lr=0.5)
        warmup_steps = 20
        sched = InfiniteLRScheduler(
            opt, warmup_steps=warmup_steps, cooldown_steps=0, constant_lr=0.1
        )

        for _ in range(warmup_steps):
            sched.step()

        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(0.5, abs=1e-7), (
            f"LR at end of warmup should equal base_lr=0.5, got {lr}"
        )

    # --- cooldown phase ---

    def test_cooldown_starts_at_base_lr(self):
        """At the beginning of cooldown, LR should be close to base_lr."""
        opt = self._make_optimizer(lr=1.0)
        warmup_steps = 5
        cooldown_steps = 20
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            constant_lr=0.1,
        )

        # Advance through warmup
        for _ in range(warmup_steps):
            sched.step()

        lr = sched.get_last_lr()[0]
        # At step=warmup_steps, cosine term: cos(0) = 1, so lr = constant_lr + (base_lr - constant_lr)/2 * 2 = base_lr
        assert lr == pytest.approx(1.0, abs=1e-6), (
            f"LR at start of cooldown should be base_lr=1.0, got {lr}"
        )

    def test_cooldown_ends_at_constant_lr(self):
        """At the end of cooldown, LR should be close to constant_lr."""
        opt = self._make_optimizer(lr=1.0)
        warmup_steps = 5
        cooldown_steps = 20
        constant_lr = 0.2
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            constant_lr=constant_lr,
        )

        # Advance through warmup + cooldown
        for _ in range(warmup_steps + cooldown_steps):
            sched.step()

        lr = sched.get_last_lr()[0]
        # At the last cooldown step (step = warmup + cooldown - 1):
        # cosine arg = pi * (cooldown_steps - 1) / cooldown_steps ~ pi
        # So cos ~ -1, LR ~ constant_lr + 0 ~ constant_lr (approximately)
        assert lr == pytest.approx(constant_lr, abs=0.05), (
            f"LR at end of cooldown should approach constant_lr={constant_lr}, got {lr}"
        )

    def test_cooldown_cosine_shape(self):
        """Cooldown LR follows cosine decay from base_lr to constant_lr."""
        opt = self._make_optimizer(lr=2.0)
        warmup_steps = 0
        cooldown_steps = 100
        constant_lr = 0.5
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            constant_lr=constant_lr,
        )

        lrs = [sched.get_last_lr()[0]]
        for _ in range(cooldown_steps):
            sched.step()
            lrs.append(sched.get_last_lr()[0])

        # Verify a midpoint value: at step=cooldown_steps/2, cosine=cos(pi/2)=0
        mid = cooldown_steps // 2
        expected_mid = constant_lr + (2.0 - constant_lr) / 2.0 * (1.0 + 0.0)
        assert lrs[mid] == pytest.approx(expected_mid, abs=0.05), (
            f"Midpoint LR should be ~{expected_mid}, got {lrs[mid]}"
        )

    # --- constant phase ---

    def test_constant_phase_with_cooldown(self):
        """After cooldown, LR should remain at constant_lr."""
        opt = self._make_optimizer(lr=1.0)
        warmup_steps = 5
        cooldown_steps = 10
        constant_lr = 0.3
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            constant_lr=constant_lr,
        )

        # Advance past warmup + cooldown
        for _ in range(warmup_steps + cooldown_steps):
            sched.step()

        # Constant phase: LR should stay at constant_lr
        for step_i in range(20):
            sched.step()
            lr = sched.get_last_lr()[0]
            assert lr == pytest.approx(constant_lr, abs=1e-7), (
                f"Constant phase step {step_i}: expected {constant_lr}, got {lr}"
            )

    def test_constant_phase_no_cooldown(self):
        """With cooldown_steps=0, constant phase should use base_lr instead."""
        opt = self._make_optimizer(lr=0.7)
        warmup_steps = 5
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=warmup_steps,
            cooldown_steps=0,
            constant_lr=0.1,  # Should be ignored when cooldown_steps=0
        )

        # Advance past warmup
        for _ in range(warmup_steps):
            sched.step()

        # Constant phase with no cooldown: LR = base_lr
        for _ in range(20):
            sched.step()
            lr = sched.get_last_lr()[0]
            assert lr == pytest.approx(0.7, abs=1e-7), (
                f"Expected base_lr=0.7 in constant phase without cooldown, got {lr}"
            )

    # --- annealing phase ---

    def test_annealing_exponential_decay(self):
        """After checkpoint_step, LR should decay exponentially toward min_lr."""
        opt = self._make_optimizer(lr=1.0)
        warmup_steps = 5
        cooldown_steps = 5
        constant_lr = 0.5
        min_lr = 1e-6
        tau = 100.0
        checkpoint_step = 20
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            constant_lr=constant_lr,
            min_lr=min_lr,
            tau=tau,
            checkpoint_step=checkpoint_step,
        )

        # Advance to just past checkpoint_step
        for _ in range(checkpoint_step):
            sched.step()

        # Now in annealing phase
        prev_lr = sched.get_last_lr()[0]
        for i in range(1, 50):
            sched.step()
            lr = sched.get_last_lr()[0]
            # LR should be decreasing
            assert lr < prev_lr or lr == pytest.approx(min_lr, abs=1e-10), (
                f"Annealing step {i}: LR should decrease, prev={prev_lr}, curr={lr}"
            )
            # Verify formula: min_lr + (constant_lr - min_lr) * exp(-t / tau)
            t = i
            expected = min_lr + (constant_lr - min_lr) * math.exp(-t / tau)
            assert lr == pytest.approx(expected, abs=1e-7), (
                f"Annealing step {i}: expected {expected}, got {lr}"
            )
            prev_lr = lr

    def test_annealing_converges_to_min_lr(self):
        """After many annealing steps, LR should converge to min_lr."""
        opt = self._make_optimizer(lr=1.0)
        constant_lr = 0.5
        min_lr = 1e-5
        tau = 10.0
        checkpoint_step = 5
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=constant_lr,
            min_lr=min_lr,
            tau=tau,
            checkpoint_step=checkpoint_step,
        )

        # Advance well past the checkpoint
        for _ in range(checkpoint_step + 10000):
            sched.step()

        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(min_lr, abs=1e-5), (
            f"After many annealing steps, LR should be ~{min_lr}, got {lr}"
        )

    # --- no warmup / no cooldown ---

    def test_no_warmup_no_cooldown(self):
        """With warmup=0 and cooldown=0, scheduler goes straight to constant (base_lr)."""
        opt = self._make_optimizer(lr=0.3)
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=0,
            cooldown_steps=0,
            constant_lr=0.1,
        )

        # First LR should already be base_lr (no warmup, no cooldown, cooldown_steps==0 means constant=base_lr)
        lr = sched.get_last_lr()[0]
        assert lr == pytest.approx(0.3, abs=1e-7)

        for _ in range(10):
            sched.step()
            lr = sched.get_last_lr()[0]
            assert lr == pytest.approx(0.3, abs=1e-7)

    # --- state dict round-trip ---

    def test_state_dict_round_trip(self):
        """state_dict / load_state_dict preserves scheduler state across save/load."""
        opt = self._make_optimizer(lr=1.0)
        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=10,
            cooldown_steps=10,
            constant_lr=0.5,
            min_lr=1e-6,
            tau=50.0,
            checkpoint_step=30,
        )

        # Advance scheduler partway through
        for _ in range(15):
            sched.step()

        saved_state = sched.state_dict()
        lr_before = sched.get_last_lr()[0]

        # Create a new scheduler and load state
        opt2 = self._make_optimizer(lr=1.0)
        sched2 = InfiniteLRScheduler(
            opt2,
            warmup_steps=10,
            cooldown_steps=10,
            constant_lr=0.5,
            min_lr=1e-6,
            tau=50.0,
            checkpoint_step=30,
        )
        sched2.load_state_dict(saved_state)

        lr_after = sched2.get_last_lr()[0]
        assert lr_before == pytest.approx(lr_after, abs=1e-10), (
            f"LR mismatch after state_dict round-trip: {lr_before} vs {lr_after}"
        )

        # Step both and verify they remain in sync
        for _ in range(20):
            sched.step()
            sched2.step()
            assert sched.get_last_lr()[0] == pytest.approx(
                sched2.get_last_lr()[0], abs=1e-10
            )

    # --- multi-parameter-group ---

    def test_multiple_param_groups(self):
        """Scheduler should handle optimizers with multiple parameter groups."""
        p1 = nn.Parameter(torch.randn(4))
        p2 = nn.Parameter(torch.randn(4))
        opt = torch.optim.SGD([
            {"params": [p1], "lr": 1.0},
            {"params": [p2], "lr": 0.5},
        ])

        sched = InfiniteLRScheduler(
            opt,
            warmup_steps=10,
            cooldown_steps=0,
            constant_lr=0.1,
        )

        # At step 0 both LRs should be 0 (warmup start)
        lrs = sched.get_last_lr()
        assert lrs[0] == pytest.approx(0.0)
        assert lrs[1] == pytest.approx(0.0)

        # Advance to step 5
        for _ in range(5):
            sched.step()

        lrs = sched.get_last_lr()
        assert lrs[0] == pytest.approx(1.0 * 5 / 10, abs=1e-7)
        assert lrs[1] == pytest.approx(0.5 * 5 / 10, abs=1e-7)

    # --- validation ---

    def test_invalid_checkpoint_step(self):
        """checkpoint_step < warmup + cooldown should raise AssertionError."""
        opt = self._make_optimizer(lr=1.0)
        with pytest.raises(AssertionError):
            InfiniteLRScheduler(
                opt,
                warmup_steps=10,
                cooldown_steps=10,
                constant_lr=0.5,
                checkpoint_step=15,  # < 10 + 10 = 20
            )

    def test_negative_warmup_raises(self):
        """Negative warmup_steps should raise."""
        opt = self._make_optimizer(lr=1.0)
        with pytest.raises(AssertionError):
            InfiniteLRScheduler(opt, warmup_steps=-1, cooldown_steps=0, constant_lr=0.1)


# ===========================================================================
# Tests for sequential_lr_factory.py
# ===========================================================================

class TestSequentialLRFactory:
    """Tests for sequential_lr_factory."""

    def _make_optimizer(self, lr=0.1):
        param = nn.Parameter(torch.randn(4))
        return torch.optim.SGD([param], lr=lr)

    def test_returns_sequential_lr(self):
        """Factory should return a SequentialLR instance."""
        opt = self._make_optimizer(lr=0.1)

        scheduler_factories = [
            partial(StepLR, step_size=5, gamma=0.5),
            partial(StepLR, step_size=10, gamma=0.1),
        ]
        milestones = [5]

        result = sequential_lr_factory(opt, scheduler_factories, milestones)
        assert isinstance(result, SequentialLR)

    def test_optimizer_forwarded_to_each_scheduler(self):
        """Each scheduler factory receives the optimizer."""
        opt = self._make_optimizer(lr=0.1)

        received_optimizers = []

        def capturing_factory(optimizer, step_size=5):
            received_optimizers.append(optimizer)
            return StepLR(optimizer, step_size=step_size)

        factories = [
            partial(capturing_factory, step_size=5),
            partial(capturing_factory, step_size=10),
        ]
        milestones = [3]

        sequential_lr_factory(opt, factories, milestones)

        assert len(received_optimizers) == 2
        assert all(o is opt for o in received_optimizers)

    def test_sequential_transition(self):
        """LR should transition between schedulers at the milestone."""
        opt = self._make_optimizer(lr=1.0)

        # Phase 1: constant (StepLR with gamma=1.0 never decays)
        # Phase 2: StepLR that decays every step
        factories = [
            partial(StepLR, step_size=1, gamma=1.0),   # No decay
            partial(StepLR, step_size=1, gamma=0.5),    # Halve each step
        ]
        milestones = [5]

        sched = sequential_lr_factory(opt, factories, milestones)

        # Steps 0-4: constant at 1.0
        for step in range(5):
            sched.step()

        # After milestone, second scheduler kicks in
        lr_before_decay = opt.param_groups[0]["lr"]
        sched.step()
        lr_after_decay = opt.param_groups[0]["lr"]

        # Second scheduler (gamma=0.5) should start decaying
        assert lr_after_decay < lr_before_decay or lr_after_decay == pytest.approx(
            lr_before_decay * 0.5, abs=1e-7
        )

    def test_last_epoch_parameter(self):
        """last_epoch parameter should be forwarded to SequentialLR."""
        opt = self._make_optimizer(lr=0.1)
        factories = [
            partial(StepLR, step_size=5, gamma=0.5),
        ]
        milestones = []

        # Should not raise
        sched = sequential_lr_factory(opt, factories, milestones, last_epoch=-1)
        assert isinstance(sched, SequentialLR)


# ===========================================================================
# Tests for subspace_proj.py
# ===========================================================================

class TestSubspaceProjector:
    """Tests for the SubspaceProjector base class."""

    def test_invalid_proj_type(self):
        """Unknown proj_type should raise."""
        with pytest.raises(Exception, match="Unknow projection type"):
            SubspaceProjector(rank=4, dim=16, proj_type="invalid", update_steps=1)

    def test_left_projection_attributes(self):
        """Left projector should set correct dimensions and einsum strings."""
        proj = SubspaceProjector(rank=4, dim=16, proj_type="left", update_steps=1)
        assert proj.proj_shape == (16, 4)
        assert proj.dim == 16

    def test_right_projection_attributes(self):
        """Right projector should set correct dimensions and einsum strings."""
        proj = SubspaceProjector(rank=4, dim=16, proj_type="right", update_steps=1)
        assert proj.proj_shape == (4, 16)
        assert proj.dim == 16

    def test_scale_computation(self):
        """scale should be sqrt(dim) / sqrt(rank)."""
        proj = SubspaceProjector(rank=4, dim=16, proj_type="left", update_steps=1)
        expected_scale = math.sqrt(16) / math.sqrt(4)
        assert proj.scale == pytest.approx(expected_scale)


class TestOnlinePCAProjector:
    """Tests for OnlinePCAProjector."""

    def test_init_left(self):
        """Left PCA projector initializes correctly."""
        proj = OnlinePCAProjector(rank=4, dim=16, proj_type="left", update_steps=1)
        assert proj.rank == 4
        assert proj.dim == 16
        assert proj.A is None  # Lazy initialization

    def test_init_right(self):
        """Right PCA projector initializes correctly."""
        proj = OnlinePCAProjector(rank=4, dim=16, proj_type="right", update_steps=1)
        assert proj.rank == 4
        assert proj.dim == 16
        assert proj.A is None

    def test_down_projection_shape_left(self):
        """Left down projection: (out, in) -> (rank, in)."""
        rank, dim = 4, 16
        proj = OnlinePCAProjector(rank=rank, dim=dim, proj_type="left", update_steps=1)
        x = torch.randn(dim, 8)  # (dim, other)
        proj.step(x)
        result = proj.down(x)
        assert result.shape == (rank, 8)

    def test_up_projection_shape_left(self):
        """Left up projection: (rank, in) -> (dim, in)."""
        rank, dim = 4, 16
        proj = OnlinePCAProjector(rank=rank, dim=dim, proj_type="left", update_steps=1)
        x = torch.randn(dim, 8)
        proj.step(x)
        down = proj.down(x)
        result = proj.up(down)
        assert result.shape == x.shape

    def test_down_projection_shape_right(self):
        """Right down projection: (out, dim) -> (rank, out)."""
        rank, dim = 4, 16
        proj = OnlinePCAProjector(rank=rank, dim=dim, proj_type="right", update_steps=1)
        x = torch.randn(8, dim)  # (other, dim)
        proj.step(x)
        result = proj.down(x)
        assert result.shape == (rank, 8)

    def test_up_projection_shape_right(self):
        """Right up projection: (rank, out) -> (out, dim)."""
        rank, dim = 4, 16
        proj = OnlinePCAProjector(rank=rank, dim=dim, proj_type="right", update_steps=1)
        x = torch.randn(8, dim)
        proj.step(x)
        down = proj.down(x)
        result = proj.up(down)
        assert result.shape == x.shape

    def test_step_initializes_projection(self):
        """After the first step, the projection matrix A should be set."""
        proj = OnlinePCAProjector(rank=4, dim=16, proj_type="left", update_steps=1)
        assert proj.A is None
        x = torch.randn(16, 8)
        proj.step(x)
        assert proj.A is not None
        assert proj.A.shape == (16, 4)

    def test_step_update_frequency(self):
        """_update should only be called every update_steps."""
        rank, dim = 4, 16
        update_steps = 3
        proj = OnlinePCAProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=update_steps
        )
        x = torch.randn(dim, 8)

        # First step (step 0 % 3 == 0): initializes A
        proj.step(x)
        assert proj.A is not None
        A_after_first = proj.A.clone()

        # Steps 1, 2: no update expected
        proj.step(x)
        assert proj.A is not None
        assert torch.equal(proj.A, A_after_first)
        proj.step(x)
        assert proj.A is not None
        assert torch.equal(proj.A, A_after_first)

        # Step 3 (3 % 3 == 0): update expected
        proj.step(x)
        # After update, A may change (depends on fitting)
        # We just verify it does not crash
        assert proj.A is not None

    def test_reconstruction_error_decreases_with_rank(self):
        """Higher rank should yield lower reconstruction error."""
        torch.manual_seed(42)
        dim = 32
        x = torch.randn(dim, 64)

        errors = []
        for rank in [2, 8, 16]:
            proj = OnlinePCAProjector(
                rank=rank, dim=dim, proj_type="left", update_steps=1
            )
            proj.step(x)
            recon = proj.up(proj.down(x))
            error = (x - recon).square().mean().item()
            errors.append(error)

        # Errors should decrease as rank increases
        assert errors[0] > errors[1] > errors[2], (
            f"Reconstruction error should decrease with rank: {errors}"
        )

    def test_orthag_qr_left(self):
        """QR orthogonalization should produce orthonormal columns for left proj."""
        proj = OnlinePCAProjector(
            rank=4, dim=16, proj_type="left", update_steps=1, orthag="qr"
        )
        x = torch.randn(16, 8)
        proj.step(x)

        # A columns should be approximately orthonormal
        assert proj.A is not None
        AtA = proj.A.T @ proj.A
        eye = torch.eye(4)
        assert torch.allclose(AtA, eye, atol=1e-5), (
            "QR orthogonalized projection should have orthonormal columns"
        )

    def test_invalid_orthag_raises(self):
        """Unknown orthagonalization method should raise."""
        with pytest.raises(Exception, match="Unknow orthagonalization"):
            OnlinePCAProjector(
                rank=4, dim=16, proj_type="left", update_steps=1, orthag="bad"
            )


class TestRandProjector:
    """Tests for RandProjector."""

    def test_init_lazy(self):
        """Lazy RandProjector should not allocate A until needed."""
        proj = RandProjector(rank=4, dim=16, proj_type="left", update_steps=1, lazy=True)
        assert proj.A is None

    def test_init_not_lazy(self):
        """Non-lazy RandProjector should allocate A on first step."""
        proj = RandProjector(
            rank=4, dim=16, proj_type="left", update_steps=1, lazy=False
        )
        x = torch.randn(16, 8)
        proj.step(x)
        assert proj.A is not None
        assert proj.A.shape == (16, 4)

    def test_down_projection_shape_left(self):
        """Left down projection shape check for RandProjector."""
        rank, dim = 4, 16
        proj = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=1, lazy=True, seed=42
        )
        x = torch.randn(dim, 8)
        proj.step(x)
        result = proj.down(x)
        assert result.shape == (rank, 8)

    def test_down_projection_shape_right(self):
        """Right down projection shape check for RandProjector."""
        rank, dim = 4, 16
        proj = RandProjector(
            rank=rank, dim=dim, proj_type="right", update_steps=1, lazy=True, seed=42
        )
        x = torch.randn(8, dim)
        proj.step(x)
        result = proj.down(x)
        assert result.shape == (rank, 8)

    def test_up_projection_shape_left(self):
        """Left up projection shape check for RandProjector."""
        rank, dim = 4, 16
        proj = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=1, lazy=True, seed=42
        )
        x = torch.randn(dim, 8)
        proj.step(x)
        down = proj.down(x)
        result = proj.up(down)
        assert result.shape == x.shape

    def test_up_projection_shape_right(self):
        """Right up projection shape check for RandProjector."""
        rank, dim = 4, 16
        proj = RandProjector(
            rank=rank, dim=dim, proj_type="right", update_steps=1, lazy=True, seed=42
        )
        x = torch.randn(8, dim)
        proj.step(x)
        down = proj.down(x)
        result = proj.up(down)
        assert result.shape == x.shape

    def test_determinism_with_seed_lazy(self):
        """Two lazy RandProjectors with the same seed should produce identical projections."""
        rank, dim = 4, 16
        x = torch.randn(dim, 8)

        proj1 = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=1, lazy=True, seed=123
        )
        proj1.step(x)
        result1 = proj1.down(x)

        proj2 = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=1, lazy=True, seed=123
        )
        proj2.step(x)
        result2 = proj2.down(x)

        assert torch.allclose(result1, result2, atol=1e-6), (
            "Identically seeded RandProjectors should produce identical results"
        )

    def test_determinism_with_seed_not_lazy(self):
        """Two non-lazy RandProjectors with the same seed should produce identical matrices."""
        rank, dim = 4, 16
        x = torch.randn(dim, 8)

        proj1 = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=1, lazy=False, seed=99
        )
        proj1.step(x)

        proj2 = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=1, lazy=False, seed=99
        )
        proj2.step(x)

        assert proj1.A is not None
        assert proj2.A is not None
        assert torch.allclose(proj1.A, proj2.A, atol=1e-6)

    def test_different_seeds_differ(self):
        """Different seeds should produce different projections."""
        rank, dim = 4, 16
        x = torch.randn(dim, 8)

        proj1 = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=1, lazy=True, seed=1
        )
        proj1.step(x)
        result1 = proj1.down(x)

        proj2 = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=1, lazy=True, seed=2
        )
        proj2.step(x)
        result2 = proj2.down(x)

        assert not torch.allclose(result1, result2, atol=1e-6), (
            "Different seeds should produce different projections"
        )

    def test_orthogonal_init(self):
        """RandProjector with 'orthogonal' init should work."""
        rank, dim = 4, 16
        proj = RandProjector(
            rank=rank,
            dim=dim,
            proj_type="left",
            update_steps=1,
            lazy=False,
            init="orthogonal",
            seed=42,
        )
        x = torch.randn(dim, 8)
        proj.step(x)
        assert proj.A is not None
        assert proj.A.shape == (dim, rank)

    def test_invalid_init_raises(self):
        """Unknown init method should raise."""
        proj = RandProjector(
            rank=4, dim=16, proj_type="left", update_steps=1, lazy=False, init="bad"
        )
        x = torch.randn(16, 8)
        with pytest.raises(Exception):
            proj.step(x)

    def test_update_frequency(self):
        """Projection should only be regenerated every update_steps steps."""
        rank, dim = 4, 16
        proj = RandProjector(
            rank=rank, dim=dim, proj_type="left", update_steps=3, lazy=False, seed=42
        )
        x = torch.randn(dim, 8)

        # Step 0: initialize
        proj.step(x)
        assert proj.A is not None
        A_initial = proj.A.clone()

        # Steps 1, 2: no regeneration
        proj.step(x)
        assert proj.A is not None
        assert torch.equal(proj.A, A_initial)
        proj.step(x)
        assert proj.A is not None
        assert torch.equal(proj.A, A_initial)

        # Step 3: regenerate
        proj.step(x)
        # A should change because the generator state advanced
        # (or at minimum, _update was called)
        assert proj.A is not None


# ===========================================================================
# Tests for adamw.py
# ===========================================================================

class TestAdamW:
    """Tests for the custom AdamW optimizer."""

    def _make_simple_problem(self):
        """Create a simple linear regression problem."""
        torch.manual_seed(42)
        model = nn.Linear(8, 1, bias=False)
        x = torch.randn(32, 8)
        y = torch.randn(32, 1)
        return model, x, y

    def test_loss_decreases(self):
        """A few steps of AdamW should reduce loss on a simple problem."""
        model, x, y = self._make_simple_problem()
        optimizer = AdamW(model.parameters(), lr=1e-2)

        initial_loss = nn.functional.mse_loss(model(x), y).item()

        for _ in range(50):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        final_loss = nn.functional.mse_loss(model(x), y).item()
        assert final_loss < initial_loss, (
            f"Loss should decrease: initial={initial_loss}, final={final_loss}"
        )

    def test_weight_decay_applied(self):
        """With weight_decay > 0, weights should shrink compared to no weight decay."""
        # Create identical models and shared data
        torch.manual_seed(42)
        model_wd = nn.Linear(8, 1, bias=False)
        x = torch.randn(32, 8)
        y = torch.randn(32, 1)

        torch.manual_seed(42)
        model_no_wd = nn.Linear(8, 1, bias=False)

        # Verify models start identical
        assert torch.equal(model_wd.weight.data, model_no_wd.weight.data)

        opt_wd = AdamW(model_wd.parameters(), lr=1e-2, weight_decay=0.5)
        opt_no_wd = AdamW(model_no_wd.parameters(), lr=1e-2, weight_decay=0.0)

        for _ in range(100):
            opt_wd.zero_grad()
            loss_wd = nn.functional.mse_loss(model_wd(x), y)
            loss_wd.backward()
            opt_wd.step()

            opt_no_wd.zero_grad()
            loss_no_wd = nn.functional.mse_loss(model_no_wd(x), y)
            loss_no_wd.backward()
            opt_no_wd.step()

        norm_wd = model_wd.weight.data.norm().item()
        norm_no_wd = model_no_wd.weight.data.norm().item()
        assert norm_wd < norm_no_wd, (
            f"Weight decay should reduce weight norm: with_wd={norm_wd}, without_wd={norm_no_wd}"
        )

    def test_state_dict_structure(self):
        """state_dict should contain expected keys: step, m, v."""
        model, x, y = self._make_simple_problem()
        optimizer = AdamW(model.parameters(), lr=1e-2)

        # Take one step to initialize state
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

        state_dict = optimizer.state_dict()
        assert "state" in state_dict
        assert "param_groups" in state_dict

        # Each param should have step, m, v
        for param_id, param_state in state_dict["state"].items():
            assert "step" in param_state
            assert "m" in param_state
            assert "v" in param_state

    def test_state_dict_load_round_trip(self):
        """state_dict should survive save/load round-trip."""
        model, x, y = self._make_simple_problem()
        optimizer = AdamW(model.parameters(), lr=1e-2)

        for _ in range(5):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        state_dict = optimizer.state_dict()

        # Create a new optimizer and load state
        model2, _, _ = self._make_simple_problem()
        optimizer2 = AdamW(model2.parameters(), lr=1e-2)
        # Need to take a step first to init state, then overwrite
        optimizer2.zero_grad()
        loss2 = nn.functional.mse_loss(model2(x), y)
        loss2.backward()
        optimizer2.step()

        optimizer2.load_state_dict(state_dict)

        # Verify state was loaded
        loaded_state = optimizer2.state_dict()
        for pid in state_dict["state"]:
            orig_step = state_dict["state"][pid]["step"]
            loaded_step = loaded_state["state"][pid]["step"]
            assert torch.equal(orig_step, loaded_step)

    def test_state_dict_validation_missing_key(self):
        """state_dict with missing keys should raise ValueError."""
        model, x, y = self._make_simple_problem()
        optimizer = AdamW(model.parameters(), lr=1e-2)

        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

        state_dict = optimizer.state_dict()
        # Remove a required key
        for pid in state_dict["state"]:
            del state_dict["state"][pid]["m"]
            break

        with pytest.raises(ValueError, match="missing keys"):
            optimizer.load_state_dict(state_dict)

    def test_closure_returns_loss(self):
        """AdamW step should call the closure and return its value."""
        model, x, y = self._make_simple_problem()
        optimizer = AdamW(model.parameters(), lr=1e-2)

        # Compute gradients before calling step (outside no_grad context)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()

        # Closure just returns a pre-computed loss value
        loss_val = loss.detach()

        def closure():
            return loss_val

        result = optimizer.step(closure)
        assert result is not None
        assert result.item() > 0

    def test_no_grad_parameters_skipped(self):
        """Parameters without gradients should be skipped without error."""
        model = nn.Sequential(
            nn.Linear(8, 4),
            nn.Linear(4, 1),
        )
        optimizer = AdamW(model.parameters(), lr=1e-2)

        # Only compute grad for one layer
        x = torch.randn(4, 8)
        y = torch.randn(4, 1)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()

        # Manually set one param's grad to None
        model[0].weight.grad = None  # type: ignore[assignment]

        # Should not raise
        optimizer.step()

    def test_different_betas(self):
        """AdamW should accept custom beta values."""
        model, x, y = self._make_simple_problem()
        optimizer = AdamW(model.parameters(), lr=1e-2, betas=(0.8, 0.99))

        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

        # Verify betas in param_groups
        assert optimizer.param_groups[0]["betas"] == (0.8, 0.99)


# ===========================================================================
# Tests for sgd.py
# ===========================================================================

class TestSGD:
    """Tests for the custom SGD optimizer."""

    def _make_simple_problem(self):
        """Create a simple linear regression problem."""
        torch.manual_seed(42)
        model = nn.Linear(8, 1, bias=False)
        x = torch.randn(32, 8)
        y = torch.randn(32, 1)
        return model, x, y

    def test_loss_decreases(self):
        """A few steps of SGD should reduce loss on a simple problem."""
        model, x, y = self._make_simple_problem()
        optimizer = SGD(model.parameters(), lr=1e-2)

        initial_loss = nn.functional.mse_loss(model(x), y).item()

        for _ in range(100):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        final_loss = nn.functional.mse_loss(model(x), y).item()
        assert final_loss < initial_loss, (
            f"Loss should decrease: initial={initial_loss}, final={final_loss}"
        )

    def test_update_direction_matches_negative_gradient(self):
        """SGD update should be in the direction of negative gradient."""
        torch.manual_seed(42)
        model = nn.Linear(4, 1, bias=False)
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)

        lr = 0.01
        optimizer = SGD(model.parameters(), lr=lr)

        # Record weights before step
        w_before = model.weight.data.clone()

        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        assert model.weight.grad is not None
        grad = model.weight.grad.clone()

        optimizer.step()

        w_after = model.weight.data.clone()

        # Update = w_after - w_before should equal -lr * grad
        update = w_after - w_before
        expected_update = -lr * grad

        assert torch.allclose(update, expected_update, atol=1e-6), (
            f"SGD update should be -lr * grad.\n"
            f"Actual update: {update}\n"
            f"Expected: {expected_update}"
        )

    def test_closure_returns_loss(self):
        """SGD step should call the closure and return its value."""
        model, x, y = self._make_simple_problem()
        optimizer = SGD(model.parameters(), lr=1e-2)

        # Compute gradients before calling step (outside no_grad context)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()

        loss_val = loss.detach()

        def closure():
            return loss_val

        result = optimizer.step(closure)
        assert result is not None
        assert result.item() > 0

    def test_no_grad_parameters_skipped(self):
        """Parameters without gradients should be skipped."""
        layer0 = nn.Linear(4, 4)
        layer1 = nn.Linear(4, 1)
        model = nn.Sequential(layer0, layer1)
        optimizer = SGD(model.parameters(), lr=1e-2)

        x = torch.randn(4, 4)
        y = torch.randn(4, 1)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()

        # Remove grad from one parameter
        layer0.weight.grad = None  # type: ignore[assignment]

        # Record the parameter that has no grad
        w_before = layer0.weight.data.clone()

        optimizer.step()

        # Parameter without grad should be unchanged
        assert torch.equal(layer0.weight.data, w_before), (
            "Parameter without gradient should not be modified"
        )

    def test_multiple_param_groups(self):
        """SGD should handle multiple parameter groups with different LRs."""
        layer0 = nn.Linear(4, 4)
        layer1 = nn.Linear(4, 1)
        model = nn.Sequential(layer0, layer1)

        optimizer = SGD(
            [  # type: ignore[arg-type]
                {"params": layer0.parameters(), "lr": 0.1},
                {"params": layer1.parameters(), "lr": 0.01},
            ],
            lr=0.001,  # default LR
        )

        x = torch.randn(4, 4)
        y = torch.randn(4, 1)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()

        # Record weights before step
        w0_before = layer0.weight.data.clone()
        w1_before = layer1.weight.data.clone()

        optimizer.step()

        # Both should be updated
        assert not torch.equal(layer0.weight.data, w0_before)
        assert not torch.equal(layer1.weight.data, w1_before)

        # Verify correct LR was used
        assert optimizer.param_groups[0]["lr"] == 0.1
        assert optimizer.param_groups[1]["lr"] == 0.01

    def test_zero_lr_no_update(self):
        """With lr=0, parameters should not change."""
        torch.manual_seed(42)
        model = nn.Linear(4, 1, bias=False)
        optimizer = SGD(model.parameters(), lr=0.0)

        x = torch.randn(4, 4)
        y = torch.randn(4, 1)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()

        w_before = model.weight.data.clone()
        optimizer.step()

        assert torch.equal(model.weight.data, w_before), (
            "With lr=0, weights should remain unchanged"
        )

    def test_large_lr_large_update(self):
        """A larger LR should produce larger parameter updates."""
        torch.manual_seed(42)
        model_small = nn.Linear(4, 1, bias=False)
        torch.manual_seed(42)
        model_large = nn.Linear(4, 1, bias=False)

        x = torch.randn(4, 4)
        y = torch.randn(4, 1)

        opt_small = SGD(model_small.parameters(), lr=0.001)
        opt_large = SGD(model_large.parameters(), lr=0.1)

        # Same forward/backward
        opt_small.zero_grad()
        loss_s = nn.functional.mse_loss(model_small(x), y)
        loss_s.backward()
        w_before_small = model_small.weight.data.clone()
        opt_small.step()

        opt_large.zero_grad()
        loss_l = nn.functional.mse_loss(model_large(x), y)
        loss_l.backward()
        w_before_large = model_large.weight.data.clone()
        opt_large.step()

        update_small = (model_small.weight.data - w_before_small).norm().item()
        update_large = (model_large.weight.data - w_before_large).norm().item()

        assert update_large > update_small, (
            f"Larger LR should produce larger update: "
            f"small={update_small}, large={update_large}"
        )


# ===========================================================================
# CUDA-specific tests
# ===========================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestAdamWCUDA:
    """CUDA-specific tests for AdamW."""

    def test_loss_decreases_on_cuda(self):
        """AdamW should reduce loss when model is on CUDA."""
        torch.manual_seed(42)
        model = nn.Linear(8, 1, bias=False).cuda()
        x = torch.randn(32, 8, device="cuda")
        y = torch.randn(32, 1, device="cuda")

        optimizer = AdamW(model.parameters(), lr=1e-2)

        initial_loss = nn.functional.mse_loss(model(x), y).item()

        for _ in range(50):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        final_loss = nn.functional.mse_loss(model(x), y).item()
        assert final_loss < initial_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestSGDCUDA:
    """CUDA-specific tests for SGD."""

    def test_loss_decreases_on_cuda(self):
        """SGD should reduce loss when model is on CUDA."""
        torch.manual_seed(42)
        model = nn.Linear(8, 1, bias=False).cuda()
        x = torch.randn(32, 8, device="cuda")
        y = torch.randn(32, 1, device="cuda")

        optimizer = SGD(model.parameters(), lr=1e-2)

        initial_loss = nn.functional.mse_loss(model(x), y).item()

        for _ in range(100):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        final_loss = nn.functional.mse_loss(model(x), y).item()
        assert final_loss < initial_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestRoundingUtilsCUDA:
    """CUDA-specific stochastic rounding tests."""

    def test_stochastic_rounding_deterministic_on_cuda(self):
        """Seeded stochastic rounding on CUDA should be deterministic."""
        x = torch.randn(256, dtype=torch.float32, device="cuda")

        gen1 = torch.Generator(device="cuda")
        gen1.manual_seed(42)
        result1 = fp32_to_bf16_stochastic_round(x, generator=gen1)

        gen2 = torch.Generator(device="cuda")
        gen2.manual_seed(42)
        result2 = fp32_to_bf16_stochastic_round(x, generator=gen2)

        assert torch.equal(result1, result2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
