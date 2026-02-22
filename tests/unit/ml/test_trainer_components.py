#!/usr/bin/env python3
"""
Unit tests for forgather ML trainer module components.

Tests the following components:
- PeriodicFunction: Step-based and epoch-based periodic triggering
- AMPContext: Automatic mixed precision context management
- base_trainer helper functions: logits_from_outputs, loss_from_outputs, loss_and_logits_from_outputs
- TrainerState, TrainerControl, IntervalStrategy: Trainer type dataclasses and enums
- JsonLogger: JSON logging callback
"""

import json
import os
import tempfile
from contextlib import nullcontext
from dataclasses import fields
from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import Tensor

from forgather.ml.trainer.amp import AMPContext
from forgather.ml.trainer.base_trainer import (
    logits_from_outputs,
    loss_and_logits_from_outputs,
    loss_from_outputs,
)
from forgather.ml.trainer.callbacks.default_callbacks import ProgressCallback
from forgather.ml.trainer.callbacks.json_logger import JsonLogger
from forgather.ml.trainer.periodic_function import PeriodicFunction
from forgather.ml.trainer.trainer_types import (
    IntervalStrategy,
    MinimalTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

# ---------------------------------------------------------------------------
# PeriodicFunction tests
# ---------------------------------------------------------------------------


class TestPeriodicFunctionNoStrategy:
    """Tests for PeriodicFunction with strategy='no' (disabled)."""

    def test_no_strategy_always_returns_false(self):
        """step() always returns False when strategy is 'no'."""
        pf = PeriodicFunction(global_step=0, strategy="no", period=1, epoch_period=1)
        for _ in range(20):
            assert pf.step() is False

    def test_no_strategy_sets_enabled_false(self):
        """The enabled flag is False when strategy is 'no'."""
        pf = PeriodicFunction(global_step=0, strategy="no", period=5, epoch_period=1)
        assert pf.enabled is False

    def test_no_strategy_still_increments_counters(self):
        """global_step and rel_step still increment even when disabled."""
        pf = PeriodicFunction(global_step=0, strategy="no", period=5, epoch_period=1)
        pf.step()
        pf.step()
        pf.step()
        assert pf.global_step == 3
        assert pf.rel_step == 3


class TestPeriodicFunctionStepsStrategy:
    """Tests for PeriodicFunction with strategy='steps'."""

    def test_triggers_at_exact_period(self):
        """step() returns True exactly when rel_step reaches the period."""
        period = 5
        pf = PeriodicFunction(
            global_step=0, strategy="steps", period=period, epoch_period=1
        )
        for i in range(1, period):
            assert pf.step() is False, f"Should not trigger at step {i}"
        assert pf.step() is True, f"Should trigger at step {period}"

    def test_triggers_every_period_without_reset(self):
        """Without reset(), step() triggers at period and then stays True
        for every subsequent step since rel_step >= period."""
        period = 3
        pf = PeriodicFunction(
            global_step=0, strategy="steps", period=period, epoch_period=1
        )
        results = [pf.step() for _ in range(10)]
        # Steps 1,2 => False, step 3 => True, steps 4+ => True (rel_step >= period)
        assert results == [False, False, True, True, True, True, True, True, True, True]

    def test_periodic_triggering_with_reset(self):
        """With reset() after each trigger, step() triggers exactly every period steps."""
        period = 3
        pf = PeriodicFunction(
            global_step=0, strategy="steps", period=period, epoch_period=1
        )
        triggers = []
        for _ in range(12):
            result = pf.step()
            triggers.append(result)
            if result:
                pf.reset()
        # Should trigger at steps 3, 6, 9, 12 (every 3 steps)
        assert triggers == [
            False,
            False,
            True,  # trigger at step 3
            False,
            False,
            True,  # trigger at step 6
            False,
            False,
            True,  # trigger at step 9
            False,
            False,
            True,  # trigger at step 12
        ]

    def test_period_of_one_triggers_every_step(self):
        """A period of 1 triggers on every step."""
        pf = PeriodicFunction(global_step=0, strategy="steps", period=1, epoch_period=1)
        for _ in range(10):
            assert pf.step() is True
            pf.reset()

    def test_global_step_increments_correctly(self):
        """global_step increments by 1 on each call to step()."""
        pf = PeriodicFunction(
            global_step=10, strategy="steps", period=5, epoch_period=1
        )
        assert pf.global_step == 10
        pf.step()
        assert pf.global_step == 11
        pf.step()
        assert pf.global_step == 12

    def test_rel_step_increments_and_resets(self):
        """rel_step increments on step() and resets to 0 on reset()."""
        pf = PeriodicFunction(global_step=0, strategy="steps", period=5, epoch_period=1)
        pf.step()
        pf.step()
        assert pf.rel_step == 2
        returned = pf.reset()
        assert returned == 2
        assert pf.rel_step == 0

    def test_reset_returns_rel_step_value(self):
        """reset() returns the current rel_step before resetting."""
        pf = PeriodicFunction(
            global_step=0, strategy="steps", period=10, epoch_period=1
        )
        for _ in range(7):
            pf.step()
        result = pf.reset()
        assert result == 7

    def test_reset_at_zero(self):
        """reset() returns 0 when no steps have been taken."""
        pf = PeriodicFunction(global_step=0, strategy="steps", period=5, epoch_period=1)
        assert pf.reset() == 0

    def test_enabled_is_true_for_steps(self):
        """enabled flag is True for steps strategy."""
        pf = PeriodicFunction(global_step=0, strategy="steps", period=5, epoch_period=1)
        assert pf.enabled is True


class TestPeriodicFunctionFirstStep:
    """Tests for PeriodicFunction with first_step parameter."""

    def test_first_step_delays_triggering(self):
        """step() does not trigger until global_step >= first_step."""
        pf = PeriodicFunction(
            global_step=0, strategy="steps", period=1, epoch_period=1, first_step=5
        )
        results = []
        for _ in range(7):
            results.append(pf.step())
        # global_step after each step: 1,2,3,4,5,6,7
        # first_step=5, so global_step < 5 for steps 1-4 => False
        # At step 5: global_step=5 >= first_step=5, rel_step=5 >= period=1 => True
        assert results[:4] == [False, False, False, False]
        assert results[4] is True

    def test_first_step_zero_means_no_delay(self):
        """first_step=0 (default) means triggering starts immediately."""
        pf = PeriodicFunction(
            global_step=0, strategy="steps", period=2, epoch_period=1, first_step=0
        )
        assert pf.step() is False  # rel_step=1, period=2
        assert pf.step() is True  # rel_step=2, period=2

    def test_first_step_with_nonzero_global_step(self):
        """first_step interacts correctly with a non-zero initial global_step."""
        pf = PeriodicFunction(
            global_step=3, strategy="steps", period=1, epoch_period=1, first_step=5
        )
        # global_step after step(): 4 (< 5) => False
        assert pf.step() is False
        # global_step after step(): 5 (>= 5) => True (rel_step=2 >= period=1)
        assert pf.step() is True


class TestPeriodicFunctionEpochStrategy:
    """Tests for PeriodicFunction with strategy='epoch'."""

    def test_epoch_strategy_uses_epoch_period(self):
        """epoch strategy uses epoch_period as the period value."""
        pf = PeriodicFunction(
            global_step=0, strategy="epoch", period=99, epoch_period=3
        )
        assert pf.period == 3

    def test_epoch_strategy_triggers_correctly(self):
        """epoch strategy triggers every epoch_period steps."""
        pf = PeriodicFunction(
            global_step=0, strategy="epoch", period=99, epoch_period=2
        )
        assert pf.step() is False  # rel_step=1
        assert pf.step() is True  # rel_step=2 >= period=2

    def test_epoch_strategy_enabled(self):
        """epoch strategy sets enabled=True."""
        pf = PeriodicFunction(
            global_step=0, strategy="epoch", period=99, epoch_period=5
        )
        assert pf.enabled is True


class TestPeriodicFunctionEdgeCases:
    """Edge cases and error handling for PeriodicFunction."""

    def test_unknown_strategy_raises_value_error(self):
        """An unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            PeriodicFunction(
                global_step=0, strategy="unknown", period=5, epoch_period=1
            )

    def test_zero_period_raises_assertion(self):
        """period=0 raises AssertionError."""
        with pytest.raises(AssertionError):
            PeriodicFunction(global_step=0, strategy="steps", period=0, epoch_period=1)

    def test_negative_period_raises_assertion(self):
        """Negative period raises AssertionError."""
        with pytest.raises(AssertionError):
            PeriodicFunction(global_step=0, strategy="steps", period=-1, epoch_period=1)

    def test_zero_epoch_period_raises_assertion(self):
        """epoch_period=0 raises AssertionError for epoch strategy."""
        with pytest.raises(AssertionError):
            PeriodicFunction(global_step=0, strategy="epoch", period=5, epoch_period=0)

    def test_str_representation(self):
        """__str__ returns a descriptive string."""
        pf = PeriodicFunction(
            global_step=0, strategy="steps", period=10, epoch_period=1
        )
        s = str(pf)
        assert "PeriodicFunction" in s
        assert "period=10" in s

    def test_step_accepts_extra_args_and_kwargs(self):
        """step() accepts and ignores extra positional and keyword arguments."""
        pf = PeriodicFunction(global_step=0, strategy="steps", period=1, epoch_period=1)
        # Should not raise
        result = pf.step("extra_arg", key="value")
        assert result is True

    def test_initial_state(self):
        """Check that initial state is set correctly."""
        pf = PeriodicFunction(
            global_step=42, strategy="steps", period=10, epoch_period=1, first_step=5
        )
        assert pf.global_step == 42
        assert pf.first_step == 5
        assert pf.rel_step == 0
        assert pf.enabled is True
        assert pf.period == 10


# ---------------------------------------------------------------------------
# AMPContext tests
# ---------------------------------------------------------------------------


class TestAMPContextDisabled:
    """Tests for AMPContext with mixed_precision=None (disabled)."""

    def test_disabled_state(self):
        """Disabled AMP has enabled=False, no scaler, no amp_dtype."""
        ctx = AMPContext(mixed_precision=None)
        assert ctx.enabled is False
        assert ctx.scaler is None
        assert ctx.amp_dtype is None

    def test_autocast_returns_nullcontext(self):
        """autocast() returns a nullcontext when AMP is disabled."""
        ctx = AMPContext(mixed_precision=None)
        ac = ctx.autocast()
        assert isinstance(ac, nullcontext)

    def test_scale_loss_returns_identity(self):
        """scale_loss() returns the loss unchanged when AMP is disabled."""
        ctx = AMPContext(mixed_precision=None)
        loss = torch.tensor(1.5)
        scaled = ctx.scale_loss(loss)
        assert scaled is loss

    def test_unscale_is_noop(self):
        """unscale_() does nothing when AMP is disabled."""
        ctx = AMPContext(mixed_precision=None)
        optimizer = MagicMock()
        ctx.unscale_(optimizer)
        # No exception, optimizer not called in any meaningful way

    def test_optimizer_step_calls_step_directly(self):
        """optimizer_step() calls optimizer.step() directly when AMP is disabled."""
        ctx = AMPContext(mixed_precision=None)
        optimizer = MagicMock()
        ctx.optimizer_step(optimizer)
        optimizer.step.assert_called_once()

    def test_state_dict_returns_empty(self):
        """state_dict() returns empty dict when AMP is disabled."""
        ctx = AMPContext(mixed_precision=None)
        assert ctx.state_dict() == {}

    def test_load_state_dict_is_noop(self):
        """load_state_dict() does nothing when AMP is disabled."""
        ctx = AMPContext(mixed_precision=None)
        ctx.load_state_dict({"some_key": "some_value"})
        # Should not raise


class TestAMPContextBf16:
    """Tests for AMPContext with mixed_precision='bf16'."""

    def test_bf16_state(self):
        """bf16 AMP has enabled=True, no scaler, amp_dtype=bfloat16."""
        ctx = AMPContext(mixed_precision="bf16")
        assert ctx.enabled is True
        assert ctx.scaler is None
        assert ctx.amp_dtype == torch.bfloat16

    def test_autocast_returns_torch_autocast(self):
        """autocast() returns a torch.autocast context manager for bf16."""
        ctx = AMPContext(mixed_precision="bf16", device_type="cpu")
        ac = ctx.autocast()
        # torch.autocast is returned, not nullcontext
        assert not isinstance(ac, nullcontext)

    def test_scale_loss_returns_identity(self):
        """scale_loss() returns the loss unchanged for bf16 (no scaler)."""
        ctx = AMPContext(mixed_precision="bf16")
        loss = torch.tensor(2.0)
        scaled = ctx.scale_loss(loss)
        assert scaled is loss

    def test_unscale_is_noop(self):
        """unscale_() does nothing for bf16 (no scaler)."""
        ctx = AMPContext(mixed_precision="bf16")
        optimizer = MagicMock()
        ctx.unscale_(optimizer)
        # Should not raise

    def test_optimizer_step_calls_step_directly(self):
        """optimizer_step() calls optimizer.step() directly for bf16."""
        ctx = AMPContext(mixed_precision="bf16")
        optimizer = MagicMock()
        ctx.optimizer_step(optimizer)
        optimizer.step.assert_called_once()

    def test_state_dict_returns_empty(self):
        """state_dict() returns empty dict for bf16."""
        ctx = AMPContext(mixed_precision="bf16")
        assert ctx.state_dict() == {}

    def test_load_state_dict_is_noop(self):
        """load_state_dict() does nothing for bf16."""
        ctx = AMPContext(mixed_precision="bf16")
        ctx.load_state_dict({})
        # Should not raise


class TestAMPContextFp16:
    """Tests for AMPContext with mixed_precision='fp16'."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fp16_state(self):
        """fp16 AMP has enabled=True, scaler present, amp_dtype=float16."""
        ctx = AMPContext(mixed_precision="fp16", device_type="cuda")
        assert ctx.enabled is True
        assert ctx.scaler is not None
        assert ctx.amp_dtype == torch.float16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_autocast_returns_torch_autocast(self):
        """autocast() returns a torch.autocast context manager for fp16."""
        ctx = AMPContext(mixed_precision="fp16", device_type="cuda")
        ac = ctx.autocast()
        assert not isinstance(ac, nullcontext)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_scale_loss_scales_via_scaler(self):
        """scale_loss() uses GradScaler to scale the loss for fp16."""
        ctx = AMPContext(mixed_precision="fp16", device_type="cuda")
        loss = torch.tensor(1.0, device="cuda")
        scaled = ctx.scale_loss(loss)
        # Scaled loss should be different from original (scale is 2^16 by default)
        assert scaled.item() != loss.item()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_state_dict_returns_scaler_state(self):
        """state_dict() returns GradScaler state for fp16."""
        ctx = AMPContext(mixed_precision="fp16", device_type="cuda")
        sd = ctx.state_dict()
        assert isinstance(sd, dict)
        assert len(sd) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_load_state_dict_restores_scaler(self):
        """load_state_dict() restores GradScaler state for fp16."""
        ctx = AMPContext(mixed_precision="fp16", device_type="cuda")
        original_state = ctx.state_dict()
        ctx2 = AMPContext(mixed_precision="fp16", device_type="cuda")
        ctx2.load_state_dict(original_state)
        restored_state = ctx2.state_dict()
        assert original_state["scale"] == restored_state["scale"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_custom_initial_scale(self):
        """Custom initial_scale is passed to GradScaler."""
        ctx = AMPContext(
            mixed_precision="fp16", device_type="cuda", initial_scale=1024.0
        )
        assert ctx.scaler is not None
        assert ctx.scaler.get_scale() == 1024.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_optimizer_step_uses_scaler(self):
        """optimizer_step() uses scaler.step() and scaler.update() for fp16."""
        ctx = AMPContext(mixed_precision="fp16", device_type="cuda")
        optimizer = MagicMock()
        # Mock the scaler to track calls
        ctx.scaler = MagicMock()
        ctx.scaler.get_scale.return_value = 65536.0
        ctx.optimizer_step(optimizer)
        ctx.scaler.step.assert_called_once_with(optimizer)
        ctx.scaler.update.assert_called_once()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_unscale_calls_scaler(self):
        """unscale_() calls scaler.unscale_() for fp16."""
        ctx = AMPContext(mixed_precision="fp16", device_type="cuda")
        ctx.scaler = MagicMock()
        optimizer = MagicMock()
        ctx.unscale_(optimizer)
        ctx.scaler.unscale_.assert_called_once_with(optimizer)


class TestAMPContextCPU:
    """Tests for AMPContext behavior on CPU (no CUDA required)."""

    def test_bf16_autocast_on_cpu(self):
        """bf16 autocast works on CPU."""
        ctx = AMPContext(mixed_precision="bf16", device_type="cpu")
        with ctx.autocast():
            t = torch.randn(3, 3)
            # Within autocast on CPU, operations may run in bfloat16
            result = t @ t
        # Should not raise

    def test_disabled_autocast_on_cpu(self):
        """Disabled AMP autocast on CPU returns nullcontext."""
        ctx = AMPContext(mixed_precision=None, device_type="cpu")
        ac = ctx.autocast()
        assert isinstance(ac, nullcontext)

    def test_default_device_type_is_cuda(self):
        """Default device_type is 'cuda'."""
        ctx = AMPContext(mixed_precision=None)
        assert ctx.device_type == "cuda"

    def test_mixed_precision_stored(self):
        """mixed_precision parameter is stored correctly."""
        ctx_none = AMPContext(mixed_precision=None)
        ctx_bf16 = AMPContext(mixed_precision="bf16")
        assert ctx_none.mixed_precision is None
        assert ctx_bf16.mixed_precision == "bf16"


# ---------------------------------------------------------------------------
# base_trainer helper function tests
# ---------------------------------------------------------------------------


class TestLogitsFromOutputs:
    """Tests for logits_from_outputs()."""

    def test_tensor_passthrough(self):
        """If outputs is a Tensor, return it directly as logits."""
        t = torch.randn(2, 10)
        result = logits_from_outputs(t)
        assert result is t

    def test_object_with_logits_attribute(self):
        """If outputs has a .logits attribute, return it."""
        logits = torch.randn(2, 10)
        outputs = MagicMock()
        outputs.logits = logits
        result = logits_from_outputs(outputs)
        assert result is logits

    def test_dict_with_logits_key(self):
        """A dict-like object with .logits attribute works."""

        class DictLike:
            def __init__(self):
                self.logits = torch.randn(2, 10)

        outputs = DictLike()
        result = logits_from_outputs(outputs)
        assert result is outputs.logits

    def test_object_without_logits_raises(self):
        """If outputs is not a Tensor and has no .logits, AssertionError is raised."""

        class NoLogits:
            pass

        with pytest.raises(AssertionError):
            logits_from_outputs(NoLogits())

    def test_tuple_without_logits_raises(self):
        """A tuple (which has no .logits attribute) raises AssertionError."""
        with pytest.raises(AssertionError):
            logits_from_outputs((torch.tensor(0.5), torch.randn(2, 10)))


class TestLossFromOutputs:
    """Tests for loss_from_outputs()."""

    def test_tuple_extracts_first_element(self):
        """If outputs is a tuple, the first element is the loss."""
        loss = torch.tensor(0.5)
        outputs = (loss, torch.randn(2, 10))
        result = loss_from_outputs(outputs)
        assert result is loss

    def test_object_with_loss_attribute(self):
        """If outputs has a .loss attribute, return it."""
        loss = torch.tensor(0.3)
        outputs = MagicMock()
        outputs.loss = loss
        # Ensure it's not a tuple
        result = loss_from_outputs(outputs)
        assert result is loss

    def test_tuple_first_element_must_be_tensor(self):
        """If tuple first element is not a Tensor, AssertionError is raised."""
        with pytest.raises(AssertionError):
            loss_from_outputs((0.5, torch.randn(2, 10)))

    def test_object_without_loss_raises(self):
        """If outputs is not a tuple and has no .loss, AssertionError is raised."""

        class NoLoss:
            pass

        with pytest.raises(AssertionError):
            loss_from_outputs(NoLoss())

    def test_single_element_tuple_loss(self):
        """A single-element tuple can provide the loss."""
        loss = torch.tensor(1.0)
        result = loss_from_outputs((loss,))
        assert result is loss


class TestLossAndLogitsFromOutputs:
    """Tests for loss_and_logits_from_outputs()."""

    def test_tuple_extracts_both(self):
        """If outputs is a tuple, extract (loss, logits)."""
        loss = torch.tensor(0.5)
        logits = torch.randn(2, 10)
        result_loss, result_logits = loss_and_logits_from_outputs((loss, logits))
        assert result_loss is loss
        assert result_logits is logits

    def test_object_with_loss_and_logits(self):
        """If outputs has .loss and .logits, extract both."""
        loss = torch.tensor(0.3)
        logits = torch.randn(2, 10)

        class ModelOutput:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits

        outputs = ModelOutput(loss, logits)
        result_loss, result_logits = loss_and_logits_from_outputs(outputs)
        assert result_loss is loss
        assert result_logits is logits

    def test_tuple_elements_must_be_tensors(self):
        """Both elements of the tuple must be Tensors."""
        with pytest.raises(AssertionError):
            loss_and_logits_from_outputs((0.5, torch.randn(2, 10)))
        with pytest.raises(AssertionError):
            loss_and_logits_from_outputs((torch.tensor(0.5), [1, 2, 3]))

    def test_object_missing_logits_raises(self):
        """If outputs lacks .logits, AssertionError is raised."""

        class OnlyLoss:
            loss = torch.tensor(0.5)

        with pytest.raises(AssertionError):
            loss_and_logits_from_outputs(OnlyLoss())

    def test_object_missing_loss_raises(self):
        """If outputs lacks .loss, AssertionError is raised."""

        class OnlyLogits:
            logits = torch.randn(2, 10)

        with pytest.raises(AssertionError):
            loss_and_logits_from_outputs(OnlyLogits())

    def test_namedtuple_like_output(self):
        """A namedtuple-style output works if it's a tuple subclass."""
        from collections import namedtuple

        Output = namedtuple("Output", ["loss", "logits"])
        loss = torch.tensor(0.7)
        logits = torch.randn(4, 5)
        outputs = Output(loss=loss, logits=logits)
        result_loss, result_logits = loss_and_logits_from_outputs(outputs)
        assert result_loss is loss
        assert result_logits is logits


# ---------------------------------------------------------------------------
# TrainerState tests
# ---------------------------------------------------------------------------


class TestTrainerState:
    """Tests for TrainerState dataclass."""

    def _make_state(self, **overrides: Any):
        """Helper to create a TrainerState with required fields and optional overrides."""
        defaults: dict[str, Any] = dict(
            logging_steps=50,
            eval_steps=100,
            train_batch_size=16,
            max_steps=1000,
            num_train_epochs=3,
            max_eval_steps=-1,
        )
        defaults.update(overrides)
        return TrainerState(**defaults)

    def test_construction_with_required_fields(self):
        """TrainerState can be constructed with required fields."""
        state = self._make_state()
        assert state.logging_steps == 50
        assert state.eval_steps == 100
        assert state.train_batch_size == 16
        assert state.max_steps == 1000
        assert state.num_train_epochs == 3
        assert state.max_eval_steps == -1

    def test_default_values(self):
        """TrainerState has correct default values for optional fields."""
        state = self._make_state()
        assert state.epoch == 0.0
        assert state.global_step == 0
        assert state.is_local_process_zero is True
        assert state.is_world_process_zero is True
        assert state.log_history == []
        assert state.save_steps == 0
        assert state.best_metric is None
        assert state.best_model_checkpoint is None
        assert state.num_input_tokens_seen == 0
        assert state.total_flos == 0.0
        assert state.is_hyper_param_search is False
        assert state.stateful_callbacks == []
        assert state.epoch_start_step == 0
        assert state.raw_epoch == 0

    def test_log_history_is_independent_per_instance(self):
        """Each TrainerState instance has its own log_history list."""
        state1 = self._make_state()
        state2 = self._make_state()
        state1.log_history.append({"loss": 0.5})
        assert len(state2.log_history) == 0

    def test_stateful_callbacks_is_independent_per_instance(self):
        """Each TrainerState instance has its own stateful_callbacks list."""
        state1 = self._make_state()
        state2 = self._make_state()
        state1.stateful_callbacks.append(MagicMock(spec=TrainerCallback))
        assert len(state2.stateful_callbacks) == 0

    def test_override_defaults(self):
        """Default values can be overridden at construction."""
        state = self._make_state(
            epoch=1.5,
            global_step=500,
            is_local_process_zero=False,
            best_metric=0.95,
            raw_epoch=2,
        )
        assert state.epoch == 1.5
        assert state.global_step == 500
        assert state.is_local_process_zero is False
        assert state.best_metric == 0.95
        assert state.raw_epoch == 2

    def test_kw_only_construction(self):
        """TrainerState requires keyword arguments (kw_only=True)."""
        with pytest.raises(TypeError):
            # Positional arguments should not work
            TrainerState(50, 100, 16, 1000, 3, -1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TrainerControl tests
# ---------------------------------------------------------------------------


class TestTrainerControl:
    """Tests for TrainerControl dataclass."""

    def test_default_values(self):
        """TrainerControl has all flags defaulting to False."""
        control = TrainerControl()
        assert control.should_training_stop is False
        assert control.should_epoch_stop is False
        assert control.should_save is False
        assert control.should_evaluate is False
        assert control.should_log is False
        assert control.should_abort_without_save is False

    def test_set_flags(self):
        """TrainerControl flags can be set individually."""
        control = TrainerControl()
        control.should_training_stop = True
        assert control.should_training_stop is True
        assert control.should_epoch_stop is False

    def test_construction_with_values(self):
        """TrainerControl can be constructed with specified values."""
        control = TrainerControl(
            should_training_stop=True,
            should_save=True,
            should_evaluate=True,
        )
        assert control.should_training_stop is True
        assert control.should_save is True
        assert control.should_evaluate is True
        assert control.should_epoch_stop is False
        assert control.should_log is False
        assert control.should_abort_without_save is False

    def test_has_slots(self):
        """TrainerControl uses slots for memory efficiency."""
        control = TrainerControl()
        assert hasattr(control, "__slots__")

    def test_all_fields_present(self):
        """TrainerControl has exactly the expected fields."""
        expected_fields = {
            "should_training_stop",
            "should_epoch_stop",
            "should_save",
            "should_evaluate",
            "should_log",
            "should_abort_without_save",
        }
        actual_fields = {f.name for f in fields(TrainerControl)}
        assert actual_fields == expected_fields


# ---------------------------------------------------------------------------
# IntervalStrategy tests
# ---------------------------------------------------------------------------


class TestIntervalStrategy:
    """Tests for IntervalStrategy enum."""

    def test_valid_values(self):
        """IntervalStrategy has NO, STEPS, and EPOCH members."""
        assert IntervalStrategy.NO.value == "no"
        assert IntervalStrategy.STEPS.value == "steps"
        assert IntervalStrategy.EPOCH.value == "epoch"

    def test_from_string(self):
        """IntervalStrategy can be created from string values."""
        assert IntervalStrategy("no") == IntervalStrategy.NO
        assert IntervalStrategy("steps") == IntervalStrategy.STEPS
        assert IntervalStrategy("epoch") == IntervalStrategy.EPOCH

    def test_invalid_value_raises(self):
        """An invalid value raises ValueError (DiagnosticEnum)."""
        with pytest.raises(ValueError):
            IntervalStrategy("invalid")

    def test_diagnostic_error_message(self):
        """DiagnosticEnum provides informative error messages."""
        with pytest.raises(ValueError) as exc_info:
            IntervalStrategy("invalid")
        error_msg = str(exc_info.value)
        # DiagnosticEnum includes valid options in error message
        assert "choose one of" in error_msg or "IntervalStrategy" in error_msg

    def test_all_members(self):
        """IntervalStrategy has exactly 3 members."""
        assert len(IntervalStrategy) == 3


# ---------------------------------------------------------------------------
# MinimalTrainingArguments tests
# ---------------------------------------------------------------------------


class TestMinimalTrainingArguments:
    """Tests for MinimalTrainingArguments defaults."""

    def test_default_values(self):
        """MinimalTrainingArguments has sensible defaults."""
        args = MinimalTrainingArguments()
        assert args.output_dir == "tmp_trainer"
        assert args.per_device_train_batch_size == 16
        assert args.per_device_eval_batch_size == 16
        assert args.num_train_epochs == 1
        assert args.max_steps == -1
        assert args.eval_strategy == "no"
        assert args.logging_strategy == "steps"
        assert args.logging_steps == 50
        assert args.save_strategy == "steps"
        assert args.save_steps == 1000
        assert args.save_total_limit == 2
        assert args.gradient_accumulation_steps == 1

    def test_kw_only_construction(self):
        """MinimalTrainingArguments requires keyword arguments."""
        with pytest.raises(TypeError):
            MinimalTrainingArguments("some_dir")  # type: ignore[call-arg]

    def test_override_values(self):
        """MinimalTrainingArguments values can be overridden."""
        args = MinimalTrainingArguments(
            output_dir="my_model",
            per_device_train_batch_size=32,
            max_steps=5000,
        )
        assert args.output_dir == "my_model"
        assert args.per_device_train_batch_size == 32
        assert args.max_steps == 5000


# ---------------------------------------------------------------------------
# JsonLogger callback tests
# ---------------------------------------------------------------------------


class TestJsonLoggerInit:
    """Tests for JsonLogger initialization."""

    def test_init_default_state(self):
        """JsonLogger initializes with None log_file and log_path."""
        logger = JsonLogger()
        assert logger.log_file is None
        assert logger.log_path is None
        assert logger.prefix == ""

    def test_init_with_kwargs(self):
        """JsonLogger stores extra kwargs."""
        logger = JsonLogger(experiment="test", run_id=42)
        assert logger.kwargs == {"experiment": "test", "run_id": 42}


class TestJsonLoggerOnTrainBegin:
    """Tests for JsonLogger.on_train_begin behavior."""

    def _make_args(self, logging_dir):
        """Create a mock args object with logging_dir."""
        args = MagicMock()
        args.logging_dir = logging_dir
        return args

    def _make_state(self, is_world_process_zero=True):
        """Create a mock state object."""
        state = MagicMock()
        state.is_world_process_zero = is_world_process_zero
        return state

    def test_creates_log_file(self):
        """on_train_begin creates a trainer_logs.json file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = JsonLogger()
            args = self._make_args(tmpdir)
            state = self._make_state()
            control = TrainerControl()

            logger.on_train_begin(args, state, control)

            assert logger.log_file is not None
            assert logger.log_path is not None
            assert logger.log_path.endswith("trainer_logs.json")
            assert os.path.exists(logger.log_path)

            logger.close()

    def test_skips_non_zero_rank(self):
        """on_train_begin does nothing when not world process zero."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = JsonLogger()
            args = self._make_args(tmpdir)
            state = self._make_state(is_world_process_zero=False)
            control = TrainerControl()

            logger.on_train_begin(args, state, control)

            assert logger.log_file is None

    def test_skips_none_logging_dir(self):
        """on_train_begin does nothing when logging_dir is None."""
        logger = JsonLogger()
        args = self._make_args(logging_dir=None)
        state = self._make_state()
        control = TrainerControl()

        logger.on_train_begin(args, state, control)

        assert logger.log_file is None


class TestJsonLoggerWriteLog:
    """Tests for JsonLogger._write_log behavior."""

    def _setup_logger(self, tmpdir):
        """Set up a JsonLogger with an active log file."""
        logger = JsonLogger()
        args = MagicMock()
        args.logging_dir = tmpdir
        state = MagicMock()
        state.is_world_process_zero = True
        control = TrainerControl()
        logger.on_train_begin(args, state, control)
        return logger

    def test_write_log_writes_json(self):
        """_write_log writes a JSON record to the log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = self._setup_logger(tmpdir)

            state = MagicMock()
            state.global_step = 100
            state.epoch = 1.5

            logger._write_log(state, {"loss": 0.5, "lr": 1e-4})
            logger.close()

            assert logger.log_path is not None
            with open(logger.log_path, "r") as f:
                content = f.read()

            data = json.loads(content)
            assert len(data) == 1
            record = data[0]
            assert record["global_step"] == 100
            assert record["epoch"] == 1.5
            assert record["loss"] == 0.5
            assert record["lr"] == 1e-4
            assert "timestamp" in record

    def test_multiple_writes_produce_valid_json_array(self):
        """Multiple _write_log calls produce a valid JSON array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = self._setup_logger(tmpdir)

            state = MagicMock()
            state.global_step = 1
            state.epoch = 0.1
            logger._write_log(state, {"loss": 1.0})

            state.global_step = 2
            state.epoch = 0.2
            logger._write_log(state, {"loss": 0.8})

            state.global_step = 3
            state.epoch = 0.3
            logger._write_log(state, {"loss": 0.6})

            logger.close()

            assert logger.log_path is not None
            with open(logger.log_path, "r") as f:
                data = json.load(f)

            assert len(data) == 3
            assert data[0]["global_step"] == 1
            assert data[1]["global_step"] == 2
            assert data[2]["global_step"] == 3
            assert data[0]["loss"] == 1.0
            assert data[2]["loss"] == 0.6

    def test_prefix_changes_after_first_write(self):
        """After first _write_log call, prefix changes from '' to ',\\n'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = self._setup_logger(tmpdir)
            assert logger.prefix == ""

            state = MagicMock()
            state.global_step = 1
            state.epoch = 0.0
            logger._write_log(state, {})
            assert logger.prefix == ",\n"

            logger.close()


class TestJsonLoggerOnLog:
    """Tests for JsonLogger.on_log event handler."""

    def test_on_log_writes_when_file_open(self):
        """on_log writes to the log file when it is open."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = JsonLogger()
            args = MagicMock()
            args.logging_dir = tmpdir
            state = MagicMock()
            state.is_world_process_zero = True
            state.global_step = 10
            state.epoch = 0.5
            control = TrainerControl()

            logger.on_train_begin(args, state, control)
            logger.on_log(args, state, control, logs={"loss": 0.42})
            logger.close()

            assert logger.log_path is not None
            with open(logger.log_path, "r") as f:
                data = json.load(f)

            assert len(data) == 1
            assert data[0]["loss"] == 0.42

    def test_on_log_noop_when_file_not_open(self):
        """on_log does nothing when log_file is None."""
        logger = JsonLogger()
        state = MagicMock()
        control = TrainerControl()
        args = MagicMock()
        # Should not raise even though no file is open
        logger.on_log(args, state, control, logs={"loss": 0.5})


class TestJsonLoggerOnEvaluate:
    """Tests for JsonLogger.on_evaluate event handler."""

    def test_on_evaluate_writes_metrics(self):
        """on_evaluate writes evaluation metrics to the log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = JsonLogger()
            args = MagicMock()
            args.logging_dir = tmpdir
            state = MagicMock()
            state.is_world_process_zero = True
            state.global_step = 50
            state.epoch = 1.0
            control = TrainerControl()

            logger.on_train_begin(args, state, control)
            logger.on_evaluate(
                args, state, control, metrics={"eval_loss": 0.35, "eval_accuracy": 0.9}
            )
            logger.close()

            assert logger.log_path is not None
            with open(logger.log_path, "r") as f:
                data = json.load(f)

            assert len(data) == 1
            assert data[0]["eval_loss"] == 0.35
            assert data[0]["eval_accuracy"] == 0.9

    def test_on_evaluate_noop_when_file_not_open(self):
        """on_evaluate does nothing when log_file is None."""
        logger = JsonLogger()
        state = MagicMock()
        control = TrainerControl()
        args = MagicMock()
        # Should not raise
        logger.on_evaluate(args, state, control, metrics={"eval_loss": 0.5})


class TestJsonLoggerOnTrainEnd:
    """Tests for JsonLogger.on_train_end event handler."""

    def test_on_train_end_closes_file(self):
        """on_train_end closes the log file properly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = JsonLogger()
            args = MagicMock()
            args.logging_dir = tmpdir
            state = MagicMock()
            state.is_world_process_zero = True
            state.global_step = 0
            state.epoch = 0.0
            control = TrainerControl()

            logger.on_train_begin(args, state, control)
            assert logger.log_file is not None

            logger.on_train_end(args, state, control)
            assert logger.log_file is None


class TestJsonLoggerClose:
    """Tests for JsonLogger.close behavior."""

    def test_close_idempotent(self):
        """Calling close() multiple times is safe."""
        logger = JsonLogger()
        logger.close()  # No file open, should be fine
        logger.close()  # Should be fine again

    def test_close_writes_closing_bracket(self):
        """close() writes the closing bracket to produce valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = JsonLogger()
            args = MagicMock()
            args.logging_dir = tmpdir
            state = MagicMock()
            state.is_world_process_zero = True
            state.global_step = 5
            state.epoch = 0.1
            control = TrainerControl()

            logger.on_train_begin(args, state, control)
            logger._write_log(state, {"loss": 0.5})
            logger.close()

            assert logger.log_path is not None
            with open(logger.log_path, "r") as f:
                content = f.read()

            # Should be valid JSON array
            assert content.startswith("[")
            assert content.rstrip().endswith("]")
            data = json.loads(content)
            assert isinstance(data, list)


class TestJsonLoggerFullLifecycle:
    """Integration-style tests for the full JsonLogger lifecycle."""

    def test_full_training_lifecycle(self):
        """Test a complete training lifecycle: begin -> log -> evaluate -> end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = JsonLogger()
            args = MagicMock()
            args.logging_dir = tmpdir
            state = MagicMock()
            state.is_world_process_zero = True
            control = TrainerControl()

            # Begin training
            logger.on_train_begin(args, state, control)

            # Log some training steps
            state.global_step = 50
            state.epoch = 0.5
            logger.on_log(args, state, control, logs={"loss": 1.0, "lr": 1e-4})

            state.global_step = 100
            state.epoch = 1.0
            logger.on_log(args, state, control, logs={"loss": 0.5, "lr": 5e-5})

            # Evaluate
            state.global_step = 100
            state.epoch = 1.0
            logger.on_evaluate(args, state, control, metrics={"eval_loss": 0.45})

            # End training
            logger.on_train_end(args, state, control)

            # Validate output
            assert logger.log_path is not None
            with open(logger.log_path, "r") as f:
                data = json.load(f)

            assert len(data) == 3
            # First entry: training log
            assert data[0]["loss"] == 1.0
            assert data[0]["global_step"] == 50
            # Second entry: training log
            assert data[1]["loss"] == 0.5
            assert data[1]["global_step"] == 100
            # Third entry: evaluation
            assert data[2]["eval_loss"] == 0.45


# ---------------------------------------------------------------------------
# ProgressCallback speed metric tests
# ---------------------------------------------------------------------------


class TestProgressCallbackSpeedMetrics:
    """Tests for ProgressCallback tok/s and MFU computation.

    tok/s should use wall-clock time between log steps (real throughput).
    MFU should use accumulated train step time (forward/backward only).
    """

    def _make_callback(self, **kwargs):
        defaults = dict(
            use_tqdm=False,
            show_tokens_per_second=True,
            show_loss=False,
            show_grad_norm=False,
            show_learning_rate=False,
            show_epoch=False,
            output_stream="stdout",
        )
        defaults.update(kwargs)
        return ProgressCallback(**defaults)

    def _make_state(self, global_step=0, max_steps=100, total_flos=0.0):
        state = MagicMock()
        state.is_world_process_zero = True
        state.global_step = global_step
        state.max_steps = max_steps
        state.total_flos = total_flos
        return state

    def _make_args(self):
        return MagicMock()

    def test_tok_s_uses_wall_clock_time(self):
        """tok/s should reflect wall-clock delta between log steps, not just
        forward/backward time."""
        cb = self._make_callback()
        args = self._make_args()
        state = self._make_state()
        control = MagicMock()

        cb.on_train_begin(args, state, control)

        # Simulate first log step to initialize _last_log_time
        with patch(
            "forgather.ml.trainer.callbacks.default_callbacks.time"
        ) as mock_time:
            mock_time.monotonic.return_value = 100.0
            cb.on_log(args, state, control, logs={"tokens": 1000, "total_flos": 0.0})

        # Simulate a training step that takes 0.5s (forward/backward)
        with patch(
            "forgather.ml.trainer.callbacks.default_callbacks.time"
        ) as mock_time:
            mock_time.monotonic.return_value = 101.0
            cb.on_step_begin(args, state, control)
            mock_time.monotonic.return_value = 101.5
            cb.on_step_end(args, state, control)

        # But wall-clock has advanced 2.0s total (optimizer + data loading = 1.5s extra)
        state.global_step = 10
        display_logs = {}
        with patch(
            "forgather.ml.trainer.callbacks.default_callbacks.time"
        ) as mock_time:
            mock_time.monotonic.return_value = 102.0
            # Capture the display_logs by intercepting the format call
            with patch(
                "forgather.ml.trainer.callbacks.default_callbacks.format_train_log"
            ) as mock_fmt:
                mock_fmt.return_value = ""
                cb.on_log(
                    args, state, control, logs={"tokens": 2000, "total_flos": 0.0}
                )
                if mock_fmt.called:
                    display_logs = mock_fmt.call_args[0][1]

        # Wall-clock delta = 102.0 - 100.0 = 2.0s, tokens = 2000
        # Expected tok/s = 2000 / 2.0 = 1000
        assert "tok/s" in display_logs
        assert display_logs["tok/s"] == 1000

    def test_mfu_uses_train_step_time(self):
        """MFU should use accumulated on_step_begin/on_step_end time, not
        wall-clock time."""
        peak_flops = 1000.0  # Simplified peak for easy math
        cb = self._make_callback(peak_hardware_flops=peak_flops)
        args = self._make_args()
        state = self._make_state()
        control = MagicMock()

        cb.on_train_begin(args, state, control)

        # First log step to initialize tracking
        with patch(
            "forgather.ml.trainer.callbacks.default_callbacks.time"
        ) as mock_time:
            mock_time.monotonic.return_value = 100.0
            cb.on_log(args, state, control, logs={"tokens": 0, "total_flos": 0.0})

        # Simulate training step: 0.5s of forward/backward
        with patch(
            "forgather.ml.trainer.callbacks.default_callbacks.time"
        ) as mock_time:
            mock_time.monotonic.return_value = 101.0
            cb.on_step_begin(args, state, control)
            mock_time.monotonic.return_value = 101.5
            cb.on_step_end(args, state, control)

        # Wall-clock is 2.0s but train time is only 0.5s
        # delta_flos = 500, so achieved = 500 / 0.5 = 1000 FLOP/s
        # MFU = 1000 / 1000 = 100%
        display_logs = {}
        with patch(
            "forgather.ml.trainer.callbacks.default_callbacks.time"
        ) as mock_time:
            mock_time.monotonic.return_value = 102.0
            with patch(
                "forgather.ml.trainer.callbacks.default_callbacks.format_train_log"
            ) as mock_fmt:
                mock_fmt.return_value = ""
                cb.on_log(
                    args, state, control, logs={"tokens": 2000, "total_flos": 500.0}
                )
                if mock_fmt.called:
                    display_logs = mock_fmt.call_args[0][1]

        assert "mfu" in display_logs
        assert display_logs["mfu"] == "100.0%"

    def test_no_tok_s_on_first_log(self):
        """tok/s should not appear on the very first log step since there is
        no previous wall-clock reference."""
        cb = self._make_callback()
        args = self._make_args()
        state = self._make_state()
        control = MagicMock()

        cb.on_train_begin(args, state, control)

        display_logs = {}
        with patch(
            "forgather.ml.trainer.callbacks.default_callbacks.time"
        ) as mock_time:
            mock_time.monotonic.return_value = 100.0
            with patch(
                "forgather.ml.trainer.callbacks.default_callbacks.format_train_log"
            ) as mock_fmt:
                mock_fmt.return_value = ""
                cb.on_log(
                    args, state, control, logs={"tokens": 5000, "total_flos": 0.0}
                )
                if mock_fmt.called:
                    display_logs = mock_fmt.call_args[0][1]

        assert "tok/s" not in display_logs

    def test_accumulated_train_time_resets_between_intervals(self):
        """_accumulated_train_time should reset after each on_log call."""
        cb = self._make_callback()
        args = self._make_args()
        state = self._make_state()
        control = MagicMock()

        cb.on_train_begin(args, state, control)

        # Simulate a step
        with patch(
            "forgather.ml.trainer.callbacks.default_callbacks.time"
        ) as mock_time:
            mock_time.monotonic.return_value = 0.0
            cb.on_step_begin(args, state, control)
            mock_time.monotonic.return_value = 1.0
            cb.on_step_end(args, state, control)

        assert cb._accumulated_train_time == 1.0

        # Log step resets it
        with patch(
            "forgather.ml.trainer.callbacks.default_callbacks.time"
        ) as mock_time:
            mock_time.monotonic.return_value = 2.0
            cb.on_log(args, state, control, logs={"tokens": 100, "total_flos": 0.0})

        assert cb._accumulated_train_time == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
