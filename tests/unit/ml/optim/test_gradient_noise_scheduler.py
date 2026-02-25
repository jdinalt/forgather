"""
Tests for GradientNoiseScheduler.
"""

import math
import random

import torch
import torch.nn as nn

from forgather.ml.optim.gradient_noise_scheduler import GradientNoiseScheduler


def _make_optimizer(lr=1e-3):
    """Create a simple optimizer for testing."""
    model = nn.Linear(10, 10)
    return torch.optim.SGD(model.parameters(), lr=lr)


def _get_lr(scheduler):
    """Get the current LR from the scheduler."""
    return scheduler.get_last_lr()[0]


def _step(sched, grad_norm):
    """Feed a grad norm and advance the scheduler by one step."""
    sched.update_grad_norm(grad_norm)
    sched.step()


def test_warmup_lr_unchanged():
    """During warmup, LR should remain at base_lr."""
    base_lr = 1e-3
    warmup = 100
    opt = _make_optimizer(lr=base_lr)
    sched = GradientNoiseScheduler(opt, calibration_steps=warmup)

    for i in range(warmup):
        _step(sched, 5.0 + random.gauss(0, 1.0))
        lr = _get_lr(sched)
        assert (
            abs(lr - base_lr) < 1e-12
        ), f"Step {i}: LR {lr} != base_lr {base_lr} during warmup"


def test_auto_calibration():
    """target_std should be set from EMA at end of warmup when not specified."""
    warmup = 200
    opt = _make_optimizer()
    sched = GradientNoiseScheduler(
        opt, calibration_steps=warmup, ema_decay=0.99, spike_threshold_std=None
    )

    assert sched.target_std is None

    # Feed grad norms with known mean and std
    rng = random.Random(42)
    for _ in range(warmup):
        gn = 10.0 + rng.gauss(0, 2.0)
        _step(sched, gn)

    assert sched.target_std is not None
    assert sched.target_std > 0.0
    # The calibrated std should be in a reasonable range given our inputs
    assert 0.5 < sched.target_std < 5.0, f"Unexpected target_std: {sched.target_std}"


def test_manual_target_not_overridden():
    """A manually specified target_std should not be overridden by warmup."""
    manual_target = 3.14
    warmup = 50
    opt = _make_optimizer()
    sched = GradientNoiseScheduler(
        opt, calibration_steps=warmup, target_std=manual_target
    )

    rng = random.Random(42)
    for _ in range(warmup + 50):
        _step(sched, 10.0 + rng.gauss(0, 1.0))

    assert sched.target_std == manual_target


def test_feedback_reduces_lr():
    """When grad norm std exceeds target, LR should decrease."""
    warmup = 100
    base_lr = 1e-3
    opt = _make_optimizer(lr=base_lr)
    sched = GradientNoiseScheduler(
        opt,
        calibration_steps=warmup,
        target_std=1.0,
        feedback_strength=1e-3,  # Strong for fast test response
        ema_decay=0.99,
    )

    # Warmup with moderate noise
    rng = random.Random(42)
    for _ in range(warmup):
        _step(sched, 10.0 + rng.gauss(0, 1.0))

    lr_after_warmup = _get_lr(sched)

    # Now feed grad norms with much higher std than target
    for _ in range(500):
        _step(sched, 10.0 + rng.gauss(0, 5.0))

    lr_after_noise = _get_lr(sched)
    assert (
        lr_after_noise < lr_after_warmup
    ), f"LR should decrease: {lr_after_noise} >= {lr_after_warmup}"


def test_feedback_increases_lr():
    """When grad norm std is below target (and unclamped), LR should increase."""
    warmup = 100
    base_lr = 1e-3
    opt = _make_optimizer(lr=base_lr)
    sched = GradientNoiseScheduler(
        opt,
        calibration_steps=warmup,
        target_std=5.0,  # High target
        feedback_strength=1e-3,
        ema_decay=0.99,
        max_lr_scale=None,  # No ceiling
    )

    # Warmup
    rng = random.Random(42)
    for _ in range(warmup):
        _step(sched, 10.0 + rng.gauss(0, 5.0))

    lr_after_warmup = _get_lr(sched)

    # Feed with very low std (well below target of 5.0)
    for _ in range(500):
        _step(sched, 10.0 + rng.gauss(0, 0.1))

    lr_after = _get_lr(sched)
    assert (
        lr_after > lr_after_warmup
    ), f"LR should increase: {lr_after} <= {lr_after_warmup}"


def test_min_lr_scale_respected():
    """LR should not drop below min_lr_scale * base_lr."""
    base_lr = 1e-3
    min_scale = 0.1
    opt = _make_optimizer(lr=base_lr)
    sched = GradientNoiseScheduler(
        opt,
        calibration_steps=10,
        target_std=0.01,  # Very low target
        feedback_strength=0.1,  # Very aggressive
        ema_decay=0.9,
        min_lr_scale=min_scale,
    )

    rng = random.Random(42)
    # Warmup
    for _ in range(10):
        _step(sched, 10.0 + rng.gauss(0, 1.0))

    # Drive with extreme noise to push LR down hard
    for _ in range(1000):
        _step(sched, 10.0 + rng.gauss(0, 50.0))

    lr = _get_lr(sched)
    floor = base_lr * min_scale
    assert lr >= floor - 1e-15, f"LR {lr} below floor {floor}"


def test_max_lr_scale_respected():
    """LR should not exceed max_lr_scale * base_lr."""
    base_lr = 1e-3
    max_scale = 1.0
    opt = _make_optimizer(lr=base_lr)
    sched = GradientNoiseScheduler(
        opt,
        calibration_steps=10,
        target_std=100.0,  # Very high target
        feedback_strength=0.1,
        ema_decay=0.9,
        max_lr_scale=max_scale,
    )

    rng = random.Random(42)
    for _ in range(10):
        _step(sched, 10.0 + rng.gauss(0, 1.0))

    # Feed with very low std (far below target) to push LR up
    for _ in range(1000):
        _step(sched, 10.0 + rng.gauss(0, 0.001))

    lr = _get_lr(sched)
    ceiling = base_lr * max_scale
    assert lr <= ceiling + 1e-15, f"LR {lr} above ceiling {ceiling}"


def test_no_ceiling_allows_increase():
    """With max_lr_scale=None, LR can exceed base_lr."""
    base_lr = 1e-3
    opt = _make_optimizer(lr=base_lr)
    sched = GradientNoiseScheduler(
        opt,
        calibration_steps=10,
        target_std=100.0,  # Very high target
        feedback_strength=0.1,
        ema_decay=0.9,
        max_lr_scale=None,
    )

    rng = random.Random(42)
    for _ in range(10):
        _step(sched, 10.0 + rng.gauss(0, 1.0))

    for _ in range(1000):
        _step(sched, 10.0 + rng.gauss(0, 0.001))

    lr = _get_lr(sched)
    assert lr > base_lr, f"LR {lr} should exceed base_lr {base_lr}"


def test_smoothness():
    """LR changes should be gradual (no large jumps)."""
    warmup = 50
    base_lr = 1e-3
    opt = _make_optimizer(lr=base_lr)
    sched = GradientNoiseScheduler(
        opt,
        calibration_steps=warmup,
        feedback_strength=1e-4,
        ema_decay=0.999,
    )

    rng = random.Random(42)
    # Warmup
    for _ in range(warmup):
        _step(sched, 10.0 + rng.gauss(0, 1.0))

    # Active phase with varying noise
    prev_lr = _get_lr(sched)
    max_ratio = 0.0
    for i in range(1000):
        # Gradually increase std
        std = 1.0 + i * 0.01
        _step(sched, 10.0 + rng.gauss(0, std))
        lr = _get_lr(sched)
        if prev_lr > 0:
            ratio = abs(lr - prev_lr) / prev_lr
            max_ratio = max(max_ratio, ratio)
        prev_lr = lr

    # No individual step should change LR by more than 1%
    assert max_ratio < 0.01, f"Max step-to-step LR ratio change: {max_ratio:.6f}"


def test_state_dict_round_trip():
    """State dict save/load should preserve exact behavior."""
    warmup = 50
    opt1 = _make_optimizer(lr=1e-3)
    sched1 = GradientNoiseScheduler(
        opt1, calibration_steps=warmup, feedback_strength=1e-3, ema_decay=0.99
    )

    rng = random.Random(42)
    for _ in range(warmup + 100):
        _step(sched1, 10.0 + rng.gauss(0, 2.0))

    state = sched1.state_dict()
    lr_before = _get_lr(sched1)
    std_before = sched1.current_std
    mean_before = sched1.current_mean
    scale_before = sched1.lr_scale

    # Create new scheduler and load state
    opt2 = _make_optimizer(lr=1e-3)
    sched2 = GradientNoiseScheduler(
        opt2, calibration_steps=warmup, feedback_strength=1e-3, ema_decay=0.99
    )
    sched2.load_state_dict(state)

    assert abs(_get_lr(sched2) - lr_before) < 1e-15
    assert abs(sched2.current_std - std_before) < 1e-15
    assert abs(sched2.current_mean - mean_before) < 1e-15
    assert abs(sched2.lr_scale - scale_before) < 1e-15

    # Verify subsequent steps produce identical results
    rng2 = random.Random(99)
    for _ in range(50):
        gn = 10.0 + rng2.gauss(0, 2.0)
        _step(sched1, gn)
        _step(sched2, gn)
        assert abs(_get_lr(sched1) - _get_lr(sched2)) < 1e-15


def test_no_grad_norm_step():
    """step() without prior update_grad_norm should not change feedback state."""
    warmup = 50
    opt = _make_optimizer(lr=1e-3)
    sched = GradientNoiseScheduler(
        opt, calibration_steps=warmup, target_std=1.0, feedback_strength=1e-3
    )

    rng = random.Random(42)
    for _ in range(warmup + 10):
        _step(sched, 10.0 + rng.gauss(0, 1.0))

    scale_before = sched.lr_scale
    std_before = sched.current_std

    # Steps without grad norm should not change feedback state
    for _ in range(100):
        sched.step()

    assert sched.lr_scale == scale_before
    assert sched.current_std == std_before


def test_tensor_input():
    """update_grad_norm() should accept torch.Tensor grad norms."""
    opt = _make_optimizer()
    sched = GradientNoiseScheduler(opt, calibration_steps=10)

    for i in range(20):
        gn = torch.tensor(5.0 + i * 0.1)
        _step(sched, gn)

    assert sched.current_mean > 0.0


def test_multiple_param_groups():
    """Scheduler should scale all param groups identically."""
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 5))
    opt = torch.optim.SGD(
        [
            {"params": model[0].parameters(), "lr": 1e-3},
            {"params": model[1].parameters(), "lr": 1e-4},
        ]
    )
    sched = GradientNoiseScheduler(
        opt,
        calibration_steps=10,
        target_std=0.1,
        feedback_strength=1e-2,
        ema_decay=0.9,
    )

    rng = random.Random(42)
    for _ in range(10):
        _step(sched, 10.0 + rng.gauss(0, 1.0))

    for _ in range(100):
        _step(sched, 10.0 + rng.gauss(0, 5.0))

    lrs = sched.get_last_lr()
    assert len(lrs) == 2
    # Both groups should have the same scale applied
    scale = sched.lr_scale
    assert abs(lrs[0] - 1e-3 * scale) < 1e-12
    assert abs(lrs[1] - 1e-4 * scale) < 1e-12


def test_zero_warmup_with_manual_target():
    """calibration_steps=0 with manual target should activate immediately."""
    base_lr = 1e-3
    opt = _make_optimizer(lr=base_lr)
    sched = GradientNoiseScheduler(
        opt,
        calibration_steps=0,
        target_std=0.1,
        feedback_strength=1e-2,
        ema_decay=0.9,
    )

    rng = random.Random(42)
    # First step should already trigger feedback
    for _ in range(100):
        _step(sched, 10.0 + rng.gauss(0, 5.0))

    assert sched.lr_scale < 1.0, "Feedback should have reduced LR"


def test_spike_filtering_skips_outliers():
    """Spikes above threshold should be discarded after min_samples."""
    min_samples = 50
    opt = _make_optimizer()
    sched = GradientNoiseScheduler(
        opt,
        calibration_steps=0,
        target_std=1.0,
        spike_threshold_std=1.645,
        min_samples_for_spike_filter=min_samples,
        feedback_strength=1e-3,
        ema_decay=0.99,
    )

    # Accumulate enough clean samples to activate filtering
    rng = random.Random(42)
    for _ in range(min_samples + 50):
        _step(sched, 10.0 + rng.gauss(0, 0.5))

    # Record state before spike
    mean_before = sched.current_mean
    std_before = sched.current_std
    scale_before = sched.lr_scale

    # Inject a massive spike (far above any reasonable threshold)
    _step(sched, 10000.0)

    # State should be unchanged -- spike was discarded
    assert (
        sched.current_mean == mean_before
    ), f"Mean changed after spike: {sched.current_mean} != {mean_before}"
    assert (
        sched.current_std == std_before
    ), f"Std changed after spike: {sched.current_std} != {std_before}"
    assert (
        sched.lr_scale == scale_before
    ), f"LR scale changed after spike: {sched.lr_scale} != {scale_before}"


def test_spike_filtering_inactive_before_min_samples():
    """Before min_samples, spikes should be incorporated normally."""
    min_samples = 200
    opt = _make_optimizer()
    sched = GradientNoiseScheduler(
        opt,
        calibration_steps=0,
        target_std=1.0,
        spike_threshold_std=1.645,
        min_samples_for_spike_filter=min_samples,
        feedback_strength=1e-3,
        ema_decay=0.99,
    )

    # Feed some normal samples (fewer than min_samples)
    rng = random.Random(42)
    for _ in range(50):
        _step(sched, 10.0 + rng.gauss(0, 0.5))

    mean_before = sched.current_mean

    # Inject a large value -- should NOT be filtered (below min_samples)
    _step(sched, 1000.0)

    assert (
        sched.current_mean != mean_before
    ), "Spike should have been incorporated before min_samples"


def test_spike_filtering_disabled_when_none():
    """With spike_threshold_std=None, all samples are accepted."""
    opt = _make_optimizer()
    sched = GradientNoiseScheduler(
        opt,
        calibration_steps=0,
        target_std=1.0,
        spike_threshold_std=None,
        min_samples_for_spike_filter=10,
        feedback_strength=1e-3,
        ema_decay=0.99,
    )

    rng = random.Random(42)
    for _ in range(50):
        _step(sched, 10.0 + rng.gauss(0, 0.5))

    mean_before = sched.current_mean

    # Spike should be accepted since filtering is disabled
    _step(sched, 10000.0)

    assert (
        sched.current_mean != mean_before
    ), "Spike should have been incorporated when filtering is disabled"


def test_spike_threshold_property():
    """spike_threshold property should return expected values."""
    opt = _make_optimizer()
    sched = GradientNoiseScheduler(
        opt,
        calibration_steps=0,
        target_std=1.0,
        spike_threshold_std=2.0,
        min_samples_for_spike_filter=20,
        ema_decay=0.99,
    )

    # Before min_samples: threshold should be 0 (inactive)
    for _ in range(10):
        _step(sched, 10.0)
    assert sched.spike_threshold == 0.0

    # After min_samples: threshold should be mean + 2*std
    rng = random.Random(42)
    for _ in range(20):
        _step(sched, 10.0 + rng.gauss(0, 1.0))

    threshold = sched.spike_threshold
    expected = sched.current_mean + 2.0 * sched.current_std
    assert (
        abs(threshold - expected) < 1e-10
    ), f"Threshold {threshold} != expected {expected}"

    # With filtering disabled: always 0
    opt2 = _make_optimizer()
    sched2 = GradientNoiseScheduler(opt2, calibration_steps=0, spike_threshold_std=None)
    for _ in range(50):
        _step(sched2, 10.0)
    assert sched2.spike_threshold == 0.0


def test_spike_filtering_state_dict_round_trip():
    """Spike filtering params should survive state_dict save/load."""
    opt1 = _make_optimizer()
    sched1 = GradientNoiseScheduler(
        opt1,
        calibration_steps=10,
        spike_threshold_std=2.5,
        min_samples_for_spike_filter=50,
        ema_decay=0.99,
    )

    rng = random.Random(42)
    for _ in range(100):
        _step(sched1, 10.0 + rng.gauss(0, 1.0))

    state = sched1.state_dict()
    threshold_before = sched1.spike_threshold

    # Load into a new scheduler (with different constructor defaults)
    opt2 = _make_optimizer()
    sched2 = GradientNoiseScheduler(
        opt2,
        calibration_steps=10,
        spike_threshold_std=1.0,  # Different from saved
        min_samples_for_spike_filter=10,  # Different from saved
        ema_decay=0.99,
    )
    sched2.load_state_dict(state)

    # Should have restored the saved values, not the constructor defaults
    assert sched2.spike_threshold_std == 2.5
    assert sched2.min_samples_for_spike_filter == 50
    assert abs(sched2.spike_threshold - threshold_before) < 1e-10


def test_on_train_metrics_callback():
    """on_train_metrics callback should feed grad_norm to the scheduler."""
    from forgather.ml.trainer.trainer_types import TrainerCallback

    opt = _make_optimizer()
    sched = GradientNoiseScheduler(
        opt,
        calibration_steps=0,
        target_std=1.0,
        feedback_strength=1e-3,
        ema_decay=0.99,
        spike_threshold_std=None,
    )

    # Verify it is a TrainerCallback
    assert isinstance(sched, TrainerCallback)

    # Feed grad_norm via on_train_metrics, then step
    rng = random.Random(42)
    for _ in range(50):
        gn = 10.0 + rng.gauss(0, 1.0)
        sched.on_train_metrics(None, None, None, grad_norm=gn)
        sched.step()

    assert sched.current_mean > 0.0
    assert sched.current_std > 0.0


def test_linear_warmup():
    """LR should ramp linearly from 0 during warmup."""
    base_lr = 1e-3
    warmup = 100
    opt = _make_optimizer(lr=base_lr)
    sched = GradientNoiseScheduler(
        opt,
        warmup_steps=warmup,
        calibration_steps=0,
        target_std=1.0,
        spike_threshold_std=None,
    )

    # LR at step 0 should be 0
    assert _get_lr(sched) == 0.0

    # Step through warmup
    for i in range(1, warmup + 1):
        _step(sched, 10.0)
        lr = _get_lr(sched)
        expected = base_lr * (i / warmup)
        assert abs(lr - expected) < 1e-12, f"Step {i}: LR {lr} != expected {expected}"

    # After warmup, LR should be at base_lr (feedback hasn't kicked in yet)
    _step(sched, 10.0)
    assert abs(_get_lr(sched) - base_lr) < 1e-12


def test_warmup_skips_stats_tracking():
    """Grad norm statistics should not be tracked during LR warmup."""
    warmup = 50
    opt = _make_optimizer()
    sched = GradientNoiseScheduler(
        opt,
        warmup_steps=warmup,
        calibration_steps=10,
        target_std=1.0,
        spike_threshold_std=None,
        ema_decay=0.99,
    )

    # Feed grad norms during LR warmup -- they should be ignored
    rng = random.Random(42)
    for _ in range(warmup):
        _step(sched, 10.0 + rng.gauss(0, 5.0))

    assert (
        sched._feedback_step == 0
    ), f"Stats tracked during warmup: {sched._feedback_step} steps"

    # After warmup, stats should start tracking
    for _ in range(20):
        _step(sched, 10.0 + rng.gauss(0, 1.0))

    assert sched._feedback_step == 20


def test_no_warmup_default():
    """With warmup_steps=0 (default), LR starts at base_lr."""
    base_lr = 1e-3
    opt = _make_optimizer(lr=base_lr)
    sched = GradientNoiseScheduler(opt, calibration_steps=10, target_std=1.0)
    assert abs(_get_lr(sched) - base_lr) < 1e-12


if __name__ == "__main__":
    test_warmup_lr_unchanged()
    print("PASS: test_warmup_lr_unchanged")

    test_auto_calibration()
    print("PASS: test_auto_calibration")

    test_manual_target_not_overridden()
    print("PASS: test_manual_target_not_overridden")

    test_feedback_reduces_lr()
    print("PASS: test_feedback_reduces_lr")

    test_feedback_increases_lr()
    print("PASS: test_feedback_increases_lr")

    test_min_lr_scale_respected()
    print("PASS: test_min_lr_scale_respected")

    test_max_lr_scale_respected()
    print("PASS: test_max_lr_scale_respected")

    test_no_ceiling_allows_increase()
    print("PASS: test_no_ceiling_allows_increase")

    test_smoothness()
    print("PASS: test_smoothness")

    test_state_dict_round_trip()
    print("PASS: test_state_dict_round_trip")

    test_no_grad_norm_step()
    print("PASS: test_no_grad_norm_step")

    test_tensor_input()
    print("PASS: test_tensor_input")

    test_multiple_param_groups()
    print("PASS: test_multiple_param_groups")

    test_zero_warmup_with_manual_target()
    print("PASS: test_zero_warmup_with_manual_target")

    test_spike_filtering_skips_outliers()
    print("PASS: test_spike_filtering_skips_outliers")

    test_spike_filtering_inactive_before_min_samples()
    print("PASS: test_spike_filtering_inactive_before_min_samples")

    test_spike_filtering_disabled_when_none()
    print("PASS: test_spike_filtering_disabled_when_none")

    test_spike_threshold_property()
    print("PASS: test_spike_threshold_property")

    test_spike_filtering_state_dict_round_trip()
    print("PASS: test_spike_filtering_state_dict_round_trip")

    test_on_train_metrics_callback()
    print("PASS: test_on_train_metrics_callback")

    test_linear_warmup()
    print("PASS: test_linear_warmup")

    test_no_warmup_default()
    print("PASS: test_no_warmup_default")

    test_warmup_skips_stats_tracking()
    print("PASS: test_warmup_skips_stats_tracking")

    print("\nAll tests passed.")
