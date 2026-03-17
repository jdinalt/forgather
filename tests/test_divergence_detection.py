"""
Unit tests for divergence detection callbacks and checkpoint preservation.
"""

import json
import os
from dataclasses import dataclass

import pytest

from forgather.ml.trainer.callbacks.divergence_detector import (
    DivergenceDetector,
    DualTimeScaleDivergenceDetector,
    DualWindowDivergenceDetector,
)


@dataclass
class MockControl:
    """Mock TrainerControl for testing."""

    should_training_stop: bool = False
    should_abort_without_save: bool = False


class TestDivergenceDetector:
    """Tests for DivergenceDetector."""

    def test_initialization(self):
        """Test detector initializes with correct defaults."""
        detector = DivergenceDetector()

        assert detector.smoothing == 0.3
        assert detector.threshold == 1.0
        assert detector.relative_threshold is None
        assert detector.patience == 3
        assert detector.warmup == 10
        assert detector.action == "stop"
        assert detector.use_eval_loss is False
        assert detector.smoothed_loss is None
        assert detector.best_smoothed_loss is None
        assert detector.observation_count == 0
        assert detector.consecutive_above == 0

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            DivergenceDetector(smoothing=-0.1)

        with pytest.raises(ValueError):
            DivergenceDetector(smoothing=1.5)

        with pytest.raises(ValueError):
            DivergenceDetector(smoothing=0.0)

        with pytest.raises(ValueError):
            DivergenceDetector(threshold=-1.0)

        with pytest.raises(ValueError):
            DivergenceDetector(relative_threshold=0.5)  # Must be > 1

        with pytest.raises(ValueError):
            DivergenceDetector(relative_threshold=1.0)  # Must be > 1

        with pytest.raises(ValueError):
            DivergenceDetector(threshold=None, relative_threshold=None)

        with pytest.raises(ValueError):
            DivergenceDetector(patience=0)

        with pytest.raises(ValueError):
            DivergenceDetector(warmup=-1)

    def test_state_dict_protocol(self):
        """Test Stateful protocol implementation."""
        detector = DivergenceDetector(warmup=0)

        # Initial state
        state = detector.state_dict()
        assert state == {
            "smoothed_loss": None,
            "best_smoothed_loss": None,
            "observation_count": 0,
            "consecutive_above": 0,
        }

        # After initialization
        control = MockControl()
        detector.on_log(None, None, control, logs={"loss": 2.5})
        state = detector.state_dict()
        assert state["smoothed_loss"] == 2.5
        assert state["best_smoothed_loss"] == 2.5
        assert state["observation_count"] == 1

    def test_state_load(self):
        """Test state can be loaded correctly."""
        detector = DivergenceDetector()

        state = {
            "smoothed_loss": 3.5,
            "best_smoothed_loss": 2.8,
            "observation_count": 50,
            "consecutive_above": 1,
        }

        detector.load_state_dict(state)
        assert detector.smoothed_loss == 3.5
        assert detector.best_smoothed_loss == 2.8
        assert detector.observation_count == 50
        assert detector.consecutive_above == 1

    def test_normal_training_no_trigger(self):
        """Test detector doesn't trigger during normal stable training."""
        detector = DivergenceDetector(
            smoothing=0.3,
            threshold=1.0,
            patience=3,
            warmup=0,
        )

        control = MockControl()

        # Simulate normal training with small fluctuations
        for i in range(50):
            logs = {"loss": 2.8 + 0.1 * (i % 3 - 1)}
            detector.on_log(None, None, control, logs=logs)

        assert not control.should_training_stop

    def test_decreasing_loss_no_trigger(self):
        """Test detector doesn't trigger during loss decrease (normal training)."""
        detector = DivergenceDetector(
            smoothing=0.3,
            threshold=1.0,
            patience=3,
            warmup=0,
        )

        control = MockControl()

        # Loss decreasing from 10 to 3 (normal training curve)
        for i in range(100):
            loss = 10.0 - 7.0 * (i / 100)
            detector.on_log(None, None, control, logs={"loss": loss})

        assert not control.should_training_stop

    def test_spike_detection_absolute(self):
        """Test detector triggers on loss spike with absolute threshold."""
        detector = DivergenceDetector(
            smoothing=0.3,
            threshold=1.0,
            patience=3,
            warmup=0,
        )

        control = MockControl()

        # Normal training at loss ~2.8
        for i in range(20):
            detector.on_log(None, None, control, logs={"loss": 2.8})

        assert not control.should_training_stop

        # Spike to 8.0
        for i in range(10):
            detector.on_log(None, None, control, logs={"loss": 8.0})

        assert control.should_training_stop

    def test_spike_detection_relative(self):
        """Test detector triggers on loss spike with relative threshold."""
        detector = DivergenceDetector(
            smoothing=0.3,
            threshold=None,
            relative_threshold=1.5,
            patience=3,
            warmup=0,
        )

        control = MockControl()

        # Normal training at loss ~2.8
        for i in range(20):
            detector.on_log(None, None, control, logs={"loss": 2.8})

        assert not control.should_training_stop

        # Spike to 8.0
        for i in range(10):
            detector.on_log(None, None, control, logs={"loss": 8.0})

        assert control.should_training_stop

    def test_patience_required(self):
        """Test that patience is respected - single spike doesn't trigger."""
        detector = DivergenceDetector(
            smoothing=0.3,
            threshold=1.0,
            patience=5,
            warmup=0,
        )

        control = MockControl()

        # Normal training
        for i in range(20):
            detector.on_log(None, None, control, logs={"loss": 2.8})

        # Single spike then return to normal
        detector.on_log(None, None, control, logs={"loss": 10.0})
        for i in range(20):
            detector.on_log(None, None, control, logs={"loss": 2.8})

        assert not control.should_training_stop

    def test_warmup_skips_early_observations(self):
        """Test that warmup period prevents false positives from high initial loss."""
        detector = DivergenceDetector(
            smoothing=0.3,
            threshold=1.0,
            patience=1,
            warmup=5,
        )

        control = MockControl()

        # High initial loss that quickly drops (would trigger without warmup)
        losses = [10.0, 8.0, 6.0, 4.0, 3.0, 2.5, 2.3, 2.2, 2.1, 2.0]
        for loss in losses:
            detector.on_log(None, None, control, logs={"loss": loss})

        assert not control.should_training_stop

    def test_nan_detection(self):
        """Test that NaN loss triggers immediately without patience."""
        detector = DivergenceDetector(warmup=0, patience=10)

        control = MockControl()

        # Normal training
        for i in range(5):
            detector.on_log(None, None, control, logs={"loss": 2.8})

        # NaN should trigger immediately
        detector.on_log(None, None, control, logs={"loss": float("nan")})
        assert control.should_training_stop

    def test_inf_detection(self):
        """Test that Inf loss triggers immediately without patience."""
        detector = DivergenceDetector(warmup=0, patience=10)

        control = MockControl()

        detector.on_log(None, None, control, logs={"loss": 2.8})
        detector.on_log(None, None, control, logs={"loss": float("inf")})
        assert control.should_training_stop

    def test_abort_action(self):
        """Test abort action sets both flags."""
        detector = DivergenceDetector(
            smoothing=0.3,
            threshold=1.0,
            patience=1,
            warmup=0,
            action="abort",
        )

        control = MockControl()

        for i in range(20):
            detector.on_log(None, None, control, logs={"loss": 2.8})

        for i in range(5):
            detector.on_log(None, None, control, logs={"loss": 8.0})

        assert control.should_training_stop
        assert control.should_abort_without_save

    def test_eval_loss_monitoring(self):
        """Test monitoring eval_loss instead of train loss."""
        detector = DivergenceDetector(
            use_eval_loss=True,
            warmup=0,
        )

        control = MockControl()

        # Train logs with only 'loss' should be ignored
        detector.on_log(None, None, control, logs={"loss": 10.0})
        assert detector.smoothed_loss is None  # Not initialized

        # Eval metrics with eval_loss should be processed
        detector.on_evaluate(None, None, control, metrics={"eval_loss": 2.8})
        assert detector.smoothed_loss == 2.8

    def test_custom_metric_monitoring(self):
        """Test monitoring custom metric."""
        detector = DivergenceDetector(
            metric_key="grad_norm",
            threshold=5.0,
            warmup=0,
        )

        control = MockControl()

        logs = {"loss": 2.8, "eval_loss": 2.5, "grad_norm": 3.2}
        detector.on_log(None, None, control, logs=logs)

        # Should initialize with grad_norm
        assert detector.smoothed_loss == 3.2

    def test_missing_metric_ignored(self):
        """Test that missing metric key is silently ignored."""
        detector = DivergenceDetector(warmup=0)

        control = MockControl()
        detector.on_log(None, None, control, logs={"other_metric": 5.0})
        assert detector.smoothed_loss is None

    def test_backward_compat_aliases(self):
        """Test that old class names still work as aliases."""
        assert DualTimeScaleDivergenceDetector is DivergenceDetector
        assert DualWindowDivergenceDetector is DivergenceDetector


class TestDivergenceDetectorOnRealLogs:
    """Test DivergenceDetector against real training logs from the pretrain project."""

    LOG_BASE = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "pretrain",
        "small-llm",
        "output_models",
        "sllm",
        "runs",
    )

    DIVERGED_RUN = "lr_5e-4_2026-03-16T18-16-52"
    GOOD_RUNS = [
        "lr_1e-4_2026-03-16T15-50-55",
        "log_2026-03-16T08-05-07",
    ]

    @staticmethod
    def _load_train_losses(log_path):
        """Load (step, loss) pairs from a trainer_logs.json file."""
        with open(log_path) as f:
            entries = json.load(f)
        return [
            (e["global_step"], e["loss"])
            for e in entries
            if "loss" in e and "eval_loss" not in e
        ]

    @staticmethod
    def _simulate(losses, **detector_kwargs):
        """Run detector on loss sequence, return step where triggered or None."""
        detector = DivergenceDetector(**detector_kwargs)
        control = MockControl()
        for step, loss in losses:
            detector.on_log(None, None, control, logs={"loss": loss})
            if control.should_training_stop:
                return step
        return None

    @pytest.fixture
    def diverged_losses(self):
        log_path = os.path.join(self.LOG_BASE, self.DIVERGED_RUN, "trainer_logs.json")
        if not os.path.exists(log_path):
            pytest.skip(f"Training log not found: {log_path}")
        return self._load_train_losses(log_path)

    @pytest.fixture
    def good_losses_list(self):
        result = []
        for run in self.GOOD_RUNS:
            log_path = os.path.join(self.LOG_BASE, run, "trainer_logs.json")
            if not os.path.exists(log_path):
                continue
            result.append((run, self._load_train_losses(log_path)))
        if not result:
            pytest.skip("No good training logs found")
        return result

    def test_detects_diverged_run_default_params(self, diverged_losses):
        """Default parameters should catch the divergence within ~200 steps of spike."""
        # Spike happens at step 3552
        step = self._simulate(diverged_losses)
        assert step is not None, "Should detect divergence in the diverged run"
        assert (
            step <= 3800
        ), f"Should detect within ~250 steps of spike (step 3552), got step {step}"

    def test_no_false_positive_good_runs(self, good_losses_list):
        """Default parameters should not trigger on healthy training runs."""
        for run_name, losses in good_losses_list:
            step = self._simulate(losses)
            assert step is None, f"False positive on good run {run_name} at step {step}"

    @pytest.mark.parametrize(
        "smoothing,threshold,patience",
        [
            (0.1, 1.0, 3),
            (0.2, 1.0, 3),
            (0.3, 1.0, 3),
            (0.3, 1.5, 3),
            (0.3, 1.0, 5),
        ],
    )
    def test_various_params_detect_divergence(
        self, diverged_losses, smoothing, threshold, patience
    ):
        """Multiple parameter combinations should all detect the divergence."""
        step = self._simulate(
            diverged_losses, smoothing=smoothing, threshold=threshold, patience=patience
        )
        assert (
            step is not None
        ), f"Failed to detect with smoothing={smoothing}, threshold={threshold}, patience={patience}"
        # Should detect within 500 steps of the spike at 3552
        assert step <= 4050, f"Detection too late at step {step}"

    @pytest.mark.parametrize(
        "smoothing,threshold,patience",
        [
            (0.1, 1.0, 3),
            (0.2, 1.0, 3),
            (0.3, 1.0, 3),
            (0.3, 1.5, 3),
            (0.3, 1.0, 5),
        ],
    )
    def test_various_params_no_false_positive(
        self, good_losses_list, smoothing, threshold, patience
    ):
        """Multiple parameter combinations should produce no false positives."""
        for run_name, losses in good_losses_list:
            step = self._simulate(
                losses, smoothing=smoothing, threshold=threshold, patience=patience
            )
            assert step is None, (
                f"False positive on {run_name} at step {step} with "
                f"smoothing={smoothing}, threshold={threshold}, patience={patience}"
            )

    def test_relative_threshold_detects_divergence(self, diverged_losses):
        """Relative threshold should also detect the divergence."""
        step = self._simulate(
            diverged_losses,
            smoothing=0.3,
            threshold=None,
            relative_threshold=1.5,
            patience=3,
        )
        assert step is not None, "Relative threshold should detect divergence"

    def test_relative_threshold_no_false_positive(self, good_losses_list):
        """Relative threshold should not trigger on healthy runs."""
        for run_name, losses in good_losses_list:
            step = self._simulate(
                losses,
                smoothing=0.3,
                threshold=None,
                relative_threshold=1.5,
                patience=3,
            )
            assert (
                step is None
            ), f"Relative threshold false positive on {run_name} at step {step}"


class TestCheckpointPreservation:
    """Tests for checkpoint preservation logic."""

    def test_best_checkpoints_tracking(self):
        """Test tracking N best checkpoints."""
        best_checkpoints = []
        preserve_n_best = 3

        checkpoints = [
            ("checkpoint-1", 2.5),
            ("checkpoint-2", 2.3),
            ("checkpoint-3", 2.7),
            ("checkpoint-4", 2.1),  # Best so far
            ("checkpoint-5", 2.4),
            ("checkpoint-6", 1.9),  # New best
        ]

        for path, metric in checkpoints:
            # Determine if this should be preserved
            is_best = False

            if len(best_checkpoints) < preserve_n_best:
                is_best = True
            else:
                worst_best = max(best_checkpoints, key=lambda x: x[1])
                is_best = metric < worst_best[1]

            if is_best:
                best_checkpoints.append((path, metric))
                best_checkpoints.sort(key=lambda x: x[1])
                best_checkpoints = best_checkpoints[:preserve_n_best]

        # Verify we kept the 3 best
        expected = [
            ("checkpoint-6", 1.9),
            ("checkpoint-4", 2.1),
            ("checkpoint-2", 2.3),
        ]
        assert best_checkpoints == expected

    def test_greater_is_better_tracking(self):
        """Test tracking with greater_is_better=True (e.g., accuracy)."""
        best_checkpoints = []
        preserve_n_best = 3

        checkpoints = [
            ("checkpoint-1", 0.85),
            ("checkpoint-2", 0.87),
            ("checkpoint-3", 0.83),
            ("checkpoint-4", 0.89),  # Best so far
            ("checkpoint-5", 0.86),
            ("checkpoint-6", 0.91),  # New best
        ]

        for path, metric in checkpoints:
            is_best = False

            if len(best_checkpoints) < preserve_n_best:
                is_best = True
            else:
                worst_best = min(
                    best_checkpoints, key=lambda x: x[1]
                )  # Min for greater_is_better
                is_best = metric > worst_best[1]

            if is_best:
                best_checkpoints.append((path, metric))
                best_checkpoints.sort(key=lambda x: x[1], reverse=True)  # Descending
                best_checkpoints = best_checkpoints[:preserve_n_best]

        # Verify we kept the 3 best
        expected = [
            ("checkpoint-6", 0.91),
            ("checkpoint-4", 0.89),
            ("checkpoint-2", 0.87),
        ]
        assert best_checkpoints == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
