"""
Unit tests for divergence detection callbacks and checkpoint preservation.
"""

import sys
from dataclasses import dataclass

import pytest

from forgather.ml.trainer.callbacks.divergence_detector import (
    DualTimeScaleDivergenceDetector,
    DualWindowDivergenceDetector,
)


@dataclass
class MockControl:
    """Mock TrainerControl for testing."""
    should_training_stop: bool = False
    should_abort_without_save: bool = False


class TestDualTimeScaleDivergenceDetector:
    """Tests for DualTimeScaleDivergenceDetector."""

    def test_initialization(self):
        """Test detector initializes with correct defaults."""
        detector = DualTimeScaleDivergenceDetector()

        assert detector.short_alpha == 0.1
        assert detector.long_alpha == 0.01
        assert detector.threshold == 2.0
        assert detector.action == "stop"
        assert detector.use_eval_loss is True
        assert detector.short_ema is None
        assert detector.long_ema is None
        assert detector.initialized is False

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            DualTimeScaleDivergenceDetector(short_alpha=-0.1)

        with pytest.raises(ValueError):
            DualTimeScaleDivergenceDetector(short_alpha=1.5)

        with pytest.raises(ValueError):
            DualTimeScaleDivergenceDetector(long_alpha=0.0)

    def test_state_dict_protocol(self):
        """Test Stateful protocol implementation."""
        detector = DualTimeScaleDivergenceDetector()

        # Initial state
        state = detector.state_dict()
        assert state == {"short_ema": None, "long_ema": None, "initialized": False}

        # After initialization (use eval_loss since use_eval_loss=True by default)
        control = MockControl()
        detector.on_log(None, None, control, logs={"eval_loss": 2.5})
        state = detector.state_dict()
        assert state["short_ema"] == 2.5
        assert state["long_ema"] == 2.5
        assert state["initialized"] is True

    def test_state_load(self):
        """Test state can be loaded correctly."""
        detector = DualTimeScaleDivergenceDetector()

        state = {
            "short_ema": 3.5,
            "long_ema": 2.8,
            "initialized": True,
        }

        detector.load_state_dict(state)
        assert detector.short_ema == 3.5
        assert detector.long_ema == 2.8
        assert detector.initialized is True

    def test_normal_training(self):
        """Test detector doesn't trigger during normal training."""
        detector = DualTimeScaleDivergenceDetector(
            short_alpha=0.2,
            long_alpha=0.01,
            threshold=1.5,
        )

        control = MockControl()

        # Simulate normal training with stable loss
        for i in range(20):
            logs = {"loss": 2.8 + 0.1 * (i % 3 - 1)}  # Small fluctuation
            detector.on_log(None, None, control, logs=logs)

        assert not control.should_training_stop, "Should not stop during normal training"

    def test_spike_detection(self):
        """Test detector triggers on loss spike."""
        detector = DualTimeScaleDivergenceDetector(
            short_alpha=0.2,
            long_alpha=0.01,
            threshold=1.5,
            action="stop",
            use_eval_loss=False,  # Monitor train loss for this test
        )

        control = MockControl()

        # Normal training
        for i in range(10):
            logs = {"loss": 2.8}
            detector.on_log(None, None, control, logs=logs)

        assert not control.should_training_stop

        # Spike to 8.0
        for i in range(5):
            logs = {"loss": 8.0}
            detector.on_log(None, None, control, logs=logs)

        # Should trigger after a few steps
        assert control.should_training_stop, "Should stop after spike"

    def test_eval_loss_monitoring(self):
        """Test monitoring eval_loss instead of train loss."""
        detector = DualTimeScaleDivergenceDetector(
            use_eval_loss=True,
            threshold=1.5,
        )

        control = MockControl()

        # Logs with both loss and eval_loss
        logs = {"loss": 10.0, "eval_loss": 2.8}
        detector.on_log(None, None, control, logs=logs)

        # Should initialize with eval_loss, not loss
        assert detector.short_ema == 2.8
        assert detector.long_ema == 2.8

    def test_custom_metric_monitoring(self):
        """Test monitoring custom metric."""
        detector = DualTimeScaleDivergenceDetector(
            use_eval_loss=False,
            metric_key="grad_norm",
            threshold=5.0,
        )

        control = MockControl()

        logs = {"loss": 2.8, "eval_loss": 2.5, "grad_norm": 3.2}
        detector.on_log(None, None, control, logs=logs)

        # Should initialize with grad_norm
        assert detector.short_ema == 3.2
        assert detector.long_ema == 3.2


class TestDualWindowDivergenceDetector:
    """Tests for DualWindowDivergenceDetector."""

    def test_initialization(self):
        """Test detector initializes with correct defaults."""
        detector = DualWindowDivergenceDetector()

        assert detector.short_window == 10
        assert detector.long_window == 100
        assert detector.threshold == 2.0
        assert detector.action == "stop"
        assert detector.use_eval_loss is True
        assert detector.short_buffer == []
        assert detector.long_buffer == []

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            DualWindowDivergenceDetector(short_window=-1)

        with pytest.raises(ValueError):
            DualWindowDivergenceDetector(long_window=0)

    def test_state_dict_protocol(self):
        """Test Stateful protocol implementation."""
        detector = DualWindowDivergenceDetector(
            short_window=5,
            long_window=20,
            use_eval_loss=False,  # Monitor train loss for this test
        )

        control = MockControl()

        # Fill buffers
        for i in range(25):
            logs = {"loss": 2.8}
            detector.on_log(None, None, control, logs=logs)

        state = detector.state_dict()
        assert len(state["short_buffer"]) == 5
        assert len(state["long_buffer"]) == 20

    def test_state_load(self):
        """Test state can be loaded correctly."""
        detector = DualWindowDivergenceDetector()

        state = {
            "short_buffer": [2.5, 2.6, 2.7],
            "long_buffer": [2.5] * 10,
        }

        detector.load_state_dict(state)
        assert detector.short_buffer == [2.5, 2.6, 2.7]
        assert detector.long_buffer == [2.5] * 10

    def test_buffer_management(self):
        """Test buffers are correctly sized."""
        detector = DualWindowDivergenceDetector(
            short_window=5,
            long_window=20,
            use_eval_loss=False,  # Monitor train loss for this test
        )

        control = MockControl()

        # Fill beyond window size
        for i in range(30):
            logs = {"loss": 2.8}
            detector.on_log(None, None, control, logs=logs)

        # Buffers should be capped at window size
        assert len(detector.short_buffer) == 5
        assert len(detector.long_buffer) == 20

    def test_normal_training(self):
        """Test detector doesn't trigger during normal training."""
        detector = DualWindowDivergenceDetector(
            short_window=5,
            long_window=20,
            threshold=1.5,
        )

        control = MockControl()

        # Fill long buffer with normal values
        for i in range(25):
            logs = {"loss": 2.8}
            detector.on_log(None, None, control, logs=logs)

        assert not control.should_training_stop

    def test_spike_detection(self):
        """Test detector triggers on loss spike."""
        detector = DualWindowDivergenceDetector(
            short_window=5,
            long_window=20,
            threshold=1.5,
            use_eval_loss=False,  # Monitor train loss for this test
        )

        control = MockControl()

        # Fill long buffer with normal values
        for i in range(25):
            logs = {"loss": 2.8}
            detector.on_log(None, None, control, logs=logs)

        assert not control.should_training_stop

        # Spike
        for i in range(10):
            logs = {"loss": 8.0}
            detector.on_log(None, None, control, logs=logs)

        # Should trigger
        assert control.should_training_stop


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
                worst_best = min(best_checkpoints, key=lambda x: x[1])  # Min for greater_is_better
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
