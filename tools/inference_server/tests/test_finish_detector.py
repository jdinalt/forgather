"""
Unit tests for FinishReasonDetector.
"""

import pytest
from unittest.mock import Mock
from ..core.finish_detector import FinishReasonDetector


class TestFinishReasonDetector:
    """Test cases for finish reason detection."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.eos_token_id = 2
        return tokenizer

    @pytest.fixture
    def detector(self, mock_tokenizer):
        """Create a detector instance."""
        stop_token_ids = {2, 50256, 50257}  # EOS and custom stops
        return FinishReasonDetector(mock_tokenizer, stop_token_ids)

    def test_finish_reason_length(self, detector):
        """Test finish reason when max tokens reached."""
        token_ids = [1, 2, 3, 4, 5]
        max_tokens = 5
        stopped_by_sequence = False

        reason = detector.determine_finish_reason(
            token_ids, max_tokens, stopped_by_sequence
        )

        assert reason == "length"

    def test_finish_reason_stop_sequence(self, detector):
        """Test finish reason when stopped by custom sequence."""
        token_ids = [1, 2, 3]
        max_tokens = 10
        stopped_by_sequence = True

        reason = detector.determine_finish_reason(
            token_ids, max_tokens, stopped_by_sequence
        )

        assert reason == "stop"

    def test_finish_reason_eos_token(self, detector):
        """Test finish reason when EOS token is last."""
        token_ids = [1, 3, 4, 2]  # Ends with EOS token (2)
        max_tokens = 10
        stopped_by_sequence = False

        reason = detector.determine_finish_reason(
            token_ids, max_tokens, stopped_by_sequence
        )

        assert reason == "stop"

    def test_finish_reason_custom_stop_token(self, detector):
        """Test finish reason when custom stop token is last."""
        token_ids = [1, 3, 4, 50256]  # Ends with custom stop token
        max_tokens = 10
        stopped_by_sequence = False

        reason = detector.determine_finish_reason(
            token_ids, max_tokens, stopped_by_sequence
        )

        assert reason == "stop"

    def test_finish_reason_early_stop(self, detector):
        """Test finish reason when stopped early without obvious reason."""
        token_ids = [1, 3, 4]
        max_tokens = 10
        stopped_by_sequence = False

        reason = detector.determine_finish_reason(
            token_ids, max_tokens, stopped_by_sequence
        )

        assert reason == "stop"

    def test_finish_reason_empty_tokens(self, detector):
        """Test finish reason with empty token list."""
        token_ids = []
        max_tokens = 10
        stopped_by_sequence = False

        reason = detector.determine_finish_reason(
            token_ids, max_tokens, stopped_by_sequence
        )

        assert reason == "stop"

    def test_finish_reason_exact_max_tokens(self, detector):
        """Test when token count exactly matches max."""
        token_ids = [1, 2, 3, 4, 5]
        max_tokens = 5
        stopped_by_sequence = False

        reason = detector.determine_finish_reason(
            token_ids, max_tokens, stopped_by_sequence
        )

        assert reason == "length"

    def test_finish_reason_exceeds_max_tokens(self, detector):
        """Test when token count exceeds max (should be length)."""
        token_ids = [1, 2, 3, 4, 5, 6]
        max_tokens = 5
        stopped_by_sequence = False

        reason = detector.determine_finish_reason(
            token_ids, max_tokens, stopped_by_sequence
        )

        assert reason == "length"

    def test_streaming_finish_reason_length(self, detector):
        """Test streaming finish reason when max tokens reached."""
        completion_tokens = 100
        max_tokens = 100
        stop_sequences = ["STOP"]
        full_response = "This is a response"

        reason = detector.determine_finish_reason_streaming(
            completion_tokens, max_tokens, stop_sequences, full_response
        )

        assert reason == "length"

    def test_streaming_finish_reason_stop_sequence(self, detector):
        """Test streaming finish reason with stop sequence."""
        completion_tokens = 50
        max_tokens = 100
        stop_sequences = ["STOP"]
        full_response = "This is STOP a response"

        reason = detector.determine_finish_reason_streaming(
            completion_tokens, max_tokens, stop_sequences, full_response
        )

        assert reason == "stop"

    def test_streaming_finish_reason_no_stop(self, detector):
        """Test streaming finish reason with no specific stop."""
        completion_tokens = 50
        max_tokens = 100
        stop_sequences = ["STOP"]
        full_response = "This is a normal response"

        reason = detector.determine_finish_reason_streaming(
            completion_tokens, max_tokens, stop_sequences, full_response
        )

        assert reason == "stop"

    def test_priority_length_over_sequence(self, detector):
        """Test that length takes priority when both conditions met."""
        token_ids = [1, 2, 3, 4, 5]
        max_tokens = 5
        stopped_by_sequence = True  # Both length and sequence

        reason = detector.determine_finish_reason(
            token_ids, max_tokens, stopped_by_sequence
        )

        # Length should take priority
        assert reason == "length"

    def test_eos_not_at_end(self, detector):
        """Test that EOS in middle doesn't trigger stop."""
        token_ids = [1, 2, 3, 4]  # EOS (2) not at end
        max_tokens = 10
        stopped_by_sequence = False

        reason = detector.determine_finish_reason(
            token_ids, max_tokens, stopped_by_sequence
        )

        assert reason == "stop"
