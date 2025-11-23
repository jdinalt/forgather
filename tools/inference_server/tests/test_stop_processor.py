"""
Unit tests for StopSequenceProcessor.
"""

import pytest
import torch
from unittest.mock import Mock
from ..core.stop_processor import StopSequenceProcessor


class TestStopSequenceProcessor:
    """Test cases for stop sequence processing."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(
            side_effect=lambda text, **kwargs: list(range(len(text)))
        )
        return tokenizer

    @pytest.fixture
    def processor(self, mock_tokenizer):
        """Create a processor instance."""
        return StopSequenceProcessor(mock_tokenizer)

    def test_no_stop_sequence_found(self, processor):
        """Test when no stop sequence is present."""
        text = "This is a test response"
        token_ids = [1, 2, 3, 4, 5]
        tokens = torch.tensor(token_ids)
        stop_sequences = ["STOP", "END"]

        result_ids, result_tokens, stopped, stop_seq = processor.process(
            text, token_ids, tokens, stop_sequences
        )

        assert result_ids == token_ids
        assert torch.equal(result_tokens, tokens)
        assert stopped is False
        assert stop_seq is None

    def test_stop_sequence_found(self, processor, mock_tokenizer):
        """Test when stop sequence is found."""
        text = "This is a test STOP more text"
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        tokens = torch.tensor(token_ids)
        stop_sequences = ["STOP"]

        # Mock encode to return fewer tokens for trimmed text
        mock_tokenizer.encode.return_value = [1, 2, 3, 4]

        result_ids, result_tokens, stopped, stop_seq = processor.process(
            text, token_ids, tokens, stop_sequences
        )

        assert len(result_ids) == 4
        assert stopped is True
        assert stop_seq == "STOP"

    def test_multiple_stop_sequences_first_wins(self, processor, mock_tokenizer):
        """Test when multiple stop sequences present, first one wins."""
        text = "Start ALPHA content BETA end"
        token_ids = [1, 2, 3, 4, 5, 6, 7]
        tokens = torch.tensor(token_ids)
        stop_sequences = ["ALPHA", "BETA"]

        mock_tokenizer.encode.return_value = [1, 2]

        result_ids, result_tokens, stopped, stop_seq = processor.process(
            text, token_ids, tokens, stop_sequences
        )

        assert stopped is True
        assert stop_seq == "ALPHA"  # First in the list

    def test_process_streaming_no_stop(self, processor):
        """Test streaming with no stop sequence."""
        full_response = "This is ongoing"
        new_text = " text"
        stop_sequences = ["STOP"]

        should_stop, remaining, stop_seq = processor.process_streaming(
            full_response, new_text, stop_sequences
        )

        assert should_stop is False
        assert remaining is None
        assert stop_seq is None

    def test_process_streaming_with_stop(self, processor):
        """Test streaming when stop sequence is found."""
        full_response = "This is STOP extra"
        new_text = "P extra"
        stop_sequences = ["STOP"]

        should_stop, remaining, stop_seq = processor.process_streaming(
            full_response, new_text, stop_sequences
        )

        assert should_stop is True
        assert stop_seq == "STOP"
        # Remaining text should be the part of new_text before the stop
        assert remaining == " extra" or remaining is not None

    def test_stop_at_beginning(self, processor, mock_tokenizer):
        """Test when stop sequence is at the very beginning."""
        text = "STOP everything after"
        token_ids = [1, 2, 3]
        tokens = torch.tensor(token_ids)
        stop_sequences = ["STOP"]

        mock_tokenizer.encode.return_value = []

        result_ids, result_tokens, stopped, stop_seq = processor.process(
            text, token_ids, tokens, stop_sequences
        )

        assert stopped is True
        assert stop_seq == "STOP"

    def test_empty_stop_sequences(self, processor):
        """Test with empty stop sequences list."""
        text = "This is a test"
        token_ids = [1, 2, 3, 4]
        tokens = torch.tensor(token_ids)
        stop_sequences = []

        result_ids, result_tokens, stopped, stop_seq = processor.process(
            text, token_ids, tokens, stop_sequences
        )

        assert result_ids == token_ids
        assert stopped is False
        assert stop_seq is None

    def test_device_preservation(self, processor, mock_tokenizer):
        """Test that tensor device is preserved."""
        device = torch.device("cpu")
        text = "Test STOP"
        token_ids = [1, 2, 3]
        tokens = torch.tensor(token_ids, device=device)
        stop_sequences = ["STOP"]

        mock_tokenizer.encode.return_value = [1]

        result_ids, result_tokens, stopped, stop_seq = processor.process(
            text, token_ids, tokens, stop_sequences
        )

        assert result_tokens.device == device
