"""
Unit tests for GenerationLogger.
"""

import pytest
import logging
from unittest.mock import Mock, call
from ..core.generation_logger import GenerationLogger


class TestGenerationLogger:
    """Test cases for generation logger."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock(spec=logging.Logger)

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.decode = Mock(return_value="decoded text")
        return tokenizer

    @pytest.fixture
    def gen_logger(self, mock_logger, mock_tokenizer):
        """Create a generation logger instance."""
        return GenerationLogger(mock_logger, mock_tokenizer)

    def test_log_request_basic(self, gen_logger, mock_logger):
        """Test basic request logging."""
        gen_logger.log_request(
            request_id="req-123",
            request_type="chat",
            model="test-model",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )

        # Verify logger.info was called
        assert mock_logger.info.called
        call_args = mock_logger.info.call_args[0][0]
        assert "req-123" in call_args
        assert "chat" in call_args
        assert "test-model" in call_args

    def test_log_request_with_kwargs(self, gen_logger, mock_logger):
        """Test request logging with additional kwargs."""
        gen_logger.log_request(
            request_id="req-123",
            request_type="completion",
            model="test-model",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            custom_param="custom_value",
            another_param=42,
        )

        call_args = mock_logger.info.call_args[0][0]
        assert "custom_param=custom_value" in call_args
        assert "another_param=42" in call_args

    def test_log_prompt(self, gen_logger, mock_logger):
        """Test prompt logging."""
        gen_logger.log_prompt("req-123", "Test prompt text")

        assert mock_logger.info.called
        call_args = mock_logger.info.call_args[0][0]
        assert "req-123" in call_args
        assert "Test prompt text" in call_args

    def test_log_input_tokens(self, gen_logger, mock_logger, mock_tokenizer):
        """Test input token logging."""
        token_ids = [1, 2, 3, 4, 5]

        gen_logger.log_input_tokens("req-123", token_ids)

        # Should be called twice: once for IDs, once for decoded
        assert mock_logger.info.call_count == 2

        # Check first call has token IDs
        first_call = mock_logger.info.call_args_list[0][0][0]
        assert "req-123" in first_call
        assert "[1, 2, 3, 4, 5]" in first_call

        # Check tokenizer.decode was called
        mock_tokenizer.decode.assert_called_once_with(
            token_ids, skip_special_tokens=False
        )

    def test_log_generation_config(self, gen_logger, mock_logger):
        """Test generation config logging."""
        config = Mock()
        config.__repr__ = Mock(return_value="GenerationConfig(...)")

        gen_logger.log_generation_config("req-123", config)

        assert mock_logger.info.called
        call_args = mock_logger.info.call_args[0][0]
        assert "req-123" in call_args

    def test_log_stop_strings(self, gen_logger, mock_logger):
        """Test stop strings logging."""
        stop_strings = ["STOP", "END"]

        gen_logger.log_stop_strings("req-123", stop_strings)

        assert mock_logger.info.called
        call_args = mock_logger.info.call_args[0][0]
        assert "req-123" in call_args
        assert "STOP" in call_args
        assert "END" in call_args

    def test_log_generated_tokens(self, gen_logger, mock_logger, mock_tokenizer):
        """Test generated token logging."""
        token_ids = [10, 20, 30]

        gen_logger.log_generated_tokens("req-123", token_ids)

        # Should be called twice
        assert mock_logger.info.call_count == 2

        # Verify decode was called
        mock_tokenizer.decode.assert_called_once_with(
            token_ids, skip_special_tokens=False
        )

    def test_log_response(self, gen_logger, mock_logger):
        """Test response logging."""
        gen_logger.log_response(
            request_id="req-123",
            response_text="This is the response",
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=25,
        )

        # Should be called 3 times: response text, finish reason, token usage
        assert mock_logger.info.call_count == 3

        # Check calls contain expected info
        all_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("This is the response" in call for call in all_calls)
        assert any("stop" in call for call in all_calls)
        assert any("50" in call and "25" in call and "75" in call for call in all_calls)

    def test_log_stop_sequence_triggered(self, gen_logger, mock_logger):
        """Test stop sequence trigger logging."""
        gen_logger.log_stop_sequence_triggered("req-123", "STOP")

        assert mock_logger.info.called
        call_args = mock_logger.info.call_args[0][0]
        assert "req-123" in call_args
        assert "STOP" in call_args
        assert "stop sequence" in call_args.lower()

    def test_log_eos_token(self, gen_logger, mock_logger):
        """Test EOS token logging."""
        gen_logger.log_eos_token("req-123")

        assert mock_logger.info.called
        call_args = mock_logger.info.call_args[0][0]
        assert "req-123" in call_args
        assert "EOS" in call_args

    def test_log_stop_token(self, gen_logger, mock_logger):
        """Test stop token logging."""
        gen_logger.log_stop_token("req-123", 50256)

        assert mock_logger.info.called
        call_args = mock_logger.info.call_args[0][0]
        assert "req-123" in call_args
        assert "50256" in call_args

    def test_log_streaming_error(self, gen_logger, mock_logger):
        """Test streaming error logging."""
        error = Exception("Test error")

        gen_logger.log_streaming_error("req-123", error)

        assert mock_logger.error.called
        call_args = mock_logger.error.call_args[0][0]
        assert "req-123" in call_args
        assert "Test error" in call_args

    def test_log_messages(self, gen_logger, mock_logger):
        """Test chat message logging."""
        messages = [
            Mock(role="user", content="Hello"),
            Mock(role="assistant", content="Hi there"),
        ]

        gen_logger.log_messages("req-123", messages)

        # Should be called once for each message
        assert mock_logger.info.call_count == 2

        # Check calls contain message info
        first_call = mock_logger.info.call_args_list[0][0][0]
        assert "req-123" in first_call
        assert "user" in first_call
        assert "Hello" in first_call

        second_call = mock_logger.info.call_args_list[1][0][0]
        assert "assistant" in second_call
        assert "Hi there" in second_call

    def test_empty_messages_list(self, gen_logger, mock_logger):
        """Test logging with empty messages list."""
        gen_logger.log_messages("req-123", [])

        # Should not be called for empty list
        assert mock_logger.info.call_count == 0

    def test_multiple_request_ids(self, gen_logger, mock_logger):
        """Test that different request IDs are logged correctly."""
        gen_logger.log_prompt("req-1", "Prompt 1")
        gen_logger.log_prompt("req-2", "Prompt 2")

        calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("req-1" in call and "Prompt 1" in call for call in calls)
        assert any("req-2" in call and "Prompt 2" in call for call in calls)
