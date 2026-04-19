"""
Integration tests for generation strategies.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from ..models.chat import ChatCompletionRequest, ChatMessage
from ..models.completion import CompletionRequest
from ..strategies import (
    ChatGenerationStrategy,
    CompletionGenerationStrategy,
    StreamingChatStrategy,
    StreamingCompletionStrategy,
)


class TestChatGenerationStrategy:
    """Integration tests for ChatGenerationStrategy."""

    @pytest.fixture
    def mock_service(self):
        """Create a mock inference service."""
        service = Mock()

        # Mock tokenizer
        service.tokenizer = Mock()
        service.tokenizer.decode = Mock(return_value="Generated response")
        service.tokenizer.eos_token_id = 2
        service.tokenizer.bos_token = "<s>"
        service.tokenizer.eos_token = "</s>"

        # Mock logger
        service.logger = Mock()
        service.logger.log_request = Mock()
        service.logger.log_messages = Mock()
        service.logger.log_prompt = Mock()
        service.logger.log_input_tokens = Mock()
        service.logger.log_generation_config = Mock()
        service.logger.log_stop_strings = Mock()
        service.logger.log_generated_tokens = Mock()
        service.logger.log_response = Mock()
        service.logger.log_stop_sequence_triggered = Mock()

        # Mock model
        service.model = Mock()
        mock_output = Mock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        service.model.generate = Mock(return_value=mock_output)

        # Mock utilities
        service.tokenizer_wrapper = Mock()
        service.tokenizer_wrapper.tokenize_and_move_to_device = Mock(
            return_value={
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "prompt_tokens": 5,
            }
        )

        service.stop_processor = Mock()
        service.stop_processor.process = Mock(
            return_value=(
                [6, 7, 8],  # token_ids
                torch.tensor([6, 7, 8]),  # tokens
                False,  # stopped_by_sequence
                None,  # stop_sequence_found
            )
        )

        service.finish_detector = Mock()
        service.finish_detector.determine_finish_reason = Mock(return_value="stop")

        service.stop_sequences = []
        service.format_messages = Mock(return_value="User: Hello\n\nAssistant: ")
        service._build_generation_config = Mock(return_value=Mock())

        return service

    def test_chat_generation_basic(self, mock_service):
        """Test basic chat generation."""
        strategy = ChatGenerationStrategy(mock_service)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )

        response = strategy.generate(request)

        # Verify response structure
        assert response.id.startswith("chatcmpl-")
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert response.usage.prompt_tokens == 5
        assert response.usage.completion_tokens == 3  # len([6, 7, 8])

        # Verify service methods were called
        mock_service.format_messages.assert_called_once()
        mock_service.model.generate.assert_called_once()
        mock_service.stop_processor.process.assert_called_once()
        mock_service.finish_detector.determine_finish_reason.assert_called_once()

    def test_chat_generation_with_stop_sequence(self, mock_service):
        """Test chat generation with stop sequence."""
        # Configure stop processor to return stopped=True
        mock_service.stop_processor.process = Mock(
            return_value=(
                [6, 7],  # trimmed token_ids
                torch.tensor([6, 7]),  # trimmed tokens
                True,  # stopped_by_sequence
                "STOP",  # stop_sequence_found
            )
        )

        strategy = ChatGenerationStrategy(mock_service)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=100,
        )

        response = strategy.generate(request)

        # Verify stop sequence was logged
        mock_service.logger.log_stop_sequence_triggered.assert_called_once_with(
            response.id, "STOP"
        )

    def test_chat_generation_logging(self, mock_service):
        """Test that all logging methods are called."""
        strategy = ChatGenerationStrategy(mock_service)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=100,
        )

        response = strategy.generate(request)

        # Verify all logging calls
        mock_service.logger.log_request.assert_called_once()
        mock_service.logger.log_messages.assert_called_once()
        mock_service.logger.log_prompt.assert_called_once()
        mock_service.logger.log_input_tokens.assert_called_once()
        mock_service.logger.log_generation_config.assert_called_once()
        mock_service.logger.log_stop_strings.assert_called_once()
        mock_service.logger.log_generated_tokens.assert_called_once()
        mock_service.logger.log_response.assert_called_once()


class TestCompletionGenerationStrategy:
    """Integration tests for CompletionGenerationStrategy."""

    @pytest.fixture
    def mock_service(self):
        """Create a mock inference service."""
        service = Mock()

        # Mock tokenizer
        service.tokenizer = Mock()
        service.tokenizer.decode = Mock(return_value="completed text")
        service.tokenizer.eos_token_id = 2

        # Mock logger
        service.logger = Mock()
        service.logger.log_request = Mock()
        service.logger.log_prompt = Mock()
        service.logger.log_input_tokens = Mock()
        service.logger.log_generation_config = Mock()
        service.logger.log_stop_strings = Mock()
        service.logger.log_generated_tokens = Mock()
        service.logger.log_response = Mock()
        service.logger.log_stop_sequence_triggered = Mock()

        # Mock model
        service.model = Mock()
        mock_output = Mock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        service.model.generate = Mock(return_value=mock_output)

        # Mock utilities
        service.tokenizer_wrapper = Mock()
        service.tokenizer_wrapper.tokenize_and_move_to_device = Mock(
            return_value={
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "prompt_tokens": 5,
            }
        )

        service.stop_processor = Mock()
        service.stop_processor.process = Mock(
            return_value=([6, 7, 8], torch.tensor([6, 7, 8]), False, None)
        )

        service.finish_detector = Mock()
        service.finish_detector.determine_finish_reason = Mock(return_value="stop")

        service.stop_sequences = []
        service._build_generation_config = Mock(return_value=Mock())

        return service

    def test_completion_generation_basic(self, mock_service):
        """Test basic completion generation."""
        strategy = CompletionGenerationStrategy(mock_service)

        request = CompletionRequest(
            model="test-model", prompt="Once upon a time", max_tokens=50
        )

        response = strategy.generate(request)

        # Verify response structure
        assert response.id.startswith("cmpl-")
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.choices[0].text == "completed text"
        assert response.usage.prompt_tokens == 5

    def test_completion_with_echo(self, mock_service):
        """Test completion with echo parameter."""
        strategy = CompletionGenerationStrategy(mock_service)

        request = CompletionRequest(
            model="test-model", prompt="Test prompt", max_tokens=50, echo=True
        )

        response = strategy.generate(request)

        # Verify prompt was echoed (should be "Test prompt" + "completed text")
        assert "Test prompt" in response.choices[0].text

    def test_completion_with_stop_sequences(self, mock_service):
        """Test completion with request stop sequences."""
        strategy = CompletionGenerationStrategy(mock_service)

        request = CompletionRequest(
            model="test-model", prompt="Test", max_tokens=50, stop=["STOP", "END"]
        )

        response = strategy.generate(request)

        # Verify response was generated
        assert response is not None
        assert response.choices[0].text == "completed text"

    def test_completion_with_list_prompt(self, mock_service):
        """Test completion with list prompt."""
        strategy = CompletionGenerationStrategy(mock_service)

        request = CompletionRequest(
            model="test-model", prompt=["Single prompt"], max_tokens=50
        )

        response = strategy.generate(request)

        # Should handle list with single element
        assert response is not None

    def test_completion_logging(self, mock_service):
        """Test that all logging methods are called."""
        strategy = CompletionGenerationStrategy(mock_service)

        request = CompletionRequest(model="test-model", prompt="Test", max_tokens=50)

        response = strategy.generate(request)

        # Verify logging calls
        mock_service.logger.log_request.assert_called_once()
        mock_service.logger.log_prompt.assert_called_once()
        mock_service.logger.log_response.assert_called_once()


class TestStreamingStrategies:
    """Integration tests for streaming strategies."""

    @pytest.fixture
    def mock_service(self):
        """Create a mock inference service for streaming."""
        service = Mock()

        # Mock tokenizer
        service.tokenizer = Mock()
        service.tokenizer.encode = Mock(return_value=[1, 2, 3])
        service.tokenizer.bos_token = "<s>"
        service.tokenizer.eos_token = "</s>"

        # Mock logger
        service.logger = Mock()
        service.logger.log_request = Mock()
        service.logger.log_messages = Mock()
        service.logger.log_prompt = Mock()
        service.logger.log_input_tokens = Mock()
        service.logger.log_generation_config = Mock()
        service.logger.log_generated_tokens = Mock()
        service.logger.log_response = Mock()
        service.logger.log_stop_sequence_triggered = Mock()

        # Mock tokenizer wrapper
        service.tokenizer_wrapper = Mock()
        service.tokenizer_wrapper.get_device = Mock(return_value=torch.device("cpu"))

        # Mock stop processor for streaming
        service.stop_processor = Mock()
        service.stop_processor.process_streaming = Mock(
            return_value=(False, None, None)  # no stop
        )

        service.finish_detector = Mock()
        service.finish_detector.determine_finish_reason_streaming = Mock(
            return_value="stop"
        )

        service.stop_sequences = []
        service.jinja_env = Mock()
        service.chat_template = "{{ messages }}"
        service._build_generation_config = Mock(return_value=Mock())

        return service

    def test_streaming_chat_basic(self, mock_service):
        """Test basic streaming chat generation."""
        # Mock the template rendering
        template_mock = Mock()
        template_mock.render = Mock(return_value="User: Hello\n\nAssistant: ")
        mock_service.jinja_env.from_string = Mock(return_value=template_mock)

        strategy = StreamingChatStrategy(mock_service)

        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=100,
        )

        # Mock the streamer to yield some text
        with patch(
            "..strategies.streaming_chat.TextIteratorStreamer"
        ) as mock_streamer_class:
            mock_streamer = Mock()
            mock_streamer.__iter__ = Mock(return_value=iter(["Hello", " world"]))
            mock_streamer_class.return_value = mock_streamer

            with patch("..strategies.streaming_chat.Thread") as mock_thread_class:
                mock_thread = Mock()
                mock_thread_class.return_value = mock_thread

                # Generate and collect chunks
                chunks = list(strategy.generate(request))

                # Verify chunks were generated
                assert len(chunks) > 0
                assert any("data:" in chunk for chunk in chunks)
                assert any("[DONE]" in chunk for chunk in chunks)

    def test_streaming_completion_basic(self, mock_service):
        """Test basic streaming completion generation."""
        strategy = StreamingCompletionStrategy(mock_service)

        request = CompletionRequest(model="test-model", prompt="Test", max_tokens=50)

        # Mock the streamer
        with patch(
            "..strategies.streaming_completion.TextIteratorStreamer"
        ) as mock_streamer_class:
            mock_streamer = Mock()
            mock_streamer.__iter__ = Mock(return_value=iter(["Generated", " text"]))
            mock_streamer_class.return_value = mock_streamer

            with patch("..strategies.streaming_completion.Thread") as mock_thread_class:
                mock_thread = Mock()
                mock_thread_class.return_value = mock_thread

                # Generate and collect chunks
                chunks = list(strategy.generate(request))

                # Verify chunks were generated
                assert len(chunks) > 0
                assert any("data:" in chunk for chunk in chunks)
                assert any("[DONE]" in chunk for chunk in chunks)
