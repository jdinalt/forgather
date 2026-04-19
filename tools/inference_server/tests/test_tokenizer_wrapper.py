"""
Unit tests for TokenizerWrapper.
"""

from unittest.mock import MagicMock, Mock

import pytest
import torch

from ..core.tokenizer_wrapper import TokenizerWrapper


class TestTokenizerWrapper:
    """Test cases for tokenizer wrapper."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()

        # Mock the tokenizer call to return a dict with input_ids
        def tokenize_side_effect(text, **kwargs):
            # Simulate tokenization
            token_ids = torch.tensor([[1, 2, 3, 4, 5]])
            return {"input_ids": token_ids}

        tokenizer.side_effect = tokenize_side_effect
        return tokenizer

    @pytest.fixture
    def mock_model_with_device(self):
        """Create a mock model with device attribute."""
        model = Mock()
        model.device = torch.device("cuda:0")
        return model

    @pytest.fixture
    def mock_model_with_params(self):
        """Create a mock model with parameters."""
        model = Mock()
        delattr(model, "device")  # Remove device attribute

        # Mock parameter
        param = Mock()
        param.device = torch.device("cpu")
        model.parameters.return_value = [param]

        return model

    def test_tokenize_with_device_attribute(
        self, mock_tokenizer, mock_model_with_device
    ):
        """Test tokenization when model has device attribute."""
        wrapper = TokenizerWrapper(mock_tokenizer, mock_model_with_device)

        result = wrapper.tokenize_and_move_to_device("test text")

        assert "input_ids" in result
        assert "prompt_tokens" in result
        assert result["prompt_tokens"] == 5  # Should be shape[1]

    def test_tokenize_with_params_device(self, mock_tokenizer, mock_model_with_params):
        """Test tokenization when using parameters device."""
        wrapper = TokenizerWrapper(mock_tokenizer, mock_model_with_params)

        result = wrapper.tokenize_and_move_to_device("test text")

        assert "input_ids" in result
        assert "prompt_tokens" in result

    def test_tokenize_with_max_length(self, mock_tokenizer, mock_model_with_device):
        """Test tokenization with max_length parameter."""
        wrapper = TokenizerWrapper(mock_tokenizer, mock_model_with_device)

        result = wrapper.tokenize_and_move_to_device("test text", max_length=512)

        assert "input_ids" in result
        assert "prompt_tokens" in result

    def test_tokenize_without_padding(self, mock_tokenizer, mock_model_with_device):
        """Test tokenization without padding."""
        wrapper = TokenizerWrapper(mock_tokenizer, mock_model_with_device)

        result = wrapper.tokenize_and_move_to_device("test text", padding=False)

        assert "input_ids" in result

    def test_tokenize_without_truncation(self, mock_tokenizer, mock_model_with_device):
        """Test tokenization without truncation."""
        wrapper = TokenizerWrapper(mock_tokenizer, mock_model_with_device)

        result = wrapper.tokenize_and_move_to_device("test text", truncation=False)

        assert "input_ids" in result

    def test_get_device_with_attribute(self, mock_tokenizer, mock_model_with_device):
        """Test get_device when model has device attribute."""
        wrapper = TokenizerWrapper(mock_tokenizer, mock_model_with_device)

        device = wrapper.get_device()

        assert device == torch.device("cuda:0")

    def test_get_device_from_params(self, mock_tokenizer, mock_model_with_params):
        """Test get_device from model parameters."""
        wrapper = TokenizerWrapper(mock_tokenizer, mock_model_with_params)

        device = wrapper.get_device()

        assert device == torch.device("cpu")

    def test_get_device_cpu_fallback(self, mock_tokenizer):
        """Test get_device falls back to CPU when no CUDA."""
        model = Mock()
        delattr(model, "device")
        model.parameters.return_value = []

        wrapper = TokenizerWrapper(mock_tokenizer, model)

        # When no params and no CUDA, should return CPU
        device = wrapper.get_device()
        assert device.type == "cpu"

    def test_prompt_tokens_calculation(self, mock_tokenizer, mock_model_with_device):
        """Test that prompt tokens is correctly calculated from shape."""
        wrapper = TokenizerWrapper(mock_tokenizer, mock_model_with_device)

        result = wrapper.tokenize_and_move_to_device("test text")

        # Shape is [1, 5], so prompt_tokens should be 5
        assert result["prompt_tokens"] == 5

    def test_return_tensors_pt(self, mock_tokenizer, mock_model_with_device):
        """Test that return_tensors='pt' is used by default."""
        wrapper = TokenizerWrapper(mock_tokenizer, mock_model_with_device)

        result = wrapper.tokenize_and_move_to_device("test text")

        # Should return pytorch tensors
        assert isinstance(result["input_ids"], torch.Tensor)

    def test_multiple_tokenization_calls(self, mock_tokenizer, mock_model_with_device):
        """Test multiple tokenization calls work correctly."""
        wrapper = TokenizerWrapper(mock_tokenizer, mock_model_with_device)

        result1 = wrapper.tokenize_and_move_to_device("first text")
        result2 = wrapper.tokenize_and_move_to_device("second text")

        assert "input_ids" in result1
        assert "input_ids" in result2
        assert "prompt_tokens" in result1
        assert "prompt_tokens" in result2
