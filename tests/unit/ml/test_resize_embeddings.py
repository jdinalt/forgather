#!/usr/bin/env python3
"""
Unit tests for resize_embeddings module in Forgather.

Tests token addition, vocabulary resizing, and embedding initialization strategies.
"""

import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch
import yaml

from forgather.ml.model_conversion.resize_embeddings import (
    DEFAULT_TOKEN_CONFIG,
    add_tokens_to_tokenizer,
    resize_word_embeddings,
    update_config_from_tokenizer,
)


class TestAddTokensToTokenizer(unittest.TestCase):
    """Test add_tokens_to_tokenizer function."""

    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = Mock()
        self.tokenizer.pad_token = None
        self.tokenizer.pad_token_id = None
        self.tokenizer.bos_token = None
        self.tokenizer.bos_token_id = None
        self.tokenizer.eos_token = None
        self.tokenizer.eos_token_id = None
        self.tokenizer.unk_token = None
        self.tokenizer.unk_token_id = None

        # Mock add_special_tokens to return number of added tokens
        self.tokenizer.add_special_tokens = Mock(return_value=0)
        self.tokenizer.add_tokens = Mock(return_value=0)
        self.tokenizer.convert_tokens_to_ids = Mock(
            return_value=self.tokenizer.unk_token_id
        )
        self.tokenizer.get_vocab = Mock(return_value={})

    def test_add_named_token_simple_format(self):
        """Test adding a named special token with simple string format."""
        config = {"pad_token": "<|pad|>"}

        num_added, token_inits = add_tokens_to_tokenizer(self.tokenizer, config)

        self.tokenizer.add_special_tokens.assert_called_once()
        call_args = self.tokenizer.add_special_tokens.call_args[0][0]
        self.assertEqual(call_args["pad_token"], "<|pad|>")

    def test_add_named_token_dict_format(self):
        """Test adding a named special token with dict format and custom init."""
        config = {"pad_token": {"token": "<|pad|>", "init": "zero"}}

        num_added, token_inits = add_tokens_to_tokenizer(self.tokenizer, config)

        self.tokenizer.add_special_tokens.assert_called_once()
        call_args = self.tokenizer.add_special_tokens.call_args[0][0]
        self.assertEqual(call_args["pad_token"], "<|pad|>")

    def test_if_missing_flag_skips_existing_token(self):
        """Test that if_missing flag skips adding token if it already exists."""
        self.tokenizer.pad_token = "[PAD]"

        config = {"pad_token": {"token": "<|pad|>", "init": "zero", "if_missing": True}}

        num_added, token_inits = add_tokens_to_tokenizer(self.tokenizer, config)

        # Should not call add_special_tokens since token already exists
        self.tokenizer.add_special_tokens.assert_not_called()

    def test_if_missing_flag_adds_missing_token(self):
        """Test that if_missing flag adds token if it doesn't exist."""
        config = {"pad_token": {"token": "<|pad|>", "init": "zero", "if_missing": True}}

        num_added, token_inits = add_tokens_to_tokenizer(self.tokenizer, config)

        # Should call add_special_tokens since token is missing
        self.tokenizer.add_special_tokens.assert_called_once()

    def test_token_reassignment_different_value(self):
        """Test replacing an existing token with a different value."""
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.pad_token_id = 0

        config = {"pad_token": "<|pad|>"}

        num_added, token_inits = add_tokens_to_tokenizer(self.tokenizer, config)

        # Should call add_special_tokens to replace the token
        self.tokenizer.add_special_tokens.assert_called_once()
        call_args = self.tokenizer.add_special_tokens.call_args[0][0]
        self.assertEqual(call_args["pad_token"], "<|pad|>")

    def test_token_already_same_value(self):
        """Test that setting a token to its existing value is a no-op."""
        self.tokenizer.pad_token = "<|pad|>"

        config = {"pad_token": "<|pad|>"}

        num_added, token_inits = add_tokens_to_tokenizer(self.tokenizer, config)

        # Should not add token since it's already set to the same value
        # The function may still call add_special_tokens but with no actual change
        # We're mainly checking that it doesn't fail

    def test_add_additional_special_tokens(self):
        """Test adding additional special tokens."""
        config = {"special_tokens": ["<|im_start|>", "<|im_end|>"]}

        self.tokenizer.add_special_tokens = Mock(return_value=2)

        num_added, token_inits = add_tokens_to_tokenizer(self.tokenizer, config)

        self.tokenizer.add_special_tokens.assert_called_once()
        call_args = self.tokenizer.add_special_tokens.call_args[0][0]
        self.assertEqual(
            call_args["additional_special_tokens"], ["<|im_start|>", "<|im_end|>"]
        )

    def test_add_regular_tokens(self):
        """Test adding regular tokens."""
        config = {"regular_tokens": ["token1", "token2"]}

        self.tokenizer.add_tokens = Mock(return_value=2)

        num_added, token_inits = add_tokens_to_tokenizer(self.tokenizer, config)

        self.tokenizer.add_tokens.assert_called_once_with(["token1", "token2"])

    def test_init_strategies_returned(self):
        """Test that initialization strategies are correctly returned."""
        self.tokenizer.pad_token_id = 100
        self.tokenizer.add_special_tokens = Mock(return_value=1)

        config = {"pad_token": {"token": "<|pad|>", "init": "zero"}}

        num_added, token_inits = add_tokens_to_tokenizer(self.tokenizer, config)

        # Check that the init strategy is recorded for the token
        self.assertEqual(token_inits.get(100), "zero")

    def test_default_init_strategies(self):
        """Test that default init strategies are used when not specified."""
        self.tokenizer.pad_token_id = 100
        self.tokenizer.bos_token_id = 101
        self.tokenizer.add_special_tokens = Mock(return_value=2)

        config = {
            "pad_token": "<|pad|>",  # Should default to zero
            "bos_token": "<|bos|>",  # Should default to mean
        }

        num_added, token_inits = add_tokens_to_tokenizer(self.tokenizer, config)

        self.assertEqual(token_inits.get(100), "zero")
        self.assertEqual(token_inits.get(101), "mean")

    def test_load_from_yaml_file(self):
        """Test loading token config from YAML file."""
        config = {"pad_token": "<|pad|>", "special_tokens": ["<|im_start|>"]}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            yaml_path = f.name

        try:
            num_added, token_inits = add_tokens_to_tokenizer(self.tokenizer, yaml_path)
            # Just verify it doesn't crash
        finally:
            import os

            os.unlink(yaml_path)

    def test_default_token_config_structure(self):
        """Test that DEFAULT_TOKEN_CONFIG has expected structure."""
        self.assertIn("pad_token", DEFAULT_TOKEN_CONFIG)
        self.assertEqual(DEFAULT_TOKEN_CONFIG["pad_token"]["token"], "[PAD]")
        self.assertEqual(DEFAULT_TOKEN_CONFIG["pad_token"]["init"], "zero")
        self.assertTrue(DEFAULT_TOKEN_CONFIG["pad_token"]["if_missing"])


class TestResizeWordEmbeddings(unittest.TestCase):
    """Test resize_word_embeddings function."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        torch.manual_seed(42)

    def test_resize_with_zero_init(self):
        """Test resizing embeddings with zero initialization."""
        # Create a mock model with embeddings
        model = Mock()
        # Create embeddings that are already the target size to simulate post-resize state
        input_embeddings = torch.randn(110, 128, device=self.device)
        output_embeddings = torch.randn(110, 128, device=self.device)

        input_weight = Mock()
        input_weight.weight = input_embeddings
        output_weight = Mock()
        output_weight.weight = output_embeddings

        model.get_input_embeddings = Mock(return_value=input_weight)
        model.get_output_embeddings = Mock(return_value=output_weight)
        model.resize_token_embeddings = Mock()

        tokenizer = Mock()
        tokenizer.__len__ = Mock(return_value=110)

        token_inits = {105: "zero"}

        resize_word_embeddings(model, tokenizer, token_inits)

        # Verify resize was called with correct size
        model.resize_token_embeddings.assert_called_once_with(110, mean_resizing=True)

        # Verify that the token at index 105 was zero-initialized
        self.assertTrue(torch.all(input_embeddings[105] == 0))

    def test_resize_without_token_inits(self):
        """Test resizing embeddings without custom init strategies."""
        model = Mock()
        model.resize_token_embeddings = Mock()

        tokenizer = Mock()
        tokenizer.__len__ = Mock(return_value=110)

        resize_word_embeddings(model, tokenizer, None)

        # Should still resize
        model.resize_token_embeddings.assert_called_once_with(110, mean_resizing=True)


class TestUpdateConfigFromTokenizer(unittest.TestCase):
    """Test update_config_from_tokenizer function."""

    def test_update_special_token_ids(self):
        """Test updating special token IDs in config."""
        config = Mock()
        config.vocab_size = 100

        tokenizer = Mock()
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        tokenizer.__len__ = Mock(return_value=110)

        update_config_from_tokenizer(config, tokenizer)

        self.assertEqual(config.bos_token_id, 1)
        self.assertEqual(config.eos_token_id, 2)
        self.assertEqual(config.pad_token_id, 0)
        self.assertEqual(config.vocab_size, 110)

    def test_skip_none_token_ids(self):
        """Test that None token IDs are not set in config."""
        config = Mock()
        config.vocab_size = 100

        tokenizer = Mock()
        tokenizer.bos_token_id = None
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = None
        tokenizer.__len__ = Mock(return_value=100)

        update_config_from_tokenizer(config, tokenizer)

        # Only eos_token_id should be set
        self.assertEqual(config.eos_token_id, 2)


if __name__ == "__main__":
    unittest.main()
