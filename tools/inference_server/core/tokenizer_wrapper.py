"""
Tokenization utilities with device placement handling.
"""

from typing import Any, Dict, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class TokenizerWrapper:
    """Handles tokenization and device placement for model inputs."""

    def __init__(self, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> None:
        """
        Initialize tokenizer wrapper.

        Args:
            tokenizer: HuggingFace tokenizer instance
            model: HuggingFace model instance
        """
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.model: PreTrainedModel = model

    def tokenize_and_move_to_device(
        self,
        text: str,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = None,
        return_token_type_ids: bool = False,
    ) -> Dict[str, Any]:
        """
        Tokenize text and move to model device.

        Args:
            text: Text to tokenize
            return_tensors: Format for return tensors
            padding: Whether to pad
            truncation: Whether to truncate
            max_length: Maximum length (optional)
            return_token_type_ids: Whether to return token type IDs

        Returns:
            Dictionary with input_ids moved to model device
        """
        # Tokenize
        tokenize_kwargs = {
            "return_tensors": return_tensors,
            "padding": padding,
            "truncation": truncation,
            "return_token_type_ids": return_token_type_ids,
        }
        if max_length is not None:
            tokenize_kwargs["max_length"] = max_length

        inputs = self.tokenizer(text, **tokenize_kwargs)

        # Move to model device
        input_ids = inputs["input_ids"]
        if hasattr(self.model, "device"):
            input_ids = input_ids.to(self.model.device)
        elif torch.cuda.is_available():
            input_ids = input_ids.to(next(self.model.parameters()).device)

        return {"input_ids": input_ids, "prompt_tokens": input_ids.shape[1]}

    def get_device(self) -> torch.device:
        """Get the device where the model is located."""
        if hasattr(self.model, "device"):
            return self.model.device
        elif torch.cuda.is_available():
            return next(self.model.parameters()).device
        else:
            return torch.device("cpu")
