"""
Stop sequence processing for generation trimming.
"""

from typing import Any, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizer


class StopSequenceProcessor:
    """
    Handles trimming of generated text at stop sequences.

    This utility processes generated text to detect and trim at stop sequences,
    which are special strings that signal the model to stop generating.

    Example:
        Basic usage:
        >>> processor = StopSequenceProcessor(tokenizer)
        >>> text = "Generated text STOP extra content"
        >>> token_ids = [1, 2, 3, 4, 5, 6]
        >>> tokens = torch.tensor(token_ids)
        >>> stop_sequences = ["STOP"]
        >>>
        >>> trimmed_ids, trimmed_tokens, stopped, seq = processor.process(
        ...     text, token_ids, tokens, stop_sequences
        ... )
        >>> print(stopped)  # True
        >>> print(seq)  # "STOP"

        Streaming usage:
        >>> full_response = "Hello STOP world"
        >>> new_text = "P world"
        >>> should_stop, remaining, seq = processor.process_streaming(
        ...     full_response, new_text, ["STOP"]
        ... )
        >>> print(should_stop)  # True
    """

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        """
        Initialize stop sequence processor.

        Args:
            tokenizer: HuggingFace tokenizer instance
        """
        self.tokenizer: PreTrainedTokenizer = tokenizer

    def process(
        self,
        generated_text: str,
        generated_token_ids: List[int],
        generated_tokens: torch.Tensor,
        stop_sequences: List[str],
    ) -> Tuple[List[int], torch.Tensor, bool, Optional[str]]:
        """
        Check for stop sequences in generated text and trim if found.

        Args:
            generated_text: Raw generated text (with special tokens)
            generated_token_ids: List of generated token IDs
            generated_tokens: Tensor of generated tokens
            stop_sequences: List of stop sequences to check

        Returns:
            Tuple of (trimmed_token_ids, trimmed_tokens, stopped_by_sequence, stop_sequence_found)
        """
        stopped_by_sequence = False
        stop_sequence_found = None

        for sequence in stop_sequences:
            if sequence in generated_text:
                stopped_by_sequence = True
                stop_sequence_found = sequence

                # Trim the generated text at the stop sequence
                stop_index = generated_text.find(sequence)
                trimmed_text = generated_text[:stop_index]

                # Re-encode to get the trimmed tokens
                trimmed_tokens = self.tokenizer.encode(
                    trimmed_text, add_special_tokens=False
                )

                if len(trimmed_tokens) < len(generated_token_ids):
                    generated_token_ids = trimmed_tokens
                    generated_tokens = torch.tensor(
                        generated_token_ids, device=generated_tokens.device
                    )
                break

        return (
            generated_token_ids,
            generated_tokens,
            stopped_by_sequence,
            stop_sequence_found,
        )

    def process_streaming(
        self,
        full_response: str,
        new_text: str,
        stop_sequences: List[str],
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check for stop sequences in streaming generation.

        Args:
            full_response: Accumulated response text so far
            new_text: Latest text chunk
            stop_sequences: List of stop sequences to check

        Returns:
            Tuple of (should_stop, remaining_text, stop_sequence_found)
            - should_stop: Whether to stop streaming
            - remaining_text: Text to send before stopping (may be None)
            - stop_sequence_found: Which sequence triggered the stop
        """
        for stop_seq in stop_sequences:
            if stop_seq in full_response:
                # Trim at stop sequence
                stop_index = full_response.find(stop_seq)
                trimmed_response = full_response[:stop_index]
                remaining_text = trimmed_response[len(full_response) - len(new_text) :]

                return True, remaining_text if remaining_text else None, stop_seq

        return False, None, None
