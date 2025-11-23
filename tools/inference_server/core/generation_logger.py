"""
Unified logging for generation operations.
"""

import logging
from typing import List, Optional, Any
from transformers import PreTrainedTokenizer


class GenerationLogger:
    """Handles consistent logging across all generation methods."""

    def __init__(
        self, logger: logging.Logger, tokenizer: Optional[PreTrainedTokenizer]
    ) -> None:
        """
        Initialize generation logger.

        Args:
            logger: Logger instance to use
            tokenizer: HuggingFace tokenizer for decoding tokens (optional, set after initialization)
        """
        self.logger: logging.Logger = logger
        self.tokenizer: Optional[PreTrainedTokenizer] = tokenizer

    def log_request(
        self,
        request_id: str,
        request_type: str,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs: Any,
    ) -> None:
        """
        Log incoming request details.

        Args:
            request_id: Unique request identifier
            request_type: Type of request (chat/completion)
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional request-specific parameters
        """
        log_parts = [
            f"[{request_id}] New {request_type} request:",
            f"model={model}",
            f"max_tokens={max_tokens}",
            f"temperature={temperature}",
            f"top_p={top_p}",
        ]
        for key, value in kwargs.items():
            log_parts.append(f"{key}={value}")

        self.logger.info(", ".join(log_parts))

    def log_prompt(self, request_id: str, prompt: str):
        """Log formatted prompt."""
        self.logger.info(f"[{request_id}] Formatted prompt: {repr(prompt)}")

    def log_input_tokens(
        self,
        request_id: str,
        input_token_ids: List[int],
    ):
        """Log input token details."""
        self.logger.info(f"[{request_id}] Input token IDs: {input_token_ids}")
        decoded_with_special = self.tokenizer.decode(
            input_token_ids, skip_special_tokens=False
        )
        self.logger.info(
            f"[{request_id}] Input tokens with special tokens: {repr(decoded_with_special)}"
        )

    def log_generation_config(self, request_id: str, generation_config):
        """Log generation configuration."""
        self.logger.info(f"[{request_id}] Generation config: {repr(generation_config)}")

    def log_stop_strings(self, request_id: str, stop_strings: List[str]):
        """Log stop strings passed to model.generate."""
        self.logger.info(
            f"[{request_id}] Passing stop_strings to model.generate: {stop_strings}"
        )

    def log_generated_tokens(
        self,
        request_id: str,
        generated_token_ids: List[int],
    ):
        """Log generated token details."""
        self.logger.info(f"[{request_id}] Generated token IDs: {generated_token_ids}")
        decoded_with_special = self.tokenizer.decode(
            generated_token_ids, skip_special_tokens=False
        )
        self.logger.info(
            f"[{request_id}] Generated tokens with special tokens: {repr(decoded_with_special)}"
        )

    def log_response(
        self,
        request_id: str,
        response_text: str,
        finish_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
    ):
        """
        Log final response details.

        Args:
            request_id: Request identifier
            response_text: Final response text (cleaned)
            finish_reason: Why generation stopped
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
        """
        self.logger.info(f"[{request_id}] Response text (clean): {repr(response_text)}")
        self.logger.info(f"[{request_id}] Finish reason: {finish_reason}")
        self.logger.info(
            f"[{request_id}] Token usage: prompt={prompt_tokens}, "
            f"completion={completion_tokens}, total={prompt_tokens + completion_tokens}"
        )

    def log_stop_sequence_triggered(
        self,
        request_id: str,
        stop_sequence: str,
    ):
        """Log when a stop sequence triggers."""
        self.logger.info(
            f"[{request_id}] Generation stopped due to stop sequence: {repr(stop_sequence)}"
        )

    def log_eos_token(self, request_id: str):
        """Log when EOS token triggers stopping."""
        self.logger.info(f"[{request_id}] Generation stopped due to EOS token")

    def log_stop_token(self, request_id: str, token_id: int):
        """Log when a stop token triggers stopping."""
        self.logger.info(
            f"[{request_id}] Generation stopped due to stop token ID: {token_id}"
        )

    def log_streaming_error(self, request_id: str, error: Exception):
        """Log streaming generation error."""
        self.logger.error(f"[{request_id}] Streaming generation failed: {str(error)}")

    def log_messages(self, request_id: str, messages: List):
        """Log individual chat messages."""
        for i, msg in enumerate(messages):
            self.logger.info(
                f"[{request_id}] Message {i}: role={msg.role}, content={repr(msg.content)}"
            )
