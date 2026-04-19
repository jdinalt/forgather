"""
Non-streaming chat completion generation strategy.
"""

import time

import torch

from ..models.chat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from .non_streaming_base import NonStreamingStrategy


class ChatGenerationStrategy(NonStreamingStrategy):
    """Generates non-streaming chat completions."""

    def _get_request_id_prefix(self) -> str:
        """Return chat completion request ID prefix."""
        return "chatcmpl-"

    def _prepare_prompt(self, request: ChatCompletionRequest, request_id: str) -> str:
        """Format messages using chat template."""
        # Log messages first
        self.service.logger.log_messages(request_id, request.messages)
        # Format using chat template
        return self.service.format_messages(request.messages)

    def _get_stop_sequences(self, request: ChatCompletionRequest) -> list:
        """Get stop sequences (use server defaults for chat)."""
        return self.service.stop_sequences.copy()

    def _tokenize_input(self, prompt: str, request: ChatCompletionRequest) -> dict:
        """Tokenize chat prompt with no max_length."""
        return self.service.tokenizer_wrapper.tokenize_and_move_to_device(
            prompt,
            max_length=None,  # No max_length for chat
        )

    def _process_response_text(
        self,
        generated_tokens: torch.Tensor,
        request: ChatCompletionRequest,
        original_prompt: str,
    ) -> str:
        """Decode generated tokens (no special processing for chat)."""
        return self.service.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _build_response(
        self,
        request_id: str,
        request: ChatCompletionRequest,
        response_text: str,
        finish_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> ChatCompletionResponse:
        """Build ChatCompletionResponse object."""
        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason=finish_reason,
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
