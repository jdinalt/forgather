"""
Non-streaming text completion generation strategy.
"""

import time
import torch
from fastapi import HTTPException
from .non_streaming_base import NonStreamingStrategy
from ..models.completion import (
    CompletionResponse,
    CompletionChoice,
    ChatCompletionUsage,
    CompletionRequest,
)


class CompletionGenerationStrategy(NonStreamingStrategy):
    """Generates non-streaming text completions."""

    def _get_request_id_prefix(self) -> str:
        """Return completion request ID prefix."""
        return "cmpl-"

    def _prepare_prompt(self, request: CompletionRequest, request_id: str) -> str:
        """Handle single prompt vs list of prompts."""
        if isinstance(request.prompt, list):
            if len(request.prompt) != 1:
                raise HTTPException(
                    status_code=400, detail="Multiple prompts not supported yet"
                )
            return request.prompt[0]
        return request.prompt

    def _get_stop_sequences(self, request: CompletionRequest) -> list:
        """Get stop sequences (merge request and server defaults)."""
        # Parse stop sequences from request
        request_stop_sequences = []
        if request.stop:
            if isinstance(request.stop, str):
                request_stop_sequences = [request.stop] if request.stop else []
            else:
                request_stop_sequences = request.stop

        # Combine with server stop sequences and filter out empty strings
        return [s for s in (self.service.stop_sequences + request_stop_sequences) if s]

    def _tokenize_input(self, prompt: str, request: CompletionRequest) -> dict:
        """Tokenize completion prompt with max_length=2048."""
        return self.service.tokenizer_wrapper.tokenize_and_move_to_device(
            prompt,
            max_length=2048,
        )

    def _process_response_text(
        self,
        generated_tokens: torch.Tensor,
        request: CompletionRequest,
        original_prompt: str,
    ) -> str:
        """Decode generated tokens and handle echo parameter."""
        response_text = self.service.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        # Handle echo parameter (include original prompt in response)
        if request.echo:
            response_text = original_prompt + response_text

        return response_text

    def _build_response(
        self,
        request_id: str,
        request: CompletionRequest,
        response_text: str,
        finish_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> CompletionResponse:
        """Build CompletionResponse object."""
        return CompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    text=response_text, index=0, finish_reason=finish_reason
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
