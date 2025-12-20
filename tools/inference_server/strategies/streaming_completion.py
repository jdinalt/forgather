"""
Streaming text completion generation strategy.
"""

import time
from typing import Iterator, Optional
from transformers import TextIteratorStreamer

from .streaming_base import StreamingStrategy
from ..models.completion import (
    CompletionStreamResponse,
    CompletionStreamChoice,
    CompletionRequest,
)


class StreamingCompletionStrategy(StreamingStrategy):
    """Generates streaming text completions."""

    def _get_request_id_prefix(self) -> str:
        """Return completion request ID prefix."""
        return "cmpl-"

    def _prepare_prompt(self, request: CompletionRequest, request_id: str) -> str:
        """Return prompt as-is for completion."""
        return request.prompt

    def _get_stop_sequences(self, request: CompletionRequest) -> list:
        """Get stop sequences (use server defaults for completion)."""
        return self.service.stop_sequences

    def _create_streamer(self, request: CompletionRequest) -> TextIteratorStreamer:
        """Create text iterator streamer for completion (handle echo parameter)."""
        return TextIteratorStreamer(
            self.service.tokenizer,
            timeout=60.0,
            skip_prompt=not request.echo,  # Include prompt if echo=True
            skip_special_tokens=True,
        )

    def _create_initial_chunk(
        self, request_id: str, created: int, model: str
    ) -> Optional[str]:
        """No initial chunk for completion (return None)."""
        return None

    def _create_chunk(
        self,
        request_id: str,
        created: int,
        model: str,
        text: str,
        finish_reason: Optional[str],
    ) -> str:
        """Create SSE-formatted chunk for completion."""
        chunk = CompletionStreamResponse(
            id=request_id,
            created=created,
            model=model,
            choices=[
                CompletionStreamChoice(
                    index=0,
                    text=text,
                    finish_reason=finish_reason if finish_reason else None,
                )
            ],
        )
        return f"data: {chunk.model_dump_json()}\n\n"

    def _create_error_chunk(self, request_id: str, model: str) -> str:
        """Create error chunk for completion."""
        chunk = CompletionStreamResponse(
            id=request_id,
            created=int(time.time()),
            model=model,
            choices=[CompletionStreamChoice(index=0, text="", finish_reason="stop")],
        )
        return f"data: {chunk.model_dump_json()}\n\n"
