"""
Streaming chat completion generation strategy.
"""

import time
from typing import Iterator, Optional
from transformers import TextIteratorStreamer

from .streaming_base import StreamingStrategy
from ..models.chat import (
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamDelta,
    ChatCompletionRequest,
)


class StreamingChatStrategy(StreamingStrategy):
    """Generates streaming chat completions."""

    def _get_request_id_prefix(self) -> str:
        """Return chat completion request ID prefix."""
        return "chatcmpl-"

    def _prepare_prompt(self, request: ChatCompletionRequest, request_id: str) -> str:
        """Format messages using chat template."""
        # Log each message
        self.service.logger.log_messages(request_id, request.messages)

        # Format messages using chat template
        template = self.service.jinja_env.from_string(self.service.chat_template)
        formatted_prompt = template.render(
            messages=request.messages,
            bos_token=self.service.tokenizer.bos_token,
            eos_token=self.service.tokenizer.eos_token,
            add_generation_prompt=True,
        )
        return formatted_prompt

    def _get_stop_sequences(self, request: ChatCompletionRequest) -> list:
        """Get stop sequences (use server defaults for chat)."""
        return self.service.stop_sequences

    def _create_streamer(self, request: ChatCompletionRequest) -> TextIteratorStreamer:
        """Create text iterator streamer for chat (skip prompt, skip special tokens)."""
        return TextIteratorStreamer(
            self.service.tokenizer,
            timeout=60.0,
            skip_prompt=True,
            skip_special_tokens=True,
        )

    def _create_initial_chunk(
        self, request_id: str, created: int, model: str
    ) -> Optional[str]:
        """Create initial chunk with assistant role."""
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatCompletionStreamDelta(role="assistant", content=""),
                    finish_reason=None,
                )
            ],
        )
        return f"data: {chunk.model_dump_json()}\n\n"

    def _create_chunk(
        self,
        request_id: str,
        created: int,
        model: str,
        text: str,
        finish_reason: Optional[str],
    ) -> str:
        """Create SSE-formatted chunk for chat."""
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=(
                        ChatCompletionStreamDelta(content=text)
                        if text
                        else ChatCompletionStreamDelta()
                    ),
                    finish_reason=finish_reason if finish_reason else None,
                )
            ],
        )
        return f"data: {chunk.model_dump_json()}\n\n"

    def _create_error_chunk(self, request_id: str, model: str) -> str:
        """Create error chunk for chat."""
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0, delta=ChatCompletionStreamDelta(), finish_reason="stop"
                )
            ],
        )
        return f"data: {chunk.model_dump_json()}\n\n"
