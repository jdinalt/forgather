"""Pydantic models for inference server API."""

from .chat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamDelta,
    ChatCompletionStreamResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from .completion import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamChoice,
    CompletionStreamResponse,
)

__all__ = [
    # Chat models
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionChoice",
    "ChatCompletionUsage",
    "ChatCompletionResponse",
    "ChatCompletionStreamDelta",
    "ChatCompletionStreamChoice",
    "ChatCompletionStreamResponse",
    # Completion models
    "CompletionRequest",
    "CompletionChoice",
    "CompletionResponse",
    "CompletionStreamChoice",
    "CompletionStreamResponse",
]
