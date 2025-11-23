"""Pydantic models for inference server API."""

from .chat import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatCompletionResponse,
    ChatCompletionStreamDelta,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
)
from .completion import (
    CompletionRequest,
    CompletionChoice,
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
