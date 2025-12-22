"""
Pydantic models for text completion API.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel
from .chat import ChatCompletionUsage  # Reuse usage model


class CompletionRequest(BaseModel):
    # OpenAI API parameters
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    # Additional HuggingFace generation parameters
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    no_repeat_ngram_size: Optional[int] = None
    encoder_no_repeat_ngram_size: Optional[int] = None
    bad_words_ids: Optional[List[List[int]]] = None
    min_length: Optional[int] = None
    max_new_tokens: Optional[int] = None
    early_stopping: Optional[bool] = None
    num_beams: Optional[int] = None
    num_beam_groups: Optional[int] = None
    diversity_penalty: Optional[float] = None
    temperature_last_layer: Optional[bool] = None
    top_k: Optional[int] = None
    typical_p: Optional[float] = None
    epsilon_cutoff: Optional[float] = None
    eta_cutoff: Optional[float] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: ChatCompletionUsage  # Same usage format


# Streaming response models
class CompletionStreamChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionStreamChoice]
