"""
Pydantic models for chat completion API.
"""

from typing import List, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False

    # Non-standard: which role the model should generate as. Default
    # "assistant" matches normal chat semantics. Setting "user" enables
    # "impersonate" / prefix-continuation: the chat template renders up
    # to where a user turn opens, with no generation-prompt closing
    # marker, so the model continues in the user's voice. Implemented
    # via Jinja's ``continue_final_message=True`` plus an empty trailing
    # user message — supported by all modern HF chat templates
    # (Llama 3, ChatML, Mistral, Qwen, Gemma, …).
    next_role: Optional[str] = None

    # Additional HuggingFace generation parameters
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    no_repeat_ngram_size: Optional[int] = None
    encoder_no_repeat_ngram_size: Optional[int] = None
    bad_words_ids: Optional[List[List[int]]] = None
    min_length: Optional[int] = None
    min_new_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = None
    early_stopping: Optional[bool] = None
    num_beams: Optional[int] = None
    num_beam_groups: Optional[int] = None
    diversity_penalty: Optional[float] = None
    temperature_last_layer: Optional[bool] = None
    top_k: Optional[int] = None
    typical_p: Optional[float] = None
    min_p: Optional[float] = None
    epsilon_cutoff: Optional[float] = None
    eta_cutoff: Optional[float] = None
    guidance_scale: Optional[float] = None
    penalty_alpha: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    do_sample: Optional[bool] = None
    seed: Optional[int] = None
    ignore_eos: Optional[bool] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


# Streaming response models
class ChatCompletionStreamDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: ChatCompletionStreamDelta
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]
