"""
Pydantic models for the /tokenize endpoint (vLLM-compatible).

Request shape mirrors vLLM's ``TokenizeRequest`` (see
``vllm/entrypoints/openai/protocol.py``): a single body that may carry
either ``prompt`` (raw string) or ``messages`` (chat-template input).
Forgather adds two extensions:

  - ``next_role`` on the request, mirroring
    ``ChatCompletionRequest.next_role`` (selects which role the
    template opens at — used by the impersonate UI).
  - ``prompt`` on the response, the rendered template string. vLLM's
    response only carries ``tokens`` / ``token_strs``; clients that
    want the rendered text would normally have to detokenize. Bundling
    the prompt avoids the round trip and is the primary reason this
    endpoint exists for our webui.
"""

from typing import List, Optional

from pydantic import BaseModel

from .chat import ChatMessage


class TokenizeRequest(BaseModel):
    model: Optional[str] = None
    # Exactly one of ``messages`` / ``prompt`` should be set. If both
    # are present, ``messages`` wins (matches vLLM's discriminated-union
    # behavior — chat path takes precedence).
    messages: Optional[List[ChatMessage]] = None
    prompt: Optional[str] = None

    # Chat-template controls (used when ``messages`` is set)
    add_generation_prompt: bool = True
    continue_final_message: bool = False

    # Forgather extension: when "user", overrides the two flags above
    # and opens the template at a user turn instead of an assistant
    # turn (the impersonate path).
    next_role: Optional[str] = None

    # Tokenization controls
    add_special_tokens: bool = False
    return_token_strs: bool = False


class TokenizeResponse(BaseModel):
    count: int
    max_model_len: int
    tokens: List[int]
    token_strs: Optional[List[str]] = None
    # Forgather extension: the rendered template string. Optional only
    # for forward compat with vLLM clients that ignore unknown fields;
    # we always populate it.
    prompt: Optional[str] = None
