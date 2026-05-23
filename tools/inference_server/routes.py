"""
FastAPI route handlers for inference server.
"""

from __future__ import annotations

import hmac
import time
import traceback
from typing import AsyncIterator, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from .models.chat import ChatCompletionRequest, ChatCompletionUsage
from .models.completion import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
)
from .models.tokenize import TokenizeRequest, TokenizeResponse
from .service import InferenceService, ModelNotFoundError
from .strategies import (
    ChatGenerationStrategy,
    CompletionGenerationStrategy,
    StreamingChatStrategy,
    StreamingCompletionStrategy,
)

# Global inference service instance
inference_service: Optional[InferenceService] = None


def _make_verify_bearer(auth_token: str):
    """Build a FastAPI dependency that enforces ``Authorization: Bearer <token>``.

    Constant-time compare via hmac.compare_digest so a partial-match
    timing leak can't fingerprint the token.
    """
    expected = auth_token

    async def verify_bearer(authorization: Optional[str] = Header(default=None)):
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(
                status_code=401,
                detail="authentication required",
                headers={"WWW-Authenticate": 'Bearer realm="forgather-inference"'},
            )
        token = authorization.split(" ", 1)[1].strip()
        if not hmac.compare_digest(token, expected):
            raise HTTPException(
                status_code=401,
                detail="authentication required",
                headers={"WWW-Authenticate": 'Bearer realm="forgather-inference"'},
            )
        return None

    return verify_bearer


def create_app(auth_token: Optional[str] = None) -> FastAPI:
    """Create and configure FastAPI application.

    When ``auth_token`` is None, no auth is enforced — matches
    ``--no-auth``. Otherwise every route except ``/health`` requires
    ``Authorization: Bearer <auth_token>``. Health is intentionally
    open so the proxy can probe before the model finishes loading.
    """
    app = FastAPI(title="HuggingFace OpenAI API Server", version="1.0.0")

    # ModelNotFoundError is raised from inside ``service.acquire(name)``
    # when a multi-model request names an unknown entry. The service
    # module deliberately raises a domain exception (not HTTPException)
    # so it stays transport-agnostic; we translate to 404 here.
    @app.exception_handler(ModelNotFoundError)
    async def _model_not_found_handler(_request, exc: ModelNotFoundError):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=404, content={"detail": str(exc)})

    deps = [Depends(_make_verify_bearer(auth_token))] if auth_token else []

    @app.get("/v1/models", dependencies=deps)
    async def list_models():
        """List configured models, with a Forgather-extension ``x_state`` field."""
        if inference_service is None:
            return {"object": "list", "data": []}
        return {
            "object": "list",
            "data": [
                {
                    "id": entry.name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "huggingface",
                    "x_state": entry.state,
                    "x_model_path": entry.model_path,
                }
                for entry in inference_service.list_entries()
            ],
        }

    @app.post("/v1/chat/completions", dependencies=deps)
    async def create_chat_completion(request: ChatCompletionRequest):
        """Create a chat completion."""
        if inference_service is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        try:
            if request.stream:
                strategy = StreamingChatStrategy(inference_service)
                return StreamingResponse(
                    _stream_under_lock(
                        inference_service, request.model, strategy, request
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "close"},
                )
            else:
                async with inference_service.acquire(request.model):
                    strategy = ChatGenerationStrategy(inference_service)
                    return strategy.generate(request)
        except (HTTPException, ModelNotFoundError):
            # ModelNotFoundError reaches the exception_handler installed
            # on the app for translation to 404; re-raising HTTPException
            # preserves its status code (e.g. the 400 above).
            raise
        except Exception as e:
            traceback.print_exception(e)
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    @app.post("/v1/completions", dependencies=deps)
    async def create_completion(request: CompletionRequest):
        """Create a text completion."""
        if inference_service is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        if request.n != 1:
            raise HTTPException(status_code=400, detail="n > 1 not supported yet")

        # Scoring path: echo + logprobs + max_tokens=0 → single forward
        # pass returning per-token logprobs in OpenAI's legacy-completions
        # shape. Matches vLLM's behavior for the same request shape;
        # bypasses generate() entirely.
        if (
            request.echo
            and request.max_tokens == 0
            and request.logprobs is not None
            and request.logprobs > 0
            and not request.stream
        ):
            # Cap top-K. OpenAI's spec maxes at 5, vLLM at 20; a client
            # requesting ``logprobs=128000`` would force a full-vocab
            # topk + per-id decode loop, expensive and useless.
            top_k = min(request.logprobs, 20)
            try:
                prompt_text = (
                    request.prompt[0]
                    if isinstance(request.prompt, list)
                    else request.prompt
                )
                async with inference_service.acquire(request.model):
                    score_kwargs = {"top_k": top_k}
                    if request.score_max_length is not None:
                        score_kwargs["max_length"] = request.score_max_length
                    scores = inference_service.score_prompt(prompt_text, **score_kwargs)
                prompt_tokens = len(scores["tokens"])
                return CompletionResponse(
                    id=f"cmpl-{int(time.time() * 1000):x}",
                    created=int(time.time()),
                    model=request.model,
                    choices=[
                        CompletionChoice(
                            text=prompt_text,
                            index=0,
                            logprobs=scores,
                            finish_reason="length",
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        total_tokens=prompt_tokens,
                    ),
                )
            except (HTTPException, ModelNotFoundError):
                raise
            except Exception as e:
                traceback.print_exception(e)
                raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

        try:
            if request.stream:
                strategy = StreamingCompletionStrategy(inference_service)
                return StreamingResponse(
                    _stream_under_lock(
                        inference_service, request.model, strategy, request
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "close"},
                )
            else:
                async with inference_service.acquire(request.model):
                    strategy = CompletionGenerationStrategy(inference_service)
                    return strategy.generate(request)
        except (HTTPException, ModelNotFoundError):
            # ModelNotFoundError reaches the exception_handler installed
            # on the app for translation to 404; re-raising HTTPException
            # preserves its status code (e.g. the 400 above).
            raise
        except Exception as e:
            traceback.print_exception(e)
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    async def _tokenize(request: TokenizeRequest) -> TokenizeResponse:
        """vLLM-compatible /tokenize: render chat template and/or
        tokenize a prompt, return token ids and (Forgather extension)
        the rendered prompt text.

        If both ``messages`` and ``prompt`` are present, ``messages``
        wins — matches vLLM's discriminated-union behavior.
        """
        if inference_service is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        if not request.messages and request.prompt is None:
            raise HTTPException(
                status_code=400,
                detail="Either 'messages' or 'prompt' must be provided.",
            )

        try:
            async with inference_service.acquire(request.model):
                if request.messages:
                    rendered = inference_service.format_messages(
                        request.messages,
                        next_role=request.next_role,
                        add_generation_prompt=request.add_generation_prompt,
                        continue_final_message=request.continue_final_message,
                    )
                else:
                    rendered = request.prompt or ""

                tokens = inference_service.tokenize(
                    rendered, add_special_tokens=request.add_special_tokens
                )
                token_strs: Optional[list] = None
                if request.return_token_strs:
                    token_strs = [
                        inference_service.tokenizer.convert_ids_to_tokens(t)
                        for t in tokens
                    ]

                return TokenizeResponse(
                    count=len(tokens),
                    max_model_len=inference_service.get_max_model_len(),
                    tokens=tokens,
                    token_strs=token_strs,
                    prompt=rendered,
                )
        except (HTTPException, ModelNotFoundError):
            raise
        except Exception as e:
            traceback.print_exception(e)
            raise HTTPException(status_code=500, detail=f"Tokenize failed: {str(e)}")

    # vLLM serves /tokenize without the /v1 prefix; register both.
    app.post("/tokenize", response_model=TokenizeResponse, dependencies=deps)(_tokenize)
    app.post("/v1/tokenize", response_model=TokenizeResponse, dependencies=deps)(
        _tokenize
    )

    # /health stays open so the proxy can probe before any model loads.
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_loaded": (
                inference_service is not None and inference_service.active is not None
            ),
            "models_configured": (
                [e.name for e in inference_service.list_entries()]
                if inference_service is not None
                else []
            ),
        }

    return app


async def _stream_under_lock(
    service: InferenceService,
    model_name: Optional[str],
    strategy,
    request,
) -> AsyncIterator[str]:
    """Acquire the swap-lock, then iterate the strategy's sync SSE
    generator and re-yield chunks into the FastAPI event loop.

    Holding the lock for the entire stream is what keeps the active
    model on GPU until the stream completes — otherwise the next
    request could swap it out mid-flight.
    """
    async with service.acquire(model_name):
        for chunk in strategy.generate(request):
            yield chunk


def set_inference_service(service: InferenceService):
    """Set the global inference service instance."""
    global inference_service
    inference_service = service


def get_inference_service() -> Optional[InferenceService]:
    """Get the global inference service instance."""
    return inference_service
