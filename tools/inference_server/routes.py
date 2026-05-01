"""
FastAPI route handlers for inference server.
"""

import time
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .models.chat import ChatCompletionRequest
from .models.completion import CompletionRequest
from .models.tokenize import TokenizeRequest, TokenizeResponse
from .service import InferenceService
from .strategies import (
    ChatGenerationStrategy,
    CompletionGenerationStrategy,
    StreamingChatStrategy,
    StreamingCompletionStrategy,
)

# Global inference service instance
inference_service: Optional[InferenceService] = None


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(title="HuggingFace OpenAI API Server", version="1.0.0")

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        return {
            "object": "list",
            "data": [
                {
                    "id": (
                        inference_service.model_path.split("/")[-1]
                        if inference_service
                        else "unknown"
                    ),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "huggingface",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest):
        """Create a chat completion."""
        if inference_service is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        try:
            if request.stream:
                strategy = StreamingChatStrategy(inference_service)
                return StreamingResponse(
                    strategy.generate(request),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "close"},
                )
            else:
                strategy = ChatGenerationStrategy(inference_service)
                return strategy.generate(request)
        except Exception as e:
            traceback.print_exception(e)
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    @app.post("/v1/completions")
    async def create_completion(request: CompletionRequest):
        """Create a text completion."""
        if inference_service is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        if request.n != 1:
            raise HTTPException(status_code=400, detail="n > 1 not supported yet")

        try:
            if request.stream:
                strategy = StreamingCompletionStrategy(inference_service)
                return StreamingResponse(
                    strategy.generate(request),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "close"},
                )
            else:
                strategy = CompletionGenerationStrategy(inference_service)
                return strategy.generate(request)
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
                    inference_service.tokenizer.convert_ids_to_tokens(t) for t in tokens
                ]

            return TokenizeResponse(
                count=len(tokens),
                max_model_len=inference_service.get_max_model_len(),
                tokens=tokens,
                token_strs=token_strs,
                prompt=rendered,
            )
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exception(e)
            raise HTTPException(status_code=500, detail=f"Tokenize failed: {str(e)}")

    # vLLM serves /tokenize without the /v1 prefix; we register both
    # paths so existing vLLM clients work and our other-endpoint
    # symmetry is preserved.
    app.post("/tokenize", response_model=TokenizeResponse)(_tokenize)
    app.post("/v1/tokenize", response_model=TokenizeResponse)(_tokenize)

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "model_loaded": inference_service is not None}

    return app


def set_inference_service(service: InferenceService):
    """Set the global inference service instance."""
    global inference_service
    inference_service = service


def get_inference_service() -> Optional[InferenceService]:
    """Get the global inference service instance."""
    return inference_service
