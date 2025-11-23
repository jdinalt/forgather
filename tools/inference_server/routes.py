"""
FastAPI route handlers for inference server.
"""

import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .service import InferenceService
from .models.chat import ChatCompletionRequest
from .models.completion import CompletionRequest
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
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

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
