"""
Base class for streaming generation strategies using Template Method pattern.
"""

import time
from abc import abstractmethod
from threading import Thread
from typing import Iterator, Optional, Union

import torch
from transformers import TextIteratorStreamer

from ..models.chat import ChatCompletionRequest
from ..models.completion import CompletionRequest
from .base import GenerationStrategy


class StreamingStrategy(GenerationStrategy):
    """
    Base class for streaming generation using Template Method pattern.

    This class implements the common flow for streaming generation,
    with hook methods for strategy-specific behavior (chat vs completion).
    """

    def generate(
        self, request: Union[ChatCompletionRequest, CompletionRequest]
    ) -> Iterator[str]:
        """
        Template method for streaming generation.

        This method orchestrates the complete streaming generation flow,
        calling hook methods at appropriate points for strategy-specific behavior.

        Args:
            request: ChatCompletionRequest or CompletionRequest instance

        Yields:
            Server-sent event strings in format "data: {...}\\n\\n"
        """
        # 1. Generate request ID
        request_id = self._generate_request_id()

        # 2. Log request start
        self._log_request(request_id, request)

        try:
            # 3. Prepare prompt (hook method)
            prompt = self._prepare_prompt(request, request_id)
            self.service.logger.log_prompt(request_id, prompt)

            # 4. Tokenize input
            device = self.service.tokenizer_wrapper.get_device()
            inputs = self.service.tokenizer(
                prompt, return_tensors="pt", return_token_type_ids=False
            ).to(device)
            input_ids = inputs["input_ids"]
            prompt_tokens = len(input_ids[0])

            # 5. Log input token details
            self.service.logger.log_input_tokens(request_id, input_ids[0].tolist())

            # 6. Build generation config
            generation_config = self.service._build_generation_config(request)
            self.service.logger.log_generation_config(request_id, generation_config)

            # 7. Setup streaming (hook method)
            streamer = self._create_streamer(request)
            generation_kwargs = {
                "input_ids": input_ids,
                "generation_config": generation_config,
                "streamer": streamer,
                "return_dict_in_generate": True,
                "output_scores": False,
            }

            # Track CUDA memory usage, if CUDA device.
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device=device)

            # 8. Start generation in background thread
            def generate_fn():
                with torch.inference_mode():
                    self.service.model.generate(**generation_kwargs)

            thread = Thread(target=generate_fn)
            thread.start()

            # 9. Yield initial chunk if needed (hook method)
            created = int(time.time())
            initial_chunk = self._create_initial_chunk(
                request_id, created, request.model
            )
            if initial_chunk:
                yield initial_chunk

            # 10. Stream tokens with timing
            generation_start = time.perf_counter()
            full_response = ""
            stop_sequences = self._get_stop_sequences(request)

            for new_text in streamer:
                if new_text:  # Skip empty strings
                    full_response += new_text

                    # Check for stop sequences
                    should_stop, remaining_text, stop_seq = (
                        self.service.stop_processor.process_streaming(
                            full_response, new_text, stop_sequences
                        )
                    )

                    if should_stop:
                        if remaining_text:
                            chunk = self._create_chunk(
                                request_id, created, request.model, remaining_text, None
                            )
                            yield chunk
                        break

                    # Send token chunk
                    chunk = self._create_chunk(
                        request_id, created, request.model, new_text, None
                    )
                    yield chunk

            generation_end = time.perf_counter()
            # 11. Calculate timing and rate
            total_time = generation_end - generation_start
            generated_token_ids = self.service.tokenizer.encode(
                full_response, add_special_tokens=False
            )
            completion_tokens = len(generated_token_ids)
            tokens_per_second = (
                completion_tokens / total_time if total_time > 0 else 0.0
            )

            # Get peak memory
            if device.type == "cuda":
                peak_memory = torch.cuda.max_memory_allocated()
            else:
                peak_memory = None

            # 12. Log output details
            self.service.logger.log_generated_tokens(request_id, generated_token_ids)

            # 13. Determine finish reason
            # Pass ignore_eos flag to finish detector
            ignore_eos = getattr(request, "ignore_eos", False)
            finish_reason = (
                self.service.finish_detector.determine_finish_reason_streaming(
                    completion_tokens,
                    request.max_tokens,
                    stop_sequences,
                    full_response,
                    ignore_eos=ignore_eos,
                )
            )

            # 14. Log response and performance
            self.service.logger.log_response(
                request_id,
                full_response,
                finish_reason,
                prompt_tokens,
                completion_tokens,
            )
            self.service.logger.log_generation_rate(
                request_id,
                completion_tokens,
                total_time,
                tokens_per_second,
                peak_memory,
            )

            # 15. Check for stop sequence
            if any(stop_seq in full_response for stop_seq in stop_sequences):
                stop_sequence_found = next(
                    stop_seq for stop_seq in stop_sequences if stop_seq in full_response
                )
                self.service.logger.log_stop_sequence_triggered(
                    request_id, stop_sequence_found
                )

            # 16. Send final chunk
            final_chunk = self._create_chunk(
                request_id, created, request.model, "", finish_reason
            )
            yield final_chunk

            # 17. Send [DONE] marker
            yield "data: [DONE]\n\n"

        except Exception as e:
            self.service.logger.log_streaming_error(request_id, e)
            # Send error chunk
            error_chunk = self._create_error_chunk(request_id, request.model)
            yield error_chunk
            yield "data: [DONE]\n\n"

    # Hook methods to be implemented by subclasses

    @abstractmethod
    def _get_request_id_prefix(self) -> str:
        """
        Return request ID prefix.

        Returns:
            Prefix string ('chatcmpl-' for chat, 'cmpl-' for completion)
        """
        pass

    @abstractmethod
    def _prepare_prompt(
        self, request: Union[ChatCompletionRequest, CompletionRequest], request_id: str
    ) -> str:
        """
        Prepare prompt from request.

        Args:
            request: Request object
            request_id: Request identifier for logging

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def _get_stop_sequences(
        self, request: Union[ChatCompletionRequest, CompletionRequest]
    ) -> list:
        """
        Get stop sequences for this request.

        Args:
            request: Request object

        Returns:
            List of stop sequence strings
        """
        pass

    @abstractmethod
    def _create_streamer(
        self, request: Union[ChatCompletionRequest, CompletionRequest]
    ) -> TextIteratorStreamer:
        """
        Create text iterator streamer with appropriate settings.

        Args:
            request: Request object

        Returns:
            TextIteratorStreamer instance
        """
        pass

    @abstractmethod
    def _create_initial_chunk(
        self, request_id: str, created: int, model: str
    ) -> Optional[str]:
        """
        Create initial chunk (chat sends role, completion sends nothing).

        Args:
            request_id: Request identifier
            created: Creation timestamp
            model: Model name

        Returns:
            SSE-formatted chunk string or None
        """
        pass

    @abstractmethod
    def _create_chunk(
        self,
        request_id: str,
        created: int,
        model: str,
        text: str,
        finish_reason: Optional[str],
    ) -> str:
        """
        Create SSE-formatted chunk.

        Args:
            request_id: Request identifier
            created: Creation timestamp
            model: Model name
            text: Text content for this chunk
            finish_reason: Finish reason (None if not final chunk)

        Returns:
            SSE-formatted chunk string
        """
        pass

    @abstractmethod
    def _create_error_chunk(self, request_id: str, model: str) -> str:
        """
        Create error chunk.

        Args:
            request_id: Request identifier
            model: Model name

        Returns:
            SSE-formatted error chunk string
        """
        pass

    # Common helper methods

    def _generate_request_id(self) -> str:
        """Generate unique request ID with strategy-specific prefix."""
        import uuid

        prefix = self._get_request_id_prefix()
        return f"{prefix}{uuid.uuid4().hex[:12]}"

    def _log_request(
        self, request_id: str, request: Union[ChatCompletionRequest, CompletionRequest]
    ) -> None:
        """Log request start with parameters."""
        # Get request type from prefix
        request_type = (
            "streaming chat completion"
            if "chatcmpl-" in request_id
            else "streaming completion"
        )

        # Build kwargs for strategy-specific logging
        kwargs = {}
        if hasattr(request, "messages"):
            kwargs["messages_count"] = len(request.messages)
        else:
            kwargs["prompt"] = repr(request.prompt)

        self.service.logger.log_request(
            request_id=request_id,
            request_type=request_type,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            **kwargs,
        )
