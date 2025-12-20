"""
Base class for non-streaming generation strategies using Template Method pattern.
"""

import time
from abc import abstractmethod
from typing import Union
import torch
from .base import GenerationStrategy
from ..models.chat import ChatCompletionRequest
from ..models.completion import CompletionRequest


class NonStreamingStrategy(GenerationStrategy):
    """
    Base class for non-streaming generation using Template Method pattern.

    This class implements the common flow for non-streaming generation,
    with hook methods for strategy-specific behavior (chat vs completion).
    """

    def generate(self, request: Union[ChatCompletionRequest, CompletionRequest]):
        """
        Template method defining the non-streaming generation algorithm.

        This method orchestrates the complete generation flow, calling
        hook methods at appropriate points for strategy-specific behavior.

        Args:
            request: ChatCompletionRequest or CompletionRequest instance

        Returns:
            Response object (ChatCompletionResponse or CompletionResponse)
        """
        # 1. Generate request ID
        request_id = self._generate_request_id()

        # 2. Log request start
        self._log_request(request_id, request)

        # 3. Prepare prompt (hook method - differs by strategy)
        prompt = self._prepare_prompt(request, request_id)
        self.service.logger.log_prompt(request_id, prompt)

        # 4. Tokenize input
        tokenize_result = self._tokenize_input(prompt, request)
        input_ids = tokenize_result["input_ids"]
        prompt_tokens = tokenize_result["prompt_tokens"]

        # 5. Log input tokens
        input_token_ids = input_ids[0].tolist()
        self.service.logger.log_input_tokens(request_id, input_token_ids)

        # 6. Build generation config
        generation_config = self.service._build_generation_config(request)
        self.service.logger.log_generation_config(request_id, generation_config)

        # 7. Prepare stop sequences (hook method)
        stop_strings = self._get_stop_sequences(request)
        self.service.logger.log_stop_strings(request_id, stop_strings)

        # 8. Generate tokens with timing
        generation_start = time.perf_counter()
        with torch.inference_mode():
            # Only pass stop_strings if not empty
            generation_kwargs = {
                "input_ids": input_ids,
                "generation_config": generation_config,
                "return_dict_in_generate": True,
                "output_scores": False,
                "tokenizer": self.service.tokenizer,
            }
            if stop_strings:
                generation_kwargs["stop_strings"] = stop_strings

            outputs = self.service.model.generate(**generation_kwargs)
        generation_end = time.perf_counter()

        # 9. Extract generated tokens
        generated_tokens = outputs.sequences[0][prompt_tokens:]
        generated_token_ids = generated_tokens.tolist()

        # 10. Log raw generated output
        raw_generated_text = self.service.tokenizer.decode(
            generated_token_ids, skip_special_tokens=False
        )
        self.service.logger.log_generated_tokens(request_id, generated_token_ids)

        # 11. Process stop sequences
        (
            generated_token_ids,
            generated_tokens,
            stopped_by_sequence,
            stop_sequence_found,
        ) = self.service.stop_processor.process(
            raw_generated_text,
            generated_token_ids,
            generated_tokens,
            stop_strings,
        )

        # 12. Determine finish reason
        finish_reason = self.service.finish_detector.determine_finish_reason(
            generated_token_ids,
            request.max_tokens,
            stopped_by_sequence,
        )

        # 13. Log stop sequence if triggered
        if stopped_by_sequence:
            self.service.logger.log_stop_sequence_triggered(
                request_id, stop_sequence_found
            )

        # 14. Decode and process final response text (hook method)
        response_text = self._process_response_text(generated_tokens, request, prompt)
        completion_tokens = len(generated_tokens)

        # 15. Calculate timing and rate
        total_time = generation_end - generation_start
        tokens_per_second = completion_tokens / total_time if total_time > 0 else 0.0

        # 16. Log response and performance
        self.service.logger.log_response(
            request_id,
            response_text,
            finish_reason,
            prompt_tokens,
            completion_tokens,
        )
        self.service.logger.log_generation_rate(
            request_id,
            completion_tokens,
            total_time,
            tokens_per_second,
        )

        # 17. Build and return response (hook method)
        return self._build_response(
            request_id=request_id,
            request=request,
            response_text=response_text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

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
    def _tokenize_input(
        self, prompt: str, request: Union[ChatCompletionRequest, CompletionRequest]
    ) -> dict:
        """
        Tokenize input prompt.

        Args:
            prompt: Formatted prompt string
            request: Request object

        Returns:
            Dict with 'input_ids' and 'prompt_tokens'
        """
        pass

    @abstractmethod
    def _process_response_text(
        self,
        generated_tokens: torch.Tensor,
        request: Union[ChatCompletionRequest, CompletionRequest],
        original_prompt: str,
    ) -> str:
        """
        Process final response text (handle echo, etc.).

        Args:
            generated_tokens: Generated token tensor
            request: Request object
            original_prompt: Original prompt string

        Returns:
            Final response text
        """
        pass

    @abstractmethod
    def _build_response(
        self,
        request_id: str,
        request: Union[ChatCompletionRequest, CompletionRequest],
        response_text: str,
        finish_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
    ):
        """
        Build final response object.

        Args:
            request_id: Request identifier
            request: Original request object
            response_text: Generated response text
            finish_reason: Why generation stopped
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Response object (ChatCompletionResponse or CompletionResponse)
        """
        pass

    # Common helper methods

    def _generate_request_id(self) -> str:
        """Generate unique request ID with strategy-specific prefix."""
        import uuid

        prefix = self._get_request_id_prefix()
        return f"{prefix}{uuid.uuid4().hex[:8]}"

    def _log_request(
        self, request_id: str, request: Union[ChatCompletionRequest, CompletionRequest]
    ) -> None:
        """Log request start with parameters."""
        # Get request type from prefix
        request_type = "chat completion" if "chatcmpl-" in request_id else "completion"

        # Build kwargs for strategy-specific logging
        kwargs = {}
        if hasattr(request, "messages"):
            kwargs["messages_count"] = len(request.messages)
        else:
            kwargs["prompt_length"] = (
                len(request.prompt)
                if isinstance(request.prompt, str)
                else len(request.prompt[0])
            )

        self.service.logger.log_request(
            request_id=request_id,
            request_type=request_type,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            **kwargs,
        )
