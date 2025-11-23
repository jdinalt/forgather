"""
Non-streaming text completion generation strategy.
"""

import time
import uuid
import torch
from fastapi import HTTPException
from .base import GenerationStrategy
from ..models.completion import (
    CompletionResponse,
    CompletionChoice,
    ChatCompletionUsage,
)


class CompletionGenerationStrategy(GenerationStrategy):
    """Generates non-streaming text completions."""

    def generate(self, request):
        """
        Generate a text completion response.

        Args:
            request: CompletionRequest instance

        Returns:
            CompletionResponse instance
        """
        # Handle single prompt vs list of prompts
        if isinstance(request.prompt, list):
            if len(request.prompt) != 1:
                raise HTTPException(
                    status_code=400, detail="Multiple prompts not supported yet"
                )
            prompt = request.prompt[0]
        else:
            prompt = request.prompt

        # Generate request ID
        request_id = f"cmpl-{uuid.uuid4().hex[:8]}"

        # Log request
        self.service.logger.log_request(
            request_id=request_id,
            request_type="completion",
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            prompt_length=len(prompt),
        )
        self.service.logger.log_prompt(request_id, prompt)

        # Parse stop sequences from request
        request_stop_sequences = []
        if request.stop:
            if isinstance(request.stop, str):
                request_stop_sequences = [request.stop] if request.stop else []
            else:
                request_stop_sequences = request.stop

        # Combine with server stop sequences and filter out empty strings
        all_stop_sequences = [
            s for s in (self.service.stop_sequences + request_stop_sequences) if s
        ]

        # Tokenize and move to device
        tokenize_result = self.service.tokenizer_wrapper.tokenize_and_move_to_device(
            prompt,
            max_length=2048,
        )
        input_ids = tokenize_result["input_ids"]
        prompt_tokens = tokenize_result["prompt_tokens"]

        # Log input tokens
        input_token_ids = input_ids[0].tolist()
        self.service.logger.log_input_tokens(request_id, input_token_ids)

        # Build generation configuration
        generation_config = self.service._build_generation_config(request)
        self.service.logger.log_generation_config(request_id, generation_config)

        # Prepare stop sequences
        stop_strings = all_stop_sequences.copy()
        self.service.logger.log_stop_strings(request_id, stop_strings)

        # Generate
        with torch.no_grad():
            outputs = self.service.model.generate(
                input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                stop_strings=stop_strings,
                tokenizer=self.service.tokenizer,
            )

        generated_tokens = outputs.sequences[0][prompt_tokens:]
        generated_token_ids = generated_tokens.tolist()

        # Log raw generated output
        generated_text_with_special = self.service.tokenizer.decode(
            generated_token_ids, skip_special_tokens=False
        )
        self.service.logger.log_generated_tokens(request_id, generated_token_ids)

        # Process stop sequences
        (
            generated_token_ids,
            generated_tokens,
            stopped_by_sequence,
            stop_sequence_found,
        ) = self.service.stop_processor.process(
            generated_text_with_special,
            generated_token_ids,
            generated_tokens,
            all_stop_sequences,
        )

        # Determine finish reason
        finish_reason = self.service.finish_detector.determine_finish_reason(
            generated_token_ids,
            request.max_tokens,
            stopped_by_sequence,
        )

        # Log stop sequence if triggered
        if stopped_by_sequence:
            self.service.logger.log_stop_sequence_triggered(
                request_id, stop_sequence_found
            )

        # Decode final response
        response_text = self.service.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        completion_tokens = len(generated_tokens)

        # Handle echo parameter (include original prompt in response)
        if request.echo:
            response_text = prompt + response_text

        # Log response
        self.service.logger.log_response(
            request_id,
            response_text,
            finish_reason,
            prompt_tokens,
            completion_tokens,
        )

        # Build response
        return CompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    text=response_text, index=0, finish_reason=finish_reason
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
