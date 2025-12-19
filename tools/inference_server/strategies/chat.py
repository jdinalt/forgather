"""
Non-streaming chat completion generation strategy.
"""

import time
import uuid
import torch
from .base import GenerationStrategy
from ..models.chat import (
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    ChatCompletionUsage,
)


class ChatGenerationStrategy(GenerationStrategy):
    """Generates non-streaming chat completions."""

    def generate(self, request):
        """
        Generate a chat completion response.

        Args:
            request: ChatCompletionRequest instance

        Returns:
            ChatCompletionResponse instance
        """
        # Generate request ID
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        # Log request
        self.service.logger.log_request(
            request_id=request_id,
            request_type="chat completion",
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            messages_count=len(request.messages),
        )

        # Log messages
        self.service.logger.log_messages(request_id, request.messages)

        # Format messages using chat template
        prompt = self.service.format_messages(request.messages)
        self.service.logger.log_prompt(request_id, prompt)

        # Tokenize and move to device
        tokenize_result = self.service.tokenizer_wrapper.tokenize_and_move_to_device(
            prompt,
            max_length=None,  # No max_length for chat
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
        stop_strings = self.service.stop_sequences.copy()
        self.service.logger.log_stop_strings(request_id, stop_strings)

        # Generate
        with torch.inference_mode():
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
        raw_generated_text = self.service.tokenizer.decode(
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
            raw_generated_text,
            generated_token_ids,
            generated_tokens,
            self.service.stop_sequences,
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

        # Log response
        self.service.logger.log_response(
            request_id,
            response_text,
            finish_reason,
            prompt_tokens,
            completion_tokens,
        )

        # Build response
        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason=finish_reason,
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
