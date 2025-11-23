"""
Streaming chat completion generation strategy.
"""

import time
import uuid
from threading import Thread
from typing import Iterator
from transformers import TextIteratorStreamer
from .base import GenerationStrategy
from ..models.chat import (
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamDelta,
)


class StreamingChatStrategy(GenerationStrategy):
    """Generates streaming chat completions."""

    def generate(self, request) -> Iterator[str]:
        """
        Generate a streaming chat completion response.

        Args:
            request: ChatCompletionRequest instance

        Yields:
            Server-sent event strings in format "data: {...}\\n\\n"
        """
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        # Log request details
        self.service.logger.log_request(
            request_id=request_id,
            request_type="streaming chat completion",
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            messages_count=len(request.messages),
        )

        # Log each message
        self.service.logger.log_messages(request_id, request.messages)

        try:
            # Format messages using chat template
            template = self.service.jinja_env.from_string(self.service.chat_template)
            formatted_prompt = template.render(
                messages=request.messages,
                bos_token=self.service.tokenizer.bos_token,
                eos_token=self.service.tokenizer.eos_token,
                add_generation_prompt=True,
            )
            self.service.logger.log_prompt(request_id, formatted_prompt)

            # Tokenize input
            inputs = self.service.tokenizer(
                formatted_prompt, return_tensors="pt", return_token_type_ids=False
            ).to(self.service.tokenizer_wrapper.get_device())
            input_ids = inputs["input_ids"]
            prompt_tokens = len(input_ids[0])

            # Log input token details
            self.service.logger.log_input_tokens(request_id, input_ids[0].tolist())

            # Build generation config
            generation_config = self.service._build_generation_config(request)
            self.service.logger.log_generation_config(request_id, generation_config)

            # Setup streaming
            streamer = TextIteratorStreamer(
                self.service.tokenizer,
                timeout=60.0,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generation_kwargs = {
                "input_ids": input_ids,
                "generation_config": generation_config,
                "streamer": streamer,
                "return_dict_in_generate": True,
                "output_scores": False,
            }

            # Start generation in background thread
            thread = Thread(
                target=self.service.model.generate, kwargs=generation_kwargs
            )
            thread.start()

            # Send initial chunk with role
            created = int(time.time())
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=ChatCompletionStreamDelta(role="assistant", content=""),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

            # Stream tokens
            full_response = ""
            for new_text in streamer:
                if new_text:  # Skip empty strings
                    full_response += new_text

                    # Check for stop sequences
                    should_stop, remaining_text, stop_seq = (
                        self.service.stop_processor.process_streaming(
                            full_response, new_text, self.service.stop_sequences
                        )
                    )

                    if should_stop:
                        if remaining_text:
                            chunk = ChatCompletionStreamResponse(
                                id=request_id,
                                created=created,
                                model=request.model,
                                choices=[
                                    ChatCompletionStreamChoice(
                                        index=0,
                                        delta=ChatCompletionStreamDelta(
                                            content=remaining_text
                                        ),
                                        finish_reason=None,
                                    )
                                ],
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"
                        break

                    # Send token chunk
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0,
                                delta=ChatCompletionStreamDelta(content=new_text),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

            # Log output details immediately after streaming completes
            generated_token_ids = self.service.tokenizer.encode(
                full_response, add_special_tokens=False
            )
            completion_tokens = len(generated_token_ids)

            self.service.logger.log_generated_tokens(request_id, generated_token_ids)
            self.service.logger.log_response(
                request_id, full_response, "stop", prompt_tokens, completion_tokens
            )

            # Determine finish reason
            finish_reason = (
                self.service.finish_detector.determine_finish_reason_streaming(
                    completion_tokens,
                    request.max_tokens,
                    self.service.stop_sequences,
                    full_response,
                )
            )

            if any(
                stop_seq in full_response for stop_seq in self.service.stop_sequences
            ):
                stop_sequence_found = next(
                    stop_seq
                    for stop_seq in self.service.stop_sequences
                    if stop_seq in full_response
                )
                self.service.logger.log_stop_sequence_triggered(
                    request_id, stop_sequence_found
                )

            # Send final chunk
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0, delta=ChatCompletionStreamDelta(), finish_reason="stop"
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

            # Send [DONE] marker
            yield "data: [DONE]\n\n"

        except Exception as e:
            self.service.logger.log_streaming_error(request_id, e)
            # Send error as final chunk
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0, delta=ChatCompletionStreamDelta(), finish_reason="stop"
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
