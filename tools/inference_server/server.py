#!/usr/bin/env python3
"""
OpenAI API-compatible inference server for HuggingFace models.
"""

import argparse
import logging
import os
import time
import uuid
import yaml
import json
import asyncio
from typing import List, Optional, Dict, Any, Union, Iterator
from threading import Thread
from queue import Queue

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextIteratorStreamer, AutoConfig
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from jinja2 import Environment, BaseLoader, TemplateError


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    
    # Additional HuggingFace generation parameters
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    no_repeat_ngram_size: Optional[int] = None
    encoder_no_repeat_ngram_size: Optional[int] = None
    bad_words_ids: Optional[List[List[int]]] = None
    min_length: Optional[int] = None
    max_new_tokens: Optional[int] = None
    early_stopping: Optional[bool] = None
    num_beams: Optional[int] = None
    num_beam_groups: Optional[int] = None
    diversity_penalty: Optional[float] = None
    temperature_last_layer: Optional[bool] = None
    top_k: Optional[int] = None
    typical_p: Optional[float] = None
    epsilon_cutoff: Optional[float] = None
    eta_cutoff: Optional[float] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None


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


# Completions API models
class CompletionRequest(BaseModel):
    # OpenAI API parameters
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    
    # Additional HuggingFace generation parameters
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    no_repeat_ngram_size: Optional[int] = None
    encoder_no_repeat_ngram_size: Optional[int] = None
    bad_words_ids: Optional[List[List[int]]] = None
    min_length: Optional[int] = None
    max_new_tokens: Optional[int] = None  # Alternative to max_tokens
    early_stopping: Optional[bool] = None
    num_beams: Optional[int] = None
    num_beam_groups: Optional[int] = None
    diversity_penalty: Optional[float] = None
    temperature_last_layer: Optional[bool] = None
    top_k: Optional[int] = None
    typical_p: Optional[float] = None
    epsilon_cutoff: Optional[float] = None
    eta_cutoff: Optional[float] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: ChatCompletionUsage  # Same usage format


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


class CompletionStreamChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionStreamChoice]


class InferenceServer:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        from_checkpoint: bool|str = False,
        chat_template_path: Optional[str] = None,
        dtype: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None
    ):
        # Create dedicated logger for this application
        self.logger = logging.getLogger('inference_server')
        
        self.model_path = model_path
        self.device = device
        self.from_checkpoint = from_checkpoint
        self.chat_template_path = chat_template_path
        self.dtype = self._resolve_dtype(dtype)
        self.stop_sequences = stop_sequences or []
        self.chat_template = None
        self.tokenizer = None
        self.model = None
        self.default_generation_config = None
        self.jinja_env = Environment(loader=BaseLoader())
        self.load_model()
        self.setup_chat_template()
        self._setup_stop_tokens()

    def load_model(self):
        self.logger.info(f"Loading model from {self.model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.from_checkpoint:
            if self.device == "auto":
                raise ValueError("Cannot use 'auto' device with checkpoint loading. Please specify a device explicitly.")
            
            from forgather.ml.sharded_checkpoint import (
                load_checkpoint,
                retie_parameters,
                find_latest_checkpoint,
                create_sharing_metadata,
            )

            if isinstance(self.from_checkpoint, bool):
                checkpoint_path = find_latest_checkpoint(self.model_path)
                if not checkpoint_path:
                    raise ValueError(
                        f"No checkpoints found in {self.model_path}. Please provide a valid model directory."
                    )
            elif isinstance(self.from_checkpoint, str):
                checkpoint_path = self.from_checkpoint
            else:
                raise ValueError("from_checkpoint must be a boolean or a string path")
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")
            
            self.logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            model_config = AutoConfig.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            # Use meta device for empty model creation
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(
                    model_config, trust_remote_code=True
                )
            
            sharing_metadata = create_sharing_metadata(model)

            # Convert model to the specified dtype and device
            model.to(dtype=self.dtype)
            model.to_empty(device=self.device)

            # When converted to empty, tied parameters are not automatically retied.
            retie_parameters(model, sharing_metadata)
            load_checkpoint(checkpoint_path, model, device=self.device, strict=True)
            self.model = model
            
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                device_map=self.device if self.device != "auto" else "auto",
                trust_remote_code=True
            )
        
            if self.device != "auto" and torch.cuda.is_available():
                self.model = self.model.to(self.device)
        
        # Load generation config from model directory if available
        self._load_generation_config()
        
        self.logger.info(f"Model loaded successfully on device: {self.model.device} with dtype: {self.dtype}")
    
    def _load_generation_config(self):
        """Load generation config from model directory if available."""
        try:
            self.default_generation_config = GenerationConfig.from_pretrained(self.model_path)
            self.logger.info(f"Loaded generation config from model directory: {self.model_path}")
            self.logger.info(f"Default generation config: {self.default_generation_config}")
        except Exception as e:
            self.logger.info(f"No generation config found in model directory or failed to load: {e}")
            # Fallback to model's generation config if available
            if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
                self.default_generation_config = self.model.generation_config
                self.logger.info("Using model's built-in generation config")
            else:
                self.default_generation_config = GenerationConfig()
                self.logger.info("Using default GenerationConfig")
        self.logger.info(f"Final default generation config: {self.default_generation_config}")
    
    def _resolve_dtype(self, dtype_str: Optional[str]) -> torch.dtype:
        """Resolve dtype string to torch.dtype with intelligent defaults."""
        if dtype_str is None:
            # Default to bfloat16 if supported, otherwise float16 on GPU, float32 on CPU
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                else:
                    return torch.float16
            else:
                return torch.float32
        
        # Parse user-specified dtype
        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float64": torch.float64,
            "fp64": torch.float64,
            "double": torch.float64,
        }
        
        dtype_str = dtype_str.lower()
        if dtype_str not in dtype_map:
            available_dtypes = ", ".join(dtype_map.keys())
            raise ValueError(f"Unsupported dtype '{dtype_str}'. Available options: {available_dtypes}")
        
        requested_dtype = dtype_map[dtype_str]
        
        # Validate bfloat16 support
        if requested_dtype == torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            self.logger.warning(f"bfloat16 not supported on this GPU, falling back to float16")
            return torch.float16
        
        return requested_dtype
    
    def setup_chat_template(self):
        """Setup chat template with priority: custom file > tokenizer > default fallback."""
        if self.chat_template_path and os.path.exists(self.chat_template_path):
            # Use custom template file
            with open(self.chat_template_path, 'r') as f:
                self.chat_template = f.read()
            self.logger.info(f"Using custom chat template from: {self.chat_template_path}")
        elif hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # Use tokenizer's built-in template
            self.chat_template = self.tokenizer.chat_template
            self.logger.info("Using tokenizer's built-in chat template")
        else:
            # Use default fallback template
            self.chat_template = self.get_default_chat_template()
            self.logger.info("Using default fallback chat template")
        self.logger.info(f"Chat template loaded: {repr(self.chat_template)}")
    
    def _setup_stop_tokens(self):
        """Setup stop token IDs from stop sequences."""
        self.stop_token_ids = set()
        
        # Always include native EOS token
        if self.tokenizer.eos_token_id is not None:
            self.stop_token_ids.add(self.tokenizer.eos_token_id)
        
        # Add custom stop sequences
        for sequence in self.stop_sequences:
            try:
                token_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
                if len(token_ids) == 1:
                    # Single token - can use as direct stopping criterion
                    self.stop_token_ids.add(token_ids[0])
                    self.logger.info(f"Added single-token stop sequence: {repr(sequence)} -> token ID {token_ids[0]}")
                else:
                    # Multi-token sequence - will need post-processing
                    self.logger.info(f"Added multi-token stop sequence: {repr(sequence)} -> token IDs {token_ids}")
            except Exception as e:
                self.logger.warning(f"Failed to tokenize stop sequence {repr(sequence)}: {e}")
        
        self.logger.info(f"Stop token IDs: {sorted(self.stop_token_ids)}")
    
    def _build_generation_config(self, request: Union[ChatCompletionRequest, CompletionRequest]) -> GenerationConfig:
        """Build a GenerationConfig from request parameters."""
        # Base parameters that are always used
        max_tokens = getattr(request, 'max_new_tokens', None) or getattr(request, 'max_tokens', 16)
        
        # Start with the loaded default generation config
        if self.default_generation_config is not None:
            # Create a copy to avoid modifying the default
            generation_config = GenerationConfig(**self.default_generation_config.to_dict())
        else:
            generation_config = GenerationConfig()
        
        # Core parameters
        generation_config.max_new_tokens = max_tokens
        generation_config.temperature = request.temperature
        generation_config.top_p = request.top_p
        generation_config.do_sample = request.temperature > 0
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.return_dict_in_generate = True
        generation_config.output_scores = False
        
        # Set early_stopping properly - only use with beam search (num_beams > 1)
        early_stopping_value = getattr(request, 'early_stopping', None)
        if early_stopping_value is not None:
            generation_config.early_stopping = early_stopping_value
        # Don't set early_stopping=True by default since it only works with beam search
        
        # Add HuggingFace specific parameters if they are not None
        hf_params = [
            'repetition_penalty', 'length_penalty', 'no_repeat_ngram_size',
            'encoder_no_repeat_ngram_size', 'bad_words_ids', 'min_length',
            'num_beams', 'num_beam_groups', 'diversity_penalty', 
            'temperature_last_layer', 'top_k', 'typical_p',
            'epsilon_cutoff', 'eta_cutoff', 'guidance_scale'
        ]
        
        for param in hf_params:
            value = getattr(request, param, None)
            if value is not None:
                setattr(generation_config, param, value)
        
        # Handle special cases
        if hasattr(request, 'seed') and request.seed is not None:
            # Set random seed for reproducibility
            torch.manual_seed(request.seed)
        
        # If using beam search, adjust sampling and early_stopping
        if generation_config.num_beams and generation_config.num_beams > 1:
            generation_config.do_sample = False  # Beam search doesn't use sampling
            # Only enable early_stopping with beam search if not explicitly set
            if early_stopping_value is None:
                generation_config.early_stopping = True
        
        return generation_config
    
    def get_default_chat_template(self) -> str:
        """Return a reasonable default chat template as Jinja2."""
        return '''{%- for message in messages %}
    {%- if message['role'] == 'system' -%}
        System: {{ message['content'] }}\\n\\n
    {%- elif message['role'] == 'user' -%}
        User: {{ message['content'] }}\\n\\n
    {%- elif message['role'] == 'assistant' -%}
        Assistant: {{ message['content'] }}\\n\\n
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    Assistant: 
{%- endif -%}'''

    def format_messages(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to a single prompt string using Jinja2 template."""
        try:
            # Prepare message data for template
            message_data = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            # Create Jinja2 template and render
            template = self.jinja_env.from_string(self.chat_template)
            formatted = template.render(
                messages=message_data,
                add_generation_prompt=True
            )
            return formatted
            
        except TemplateError as e:
            self.logger.error(f"Template error: {e}")
            # Fallback to simple formatting if template fails
            return self._fallback_format_messages(messages)
        except Exception as e:
            self.logger.error(f"Unexpected error in template rendering: {e}")
            return self._fallback_format_messages(messages)
    
    def _fallback_format_messages(self, messages: List[ChatMessage]) -> str:
        """Simple fallback formatting if template fails."""
        formatted = ""
        for message in messages:
            if message.role == "system":
                formatted += f"System: {message.content}\n\n"
            elif message.role == "user":
                formatted += f"User: {message.content}\n\n"
            elif message.role == "assistant":
                formatted += f"Assistant: {message.content}\n\n"
        
        formatted += "Assistant: "
        return formatted

    def generate_response(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        # Log incoming request details
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        self.logger.info(f"[{request_id}] New chat completion request: model={request.model}, "
                    f"max_tokens={request.max_tokens}, temperature={request.temperature}, "
                    f"top_p={request.top_p}, messages_count={len(request.messages)}")
        
        # Log individual messages
        for i, msg in enumerate(request.messages):
            self.logger.info(f"[{request_id}] Message {i}: role={msg.role}, content={repr(msg.content)}")
        
        prompt = self.format_messages(request.messages)
        self.logger.info(f"[{request_id}] Formatted prompt: {repr(prompt)}")
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            return_token_type_ids=False
        )
        
        # Move inputs to model device
        input_ids = inputs["input_ids"]
        if hasattr(self.model, 'device'):
            input_ids = input_ids.to(self.model.device)
        elif torch.cuda.is_available():
            input_ids = input_ids.to(next(self.model.parameters()).device)
        
        prompt_tokens = input_ids.shape[1]
        
        # Log tokenized input with special token representations
        input_token_ids = input_ids[0].tolist()
        self.logger.info(f"[{request_id}] Input token IDs: {input_token_ids}")
        self.logger.info(f"[{request_id}] Input tokens with special tokens: {repr(self.tokenizer.decode(input_token_ids, skip_special_tokens=False))}")
        
        # Build generation configuration
        generation_config = self._build_generation_config(request)
        self.logger.info(f"[{request_id}] Generation config: {generation_config}")
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                generation_config=generation_config
            )
        
        generated_tokens = outputs.sequences[0][prompt_tokens:]
        generated_token_ids = generated_tokens.tolist()
        
        # Check for stop sequences in generated text
        generated_text_with_special = self.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        stopped_by_sequence = False
        stop_sequence_found = None
        
        for sequence in self.stop_sequences:
            if sequence in generated_text_with_special:
                stopped_by_sequence = True
                stop_sequence_found = sequence
                # Trim the generated text at the stop sequence
                stop_index = generated_text_with_special.find(sequence)
                generated_text_with_special = generated_text_with_special[:stop_index]
                # Re-encode to get the trimmed tokens
                trimmed_tokens = self.tokenizer.encode(generated_text_with_special, add_special_tokens=False)
                if len(trimmed_tokens) < len(generated_token_ids):
                    generated_token_ids = trimmed_tokens
                    generated_tokens = torch.tensor(generated_token_ids, device=generated_tokens.device)
                break
        
        # Determine finish reason
        finish_reason = "stop"  # Default to "stop"
        if len(generated_token_ids) >= request.max_tokens:
            finish_reason = "length"  # Reached max tokens
        elif stopped_by_sequence:
            finish_reason = "stop"  # Stopped by custom sequence
            self.logger.info(f"[{request_id}] Generation stopped due to stop sequence: {repr(stop_sequence_found)}")
        elif (self.tokenizer.eos_token_id is not None and 
              len(generated_token_ids) > 0 and 
              generated_token_ids[-1] == self.tokenizer.eos_token_id):
            finish_reason = "stop"  # EOS token found
            self.logger.info(f"[{request_id}] Generation stopped due to EOS token")
        elif (len(generated_token_ids) > 0 and 
              generated_token_ids[-1] in self.stop_token_ids):
            finish_reason = "stop"  # Stopped by configured stop token
            self.logger.info(f"[{request_id}] Generation stopped due to stop token ID: {generated_token_ids[-1]}")
        elif len(generated_token_ids) < request.max_tokens:
            # Stopped early but not due to obvious reasons
            finish_reason = "stop"
        
        # Log generated tokens with special token representations
        self.logger.info(f"[{request_id}] Generated token IDs: {generated_token_ids}")
        self.logger.info(f"[{request_id}] Generated tokens with special tokens: {repr(self.tokenizer.decode(generated_token_ids, skip_special_tokens=False))}")
        
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        completion_tokens = len(generated_tokens)
        
        # Log final response details
        self.logger.info(f"[{request_id}] Response text (clean): {repr(response_text)}")
        self.logger.info(f"[{request_id}] Finish reason: {finish_reason}")
        self.logger.info(f"[{request_id}] Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={prompt_tokens + completion_tokens}")
        
        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason=finish_reason
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        """Generate text completion for the given prompt."""
        # Handle single prompt vs list of prompts
        if isinstance(request.prompt, list):
            if len(request.prompt) != 1:
                raise HTTPException(status_code=400, detail="Multiple prompts not supported yet")
            prompt = request.prompt[0]
        else:
            prompt = request.prompt

        # Log incoming request details
        request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        self.logger.info(f"[{request_id}] New completion request: model={request.model}, "
                    f"max_tokens={request.max_tokens}, temperature={request.temperature}, "
                    f"top_p={request.top_p}, prompt_length={len(prompt)}")
        self.logger.info(f"[{request_id}] Prompt: {repr(prompt)}")

        # Parse stop sequences from request
        request_stop_sequences = []
        if request.stop:
            if isinstance(request.stop, str):
                request_stop_sequences = [request.stop]
            else:
                request_stop_sequences = request.stop

        # Combine with server stop sequences
        all_stop_sequences = self.stop_sequences + request_stop_sequences

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            return_token_type_ids=False
        )

        # Move inputs to model device
        input_ids = inputs["input_ids"]
        if hasattr(self.model, 'device'):
            input_ids = input_ids.to(self.model.device)
        elif torch.cuda.is_available():
            input_ids = input_ids.to(next(self.model.parameters()).device)

        prompt_tokens = input_ids.shape[1]

        # Log tokenized input
        input_token_ids = input_ids[0].tolist()
        self.logger.info(f"[{request_id}] Input token IDs: {input_token_ids}")
        self.logger.info(f"[{request_id}] Input tokens with special tokens: {repr(self.tokenizer.decode(input_token_ids, skip_special_tokens=False))}")

        # Build generation configuration
        generation_config = self._build_generation_config(request)
        self.logger.info(f"[{request_id}] Generation config: {generation_config}")
        
        with torch.no_grad():
            outputs = self.model.generate(input_ids, generation_config=generation_config)

        generated_tokens = outputs.sequences[0][prompt_tokens:]
        generated_token_ids = generated_tokens.tolist()

        # Check for stop sequences in generated text
        generated_text_with_special = self.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        stopped_by_sequence = False
        stop_sequence_found = None

        for sequence in all_stop_sequences:
            if sequence in generated_text_with_special:
                stopped_by_sequence = True
                stop_sequence_found = sequence
                # Trim the generated text at the stop sequence
                stop_index = generated_text_with_special.find(sequence)
                generated_text_with_special = generated_text_with_special[:stop_index]
                # Re-encode to get the trimmed tokens
                trimmed_tokens = self.tokenizer.encode(generated_text_with_special, add_special_tokens=False)
                if len(trimmed_tokens) < len(generated_token_ids):
                    generated_token_ids = trimmed_tokens
                    generated_tokens = torch.tensor(generated_token_ids, device=generated_tokens.device)
                break

        # Determine finish reason
        finish_reason = "stop"
        if len(generated_token_ids) >= request.max_tokens:
            finish_reason = "length"
        elif stopped_by_sequence:
            finish_reason = "stop"
            self.logger.info(f"[{request_id}] Generation stopped due to stop sequence: {repr(stop_sequence_found)}")
        elif (self.tokenizer.eos_token_id is not None and 
              len(generated_token_ids) > 0 and 
              generated_token_ids[-1] == self.tokenizer.eos_token_id):
            finish_reason = "stop"
            self.logger.info(f"[{request_id}] Generation stopped due to EOS token")
        elif (len(generated_token_ids) > 0 and 
              generated_token_ids[-1] in self.stop_token_ids):
            finish_reason = "stop"
            self.logger.info(f"[{request_id}] Generation stopped due to stop token ID: {generated_token_ids[-1]}")

        # Log generated tokens
        self.logger.info(f"[{request_id}] Generated token IDs: {generated_token_ids}")
        self.logger.info(f"[{request_id}] Generated tokens with special tokens: {repr(self.tokenizer.decode(generated_token_ids, skip_special_tokens=False))}")

        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        completion_tokens = len(generated_tokens)

        # Handle echo parameter (include original prompt in response)
        if request.echo:
            response_text = prompt + response_text

        # Log final response details
        self.logger.info(f"[{request_id}] Response text (clean): {repr(response_text)}")
        self.logger.info(f"[{request_id}] Finish reason: {finish_reason}")
        self.logger.info(f"[{request_id}] Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={prompt_tokens + completion_tokens}")

        return CompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    text=response_text,
                    index=0,
                    finish_reason=finish_reason
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    def generate_stream_response(self, request: ChatCompletionRequest) -> Iterator[str]:
        """Generate a streaming chat completion response."""
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        
        # Log request details
        self.logger.info(f"[{request_id}] New streaming chat completion request: model={request.model}, max_tokens={request.max_tokens}, temperature={request.temperature}, top_p={request.top_p}, messages_count={len(request.messages)}")
        
        # Log each message
        for i, message in enumerate(request.messages):
            self.logger.info(f"[{request_id}] Message {i}: role={message.role}, content={repr(message.content)}")
        
        try:
            # Format messages using chat template
            template = self.jinja_env.from_string(self.chat_template)
            formatted_prompt = template.render(messages=request.messages, add_generation_prompt=True)
            self.logger.info(f"[{request_id}] Formatted prompt: {repr(formatted_prompt)}")
            
            # Tokenize input
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", return_token_type_ids=False).to(self.model.device)
            input_ids = inputs["input_ids"]
            prompt_tokens = len(input_ids[0])
            
            # Log input token details
            self.logger.info(f"[{request_id}] Input token IDs: {input_ids[0].tolist()}")
            self.logger.info(f"[{request_id}] Input tokens with special tokens: {repr(self.tokenizer.decode(input_ids[0], skip_special_tokens=False))}")
            
            # Build generation config
            generation_config = self._build_generation_config(request)
            self.logger.info(f"[{request_id}] Generation config: {generation_config}")
            
            # Setup streaming
            streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = {
                "input_ids": input_ids,
                "generation_config": generation_config,
                "streamer": streamer,
                "return_dict_in_generate": True,
                "output_scores": False,
            }
            
            # Start generation in background thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
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
                        finish_reason=None
                    )
                ]
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            
            # Stream tokens
            full_response = ""
            for new_text in streamer:
                if new_text:  # Skip empty strings
                    full_response += new_text
                    
                    # Check for stop sequences
                    should_stop = False
                    for stop_seq in self.stop_sequences:
                        if stop_seq in full_response:
                            # Trim at stop sequence
                            stop_index = full_response.find(stop_seq)
                            trimmed_response = full_response[:stop_index]
                            remaining_text = trimmed_response[len(full_response) - len(new_text):]
                            if remaining_text:
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    created=created,
                                    model=request.model,
                                    choices=[
                                        ChatCompletionStreamChoice(
                                            index=0,
                                            delta=ChatCompletionStreamDelta(content=remaining_text),
                                            finish_reason=None
                                        )
                                    ]
                                )
                                yield f"data: {chunk.model_dump_json()}\n\n"
                            should_stop = True
                            break
                    
                    if should_stop:
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
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
            
            # Send final chunk
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=ChatCompletionStreamDelta(),
                        finish_reason="stop"
                    )
                ]
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            
            # Send [DONE] marker
            yield "data: [DONE]\n\n"
            
            # Wait for generation to complete
            thread.join()
            
            # Log output details (similar to non-streaming version)
            generated_token_ids = self.tokenizer.encode(full_response, add_special_tokens=False)
            completion_tokens = len(generated_token_ids)
            
            self.logger.info(f"[{request_id}] Generated token IDs: {generated_token_ids}")
            self.logger.info(f"[{request_id}] Generated tokens with special tokens: {repr(full_response)}")
            self.logger.info(f"[{request_id}] Response text (clean): {repr(full_response)}")
            
            # Determine finish reason
            finish_reason = "stop"  # Default for streaming
            if completion_tokens >= request.max_tokens:
                finish_reason = "length"
            elif any(stop_seq in full_response for stop_seq in self.stop_sequences):
                finish_reason = "stop"
                stop_sequence_found = next(stop_seq for stop_seq in self.stop_sequences if stop_seq in full_response)
                self.logger.info(f"[{request_id}] Generation stopped due to stop sequence: {repr(stop_sequence_found)}")
            
            self.logger.info(f"[{request_id}] Finish reason: {finish_reason}")
            self.logger.info(f"[{request_id}] Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={prompt_tokens + completion_tokens}")
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Streaming generation failed: {str(e)}")
            # Send error as final chunk
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=ChatCompletionStreamDelta(),
                        finish_reason="stop"
                    )
                ]
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

    def generate_stream_completion(self, request: CompletionRequest) -> Iterator[str]:
        """Generate a streaming text completion response."""
        request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        
        # Log request details
        self.logger.info(f"[{request_id}] New streaming completion request: model={request.model}, max_tokens={request.max_tokens}, prompt={repr(request.prompt)}")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(request.prompt, return_tensors="pt", return_token_type_ids=False).to(self.model.device)
            input_ids = inputs["input_ids"]
            prompt_tokens = len(input_ids[0])
            
            # Log input token details
            self.logger.info(f"[{request_id}] Input token IDs: {input_ids[0].tolist()}")
            self.logger.info(f"[{request_id}] Input tokens with special tokens: {repr(self.tokenizer.decode(input_ids[0], skip_special_tokens=False))}")
            
            # Build generation config
            generation_config = self._build_generation_config(request)
            self.logger.info(f"[{request_id}] Generation config: {generation_config}")
            
            # Setup streaming
            streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=not request.echo, skip_special_tokens=True)
            generation_kwargs = {
                "input_ids": input_ids,
                "generation_config": generation_config,
                "streamer": streamer,
                "return_dict_in_generate": True,
                "output_scores": False,
            }
            
            # Start generation in background thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream tokens
            created = int(time.time())
            full_response = ""
            for new_text in streamer:
                if new_text:  # Skip empty strings
                    full_response += new_text
                    
                    # Check for stop sequences
                    should_stop = False
                    for stop_seq in self.stop_sequences:
                        if stop_seq in full_response:
                            # Trim at stop sequence
                            stop_index = full_response.find(stop_seq)
                            trimmed_response = full_response[:stop_index]
                            remaining_text = trimmed_response[len(full_response) - len(new_text):]
                            if remaining_text:
                                chunk = CompletionStreamResponse(
                                    id=request_id,
                                    created=created,
                                    model=request.model,
                                    choices=[
                                        CompletionStreamChoice(
                                            index=0,
                                            text=remaining_text,
                                            finish_reason=None
                                        )
                                    ]
                                )
                                yield f"data: {chunk.model_dump_json()}\n\n"
                            should_stop = True
                            break
                    
                    if should_stop:
                        break
                    
                    # Send token chunk
                    chunk = CompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=request.model,
                        choices=[
                            CompletionStreamChoice(
                                index=0,
                                text=new_text,
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
            
            # Send final chunk
            chunk = CompletionStreamResponse(
                id=request_id,
                created=created,
                model=request.model,
                choices=[
                    CompletionStreamChoice(
                        index=0,
                        text="",
                        finish_reason="stop"
                    )
                ]
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            
            # Send [DONE] marker
            yield "data: [DONE]\n\n"
            
            # Wait for generation to complete
            thread.join()
            
            # Log output details (similar to non-streaming version)
            generated_token_ids = self.tokenizer.encode(full_response, add_special_tokens=False)
            completion_tokens = len(generated_token_ids)
            
            self.logger.info(f"[{request_id}] Generated token IDs: {generated_token_ids}")
            self.logger.info(f"[{request_id}] Generated tokens with special tokens: {repr(full_response)}")
            self.logger.info(f"[{request_id}] Response text (clean): {repr(full_response)}")
            
            # Determine finish reason
            finish_reason = "stop"  # Default for streaming
            if completion_tokens >= request.max_tokens:
                finish_reason = "length"
            elif any(stop_seq in full_response for stop_seq in self.stop_sequences):
                finish_reason = "stop"
                stop_sequence_found = next(stop_seq for stop_seq in self.stop_sequences if stop_seq in full_response)
                self.logger.info(f"[{request_id}] Generation stopped due to stop sequence: {repr(stop_sequence_found)}")
            
            self.logger.info(f"[{request_id}] Finish reason: {finish_reason}")
            self.logger.info(f"[{request_id}] Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={prompt_tokens + completion_tokens}")
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Streaming generation failed: {str(e)}")
            # Send error as final chunk
            chunk = CompletionStreamResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    CompletionStreamChoice(
                        index=0,
                        text="",
                        finish_reason="stop"
                    )
                ]
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"


# Global inference server instance
inference_server: Optional[InferenceServer] = None

app = FastAPI(title="HuggingFace OpenAI API Server", version="1.0.0")


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": inference_server.model_path.split("/")[-1] if inference_server else "unknown",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "huggingface"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion."""
    if inference_server is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        if request.stream:
            return StreamingResponse(
                inference_server.generate_stream_response(request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            return inference_server.generate_response(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create a text completion."""
    if inference_server is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if request.n != 1:
        raise HTTPException(status_code=400, detail="n > 1 not supported yet")
    
    try:
        if request.stream:
            return StreamingResponse(
                inference_server.generate_stream_completion(request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            return inference_server.generate_completion(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": inference_server is not None}


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from: {config_path}")
        return config or {}
    except Exception as e:
        logging.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Merge YAML config with command line arguments, with CLI args taking precedence."""
    # Convert config keys to match argument names (replace - with _)
    normalized_config = {}
    for key, value in config.items():
        normalized_key = key.replace('-', '_')
        normalized_config[normalized_key] = value
    
    # Get default values from parser to detect which args were actually set
    defaults = {}
    for action in parser._actions:
        if action.dest != 'help' and action.dest != 'config':
            defaults[action.dest] = action.default
    
    # For each config value, set it if the argument uses the default value
    for key, value in normalized_config.items():
        if hasattr(args, key):
            current_value = getattr(args, key)
            default_value = defaults.get(key)
            
            # Only override if the current value is the default (wasn't explicitly set)
            if current_value == default_value:
                if key == 'stop_sequences':
                    # Special handling for stop_sequences which uses nargs="*"
                    setattr(args, key, value if isinstance(value, list) else [value] if value else None)
                else:
                    setattr(args, key, value)
    
    return args


def main():
    parser = argparse.ArgumentParser(description="OpenAI API-compatible inference server")
    parser.add_argument("config", nargs="?", help="YAML configuration file (optional)")
    parser.add_argument("-m", "--model", help="HuggingFace model path or name")
    parser.add_argument("-H", "--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("-p", "--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("-d", "--device", default="auto", help="Device to use (cuda, cpu, auto)")
    parser.add_argument("-t", "--chat-template", help="Path to custom Jinja2 chat template file")
    parser.add_argument("-T", "--dtype", help="Model data type (float32/fp32, float16/fp16/half, bfloat16/bf16, float64/fp64/double). Default: bfloat16 if supported, otherwise float16 on GPU, float32 on CPU")
    parser.add_argument("-s", "--stop-sequences", nargs="*", help="Custom stop sequences (e.g., --stop-sequences '<|im_end|>' '</s>'). Default includes EOS token.")
    parser.add_argument("-l", "--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("-c", "--from-checkpoint", nargs="?", const=True, default=False, help="Load model from checkpoint")
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        config = load_config_from_yaml(args.config)
        args = merge_config_with_args(config, args, parser)
    
    # Validate required arguments
    if not args.model:
        parser.error("--model is required (can be specified in config file)")
    
    # Setup logging - configure the root logger and our dedicated application logger
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ],
        force=True  # Force reconfiguration even if logging was already configured
    )
    
    # Configure our dedicated application logger
    app_logger = logging.getLogger('inference_server')
    app_logger.setLevel(log_level)
    
    # Ensure the logger propagates to the root logger
    app_logger.propagate = True
    
    global inference_server
    inference_server = InferenceServer(
        args.model, 
        args.device,
        args.from_checkpoint,
        getattr(args, 'chat_template', None), 
        args.dtype,
        args.stop_sequences
    )
    
    logging.info(f"Starting server on {args.host}:{args.port}")
    logging.info(f"OpenAI API endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    
    # Configure uvicorn to use the same log level but not override our logger
    uvicorn_log_level = args.log_level.lower()
    uvicorn.run(app, host=args.host, port=args.port, log_level=uvicorn_log_level, access_log=True)


if __name__ == "__main__":
    main()