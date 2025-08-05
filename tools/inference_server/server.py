#!/usr/bin/env python3
"""
OpenAI API-compatible inference server for HuggingFace models.
"""

import argparse
import logging
import os
import time
import uuid
from typing import List, Optional, Dict, Any, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
from fastapi import FastAPI, HTTPException
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
    low_cpu_mem_usage: Optional[bool] = None
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
    low_cpu_mem_usage: Optional[bool] = None
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


class InferenceServer:
    def __init__(self, model_path: str, device: str = "auto", chat_template_path: Optional[str] = None, dtype: Optional[str] = None, stop_sequences: Optional[List[str]] = None):
        self.model_path = model_path
        self.device = device
        self.chat_template_path = chat_template_path
        self.dtype = self._resolve_dtype(dtype)
        self.stop_sequences = stop_sequences or []
        self.chat_template = None
        self.tokenizer = None
        self.model = None
        self.jinja_env = Environment(loader=BaseLoader())
        self.load_model()
        self.setup_chat_template()
        self._setup_stop_tokens()

    def load_model(self):
        logging.info(f"Loading model from {self.model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=self.device if self.device != "auto" else "auto",
            trust_remote_code=True
        )
        
        if self.device != "auto" and torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        logging.info(f"Model loaded successfully on device: {self.model.device} with dtype: {self.dtype}")
    
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
            logging.warning(f"bfloat16 not supported on this GPU, falling back to float16")
            return torch.float16
        
        return requested_dtype
    
    def setup_chat_template(self):
        """Setup chat template with priority: custom file > tokenizer > default fallback."""
        if self.chat_template_path and os.path.exists(self.chat_template_path):
            # Use custom template file
            with open(self.chat_template_path, 'r') as f:
                self.chat_template = f.read()
            logging.info(f"Using custom chat template from: {self.chat_template_path}")
        elif hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # Use tokenizer's built-in template
            self.chat_template = self.tokenizer.chat_template
            logging.info("Using tokenizer's built-in chat template")
        else:
            # Use default fallback template
            self.chat_template = self.get_default_chat_template()
            logging.info("Using default fallback chat template")
    
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
                    logging.info(f"Added single-token stop sequence: {repr(sequence)} -> token ID {token_ids[0]}")
                else:
                    # Multi-token sequence - will need post-processing
                    logging.info(f"Added multi-token stop sequence: {repr(sequence)} -> token IDs {token_ids}")
            except Exception as e:
                logging.warning(f"Failed to tokenize stop sequence {repr(sequence)}: {e}")
        
        logging.info(f"Stop token IDs: {sorted(self.stop_token_ids)}")
    
    def _build_generation_kwargs(self, request: Union[ChatCompletionRequest, CompletionRequest]) -> Dict[str, Any]:
        """Build generation kwargs from request parameters, filtering out None values."""
        # Base parameters that are always used
        max_tokens = getattr(request, 'max_new_tokens', None) or getattr(request, 'max_tokens', 16)
        
        kwargs = {
            'max_new_tokens': max_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'do_sample': request.temperature > 0,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'early_stopping': getattr(request, 'early_stopping', None) or True,
            'return_dict_in_generate': True,
            'output_scores': False
        }
        
        # Add HuggingFace specific parameters if they are not None
        hf_params = [
            'repetition_penalty', 'length_penalty', 'no_repeat_ngram_size',
            'encoder_no_repeat_ngram_size', 'bad_words_ids', 'min_length',
            'num_beams', 'num_beam_groups', 'diversity_penalty', 
            'temperature_last_layer', 'top_k', 'typical_p',
            'epsilon_cutoff', 'eta_cutoff', 'guidance_scale', 'low_cpu_mem_usage'
        ]
        
        for param in hf_params:
            value = getattr(request, param, None)
            if value is not None:
                kwargs[param] = value
        
        # Handle special cases
        if hasattr(request, 'seed') and request.seed is not None:
            # Set random seed for reproducibility
            import torch
            torch.manual_seed(request.seed)
        
        # If using beam search, adjust sampling
        if kwargs.get('num_beams', 1) > 1:
            kwargs['do_sample'] = False  # Beam search doesn't use sampling
        
        return kwargs
    
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
            logging.error(f"Template error: {e}")
            # Fallback to simple formatting if template fails
            return self._fallback_format_messages(messages)
        except Exception as e:
            logging.error(f"Unexpected error in template rendering: {e}")
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
        logging.info(f"[{request_id}] New chat completion request: model={request.model}, "
                    f"max_tokens={request.max_tokens}, temperature={request.temperature}, "
                    f"top_p={request.top_p}, messages_count={len(request.messages)}")
        
        # Log individual messages
        for i, msg in enumerate(request.messages):
            logging.info(f"[{request_id}] Message {i}: role={msg.role}, content={repr(msg.content)}")
        
        prompt = self.format_messages(request.messages)
        logging.info(f"[{request_id}] Formatted prompt: {repr(prompt)}")
        
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
        logging.info(f"[{request_id}] Input token IDs: {input_token_ids}")
        logging.info(f"[{request_id}] Input tokens with special tokens: {repr(self.tokenizer.decode(input_token_ids, skip_special_tokens=False))}")
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,  # Stop generation when EOS is encountered
                return_dict_in_generate=True,  # Return additional info for finish reasons
                output_scores=False  # We don't need generation scores
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
            logging.info(f"[{request_id}] Generation stopped due to stop sequence: {repr(stop_sequence_found)}")
        elif (self.tokenizer.eos_token_id is not None and 
              len(generated_token_ids) > 0 and 
              generated_token_ids[-1] == self.tokenizer.eos_token_id):
            finish_reason = "stop"  # EOS token found
            logging.info(f"[{request_id}] Generation stopped due to EOS token")
        elif (len(generated_token_ids) > 0 and 
              generated_token_ids[-1] in self.stop_token_ids):
            finish_reason = "stop"  # Stopped by configured stop token
            logging.info(f"[{request_id}] Generation stopped due to stop token ID: {generated_token_ids[-1]}")
        elif len(generated_token_ids) < request.max_tokens:
            # Stopped early but not due to obvious reasons
            finish_reason = "stop"
        
        # Log generated tokens with special token representations
        logging.info(f"[{request_id}] Generated token IDs: {generated_token_ids}")
        logging.info(f"[{request_id}] Generated tokens with special tokens: {repr(self.tokenizer.decode(generated_token_ids, skip_special_tokens=False))}")
        
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        completion_tokens = len(generated_tokens)
        
        # Log final response details
        logging.info(f"[{request_id}] Response text (clean): {repr(response_text)}")
        logging.info(f"[{request_id}] Finish reason: {finish_reason}")
        logging.info(f"[{request_id}] Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={prompt_tokens + completion_tokens}")
        
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
        logging.info(f"[{request_id}] New completion request: model={request.model}, "
                    f"max_tokens={request.max_tokens}, temperature={request.temperature}, "
                    f"top_p={request.top_p}, prompt_length={len(prompt)}")
        logging.info(f"[{request_id}] Prompt: {repr(prompt)}")

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
        logging.info(f"[{request_id}] Input token IDs: {input_token_ids}")
        logging.info(f"[{request_id}] Input tokens with special tokens: {repr(self.tokenizer.decode(input_token_ids, skip_special_tokens=False))}")

        # Build generation parameters
        generation_kwargs = self._build_generation_kwargs(request)
        logging.info(f"[{request_id}] Generation parameters: {generation_kwargs}")
        
        with torch.no_grad():
            outputs = self.model.generate(input_ids, **generation_kwargs)

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
            logging.info(f"[{request_id}] Generation stopped due to stop sequence: {repr(stop_sequence_found)}")
        elif (self.tokenizer.eos_token_id is not None and 
              len(generated_token_ids) > 0 and 
              generated_token_ids[-1] == self.tokenizer.eos_token_id):
            finish_reason = "stop"
            logging.info(f"[{request_id}] Generation stopped due to EOS token")
        elif (len(generated_token_ids) > 0 and 
              generated_token_ids[-1] in self.stop_token_ids):
            finish_reason = "stop"
            logging.info(f"[{request_id}] Generation stopped due to stop token ID: {generated_token_ids[-1]}")

        # Log generated tokens
        logging.info(f"[{request_id}] Generated token IDs: {generated_token_ids}")
        logging.info(f"[{request_id}] Generated tokens with special tokens: {repr(self.tokenizer.decode(generated_token_ids, skip_special_tokens=False))}")

        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        completion_tokens = len(generated_tokens)

        # Handle echo parameter (include original prompt in response)
        if request.echo:
            response_text = prompt + response_text

        # Log final response details
        logging.info(f"[{request_id}] Response text (clean): {repr(response_text)}")
        logging.info(f"[{request_id}] Finish reason: {finish_reason}")
        logging.info(f"[{request_id}] Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={prompt_tokens + completion_tokens}")

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
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Create a chat completion."""
    if inference_server is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported yet")
    
    try:
        return inference_server.generate_response(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest) -> CompletionResponse:
    """Create a text completion."""
    if inference_server is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported yet")
    
    if request.n != 1:
        raise HTTPException(status_code=400, detail="n > 1 not supported yet")
    
    try:
        return inference_server.generate_completion(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": inference_server is not None}


def main():
    parser = argparse.ArgumentParser(description="OpenAI API-compatible inference server")
    parser.add_argument("--model", required=True, help="HuggingFace model path or name")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--device", default="auto", help="Device to use (cuda, cpu, auto)")
    parser.add_argument("--chat-template", help="Path to custom Jinja2 chat template file")
    parser.add_argument("--dtype", help="Model data type (float32/fp32, float16/fp16/half, bfloat16/bf16, float64/fp64/double). Default: bfloat16 if supported, otherwise float16 on GPU, float32 on CPU")
    parser.add_argument("--stop-sequences", nargs="*", help="Custom stop sequences (e.g., --stop-sequences '<|im_end|>' '</s>'). Default includes EOS token.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    global inference_server
    inference_server = InferenceServer(
        args.model, 
        args.device, 
        getattr(args, 'chat_template', None), 
        args.dtype,
        args.stop_sequences
    )
    
    logging.info(f"Starting server on {args.host}:{args.port}")
    logging.info(f"OpenAI API endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()