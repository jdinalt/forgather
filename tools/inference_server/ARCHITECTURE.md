# Inference Server Architecture

**Version**: 2026-01-08
**Purpose**: Technical documentation for maintainers of the HuggingFace OpenAI-compatible inference server

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Patterns](#architecture-patterns)
3. [Request Flow](#request-flow)
4. [Core Components](#core-components)
5. [Generation Pipeline](#generation-pipeline)
6. [Configuration System](#configuration-system)
7. [Critical Implementation Details](#critical-implementation-details)
8. [Testing and Debugging](#testing-and-debugging)
9. [Common Tasks](#common-tasks)

---

## Overview

### What This Is

An OpenAI API-compatible inference server for HuggingFace causal language models. Supports both chat completions and text completions with streaming and non-streaming modes.

### Key Features

- **OpenAI compatibility**: Drop-in replacement for OpenAI API endpoints
- **Multiple modes**: Chat completion, text completion, streaming/non-streaming
- **HuggingFace integration**: Full access to HuggingFace generation parameters
- **Flexible stopping**: Custom stop sequences, EOS control, max tokens
- **Performance**: torch.compile support, KV cache options, device placement
- **Debugging**: Comprehensive logging with token-level visibility

### File Structure

```
tools/inference_server/
├── server.py                    # Server entry point & CLI
├── client.py                    # CLI client for testing
├── service.py                   # Core inference service (model/tokenizer management)
├── routes.py                    # FastAPI route handlers
├── config.py                    # Configuration loading utilities
├── core/                        # Core utilities
│   ├── finish_detector.py       # Determines why generation stopped
│   ├── stop_processor.py        # Processes stop sequences
│   ├── tokenizer_wrapper.py     # Tokenization with device placement
│   └── generation_logger.py     # Unified logging
├── models/                      # Pydantic data models
│   ├── chat.py                  # Chat completion request/response models
│   └── completion.py            # Text completion request/response models
├── strategies/                  # Generation strategy pattern
│   ├── base.py                  # Abstract base strategy
│   ├── non_streaming_base.py    # Template for non-streaming generation
│   ├── streaming_base.py        # Template for streaming generation
│   ├── chat.py                  # Non-streaming chat implementation
│   ├── completion.py            # Non-streaming completion implementation
│   ├── streaming_chat.py        # Streaming chat implementation
│   └── streaming_completion.py  # Streaming completion implementation
└── tests/                       # Test suite
```

---

## Architecture Patterns

### 1. Strategy Pattern (Strategies Directory)

**Why**: Separate the concerns of chat vs completion, streaming vs non-streaming

**Structure**:
```
BaseGenerationStrategy (abstract)
├── NonStreamingGenerationBase (template method pattern)
│   ├── ChatGenerationStrategy
│   └── CompletionGenerationStrategy
└── StreamingGenerationBase (template method pattern)
    ├── StreamingChatStrategy
    └── StreamingCompletionStrategy
```

**How It Works**:
- `BaseGenerationStrategy` defines the interface: `generate(request) -> response`
- Template base classes (`NonStreamingGenerationBase`, `StreamingGenerationBase`) implement the generation flow with hook methods
- Concrete strategies override hook methods for chat vs completion specifics

**Key Hook Methods**:
- `_format_prompt()`: Convert request to prompt text
- `_process_response_text()`: Process generated text before returning
- `_build_response()`: Construct the final response object

### 2. Template Method Pattern (Generation Flow)

Both streaming and non-streaming follow a **14-step template**:

**Non-Streaming** (`non_streaming_base.py`):
1. Generate request ID
2. Log request details
3. Prepare prompt (hook method)
4. Tokenize input
5. Log input tokens
6. Build generation config
7. Get stop sequences
8. **Generate tokens** (HuggingFace `model.generate()`)
9. Extract generated tokens
10. Process stop sequences
11. Log output tokens
12. Determine finish reason
13. Log stop sequence if triggered
14. Decode and build response (hook methods)

**Streaming** (`streaming_base.py`):
1. Generate request ID
2. Log request details
3. Prepare prompt (hook method)
4. Tokenize input
5. Log input tokens
6. Build generation config
7. Get stop sequences
8. Create streamer (TextIteratorStreamer)
9. **Start generation thread**
10. **Yield chunks** (incremental stop sequence checking)
11. Wait for completion
12. Log output details
13. Determine finish reason
14. Log response and metrics

### 3. Service Layer Pattern

**InferenceService** (`service.py`) is the **single source of truth** for:
- Model and tokenizer instances
- Server-level configuration (device, dtype, stop sequences, **ignore_eos default**)
- Core utilities (logger, finish_detector, stop_processor, tokenizer_wrapper)

**Important**: Strategies receive a reference to the service and delegate to it for:
- Tokenization
- Generation config building
- Stop sequence processing
- Finish reason detection
- Logging

---

## Request Flow

### HTTP Request → Response (Non-Streaming)

```
1. Client HTTP POST → FastAPI route
   ├── /v1/chat/completions → chat_completions()
   └── /v1/completions → completions()

2. Route handler validates request (Pydantic models)
   ├── ChatCompletionRequest
   └── CompletionRequest

3. Route handler selects strategy
   ├── stream=False → ChatGenerationStrategy / CompletionGenerationStrategy
   └── stream=True → StreamingChatStrategy / StreamingCompletionStrategy

4. Strategy.generate(request) executes 14-step template
   ├── Uses service for tokenization, config building, etc.
   └── Calls model.generate() with GenerationConfig

5. Strategy returns response object
   ├── ChatCompletionResponse
   └── CompletionResponse

6. FastAPI serializes and returns JSON
```

### Streaming Flow Differences

```
4. Strategy.generate(request) executes 14-step template
   ├── Creates TextIteratorStreamer
   ├── Starts generation in background thread
   └── Yields SSE chunks as tokens arrive

5. FastAPI wraps in StreamingResponse
   └── Client receives Server-Sent Events (SSE)
```

---

## Core Components

### InferenceService (service.py)

**Responsibilities**:
- Load model and tokenizer from disk/checkpoint
- Manage server-level defaults (stop_sequences, ignore_eos, etc.)
- Build GenerationConfig from request + defaults
- Provide utilities to strategies

**Key Methods**:

```python
def __init__(
    model_path, device, dtype, stop_sequences,
    ignore_eos,  # Server-level default for EOS control
    compile_args, cache_implementation, use_cache,
    from_checkpoint, chat_template_path, attn_implementation
)
    # Loads model, tokenizer, sets up utilities
    # Stores server-level defaults (e.g., self.ignore_eos)

def _build_generation_config(request) -> GenerationConfig:
    # Merges model defaults + request parameters
    # Handles ignore_eos logic (see Critical Details)
    # Returns HuggingFace GenerationConfig object

def _setup_stop_tokens():
    # Builds self.stop_token_ids set
    # Includes EOS token + custom stop sequences

def apply_chat_template(messages) -> str:
    # Formats chat messages using Jinja2 template
    # Falls back to basic format if no template
```

**Important Attributes**:
- `self.model`: HuggingFace model
- `self.tokenizer`: HuggingFace tokenizer
- `self.ignore_eos`: Server-level default (from CLI `--ignore-eos`)
- `self.stop_sequences`: Server-level stop sequences
- `self.default_generation_config`: Loaded from model's `generation_config.json`

### FinishReasonDetector (core/finish_detector.py)

**Responsibilities**: Determine why generation stopped

**Methods**:

```python
def determine_finish_reason(
    generated_token_ids, max_tokens, stopped_by_sequence, ignore_eos
) -> str:
    # Returns "length" or "stop"
    # Priority: max_tokens > stop_sequence > EOS token
    # If ignore_eos=True, skips EOS token checking

def determine_finish_reason_streaming(
    completion_tokens, max_tokens, stop_sequences, full_response, ignore_eos
) -> str:
    # Streaming version (works with text, not token IDs)
```

**Logic Flow**:
1. If `len(tokens) >= max_tokens` → `"length"`
2. If `stopped_by_sequence` → `"stop"`
3. If `not ignore_eos` and last token is EOS → `"stop"`
4. Else → `"stop"` (fallback)

### StopSequenceProcessor (core/stop_processor.py)

**Responsibilities**: Trim generated text at stop sequences

**Method**:

```python
def process(
    generated_text, generated_token_ids, generated_tokens, stop_sequences
) -> (token_ids, tokens, stopped_by_sequence, stop_sequence_found):
    # Searches for stop sequences in decoded text
    # Trims text and re-encodes to get trimmed tokens
    # Returns updated tokens and whether stopped
```

**Important**: This runs **after** generation, not during. HuggingFace's `stop_strings` parameter handles multi-token sequences during generation.

### GenerationLogger (core/generation_logger.py)

**Responsibilities**: Unified logging with token-level visibility

**Key Methods**:

```python
def log_request(request_id, model, max_tokens, temperature, ...)
def log_input_tokens(request_id, token_ids, decoded_text)
def log_generated_tokens(request_id, token_ids)
def log_response(request_id, response_text, tokens_per_sec, peak_memory_mb)
```

**Usage**: Strategies call logger methods at each step of the template

---

## Generation Pipeline

### Building GenerationConfig

**Location**: `InferenceService._build_generation_config()`

**Steps**:

1. **Start with model defaults**:
   ```python
   if self.default_generation_config is not None:
       generation_config = GenerationConfig(**self.default_generation_config.to_dict())
   else:
       generation_config = GenerationConfig()
   ```

2. **Set core parameters**:
   ```python
   generation_config.max_new_tokens = request.max_tokens or 16
   generation_config.temperature = request.temperature  # if not None
   generation_config.top_p = request.top_p  # if not None
   generation_config.do_sample = (temperature is None or temperature > 0)
   ```

3. **Set token IDs**:
   ```python
   generation_config.pad_token_id = self.tokenizer.pad_token_id
   generation_config.bos_token_id = self.tokenizer.bos_token_id
   # eos_token_id handled specially (see Critical Details)
   ```

4. **Add HuggingFace parameters** (if present in request):
   - `repetition_penalty`, `length_penalty`, `no_repeat_ngram_size`
   - `num_beams`, `top_k`, `typical_p`, `min_length`
   - `seed` (sets `torch.manual_seed()`)

5. **Handle beam search**:
   ```python
   if num_beams > 1:
       generation_config.do_sample = False
       generation_config.early_stopping = True  # unless explicitly set
   ```

6. **Set cache options**:
   ```python
   generation_config.use_cache = self.use_cache  # if set
   generation_config.cache_implementation = self.cache_implementation  # if set
   ```

### Calling model.generate()

**Location**: `non_streaming_base.py` line ~71

```python
generation_kwargs = {
    "input_ids": input_ids,
    "generation_config": generation_config,
    "return_dict_in_generate": True,
    "output_scores": False,
    "tokenizer": self.service.tokenizer,
}
if stop_strings:
    generation_kwargs["stop_strings"] = stop_strings

with torch.inference_mode():
    outputs = self.service.model.generate(**generation_kwargs)
```

**For streaming** (`streaming_base.py` line ~65):

```python
streamer = TextIteratorStreamer(
    self.service.tokenizer,
    skip_special_tokens=False,
    skip_prompt=True,
)

generation_kwargs = {
    "input_ids": input_ids,
    "generation_config": generation_config,
    "streamer": streamer,
    "tokenizer": self.service.tokenizer,
}
if stop_strings:
    generation_kwargs["stop_strings"] = stop_strings

# Start generation in background thread
thread = Thread(
    target=self.service.model.generate,
    kwargs=generation_kwargs,
)
thread.start()

# Yield chunks as they arrive
for text in streamer:
    # ... process and yield chunk
```

---

## Configuration System

### Server Configuration (server.py)

**CLI Arguments**:

```python
--model, -m              # Model path (required)
--device, -d             # Device (cuda:0, cpu, auto)
--dtype, -T              # Data type (float32, float16, bfloat16)
--from-checkpoint, -c    # Load from checkpoint
--chat-template, -t      # Custom Jinja2 chat template
--attn-implementation    # Attention implementation (eager, sdpa, flash_attention_2, flex_attention)
--stop-sequences, -s     # Custom stop sequences (list)
--ignore-eos             # Server default: ignore EOS tokens
--compile                # Use torch.compile
--compile-args           # YAML-encoded torch.compile arguments
--cache-implementation   # KV cache implementation (dynamic, static, etc.)
--disable-kv-cache       # Disable KV cache
```

**YAML Config Support**:

You can provide a YAML config file:
```yaml
model: /path/to/model
device: cuda:0
dtype: bfloat16
stop_sequences:
  - "<|im_end|>"
  - "</s>"
ignore_eos: true
```

### Request Configuration (models/)

**Common Parameters** (chat and completion):

```python
# OpenAI standard
model: str
max_tokens: int = 512
temperature: float = None  # None = greedy (do_sample=False)
top_p: float = None
stream: bool = False

# HuggingFace parameters
repetition_penalty: float = None
length_penalty: float = None
no_repeat_ngram_size: int = None
top_k: int = None
typical_p: float = None
num_beams: int = None
min_length: int = None
seed: int = None
ignore_eos: bool = None  # Request-level override
```

**Chat-specific**:
```python
messages: List[ChatMessage]  # [{"role": "user", "content": "..."}]
```

**Completion-specific**:
```python
prompt: Union[str, List[str]]
stop: Union[str, List[str]] = None  # Stop sequences
echo: bool = None  # Include prompt in response
```

### Configuration Precedence

For all parameters:
1. **Request-level** (highest priority)
2. **Server-level defaults** (CLI/YAML)
3. **Model defaults** (`generation_config.json`)
4. **Framework defaults** (HuggingFace/Transformers)

**Example: ignore_eos**

```python
# In service.py:_build_generation_config()
request_ignore_eos = getattr(request, "ignore_eos", None)
ignore_eos = request_ignore_eos if request_ignore_eos is not None else self.ignore_eos
```

If request has `ignore_eos=True`, use that.
If request has `ignore_eos=None`, use server default (`self.ignore_eos`).
If request has `ignore_eos=False`, use that (explicit override).

---

## Critical Implementation Details

### 1. EOS Token Handling (IMPORTANT!)

**Problem**: Setting `generation_config.eos_token_id = None` **does NOT work**.

**Why**: HuggingFace's `generate()` method automatically fills `None` from model defaults:

```python
# In transformers/generation/utils.py:1794-1795
if generation_config.eos_token_id is None:
    generation_config.eos_token_id = self.generation_config.eos_token_id
```

**Solution**: Set `eos_token_id = -1` (impossible token ID).

**Implementation** (`service.py:379-392`):

```python
# Check request-level ignore_eos, fall back to server-level default
request_ignore_eos = getattr(request, "ignore_eos", None)
ignore_eos = request_ignore_eos if request_ignore_eos is not None else self.ignore_eos

if ignore_eos:
    # Set to -1 (impossible token ID) to prevent HF from stopping on EOS
    # Note: Setting to None doesn't work because HuggingFace fills it from model defaults
    generation_config.eos_token_id = -1
else:
    # Normal behavior: ensure eos_token_id is set
    if not hasattr(generation_config, "eos_token_id") or generation_config.eos_token_id is None:
        generation_config.eos_token_id = self.tokenizer.eos_token_id
```

**Testing EOS Behavior**:

```bash
# Should stop early on EOS
forgather inf client --completion "The end." --max-tokens 512 --seed 42

# Should generate full 512 tokens (or close to it)
forgather inf client --completion "The end." --max-tokens 512 --seed 42 --ignore-eos
```

### 2. Request Parameter Defaults

**CRITICAL**: Request model defaults must be `None`, not `False`, for server defaults to work.

**Why**: If `ChatCompletionRequest` has `ignore_eos: bool = False`, then:
- Request without `ignore_eos` → `request.ignore_eos = False` (Pydantic default)
- Server default is never used because `False` is a valid value

**Correct** (`models/chat.py`, `models/completion.py`):

```python
ignore_eos: Optional[bool] = None  # ✓ Correct
```

**Incorrect**:

```python
ignore_eos: Optional[bool] = False  # ✗ Wrong - breaks server defaults
```

**Handling in service.py**:

```python
request_ignore_eos = getattr(request, "ignore_eos", None)
# If request.ignore_eos is None, fall back to self.ignore_eos (server default)
ignore_eos = request_ignore_eos if request_ignore_eos is not None else self.ignore_eos
```

### 3. Stop Sequences: Two Systems

**System 1: HuggingFace `stop_strings`** (during generation)

```python
# In strategies/non_streaming_base.py
generation_kwargs = {
    "stop_strings": stop_strings,  # Passed to model.generate()
    # ...
}
```

- Handles multi-token stop sequences
- Stops generation when sequence appears
- More efficient (stops during generation, not after)

**System 2: Post-processing** (`StopSequenceProcessor`)

```python
# After generation completes
generated_token_ids, generated_tokens, stopped_by_sequence, stop_sequence_found = (
    self.service.stop_processor.process(
        generated_text,
        generated_token_ids,
        generated_tokens,
        stop_strings,
    )
)
```

- Trims text at stop sequences
- Re-encodes to get correct token count
- **Needed** because `stop_strings` might not trim perfectly

**Why Both?**

- `stop_strings` stops generation (saves compute)
- Post-processing ensures clean output (removes stop sequence from response)

### 4. Streaming Stop Sequence Handling

**Challenge**: With streaming, we yield tokens incrementally but need to check for stop sequences.

**Solution** (`streaming_base.py:80-110`):

```python
full_response = ""
for text in streamer:
    full_response += text

    # Check if any stop sequence appeared
    current_chunk = text
    for stop_seq in stop_sequences:
        if stop_seq in full_response:
            # Stop sequence found - trim and stop streaming
            stop_index = full_response.find(stop_seq)
            trimmed_response = full_response[:stop_index]

            # Calculate how much of current chunk to send
            if len(trimmed_response) > len(full_response) - len(current_chunk):
                final_chunk = trimmed_response[len(full_response) - len(current_chunk):]
                yield build_chunk(final_chunk)

            break  # Stop streaming
    else:
        # No stop sequence, yield chunk normally
        yield build_chunk(current_chunk)
```

### 5. Chat Template Handling

**Default Template** (`service.py:437-464`):

```python
def get_default_chat_template(self) -> str:
    return """
    {%- for message in messages %}
        {%- if message['role'] == 'system' %}
            {{- message['content'] + '\\n\\n' }}
        {%- elif message['role'] == 'user' %}
            {{- message['content'] + '\\n' }}
        {%- elif message['role'] == 'assistant' %}
            {{- message['content'] + '\\n' }}
        {%- endif %}
    {%- endfor %}
    """.strip()
```

**Custom Template**:

```bash
forgather inf server -m /path/to/model --chat-template /path/to/template.jinja
```

**Template Variables Available**:
- `messages`: List of `{"role": str, "content": str}`
- `bos_token`, `eos_token` (if you want to include them)

### 6. Device and Dtype Handling

**Device Resolution** (`service.py:482-498`):

```python
if device == "auto":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
```

**Dtype Resolution** (`service.py:500-537`):

```python
def _resolve_dtype(self, dtype_str: Optional[str]) -> torch.dtype:
    if dtype_str is None:
        # Auto-select based on device
        if self.device.startswith("cuda"):
            # Use bfloat16 if available, else float16
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            return torch.float32
    # ... parse dtype_str
```

**Loading Model**:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=self.dtype,
    device_map=self.device,
    attn_implementation=self.attn_implementation,
    trust_remote_code=True,
)
```

### 7. Checkpoint Loading

**From Latest Checkpoint**:

```bash
forgather inf server -c -m /path/to/model
```

**From Specific Checkpoint**:

```bash
forgather inf server -c /path/to/checkpoint -m /path/to/model
```

**Implementation** (`service.py:149-189`):

```python
if self.from_checkpoint:
    if isinstance(self.from_checkpoint, str):
        checkpoint_path = self.from_checkpoint
    else:
        checkpoint_path = find_latest_checkpoint(self.model_path)

    # Load with no_init_weights (fast loading)
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(config, ...)

    # Load checkpoint weights
    load_checkpoint(model, checkpoint_path, ...)
```

---

## Testing and Debugging

### Running the Server

```bash
# Basic
forgather inf server -m /path/to/model

# With checkpoint
forgather inf server -c -m /path/to/model

# With options
forgather inf server -m /path/to/model \
    --dtype bfloat16 \
    --device cuda:0 \
    --stop-sequences "<|im_end|>" "</s>" \
    --ignore-eos \
    --log-level DEBUG
```

### Using the Client

```bash
# Interactive chat
forgather inf client

# Single message
forgather inf client --message "Tell me a joke"

# Text completion
forgather inf client --completion "Once upon a time" --max-tokens 100

# With HF parameters
forgather inf client --completion "Once upon a time" \
    --max-tokens 100 \
    --temperature 0.7 \
    --top-k 50 \
    --repetition-penalty 1.1 \
    --seed 42 \
    --ignore-eos \
    --show-usage
```

### Debugging with Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# Chat completion
response = client.chat.completions.create(
    model="test",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100,
    extra_body={
        "ignore_eos": True,
        "repetition_penalty": 1.1,
        "seed": 42,
    }
)

print(response.choices[0].message.content)
print(f"Usage: {response.usage}")
```

### Logging

**Enable DEBUG logging**:

```bash
forgather inf server -m /path/to/model --log-level DEBUG
```

**What Gets Logged**:

- Request ID, model, parameters
- Input tokens (IDs + decoded text with special tokens)
- Generation config
- Generated tokens (IDs)
- Response text
- Performance metrics (tokens/sec, peak memory)
- Finish reason
- Stop sequences triggered

**Example Log Output**:

```
INFO - Request cmpl-abc123: model=inference-server, max_tokens=100
DEBUG - Input tokens (5): [1, 2345, 6789, ...]
DEBUG - Input text: <s>Once upon a time
DEBUG - Generated tokens (98): [234, 567, 890, ...]
INFO - Response (98 tokens, 12.5 tokens/sec, 4.2 GB peak): Once upon a time in a land...
INFO - Finish reason: length
```

### Common Issues

**1. Model stops early despite `ignore_eos=True`**

Check:
- Is `ignore_eos` actually reaching the server? (Check logs for generation config)
- Is `eos_token_id = -1` in the generation config? (Should be when ignore_eos=True)

**2. Server-level `--ignore-eos` has no effect**

Check:
- Are request models using `None` as default? (`ignore_eos: Optional[bool] = None`)
- Is service.py properly falling back to server default? (See Critical Details #2)

**3. Stop sequences not working**

Check:
- Are they being passed to `model.generate()` as `stop_strings`?
- Is tokenizer available? (Required for `stop_strings`)
- Are they multi-token sequences? (Might need post-processing)

**4. Streaming cuts off early**

Check:
- Stop sequence detection logic in streaming_base.py
- Is stop sequence appearing in middle of valid text?

**5. Chat template not working**

Check:
- Is template valid Jinja2?
- Does tokenizer have built-in chat template? (Might conflict)
- Try with `--log-level DEBUG` to see formatted prompt

---

## Common Tasks

### Adding a New Generation Parameter

**1. Add to request models**:

```python
# models/chat.py and models/completion.py
class ChatCompletionRequest(BaseModel):
    # ... existing fields ...
    my_new_param: Optional[float] = None
```

**2. Pass to generation config**:

```python
# service.py:_build_generation_config()
hf_params = [
    # ... existing params ...
    "my_new_param",
]
```

**3. Add to client** (optional):

```python
# client.py:completion()
def completion(
    self,
    # ... existing params ...
    my_new_param: Optional[float] = None,
):
    extra_body = {
        # ... existing params ...
        "my_new_param": my_new_param,
    }
```

### Adding a New Stopping Criterion

**1. Update FinishReasonDetector**:

```python
# core/finish_detector.py
def determine_finish_reason(self, generated_token_ids, max_tokens, stopped_by_sequence, ignore_eos, my_new_criterion):
    if len(generated_token_ids) >= max_tokens:
        return "length"
    elif my_new_criterion:  # NEW
        return "my_reason"  # NEW
    elif stopped_by_sequence:
        return "stop"
    # ... rest of logic
```

**2. Update strategy base classes**:

```python
# strategies/non_streaming_base.py
my_criterion_triggered = check_my_criterion(...)  # NEW
finish_reason = self.service.finish_detector.determine_finish_reason(
    generated_token_ids,
    request.max_tokens,
    stopped_by_sequence,
    ignore_eos=ignore_eos,
    my_new_criterion=my_criterion_triggered,  # NEW
)
```

### Adding a New Endpoint

**1. Create route handler**:

```python
# routes.py
@router.post("/v1/my_new_endpoint")
async def my_new_endpoint(request: MyNewRequest):
    strategy = MyNewStrategy(inference_service)
    return strategy.generate(request)
```

**2. Create request/response models**:

```python
# models/my_new.py
class MyNewRequest(BaseModel):
    model: str
    # ... fields ...

class MyNewResponse(BaseModel):
    # ... fields ...
```

**3. Create strategy**:

```python
# strategies/my_new.py
class MyNewStrategy(NonStreamingGenerationBase):
    def _format_prompt(self, request):
        # Convert request to prompt
        return request.my_prompt_field

    def _process_response_text(self, generated_tokens, request, prompt):
        # Process generated text
        return self.service.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _build_response(self, request, request_id, response_text, ...):
        # Build response object
        return MyNewResponse(...)
```

### Modifying Generation Behavior

**Server-level default**: Add CLI argument in `server.py`, pass to `InferenceService.__init__()`, store as `self.my_param`

**Request-level override**: Add to request models, use in `_build_generation_config()`:

```python
# service.py
def _build_generation_config(self, request):
    # ... existing code ...

    # Handle my_param
    request_my_param = getattr(request, "my_param", None)
    my_param = request_my_param if request_my_param is not None else self.my_param

    if my_param:
        # Modify generation_config based on my_param
        generation_config.some_setting = some_value
```

### Testing Changes

**1. Unit tests**:

```python
# tests/test_my_feature.py
def test_my_feature():
    from inference_server.service import InferenceService

    service = InferenceService(
        model_path="gpt2",  # Small model for testing
        device="cpu",
    )

    # Test your feature
    assert service.my_method() == expected_value
```

**2. Integration tests**:

```bash
# Start server
forgather inf server -m gpt2 &
SERVER_PID=$!

# Test with client
forgather inf client --completion "Test" --my-param true

# Cleanup
kill $SERVER_PID
```

**3. Manual testing**:

```bash
# Terminal 1
forgather inf server -m /path/to/model --log-level DEBUG

# Terminal 2
forgather inf client --completion "Test" --show-usage
```

---

## Architecture Decision Records

### Why Strategy Pattern?

**Problem**: 4 combinations of (chat/completion) × (streaming/non-streaming) with shared logic

**Alternatives Considered**:
1. Single monolithic handler with if/else branches
2. Inheritance hierarchy
3. Strategy pattern with template method

**Chosen**: Strategy pattern with template method

**Reasoning**:
- Separates concerns (chat formatting vs completion formatting)
- Template method captures shared generation flow
- Easy to test each strategy independently
- Easy to add new modes (e.g., vision, audio)

### Why Service Layer?

**Problem**: Strategies need access to model, tokenizer, config building, logging, etc.

**Alternatives Considered**:
1. Pass all dependencies to each strategy constructor
2. Global singletons
3. Service layer with single source of truth

**Chosen**: Service layer

**Reasoning**:
- Single initialization point for model/tokenizer
- Centralized config building logic
- Easy to mock for testing
- Clear ownership of resources

### Why Two Stop Sequence Systems?

**Problem**: HuggingFace's `stop_strings` might not trim perfectly

**Alternatives Considered**:
1. Only use `stop_strings` (rely on HF)
2. Only use post-processing (inefficient)
3. Both (belt and suspenders)

**Chosen**: Both

**Reasoning**:
- `stop_strings` saves compute (stops generation early)
- Post-processing ensures clean output
- Redundancy is acceptable for correctness

---

## Future Improvements

### Potential Enhancements

1. **Batching**: Support multiple prompts in single request
2. **Model caching**: Keep multiple models loaded, switch on demand
3. **Quantization**: Support 4-bit, 8-bit quantization
4. **Advanced stopping**: Custom stopping criteria (e.g., confidence threshold)
5. **Metrics**: Prometheus metrics endpoint
6. **Authentication**: API key validation
7. **Rate limiting**: Per-user rate limits
8. **Model management**: Load/unload models via API

### Known Limitations

1. **Single model**: Server can only load one model at a time
2. **No batching**: Each request processed independently
3. **Memory**: No automatic model unloading
4. **Streaming**: Can't cancel streaming requests cleanly
5. **Chat history**: No conversation state management (stateless)

---

## References

### HuggingFace Documentation

- [GenerationConfig](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig)
- [generate() method](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationMixin.generate)
- [Text generation strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [KV cache](https://huggingface.co/docs/transformers/en/kv_cache)

### OpenAI API Compatibility

- [Chat completions API](https://platform.openai.com/docs/api-reference/chat)
- [Completions API](https://platform.openai.com/docs/api-reference/completions)

### Key Source Files (for reference)

- `~/fg/lib/python3.12/site-packages/transformers/generation/utils.py:2234` - `generate()` method
- `~/fg/lib/python3.12/site-packages/transformers/generation/configuration_utils.py:82` - `GenerationConfig`

---

**Last Updated**: 2026-01-08
**Maintainer**: Claude Code (AI assistant)
**Version**: 1.0
