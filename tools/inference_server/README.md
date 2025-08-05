# HuggingFace OpenAI API Server

A simple OpenAI API-compatible inference server for testing HuggingFace models.

## Installation

```bash
cd tools/inference_server
pip install -r requirements.txt
```

## Usage

### Start the server

```bash
python server.py --model microsoft/DialoGPT-medium
```

Options:
- `--model`: HuggingFace model path or name (required)
- `--host`: Host to bind to (default: 127.0.0.1)
- `--port`: Port to bind to (default: 8000)
- `--device`: Device to use - cuda, cpu, or auto (default: auto)
- `--chat-template`: Path to custom Jinja2 chat template file (optional)
- `--dtype`: Model data type (optional, see Data Types section)
- `--stop-sequences`: Custom stop sequences to halt generation (optional)
- `--log-level`: Logging level - DEBUG, INFO, WARNING, ERROR (default: INFO)

### Chat Template Support

The server supports three chat template sources, in order of priority:

1. **Custom template file** (via `--chat-template` argument)
2. **Tokenizer's built-in template** (if available)
3. **Default fallback template** (simple format)

#### Using ChatML template
```bash
python server.py --model /path/to/model --chat-template /path/to/chat_templates/chatml.jinja
```

#### Example custom template
Create a Jinja2 template file:
```jinja2
{%- for message in messages %}
    {%- if message['role'] == 'system' -%}
        System: {{ message['content'] }}\n\n
    {%- elif message['role'] == 'user' -%}
        User: {{ message['content'] }}\n\n
    {%- elif message['role'] == 'assistant' -%}
        Assistant: {{ message['content'] }}\n\n
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    Assistant: 
{%- endif -%}
```

### Data Types

The server supports flexible data type configuration with intelligent defaults:

**Default Behavior:**
- GPU with bfloat16 support: `bfloat16` (recommended for modern GPUs)
- GPU without bfloat16 support: `float16` 
- CPU: `float32`

**Supported Data Types:**
- `float32`, `fp32` - 32-bit floating point
- `float16`, `fp16`, `half` - 16-bit floating point
- `bfloat16`, `bf16` - 16-bit brain floating point (recommended)
- `float64`, `fp64`, `double` - 64-bit floating point

**Examples:**
```bash
# Use default (bfloat16 if supported)
python server.py --model /path/to/model

# Explicit bfloat16
python server.py --model /path/to/model --dtype bfloat16

# Use float16 for older GPUs
python server.py --model /path/to/model --dtype float16

# High precision for research
python server.py --model /path/to/model --dtype float32

# With custom stop sequences
python server.py --model /path/to/model --stop-sequences "<|im_end|>" "</s>"
```

### Stop Sequences

The server supports flexible stop sequence configuration to control when generation should halt:

**Default Behavior:**
- EOS token is always included as a stop criterion
- Models typically learn when to stop during training

**Custom Stop Sequences:**
- Single tokens: `"</s>"`, `"<|endoftext|>"`
- Multi-token sequences: `"<|im_end|>"`, `"Human:"`
- Multiple sequences: `--stop-sequences "<|im_end|>" "</s>" "Human:"`

**Examples:**
```bash
# ChatML format stopping
python server.py --model /path/to/model --stop-sequences "<|im_end|>"

# Multiple stop sequences
python server.py --model /path/to/model --stop-sequences "</s>" "<|endoftext|>" "Human:"

# Instruct format stopping
python server.py --model /path/to/model --stop-sequences "### Human:" "### Assistant:"
```

## Logging

The server uses structured logging with detailed request/response information:

### Log Levels
- `DEBUG`: Detailed debugging information
- `INFO`: Request details, token IDs, and decoded text with special tokens (default)
- `WARNING`: Important warnings (e.g., dtype fallbacks)
- `ERROR`: Error conditions

### What's Logged at INFO Level
For each request, the server logs:
- Request ID and parameters (model, max_tokens, temperature, etc.)
- Individual chat messages with content
- Formatted prompt text
- Input token IDs and decoded text with special tokens (BOS, EOS, etc.)
- Generated token IDs and decoded text with special tokens
- Final response text (cleaned)
- Token usage statistics

### Example Log Output
```
2025-08-05 18:15:03,770 - root - INFO - [chatcmpl-c814be94] New chat completion request: model=test-model, max_tokens=30, temperature=0.7, top_p=1.0, messages_count=2
2025-08-05 18:15:03,771 - root - INFO - [chatcmpl-c814be94] Message 0: role=system, content='You are a helpful assistant.'
2025-08-05 18:15:03,771 - root - INFO - [chatcmpl-c814be94] Message 1: role=user, content='What is 2+2?'
2025-08-05 18:15:03,771 - root - INFO - [chatcmpl-c814be94] Input token IDs: [1, 4444, 29901, 887, 526, 263, 8444, 20255, 29889, 13, 13, 2659, 29901, 1724, 338, 29871, 29906, 29974, 29906, 29973, 13, 13, 7900, 22137, 29901, 29871]
2025-08-05 18:15:03,772 - root - INFO - [chatcmpl-c814be94] Generated tokens with special tokens: 'The answer is 4! I'm here to provide accurate information and support in a range of topics, from math to emotional well-be</s>'
2025-08-05 18:15:03,772 - root - INFO - [chatcmpl-c814be94] Response text (clean): 'The answer is 4! I'm here to provide accurate information and support in a range of topics, from math to emotional well-be'
```

### Usage
```bash
# Default INFO logging
python server.py --model /path/to/model

# Debug logging (very verbose)
python server.py --model /path/to/model --log-level DEBUG

# Minimal logging
python server.py --model /path/to/model --log-level WARNING
```

## CLI Client Tool

A dedicated CLI client (`client.py`) is provided for easy interaction with the server using the official OpenAI Python client, ensuring full compatibility.

### Installation
```bash
pip install openai  # Or use requirements.txt
```

### Usage

#### Single Message
```bash
# Basic usage
python client.py --message "Hello, how are you?"

# With custom server URL
python client.py --message "What's 2+2?" --url http://localhost:8001/v1

# With system prompt
python client.py --message "Tell me a joke" --system "You are a funny comedian"

# Show token usage
python client.py --message "Explain AI" --show-usage --max-tokens 200
```

#### Text Completion (Completions API)
The server also supports the older `/v1/completions` endpoint for raw text completion without chat formatting:

```bash
# Basic text completion
python client.py --completion "Once upon a time"

# With custom parameters
python client.py --completion "The weather today is" --max-tokens 50 --temperature 0.8

# With stop sequences
python client.py --completion "Q: What is AI? A:" --stop "Q:" --max-tokens 100

# Echo the prompt in response
python client.py --completion "Python is" --echo --max-tokens 30

# Show detailed usage information
python client.py --completion "Hello world" --show-usage

# Advanced generation parameters for better quality
python client.py --completion "Once upon a time" --repetition-penalty 1.2 --top-k 50 --max-tokens 100

# Multiple parameters for fine control
python client.py --completion "The story begins" --repetition-penalty 1.1 --no-repeat-ngram-size 3 --top-k 40 --num-beams 2
```

#### Interactive Chat Mode
```bash
# Start interactive session
python client.py --interactive

# With system prompt
python client.py --interactive --system "You are a helpful coding assistant"

# Custom generation parameters
python client.py --interactive --temperature 0.9 --max-tokens 1000
```

#### Utility Commands
```bash
# Check server health
python client.py --health

# List available models
python client.py --list-models

# Connect to different server
python client.py --health --url http://localhost:8001/v1
```

#### Interactive Commands
When in interactive mode, use these commands:
- `/clear` - Clear conversation history
- `/system <message>` - Set or change system prompt
- `/help` - Show available commands
- `quit`, `exit`, or `q` - Exit interactive mode

#### Client Options

**Basic Options:**
- `--url` - Server base URL (default: http://localhost:8000/v1)
- `--model` - Model name (default: inference-server)
- `--max-tokens` - Maximum tokens to generate (default: 512)
- `--temperature` - Sampling temperature (default: 0.7)
- `--top-p` - Top-p sampling (default: 1.0)
- `--system` - System prompt (chat mode only)
- `--show-usage` - Display token usage information

**Completion-Specific Options:**
- `--completion` - Generate text completion for the given prompt
- `--stop` - Stop sequences for completion mode (can specify multiple)
- `--echo` - Echo the prompt in the completion response

**Advanced Generation Parameters:**
- `--repetition-penalty` - Repetition penalty (e.g., 1.2 to reduce repetition)
- `--no-repeat-ngram-size` - Size of n-grams to avoid repeating
- `--top-k` - Top-k sampling parameter
- `--num-beams` - Number of beams for beam search
- `--min-length` - Minimum length of generated sequence
- `--seed` - Random seed for reproducible generation

### Test with curl

#### Chat Completions
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/DialoGPT-medium",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'
```

#### Text Completions
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-model",
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.8
  }'
```

#### With HuggingFace Parameters
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-model",
    "prompt": "The quick brown fox",
    "max_tokens": 50,
    "temperature": 0.7,
    "repetition_penalty": 1.2,
    "top_k": 40,
    "no_repeat_ngram_size": 3
  }'
```

### Test with OpenAI Python client

#### Chat Completions
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not used but required
)

response = client.chat.completions.create(
    model="test-model",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```

#### Text Completions
```python
response = client.completions.create(
    model="test-model",
    prompt="Once upon a time",
    max_tokens=50,
    temperature=0.8
)

print(response.choices[0].text)
```

#### Using HuggingFace Parameters with extra_body
For advanced generation parameters, use the `extra_body` parameter to pass HuggingFace-specific options:

```python
response = client.completions.create(
    model="test-model",
    prompt="The story begins with",
    max_tokens=100,
    temperature=0.7,
    extra_body={
        "repetition_penalty": 1.2,
        "top_k": 40,
        "no_repeat_ngram_size": 3,
        "num_beams": 2
    }
)

print(response.choices[0].text)
```

This mechanism allows you to use any HuggingFace generation parameter while maintaining compatibility with the OpenAI client library.

## API Endpoints

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Create chat completion
- `POST /v1/completions` - Create text completion
- `GET /health` - Health check

## Features

- **OpenAI API Compatibility**: Full support for both chat completions and text completions endpoints
- **HuggingFace Generation Parameters**: Comprehensive support for all HuggingFace generation options
- **Flexible Chat Templates**: Support for custom Jinja2 templates, tokenizer templates, or fallback formatting
- **Stop Sequence Control**: Configurable stop sequences for precise generation control
- **Data Type Support**: Intelligent dtype selection with bfloat16, float16, and float32 support
- **Automatic Device Selection**: Smart GPU/CPU device placement
- **Detailed Logging**: Structured request/response logging with token-level information
- **Token Usage Tracking**: Accurate prompt, completion, and total token counts
- **EOS Token Handling**: Proper early stopping on end-of-sequence tokens

## HuggingFace Generation Parameters

The server supports all major HuggingFace generation parameters for fine-tuning model behavior:

**Repetition Control:**
- `repetition_penalty` - Penalty for repeating tokens (default: None)
- `no_repeat_ngram_size` - Prevent repeating n-grams (default: None)
- `encoder_no_repeat_ngram_size` - For encoder-decoder models (default: None)

**Sampling Parameters:**
- `top_k` - Top-k sampling (default: None, uses model default)
- `typical_p` - Typical sampling parameter (default: None)
- `temperature` - Sampling temperature (default: 0.7 for chat, 1.0 for completions)
- `top_p` - Nucleus sampling (default: 1.0)

**Beam Search:**
- `num_beams` - Number of beams for beam search (default: 1)
- `num_beam_groups` - Diverse beam search groups (default: None)
- `diversity_penalty` - Penalty for diverse beam search (default: None)
- `length_penalty` - Length penalty for beam search (default: None)

**Length Control:**
- `min_length` - Minimum generation length (default: None)
- `max_new_tokens` - Maximum new tokens to generate (default: varies)
- `early_stopping` - Stop at EOS token (default: True)

**Other Options:**
- `seed` - Random seed for reproducible generation (default: None)
- `guidance_scale` - Classifier-free guidance scale (default: None)
- `bad_words_ids` - Token IDs to avoid (default: None)

## Limitations

- No streaming support yet
- Single model per server instance
- Multi-token stop sequences require post-processing (slight performance impact)