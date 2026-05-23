#!/usr/bin/env python3
"""
CLI client for interacting with the HuggingFace OpenAI API-compatible inference server.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI Python client not installed. Run: pip install openai")
    sys.exit(1)


# ANSI escape sequences for displaying a model's reasoning trace dimmer
# than its actual content. Resolved per-call (not at import time) so
# test harnesses that capture stdout after import still get plain text,
# and so callers running the same module under a TTY and a pipe in
# different invocations see the right behavior each time.
def _dim_codes() -> tuple[str, str]:
    if sys.stdout.isatty():
        return "\033[2m", "\033[0m"
    return "", ""


def _delta_field(delta: Any, name: str) -> Optional[str]:
    """Read an optional field from a streaming-chunk delta.

    vLLM's reasoning-parser surface (``delta.reasoning`` /
    ``message.reasoning``) is not in the OpenAI SDK's typed schema,
    so we go through ``getattr`` to avoid an AttributeError on
    SDK versions that haven't caught up. Returns None when the
    field is missing or null.
    """
    val = getattr(delta, name, None)
    if val is None:
        # Some SDK versions surface unknown fields via ``model_extra``
        # rather than as real attributes.
        extra = getattr(delta, "model_extra", None) or {}
        val = extra.get(name)
    return val if isinstance(val, str) and val else None

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


class InferenceClient:
    def __init__(self, base_url: str, api_key: str = "dummy"):
        """Initialize the OpenAI client for our inference server."""
        # Build an httpx client carrying the shared CA bundle so the
        # OpenAI SDK validates our self-signed certs. Passing
        # ``http_client`` to ``OpenAI`` is the supported escape hatch
        # (``verify=`` is not exposed on the SDK constructor directly).
        #
        # Narrow the catch to ImportError: we'd rather surface a real
        # configuration failure (bad cert path, broken bundle) than
        # silently degrade to the SDK's default httpx client (system
        # trust only — would reject our self-signed certs and produce
        # a confusing TLS verification error several stack frames
        # away).
        import httpx as _httpx

        try:
            from forgather.tls import httpx_verify_for_url

            verify = httpx_verify_for_url(base_url)
        except ImportError:
            verify = True  # forgather.tls missing entirely — system trust.

        http_client = _httpx.Client(verify=verify)
        self.client = OpenAI(
            base_url=base_url, api_key=api_key, http_client=http_client
        )
        self.conversation_history: List[Dict[str, str]] = []

    def add_system_message(self, content: str):
        """Add a system message to the conversation."""
        if (
            not self.conversation_history
            or self.conversation_history[0]["role"] != "system"
        ):
            self.conversation_history.insert(0, {"role": "system", "content": content})
        else:
            self.conversation_history[0]["content"] = content

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.conversation_history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        """Add an assistant message to the conversation."""
        self.conversation_history.append({"role": "assistant", "content": content})

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()

    def _consume_chat_stream(self, response, add_to_history: bool) -> str:
        """Print a streaming chat response, separating reasoning from content.

        Returns the accumulated content string (never the reasoning).
        Conversation history is updated only when ``add_to_history`` is
        true and at least one content token arrived — a thinking model
        that exhausts its budget mid-``<think>`` produces no content,
        and pushing an empty assistant turn into history would corrupt
        the next request.
        """
        dim, reset = _dim_codes()
        assistant_message = ""
        in_reasoning = False
        for chunk in response:
            delta = chunk.choices[0].delta
            r = _delta_field(delta, "reasoning")
            if r is not None:
                if not in_reasoning:
                    print(dim, end="", flush=True)
                    in_reasoning = True
                print(r, end="", flush=True)
            # Don't `continue` after reasoning: a single chunk can carry
            # both fields, and the content branch needs to run too.
            if delta.content is not None:
                if in_reasoning:
                    print(reset + "\n", end="", flush=True)
                    in_reasoning = False
                print(delta.content, end="", flush=True)
                assistant_message += delta.content
        if in_reasoning:
            print(reset, end="", flush=True)
        if add_to_history and assistant_message:
            self.add_assistant_message(assistant_message)
        return assistant_message

    def _consume_chat_response(self, msg, add_to_history: bool) -> str:
        """Print a non-streaming chat response, separating reasoning from content.

        Returns the content string (never the reasoning). Same
        empty-content history policy as ``_consume_chat_stream``.
        """
        dim, reset = _dim_codes()
        reasoning = _delta_field(msg, "reasoning")
        if reasoning:
            # ``rstrip("\\n")`` so the dim block doesn't add a blank line
            # before the content when the parser already terminates the
            # reasoning text with a newline (common with qwen3 parser).
            print(f"{dim}{reasoning.rstrip(chr(10))}{reset}\n", end="", flush=True)
        assistant_message = msg.content or ""
        if add_to_history and assistant_message:
            self.add_assistant_message(assistant_message)
        return assistant_message

    def get_completion(
        self,
        model: str = "inference-server",
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
    ) -> str:
        """Get a completion from the server."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=self.conversation_history,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
            )

            if stream:
                return self._consume_chat_stream(response, add_to_history=True)
            else:
                return self._consume_chat_response(
                    response.choices[0].message, add_to_history=True
                )

        except Exception as e:
            return f"Error: {str(e)}"

    def single_shot(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        model: str = "inference-server",
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        show_usage: bool = False,
        stream: bool = False,
    ) -> str:
        """Send a single message and get a response without conversation history."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": message})

        try:
            if stream:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                )

                assistant_message = self._consume_chat_stream(
                    response, add_to_history=False
                )
                print()  # Add newline at end

                # Note: Usage is not available with streaming
                if show_usage:
                    print("\\nUsage: Not available with streaming")

                return assistant_message
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

                assistant_message = self._consume_chat_response(
                    response.choices[0].message, add_to_history=False
                )

                if show_usage:
                    usage = response.usage
                    print(
                        f"\\nUsage: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} total tokens"
                    )

                return assistant_message

        except Exception as e:
            return f"Error: {str(e)}"

    def completion(
        self,
        prompt: str,
        model: str = "inference-server",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        echo: Optional[bool] = None,
        show_usage: bool = False,
        stream: bool = False,
        repetition_penalty: Optional[float] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        top_k: Optional[int] = None,
        typical_p: Optional[float] = None,
        num_beams: Optional[int] = None,
        min_length: Optional[int] = None,
        seed: Optional[int] = None,
        ignore_eos: Optional[bool] = None,
    ) -> str:
        """Generate a text completion for the given prompt."""
        try:
            # Build standard OpenAI parameters
            params = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop,
                "echo": echo,
                "stream": stream,
            }

            # Filter out None values
            params = {k: v for k, v in params.items() if v is not None}

            # Build HuggingFace parameters for extra_body
            extra_body = {
                "repetition_penalty": repetition_penalty,
                "length_penalty": length_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "top_k": top_k,
                "typical_p": typical_p,
                "num_beams": num_beams,
                "min_length": min_length,
                "seed": seed,
                "ignore_eos": ignore_eos,
            }

            # Filter out None values
            extra_body = {k: v for k, v in extra_body.items() if v is not None}

            # Add extra_body if we have any HF parameters
            if extra_body:
                params["extra_body"] = extra_body

            response = self.client.completions.create(**params)

            if stream:
                completion_text = ""
                for chunk in response:
                    if chunk.choices[0].text is not None:
                        content = chunk.choices[0].text
                        print(content, end="", flush=True)
                        completion_text += content
                print()  # Add newline at end

                # Note: Usage is not available with streaming
                if show_usage:
                    print("\\nUsage: Not available with streaming")

                return completion_text
            else:
                completion_text = response.choices[0].text

                if show_usage:
                    usage = response.usage
                    print(
                        f"\\nUsage: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} total tokens"
                    )
                    print(f"Finish reason: {response.choices[0].finish_reason}")

                return completion_text

        except Exception as e:
            return f"Error: {str(e)}"

    def check_server_health(self) -> bool:
        """Check if the server is healthy."""
        try:
            # Try to list models as a health check
            models = self.client.models.list()
            return True
        except Exception as e:
            print(f"Server health check failed: {e}")
            return False

    def list_models(self) -> List[str]:
        """List available models."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []


def interactive_mode(client: InferenceClient, args: argparse.Namespace):
    """Run interactive chat mode."""
    print("Interactive Chat Mode (type 'quit', 'exit', or 'q' to quit)")
    print("Commands:")
    print("  /clear    - Clear conversation history")
    print("  /system <message> - Set system prompt")
    print("  /help     - Show this help")
    print()

    # Set system prompt if provided
    if args.system:
        client.add_system_message(args.system)
        print(f"System prompt set: {args.system}\\n")

    while True:
        try:
            user_input = input("> ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if user_input.startswith("/"):
                command = user_input[1:].split(" ", 1)
                cmd = command[0].lower()

                if cmd == "clear":
                    client.clear_history()
                    if args.system:
                        client.add_system_message(args.system)
                    print("Conversation history cleared.\\n")
                    continue

                elif cmd == "system":
                    if len(command) > 1:
                        client.add_system_message(command[1])
                        print(f"System prompt set: {command[1]}\\n")
                    else:
                        print("Usage: /system <message>\\n")
                    continue

                elif cmd == "help":
                    print("Commands:")
                    print("  /clear    - Clear conversation history")
                    print("  /system <message> - Set system prompt")
                    print("  /help     - Show this help\\n")
                    continue

                else:
                    print(f"Unknown command: {cmd}\\n")
                    continue

            if not user_input:
                continue

            client.add_user_message(user_input)

            # Get response and print without "Assistant:" prefix
            response = client.get_completion(
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=args.stream,
            )
            if not args.stream:
                print(response)
                print()  # Extra blank line for non-streaming mode
            else:
                # Streaming mode needs two newlines (no trailing newline from streaming)
                print()  # First newline after streamed response
                print()  # Second newline for proper spacing

        except KeyboardInterrupt:
            print("\\n\\nGoodbye!")
            break
        except EOFError:
            print("\\nGoodbye!")
            break


# Support both module and standalone execution
if __name__ == "__main__" and __package__ is None:
    # Running as standalone script - add parent directory to path
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    # Import as if we're a package
    from inference_server.auth_paths import read_standalone_token
    from inference_server.config import load_config_from_yaml, merge_config_with_args
else:
    # Running as module - use relative imports
    from .auth_paths import read_standalone_token
    from .config import load_config_from_yaml, merge_config_with_args


class _HelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Combine raw-text wrapping (for the multiline ``epilog``) with
    auto-appended ``(default: …)`` suffixes. Boolean flags are skipped —
    ``store_true`` carries its semantics in the name and "(default: False)"
    only adds noise."""

    def _get_help_string(self, action):
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            return action.help or ""
        return super()._get_help_string(action)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=_HelpFormatter,
        description="CLI client for HuggingFace OpenAI API-compatible inference server",
        epilog=(
            "Examples:\n"
            "\n"
            "Chat with model: ./client.py\n"
            "Respond to single message: ./client.py --message 'Hello, what is your name?'\n"
            "Text completion: './client.py --completion 'Once upon a time' --max-tokens 500\n"
        ),
    )

    # Configuration file option
    parser.add_argument(
        "config",
        nargs="?",
        type=os.path.expanduser,
        help="YAML configuration file (optional)",
    )

    # Connection options
    parser.add_argument(
        "--url",
        default="http://localhost:8137/v1",
        help="Base URL of the inference server",
    )

    # Generation options
    parser.add_argument(
        "--model",
        default="inference-server",
        help="Model name to use",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (None: use server-side default)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling (None: use server-side default)",
    )

    # Mode options
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive chat mode"
    )
    parser.add_argument("--message", help="Single message to send (chat mode)")
    parser.add_argument(
        "--completion", help="Generate text completion for the given prompt"
    )
    parser.add_argument("--system", help="System prompt to use (chat mode only)")
    parser.add_argument("--stop", nargs="*", help="Stop sequences for completion mode")
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Echo the prompt in the completion response (default for completion mode)",
    )
    parser.add_argument(
        "--no-echo",
        action="store_true",
        help="Don't echo the prompt in the completion response",
    )
    parser.add_argument(
        "--show-usage", action="store_true", help="Show token usage information"
    )

    # HuggingFace generation parameters
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        help="Repetition penalty (e.g., 1.1 to reduce repetition)",
    )
    parser.add_argument(
        "--no-repeat-ngram-size", type=int, help="Size of n-grams to avoid repeating"
    )
    parser.add_argument("--top-k", type=int, help="Top-k sampling parameter")
    parser.add_argument("--num-beams", type=int, help="Number of beams for beam search")
    parser.add_argument(
        "--min-length", type=int, help="Minimum length of generated sequence"
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducible generation"
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS tokens during generation (continue past EOS until max_tokens or stop_sequence)",
    )

    # Streaming option
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Deprecated: the default is now 'stream' See --no-stream",
    )

    # Streaming option
    parser.add_argument(
        "--no-stream", action="store_true", help="Disable streaming response"
    )

    # Auth options. Mirrors server.py — either flag overrides the default
    # 'dummy' api_key. The OpenAI SDK forwards api_key as the Bearer token
    # in the Authorization header, which is what the server validates.
    auth_group = parser.add_mutually_exclusive_group()
    auth_group.add_argument(
        "--auth-token",
        default=None,
        help="Bearer token to send to the inference server.",
    )
    auth_group.add_argument(
        "--auth-token-file",
        default=None,
        type=os.path.expanduser,
        help="Read the bearer token from this file (avoids putting it on the command line).",
    )

    # Utility options
    parser.add_argument(
        "--health", action="store_true", help="Check server health and exit"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )

    args = parser.parse_args()

    # Deprecating "stream" argument. The default is now "stream"
    if args.no_stream:
        args.stream = False
    else:
        args.stream = True

    # Load config file if provided
    if args.config:
        config = load_config_from_yaml(args.config, use_logging=False)
        args = merge_config_with_args(config, args, parser)

    # Handle stdin input for completion mode
    stdin_prompt = None
    if not sys.stdin.isatty():
        # Data is being piped in
        stdin_prompt = sys.stdin.read().strip()
        if stdin_prompt and not args.completion:
            # If we have stdin input and no explicit mode, use completion mode
            args.completion = stdin_prompt
        elif stdin_prompt and args.completion:
            # If both stdin and --completion, prefer stdin
            args.completion = stdin_prompt

    # Resolve auth token: explicit > file > standalone-server cache > "dummy"
    # placeholder. The cache lookup makes ``forgather inf server`` paired
    # with ``forgather inf client`` work without the user having to copy the
    # auto-generated token by hand; the placeholder is preserved so legacy
    # ``--no-auth`` servers keep working.
    api_key = "dummy"
    if args.auth_token:
        api_key = args.auth_token.strip()
    elif args.auth_token_file:
        try:
            api_key = Path(args.auth_token_file).read_text().strip()
        except OSError as e:
            parser.error(f"could not read --auth-token-file: {e}")
        if not api_key:
            parser.error(f"auth-token-file is empty: {args.auth_token_file}")
    else:
        cached = read_standalone_token(args.url)
        if cached:
            api_key = cached

    # Create client
    client = InferenceClient(args.url, api_key=api_key)

    # Handle utility commands
    if args.health:
        if client.check_server_health():
            print("Server is healthy!")
            sys.exit(0)
        else:
            print("Server is not responding!")
            sys.exit(1)

    if args.list_models:
        models = client.list_models()
        if models:
            print("Available models:")
            for model in models:
                print(f"  - {model}")
        else:
            print("No models available or server error")
        sys.exit(0)

    # Check server health first
    if not client.check_server_health():
        print(f"Error: Cannot connect to server at {args.url}")
        print("Make sure the inference server is running.")
        print("e.g. forgather inf server -m /path/to/model")
        sys.exit(1)

    # Interactive mode
    if args.interactive:
        interactive_mode(client, args)

    # Single message mode (chat)
    if args.message:
        response = client.single_shot(
            args.message,
            system_prompt=args.system,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            show_usage=args.show_usage,
            stream=args.stream,
        )
        if not args.stream:
            print(response)

    # Completion mode
    elif args.completion:
        # Default echo behavior for completion mode (echo unless --no-echo specified)
        echo_enabled = not args.no_echo

        response = client.completion(
            args.completion,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop=args.stop,
            echo=echo_enabled,
            show_usage=args.show_usage,
            stream=args.stream,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            top_k=args.top_k,
            num_beams=args.num_beams,
            min_length=args.min_length,
            seed=args.seed,
            ignore_eos=args.ignore_eos if args.ignore_eos else None,
        )
        if not args.stream:
            print(response)
    else:
        interactive_mode(client, args)


if __name__ == "__main__":
    main()
