#!/usr/bin/env python3
"""
CLI client for interacting with the HuggingFace OpenAI API-compatible inference server.
"""

import argparse
from argparse import RawTextHelpFormatter
import sys
from typing import List, Dict, Any, Optional, Union
import os
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI Python client not installed. Run: pip install openai")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


class InferenceClient:
    def __init__(self, base_url: str, api_key: str = "dummy"):
        """Initialize the OpenAI client for our inference server."""
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,  # Not used by our server but required by client
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

    def get_completion(
        self,
        model: str = "inference-server",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
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
                assistant_message = ""
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        assistant_message += content
                self.add_assistant_message(assistant_message)
                return assistant_message
            else:
                assistant_message = response.choices[0].message.content
                self.add_assistant_message(assistant_message)
                return assistant_message

        except Exception as e:
            return f"Error: {str(e)}"

    def single_shot(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        model: str = "inference-server",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
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

                assistant_message = ""
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        assistant_message += content
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

                assistant_message = response.choices[0].message.content

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
        max_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        echo: bool = False,
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

            # Build HuggingFace parameters for extra_body
            extra_body = {}
            hf_params = {
                "repetition_penalty": repetition_penalty,
                "length_penalty": length_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "top_k": top_k,
                "typical_p": typical_p,
                "num_beams": num_beams,
                "min_length": min_length,
                "seed": seed,
            }

            # Only add non-None values to extra_body
            for key, value in hf_params.items():
                if value is not None:
                    extra_body[key] = value

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
    from inference_server.config import load_config_from_yaml, merge_config_with_args
else:
    # Running as module - use relative imports
    from .config import load_config_from_yaml, merge_config_with_args


def main():
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
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
        default="http://localhost:8000/v1",
        help="Base URL of the inference server (default: http://localhost:8000/v1)",
    )

    # Generation options
    parser.add_argument(
        "--model",
        default="inference-server",
        help="Model name to use (default: inference-server)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0, help="Top-p sampling (default: 1.0)"
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

    # Create client
    client = InferenceClient(args.url)

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
        )
        if not args.stream:
            print(response)
    else:
        interactive_mode(client, args)


if __name__ == "__main__":
    main()
