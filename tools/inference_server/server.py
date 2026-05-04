#!/usr/bin/env python3
"""
OpenAI API-compatible inference server for HuggingFace models.
"""

import argparse
import logging
import os
import secrets
import sys
from pathlib import Path
from typing import Optional

import yaml

# Support both module and standalone execution
if __name__ == "__main__" and __package__ is None:
    # Running as standalone script - add parent directory to path
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    # Import as if we're a package
    from inference_server.config import load_config_from_yaml, merge_config_with_args
    from inference_server.routes import create_app, set_inference_service
    from inference_server.service import InferenceService
else:
    # Running as module - use relative imports
    from .config import load_config_from_yaml, merge_config_with_args
    from .service import InferenceService
    from .routes import create_app, set_inference_service

import uvicorn


def json_type(data):
    try:
        return yaml.safe_load(data)
    except yaml.YAMLError as e:
        raise argparse.ArgumentTypeError(f"Invalid YAML: {e}")


class _HelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Combine raw-text wrapping (so multiline ``epilog`` survives) with
    auto-appended ``(default: …)`` suffixes. Boolean flags are skipped —
    ``store_true`` / ``store_false`` carry their semantics in the name,
    and "(default: False)" only adds noise."""

    def _get_help_string(self, action):
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            return action.help or ""
        return super()._get_help_string(action)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=_HelpFormatter,
        description="OpenAI API-compatible inference server",
        epilog=(
            "Examples:\n"
            "\n"
            "Perform inference in bfloat16 on cuda:0: ./server.py -m ./path/to/model\n"
            "Load from latest checkpoint; don't use AutoModelForCausalLM.from_pretrained(): ./server.py -c -m ./path/to/model\n"
            "Load a specific checkpoint and run on CPU in float32: ./server.py -T float32 -d 'cpu' -c ./path/to/checkpoint -m ./path/to/model\n"
        ),
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=os.path.expanduser,
        help="YAML configuration file (optional)",
    )
    parser.add_argument(
        "-m", "--model", type=os.path.expanduser, help="HuggingFace model path or name"
    )
    parser.add_argument(
        "-a",
        "--attn-implementation",
        help="HuggingFace model path or name",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2", "flex_attention"],
    )
    parser.add_argument("-H", "--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("-p", "--port", type=int, default=8137, help="Port to bind to")
    parser.add_argument(
        "-d", "--device", default="cuda:0", help="Device to use (cuda, cpu, auto)"
    )
    parser.add_argument(
        "-t", "--chat-template", help="Path to custom Jinja2 chat template file"
    )
    parser.add_argument(
        "-T",
        "--dtype",
        help="Model data type (float32/fp32, float16/fp16/half, bfloat16/bf16, float64/fp64/double). Default: bfloat16 if supported, otherwise float16 on GPU, float32 on CPU",
    )
    parser.add_argument(
        "-s",
        "--stop-sequences",
        nargs="*",
        help="Custom stop sequences (e.g., --stop-sequences '<|im_end|>' '</s>'). Default includes EOS token.",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile on model, else eager",
    )
    parser.add_argument(
        "--compile-args",
        default=None,
        type=json_type,
        help="YAML encoded torch compile-args. See: https://docs.pytorch.org/docs/stable/generated/torch.compile.html",
    )
    parser.add_argument(
        "--cache-implementation",
        default=None,
        help="HF cache implementation e.g. 'dynamic', 'static', etc. See: https://huggingface.co/docs/transformers/en/kv_cache.",
    )
    parser.add_argument(
        "--disable-kv-cache",
        action="store_true",
        help="Set 'use_cache' to False in generation config.",
    )
    parser.add_argument(
        "-c",
        "--from-checkpoint",
        nargs="?",
        const=True,
        default=False,
        help="Load model from specific checkpoint or latest checkpoint",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS tokens during generation (continue past EOS until max_tokens or stop_sequence)",
    )

    # Bearer-token auth (default-on). Either supply a token, point at a file
    # holding one, or generate one at startup. ``--no-auth`` disables auth
    # entirely for users who explicitly opt out (matching forgather_server's
    # flag of the same name).
    auth_group = parser.add_mutually_exclusive_group()
    auth_group.add_argument(
        "--auth-token",
        default=None,
        help="Bearer token clients must present in 'Authorization: Bearer <token>'. Auto-generated if neither this nor --auth-token-file is given.",
    )
    auth_group.add_argument(
        "--auth-token-file",
        default=None,
        type=os.path.expanduser,
        help="Read the bearer token from this file (mode 0600 expected). Avoids exposing the token via argv (visible in 'ps').",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable bearer-token auth. Any local user on the host will be able to use the model — only set this if you understand the threat model.",
    )

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        config = load_config_from_yaml(args.config, use_logging=True)
        args = merge_config_with_args(config, args, parser)

    # Validate required arguments
    if not args.model:
        parser.error("--model is required (can be specified in config file)")

    # Setup logging - configure the root logger and our dedicated application logger
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
        force=True,  # Force reconfiguration even if logging was already configured
    )

    # Configure our dedicated application logger
    app_logger = logging.getLogger("inference_server")
    app_logger.setLevel(log_level)

    # Ensure the logger propagates to the root logger
    app_logger.propagate = True

    if isinstance(args.from_checkpoint, str):
        args.from_checkpoint = os.path.expanduser(args.from_checkpoint)

    compile_args = None
    if args.compile:
        if args.compile_args is not None:
            compile_args = args.compile_args
        else:
            compile_args = {}
        logging.info(f"Compile Args: {compile_args}")

    if args.disable_kv_cache:
        use_cache = False
    else:
        use_cache = None

    # Create inference service
    service = InferenceService(
        model_path=args.model,
        device=args.device,
        attn_implementation=args.attn_implementation,
        from_checkpoint=args.from_checkpoint,
        chat_template_path=getattr(args, "chat_template", None),
        dtype=args.dtype,
        stop_sequences=args.stop_sequences,
        compile_args=compile_args,
        cache_implementation=args.cache_implementation,
        use_cache=use_cache,
        ignore_eos=args.ignore_eos,
    )

    # Resolve auth token. --no-auth wins; otherwise prefer an explicit token,
    # then a file, then an auto-generated one. Auto-generation gives
    # default-secure behaviour without forcing operators to manage secrets.
    auth_token: Optional[str] = None
    if args.no_auth:
        print(
            "!! inference_server is running with --no-auth — any local user on "
            "this host can use the model",
            file=sys.stderr,
            flush=True,
        )
    else:
        if args.auth_token:
            auth_token = args.auth_token.strip()
        elif args.auth_token_file:
            try:
                auth_token = Path(args.auth_token_file).read_text().strip()
            except OSError as e:
                parser.error(f"could not read --auth-token-file: {e}")
            if not auth_token:
                parser.error(f"auth-token-file is empty: {args.auth_token_file}")
        else:
            auth_token = secrets.token_hex(32)

        # Print on stderr so it's visible in TTY logs (the scheduler captures
        # stderr) but not entangled with uvicorn's stdout request log.
        print(f"inference_server auth token: {auth_token}", file=sys.stderr, flush=True)
        print(
            "clients must send 'Authorization: Bearer <token>'",
            file=sys.stderr,
            flush=True,
        )
        print(
            f'curl -H "Authorization: Bearer {auth_token}" '
            f"http://{args.host}:{args.port}/v1/models",
            file=sys.stderr,
            flush=True,
        )

    # Create FastAPI app and set service
    app = create_app(auth_token=auth_token)
    set_inference_service(service)

    logging.info(f"Starting server on {args.host}:{args.port}")
    logging.info(
        f"OpenAI API endpoint: http://{args.host}:{args.port}/v1/chat/completions"
    )

    # Configure uvicorn to use the same log level but not override our logger
    uvicorn_log_level = args.log_level.lower()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=uvicorn_log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
