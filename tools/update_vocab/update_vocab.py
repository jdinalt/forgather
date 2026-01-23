#!/usr/bin/env python3
"""
Update vocabulary of an existing HuggingFace or Forgather model.

This tool adds tokens to a model's vocabulary and resizes embeddings accordingly.
Works with both HuggingFace and Forgather models using the same HuggingFace APIs.
"""

import argparse
import logging
import os
import sys
from argparse import RawTextHelpFormatter
from contextlib import ExitStack

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Add forgather to path
from forgather import MetaConfig

forgather_root = MetaConfig.find_workspace_dir(os.path.abspath(__file__))
if forgather_root not in sys.path:
    sys.path.insert(0, forgather_root)

from forgather.ml.model_conversion.resize_embeddings import (
    DEFAULT_TOKEN_CONFIG,
    add_tokens_to_tokenizer,
    resize_word_embeddings,
    update_config_from_tokenizer,
)
from forgather.ml.sharded_checkpoint import save_checkpoint
from forgather.ml.utils import default_dtype

logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Update vocabulary of HuggingFace or Forgather model",
        epilog=(
            "Examples:\n"
            "\n"
            "Add tokens from YAML config:\n"
            "  ./update_vocab.py --add-tokens tokens.yaml ~/models/llama ~/models/llama_updated\n"
            "\n"
            "Use default PAD token handling:\n"
            "  ./update_vocab.py ~/models/llama ~/models/llama_updated\n"
            "\n"
            "Skip default tokens and save as sharded checkpoint:\n"
            "  ./update_vocab.py --skip-default-tokens --save-format sharded ~/models/llama ~/models/llama_updated\n"
            "\n"
            "Save with safetensors format:\n"
            "  ./update_vocab.py --add-tokens tokens.yaml --safetensors ~/models/llama ~/models/llama_updated\n"
        ),
    )
    parser.add_argument(
        "model_path",
        type=os.path.expanduser,
        help="Path to source model (HuggingFace or Forgather)",
    )
    parser.add_argument(
        "output_path",
        type=os.path.expanduser,
        help="Output directory for updated model",
    )
    parser.add_argument(
        "--add-tokens",
        type=os.path.expanduser,
        default=None,
        help="Path to YAML file specifying tokens to add to vocabulary",
    )
    parser.add_argument(
        "--skip-default-tokens",
        action="store_true",
        help="Skip default token handling (e.g., adding missing PAD token)",
    )
    parser.add_argument(
        "--save-format",
        type=str,
        choices=["huggingface", "sharded"],
        default="huggingface",
        help=(
            "Save format:\n"
            "  huggingface: Use model.save_pretrained() (saves config automatically)\n"
            "  sharded: Use Forgather's sharded_checkpoint.save_checkpoint()\n"
            "Default: huggingface"
        ),
    )
    parser.add_argument(
        "--safetensors",
        action="store_true",
        help="Save using safetensors format (applies to both save formats)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Torch dtype for model loading (e.g., bfloat16, float32). If not specified, uses model's existing dtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for model operations (default: cpu)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model and tokenizer, show what would be changed, but don't save",
    )

    args = parser.parse_args(args)
    return args


def validate_paths(model_path, output_path):
    """Validate model and output paths."""
    model_path = os.path.abspath(model_path)
    output_path = os.path.abspath(output_path)

    if not os.path.isdir(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")

    if os.path.exists(output_path):
        raise ValueError(
            f"Output path already exists: {output_path}. Will not overwrite."
        )

    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        raise ValueError(f"Output directory does not exist: {output_dir}")

    return model_path, output_path


def main():
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    logger.info("Forgather Vocabulary Update Tool")
    logger.info("=" * 60)

    # Validate paths
    try:
        model_path, output_path = validate_paths(args.model_path, args.output_path)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Source model: {model_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Save format: {args.save_format}")
    logger.info(f"Safetensors: {args.safetensors}")

    # Determine token configuration
    token_config = args.add_tokens
    if token_config is None and not args.skip_default_tokens:
        logger.info("Using default token configuration (adds missing PAD token)")
        token_config = DEFAULT_TOKEN_CONFIG
    elif args.skip_default_tokens:
        logger.info("Skipping default token handling")
    else:
        logger.info(f"Loading token configuration from: {token_config}")

    # Determine dtype
    torch_dtype = None
    if args.dtype:
        from forgather.ml.construct import torch_dtype as get_torch_dtype

        torch_dtype = get_torch_dtype(args.dtype)
        logger.info(f"Using dtype: {torch_dtype}")
    else:
        logger.info("Using model's existing dtype")

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    try:
        with ExitStack() as stack:
            if torch_dtype:
                stack.enter_context(default_dtype(torch_dtype))

            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=args.trust_remote_code
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=args.trust_remote_code
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=args.trust_remote_code
            )

        logger.info(f"Model loaded: {type(model).__name__}")
        logger.info(f"Current vocab size: {len(tokenizer)}")
        logger.info(f"Model config vocab size: {config.vocab_size}")

        # Show current special tokens
        logger.info("Current special tokens:")
        logger.info(f"  BOS: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
        logger.info(f"  EOS: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        logger.info(f"  PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        logger.info(f"  UNK: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Add tokens if configuration provided
    if token_config is not None:
        logger.info("=" * 60)
        logger.info("Adding tokens to vocabulary...")

        try:
            num_added, token_inits = add_tokens_to_tokenizer(tokenizer, token_config)

            logger.info(f"Added {num_added} token(s) to vocabulary")
            logger.info(f"New vocab size: {len(tokenizer)}")

            if num_added > 0:
                logger.info("Resizing model embeddings...")
                resize_word_embeddings(model, tokenizer, token_inits)

                logger.info("Updating config from tokenizer...")
                update_config_from_tokenizer(config, tokenizer)

                # Show updated special tokens
                logger.info("Updated special tokens:")
                logger.info(
                    f"  BOS: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})"
                )
                logger.info(
                    f"  EOS: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})"
                )
                logger.info(
                    f"  PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})"
                )
                logger.info(
                    f"  UNK: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})"
                )
            else:
                logger.info("No tokens were added (all tokens already exist)")

        except Exception as e:
            logger.error(f"Failed to add tokens: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
    else:
        logger.info("No token configuration specified, vocabulary unchanged")

    # Save model
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN: Skipping save operation")
        logger.info(f"Would have saved to: {output_path}")
        logger.info(f"Save format: {args.save_format}")
        logger.info(f"Safetensors: {args.safetensors}")
        return

    logger.info("=" * 60)
    logger.info(f"Saving updated model to: {output_path}")

    try:
        if args.save_format == "huggingface":
            logger.info("Using HuggingFace save_pretrained() format")
            model.save_pretrained(output_path, safe_serialization=args.safetensors)
            tokenizer.save_pretrained(output_path)
            logger.info("Model and tokenizer saved (config saved automatically)")

        elif args.save_format == "sharded":
            logger.info("Using Forgather sharded checkpoint format")

            # Create output directory
            os.makedirs(output_path, exist_ok=True)

            # Save sharded checkpoint
            save_checkpoint(
                output_dir=output_path,
                module=model,
                safetensors=args.safetensors,
                include_param_sharing=True,
            )

            # Save tokenizer and config separately
            tokenizer.save_pretrained(output_path)
            config.save_pretrained(output_path)

            logger.info("Model saved as sharded checkpoint")
            logger.info("Tokenizer and config saved separately")

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Vocabulary update complete!")
    logger.info(f"Updated model saved to: {output_path}")


if __name__ == "__main__":
    main()
