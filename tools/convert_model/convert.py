#!/usr/bin/env python3
import os
import sys
import argparse
from argparse import RawTextHelpFormatter
import logging

# Add forgather root to sys.path to enable importing from examples/
# Find the workspace directory (forgather root)
from forgather import MetaConfig
forgather_root = MetaConfig.find_workspace_dir(os.path.abspath(__file__))
if forgather_root not in sys.path:
    sys.path.insert(0, forgather_root)

from forgather.ml.model_conversion import (
    get_converter,
    detect_model_type_from_hf,
    detect_model_type_from_forgather,
    list_converters,
)

# Import model converters to register them
# This ensures they are available in the registry
import examples.models.llama.src.converter
import examples.models.mistral.src.converter
import examples.models.qwen3.src.converter

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def validate_paths(src_model_path, dst_model_path):
    """Validate source and destination paths."""
    src_model_path = os.path.abspath(src_model_path)
    dst_model_path = os.path.abspath(dst_model_path)

    assert os.path.isdir(src_model_path), "The source path must be a directory"
    dest_dir = os.path.dirname(dst_model_path)
    assert os.path.isdir(
        dest_dir
    ), f"The destination directory, {dest_dir}, does not exist"
    assert not os.path.exists(
        dst_model_path
    ), "The destination path already exists. Will not overwrite."

    return src_model_path, dst_model_path


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Convert between Huggingface Llama/Mistral and Forgather Dynamic Llama models",
        epilog=(
            "Typical Usage:\n"
            "\n"
            "Convert from Llama or Mistral: ./convert_llama.py --dtype bfloat16 --max-length 16384 -t ../chat_templates/chatml_eos.jinja \\\n"
            "  --add-tokens example_additional_tokens.yaml ~/models/hf_llama ~/models/fg_mistral\n"
            "\n"
            "Convert to Mistral: ./convert_llama.py --reverse --model-type mistral --dtype bfloat16 \\\n"
            "  ~/models/fg_mistral ~/models/my_hf_llama"
        ),
    )
    parser.add_argument(
        "src_model_path",
        type=os.path.expanduser,
        help="Path to source model (HF Llama/Mistral/Qwen3 or Forgather model)",
    )
    parser.add_argument(
        "dst_model_path",
        type=os.path.expanduser,
        help="Output directory",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Convert from Forgather Dynamic Llama to HuggingFace Llama/Mistral/Qwen3 (default: HF to Forgather)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["llama", "mistral", "qwen3"],
        default="llama",
        help="Target model type for reverse conversion (default: llama)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Torch output dtype",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override max model length of exported model.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=os.path.expanduser,
        default=None,
        help="Path to load Forgather checkpoint from (if not specified, will use latest checkpoint in src_model_path)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Destination model test device",
    )
    parser.add_argument(
        "-g",
        "--generation-test",
        action="store_true",
        help="Test model generation with prompt",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The old bookstore at the corner of Elm Street had always held secrets, but today, something unusual caught Emma's eye.",
        help="Destination model test prompt",
    )
    parser.add_argument(
        "--debug-params",
        action="store_true",
        help="Print parameter names for debugging mapping",
    )
    parser.add_argument(
        "-t",
        "--chat-template-path",
        type=os.path.expanduser,
        default=None,
        help="Assign chat template at given path to output tokenizer.",
    )
    parser.add_argument(
        "--add-tokens",
        type=os.path.expanduser,
        default=None,
        help="Path to YAML file specifying additional tokens to add to vocabulary.",
    )

    args = parser.parse_args(args)
    return args


def convert_hf_to_forgather(args):
    """Convert HuggingFace Llama/Mistral model to Forgather format using converter plugins."""
    src_model_path, dst_model_path = validate_paths(args.src_model_path, args.dst_model_path)

    # Auto-detect model type from HuggingFace config
    print("Detecting model type...")
    model_type = detect_model_type_from_hf(src_model_path)
    print(f"Detected model type: {model_type}")

    # Get appropriate converter
    converter_class = get_converter(model_type)
    converter = converter_class()

    # Build conversion kwargs
    kwargs = {
        "debug_params": args.debug_params,
        "prompt": args.prompt,
        "chat_template_path": args.chat_template_path,
    }

    # Note: Vocabulary extension (--add-tokens) is excluded from Phase 1
    # This will be added in Phase 2
    if args.add_tokens:
        print("Warning: Vocabulary extension (--add-tokens) is not yet supported in the refactored converter.")
        print("This feature will be added in Phase 2.")

    # Delegate to converter
    converter.convert_to_forgather(
        src_model_path=src_model_path,
        dst_model_path=dst_model_path,
        dtype=args.dtype,
        max_length=args.max_length,
        **kwargs
    )


def convert_forgather_to_hf(args):
    """Convert Forgather model to HuggingFace format using converter plugins."""
    src_model_path, dst_model_path = validate_paths(args.src_model_path, args.dst_model_path)

    # Try to auto-detect model type from Forgather config
    print("Detecting model type...")
    detected_type = detect_model_type_from_forgather(src_model_path)

    # Check if detected type is a supported converter type
    available_converters = list_converters()
    model_type = None

    if detected_type and detected_type in available_converters:
        # Auto-detection successful and supported
        print(f"Detected model type: {detected_type}")
        model_type = detected_type
        if args.model_type != "llama" and args.model_type != detected_type:
            print(f"Warning: Detected type '{detected_type}' differs from --model-type '{args.model_type}'")
            print(f"Using detected type: {detected_type}")
    else:
        # Auto-detection failed or returned unsupported type, use user-specified
        if detected_type:
            print(f"Detected forgather-specific model type: {detected_type}")
        print(f"Using --model-type: {args.model_type}")
        model_type = args.model_type

    # Get appropriate converter
    converter_class = get_converter(model_type)
    converter = converter_class()

    # Build conversion kwargs
    kwargs = {
        "debug_params": args.debug_params,
        "prompt": args.prompt,
    }

    # Delegate to converter
    converter.convert_from_forgather(
        src_model_path=src_model_path,
        dst_model_path=dst_model_path,
        dtype=args.dtype,
        max_length=args.max_length,
        checkpoint_path=args.checkpoint_path,
        **kwargs
    )


def main():
    args = parse_args()

    # Show available converters
    logger.info(f"Available model converters: {list_converters()}")

    # Validate arguments
    if not args.reverse and args.model_type != "llama":
        print(
            "Warning: --model-type is only used for reverse conversion (--reverse). Ignoring."
        )

    if args.reverse:
        print(
            f"Converting Forgather model to HuggingFace {args.model_type.capitalize()}"
        )
        convert_forgather_to_hf(args)
    else:
        print("Converting HuggingFace model to Forgather format")
        convert_hf_to_forgather(args)


if __name__ == "__main__":
    main()
