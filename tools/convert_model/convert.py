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
    detect_model_type,
    list_converters,
    discover_and_register_converters,
)

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
        description="Convert between Huggingface Llama/Mistral/Qwen3 and Forgather Dynamic models",
        epilog=(
            "Typical Usage:\n"
            "\n"
            "Convert HF to Forgather (auto-detected):\n"
            "  ./convert.py --dtype bfloat16 --max-length 16384 ~/models/hf_llama ~/models/fg_llama\n"
            "\n"
            "Convert Forgather to HF (auto-detected):\n"
            "  ./convert.py --dtype bfloat16 ~/models/fg_llama ~/models/my_hf_llama\n"
            "\n"
            "Force direction with --reverse:\n"
            "  ./convert.py --reverse --dtype bfloat16 ~/models/fg_mistral ~/models/my_hf_mistral\n"
            "\n"
            "Override auto-detected model type:\n"
            "  ./convert.py --reverse --model-type qwen3 ~/models/fg_qwen ~/models/my_hf_qwen\n"
            "\n"
            "Note: Conversion direction is auto-detected from the source model's config.\n"
            "      Models converted from HF->FG store 'hf_model_type' metadata for auto-detection."
        ),
    )
    parser.add_argument(
        "src_model_path",
        type=os.path.expanduser,
        help="Path to source model (HF or Forgather - direction auto-detected)",
    )
    parser.add_argument(
        "dst_model_path",
        type=os.path.expanduser,
        help="Output directory",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Force Forgather->HF conversion (optional - direction is auto-detected by default)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["llama", "mistral", "qwen3"],
        default="llama",
        help="Override auto-detected model type for FG->HF conversion (default: llama if auto-detection fails)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Torch output dtype. If not specified, uses source model's dtype (or bfloat16 if not available)",
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
    parser.add_argument(
        "--converter-path",
        action="append",
        dest="converter_paths",
        type=os.path.expanduser,
        default=[],
        help="Additional directory path(s) to search for model converters. Can be specified multiple times.",
    )

    args = parser.parse_args(args)
    return args


def convert_hf_to_forgather(args, detected_model_type=None):
    """Convert HuggingFace Llama/Mistral/Qwen3 model to Forgather format using converter plugins."""
    src_model_path, dst_model_path = validate_paths(
        args.src_model_path, args.dst_model_path
    )

    # Use detected model type from main() if available
    if detected_model_type:
        model_type = detected_model_type
        print(f"Using detected model type: {model_type}")
    else:
        # Fallback: Try to detect from config
        from forgather.ml.model_conversion import detect_model_type_from_hf

        print("Detecting model type from config...")
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
        "add_tokens": args.add_tokens,
    }

    # Delegate to converter
    converter.convert_to_forgather(
        src_model_path=src_model_path,
        dst_model_path=dst_model_path,
        dtype=args.dtype,
        max_length=args.max_length,
        **kwargs,
    )


def convert_forgather_to_hf(args, detected_model_type=None):
    """Convert Forgather model to HuggingFace format using converter plugins."""
    src_model_path, dst_model_path = validate_paths(
        args.src_model_path, args.dst_model_path
    )

    # Use detected model type from main() if available
    if detected_model_type:
        model_type = detected_model_type
        print(f"Using detected model type: {model_type}")
    else:
        # Fallback: Use --model-type
        print(f"Warning: Could not auto-detect model type from config")
        print(
            f"This may indicate the model was not converted from HuggingFace using this tool"
        )
        print(f"Using --model-type: {args.model_type}")
        model_type = args.model_type

    # Check if model type is supported
    available_converters = list_converters()
    if model_type not in available_converters:
        raise ValueError(
            f"Model type '{model_type}' is not supported. "
            f"Available converters: {available_converters}"
        )

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
        **kwargs,
    )


def main():
    args = parse_args()

    # Discover and register converters from builtin and custom paths
    custom_paths = args.converter_paths if args.converter_paths else None
    discover_and_register_converters(custom_paths, forgather_root)

    # Show available converters
    available_converters = list_converters()
    logger.info(f"Available model converters: {available_converters}")
    if not available_converters:
        print("ERROR: No model converters found!")
        print("Make sure you're running from the Forgather root directory.")
        sys.exit(1)

    # Auto-detect conversion direction and model type
    direction = None
    detected_model_type = None

    if args.reverse:
        # Explicit --reverse flag takes precedence
        direction = "fg_to_hf"
        logger.info("Using explicit --reverse flag for FG->HF conversion")
    else:
        # Try to auto-detect direction and model type
        detection_result = detect_model_type(args.src_model_path)

        if detection_result:
            source, model_type = detection_result
            detected_model_type = model_type

            if source == "forgather":
                direction = "fg_to_hf"
                logger.info(
                    f"Auto-detected Forgather model with hf_model_type={model_type}"
                )
                print(f"Auto-detected Forgather model (original type: {model_type})")
            else:  # source == "huggingface"
                direction = "hf_to_fg"
                logger.info(
                    f"Auto-detected HuggingFace model with model_type={model_type}"
                )
                print(f"Auto-detected HuggingFace model (type: {model_type})")
        else:
            # Default to HF->FG if detection fails
            direction = "hf_to_fg"
            logger.info(
                "Could not auto-detect model type, defaulting to HF->FG conversion"
            )
            print(
                "Warning: Could not auto-detect model type, assuming HF->FG conversion"
            )

    # Validate arguments
    if direction == "hf_to_fg" and args.model_type != "llama":
        print(
            "Warning: --model-type is only used for FG->HF conversion. Ignoring for HF->FG."
        )

    if direction == "fg_to_hf":
        # Use detected model type if available, otherwise use --model-type
        target_type = detected_model_type if detected_model_type else args.model_type
        print(f"Converting Forgather model to HuggingFace {target_type.capitalize()}")
        convert_forgather_to_hf(args, detected_model_type=detected_model_type)
    else:
        print("Converting HuggingFace model to Forgather format")
        convert_hf_to_forgather(args, detected_model_type=detected_model_type)


if __name__ == "__main__":
    main()
