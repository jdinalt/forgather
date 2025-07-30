#!/usr/bin/env python3
import os
import argparse
from argparse import RawTextHelpFormatter
import logging

import torch
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig,
    AutoTokenizer,
)
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from forgather.config import ConfigEnvironment
from forgather.latent import Latent
from forgather.ml.remap_params import remap_state_dict
from forgather.ml.sharded_checkpoint import (
    save_checkpoint,
    load_checkpoint,
    create_sharing_metadata,
    retie_parameters,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

model_config_template = """
-- extends "models/dynamic_llama.yaml"

-- block model_meta_config
    == super()
-- endblock model_meta_config

-- block model_tokenizer
tokenizer: &tokenizer !singleton:transformers:AutoTokenizer.from_pretrained@tokenizer
    args:
        - "{{ tokenizer_path }}"
    kwargs:
        legacy: False
        model_max_length: {{ max_model_length }}
-- endblock model_tokenizer

-- block model_config
    == super()

    # Imported Config
    attention_dropout: !!float {{ attention_dropout }}
    hidden_size: {{ hidden_size }}
    num_attention_heads: {{ num_attention_heads }}
    num_kv_heads: {{ num_kv_heads }}
    d_head: {{ d_head }}
    num_hidden_layers: {{ num_hidden_layers }}
    dim_feedforward: {{ dim_feedforward }}
    rope_theta: !!float {{ rope_theta }}
    rms_norm_eps: !!float {{ rms_norm_eps }}
-- endblock model_config

-- block init_weights
# We are going to replace these anyway
init_weights: &init_weights !partial:.init_weights:init_pass []
-- endblock init_weights
"""

config_template = """
-- set ns = namespace()
-- set ns.forgather_dir = forgather_root
-- set ns.output_dir = model_output_dir

.define: &model_constructor_args {}
-- include "model_config"
"""

hflamma_to_dllama = [
    (r"lm_head\.", r"causal_lm.output_decoder.", []),
    (
        r"model\.",
        r"causal_lm.",
        [
            (r"embed_tokens\.", r"input_encoder.embedding.", []),
            (r"norm\.", r"layer_stack.layer_norm.", []),
            (
                r"layers\.(\d+)\.",
                r"layer_stack.layers.\1.",
                [
                    (
                        r"self_attn\.",
                        r"attention.",
                        [
                            (r"q_proj\.", r"query_linear.", []),
                            (r"k_proj\.", r"key_linear.", []),
                            (r"v_proj\.", r"value_linear.", []),
                            (r"o_proj\.", r"output_linear.", []),
                        ],
                    ),
                    (r"mlp\.", r"feedforward.", []),
                    (r"input_layernorm\.", r"norm1.", []),
                    (r"post_attention_layernorm\.", r"norm2.", []),
                ],
            ),
        ],
    ),
]

# Reverse mapping: Forgather Dynamic Llama to HuggingFace Llama
dllama_to_hflamma = [
    (r"causal_lm\.output_decoder\.", r"lm_head.", []),
    (
        r"causal_lm\.",
        r"model.",
        [
            (r"input_encoder\.embedding\.", r"embed_tokens.", []),
            (r"layer_stack\.layer_norm\.", r"norm.", []),
            (
                r"layer_stack\.layers\.(\d+)\.",
                r"layers.\1.",
                [
                    (
                        r"attention\.",
                        r"self_attn.",
                        [
                            (r"query_linear\.", r"q_proj.", []),
                            (r"key_linear\.", r"k_proj.", []),
                            (r"value_linear\.", r"v_proj.", []),
                            (r"output_linear\.", r"o_proj.", []),
                        ],
                    ),
                    (r"feedforward\.", r"mlp.", []),
                    (r"norm1\.", r"input_layernorm.", []),
                    (r"norm2\.", r"post_attention_layernorm.", []),
                ],
            ),
        ],
    ),
]


def setup_conversion(args):
    """Setup and validate paths, dtype, and directories for conversion"""
    src_model_path = os.path.abspath(args.src_model_path)
    dst_model_path = os.path.abspath(args.dst_model_path)

    from forgather.ml.construct import torch_dtype

    if args.dtype:
        new_dtype = torch_dtype(args.dtype)
    else:
        new_dtype = None

    logger.info(f"Source: {src_model_path}")
    logger.info(f"Destination: {dst_model_path}")
    logger.info(f"DType: {new_dtype}")

    assert os.path.isdir(src_model_path), "The source path must be a directory"
    dest_dir = os.path.dirname(dst_model_path)
    assert os.path.isdir(
        dest_dir
    ), f"The destination directory, {dest_dir}, does not exist"
    assert not os.path.exists(
        dst_model_path
    ), "The destination path already exists. Will not overwrite."

    return src_model_path, dst_model_path, new_dtype


def print_debug_params(model, label, args):
    """Print parameter names for debugging if requested"""
    if args.debug_params:
        print(f"{label} parameter names:")
        for name in model.state_dict().keys():
            print(f"  {name}")


def compare_model_logits(
    src_model,
    dst_model,
    tokenizer,
    prompt,
    src_label="Source",
    dst_label="Destination",
    tolerance=1e-5,
):
    """Compare logits between source and destination models"""
    src_logits = test_model_forward(src_model, tokenizer, prompt, "cpu")
    dst_logits = test_model_forward(dst_model, tokenizer, prompt, "cpu")

    if not torch.allclose(src_logits, dst_logits, atol=tolerance):
        print("Model logits are dissimilar")
        print(f"{src_label} Model Logits")
        print(src_logits.shape)
        print(src_logits)

        print(f"{dst_label} Model Logits")
        print(dst_logits.shape)
        print(dst_logits)

        print(f"Max diff: {torch.max(torch.abs(src_logits - dst_logits)).item()}")
        print(f"Mean diff: {torch.mean(torch.abs(src_logits - dst_logits)).item()}")
        print(
            f"{src_label} logits range: [{src_logits.min().item():.6f}, {src_logits.max().item():.6f}]"
        )
        print(
            f"{dst_label} logits range: [{dst_logits.min().item():.6f}, {dst_logits.max().item():.6f}]"
        )
    else:
        print("Model logits match.")


def test_generation_if_requested(model, tokenizer, args):
    """Test model generation if requested by user"""
    if not args.generation_test:
        return

    if args.device != "cpu":
        print(f"Moving model to {args.device}")
        model.to(device=args.device)

    print("Testing model generation...")
    gen_config = GenerationConfig(
        pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        do_sample=True,
        top_k=20,
        top_p=0.9,
        temperature=0.7,
        repitition_penalty=1.15,
    )

    text = generate_text(model, tokenizer, args.prompt, gen_config, 100, args.device)
    print(f"Test Prompt: {text}")


def load_state_dict_with_validation(
    model, mapped_state_dict, strict=False, assign=True, validate_rope_only=False
):
    """Load state dict with validation and optional RoPE buffer checking"""
    print("Loading mapped state dictionary...")
    result = model.load_state_dict(mapped_state_dict, strict=strict, assign=assign)
    print(f"load_state_dict() result: {result}")

    # Validate that unused parameters are only RoPE cached buffers (for reverse conversion)
    if validate_rope_only and result.unexpected_keys:
        non_rope_unexpected = [
            key
            for key in result.unexpected_keys
            if not (key.endswith(".cos_cached") or key.endswith(".sin_cached"))
        ]
        if non_rope_unexpected:
            print(
                f"Warning: Unexpected non-RoPE parameters not loaded: {non_rope_unexpected}"
            )
        else:
            print(
                f"As expected, {len(result.unexpected_keys)} RoPE cached buffers were not loaded (they will be recomputed)"
            )

    return result


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Convert between Huggingface Llama and Forgather Dynamic Llama models",
    )
    parser.add_argument(
        "src_model_path",
        type=str,
        help="Path to source model (HF Llama or Forgather model)",
    )
    parser.add_argument(
        "dst_model_path",
        help="Output directory",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Convert from Forgather Dynamic Llama to HuggingFace Llama (default: HF to Forgather)",
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

    args = parser.parse_args(args)
    return args


def generate_text(model, tokenizer, prompt, gen_config, max_new_tokens, device):
    model.to(device)
    model.eval()

    with torch.inference_mode():
        tokenizer_outputs = tokenizer(
            [prompt],
            truncation=False,
            return_length=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_ids = tokenizer_outputs["input_ids"].to(device)
        attention_mask = tokenizer_outputs["attention_mask"].to(device)
        use_cache = getattr(model, "_supports_cache_class", False)
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
            return_dict_in_generate=True,
            use_cache=use_cache,
            past_key_values=None,
            max_new_tokens=max_new_tokens,
        )

        output_text = tokenizer.decode(
            outputs.sequences[0],
            skip_special_tokens=True,
        )
        return prompt + " [START] " + output_text[len(prompt) + 1 :]


def test_model_forward(model, tokenizer, prompt, device):
    model.to(device)
    model.eval()

    tokenizer_outputs = tokenizer(
        [prompt],
        truncation=False,
        return_length=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    with torch.inference_mode():
        input_ids = tokenizer_outputs["input_ids"].to(device)
        outputs = model(input_ids, return_dict=True)
        logits = outputs.logits
    return logits


def convert_hf_to_forgather(args):
    """Convert HuggingFace Llama model to Forgather Dynamic Llama format"""
    src_model_path, dst_model_path, new_dtype = setup_conversion(args)

    from forgather.ml.construct import copy_package_files

    src_model_config = AutoConfig.from_pretrained(src_model_path)

    # Ensure only supported config values
    assert src_model_config.model_type == "llama"
    assert src_model_config.hidden_act == "silu"
    assert src_model_config.tie_word_embeddings == False
    assert src_model_config.mlp_bias == False
    assert src_model_config.attention_bias == False
    assert src_model_config.rope_scaling == None

    logger.info(src_model_config)

    tokenizer = AutoTokenizer.from_pretrained(src_model_path)

    print("Loading source model...")
    src_model = AutoModelForCausalLM.from_pretrained(src_model_path)
    logger.debug(src_model)

    if new_dtype:
        print(f"Converting model dtype to {new_dtype}")
        src_model.to(dtype=new_dtype)

    print_debug_params(src_model, "Source model", args)

    print("Remapping model weight names")
    src_state_dict = src_model.state_dict()
    mapped_state_dict = remap_state_dict(src_state_dict, hflamma_to_dllama)

    # Get script directory and make paths relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    forgather_root = os.path.dirname(script_dir)
    template_searchpath = [
        os.path.join(forgather_root, "templatelib/base"),
        os.path.join(forgather_root, "templatelib/examples"),
    ]

    env = ConfigEnvironment(searchpath=template_searchpath)
    template_loader = env.pp_environment.loader
    template_loader.add_template("model_config", model_config_template)

    max_model_length = src_model_config.max_position_embeddings
    if args.max_length:
        max_model_length = args.max_length

    pp_config = env.preprocess_from_string(
        config_template,
        forgather_root=forgather_root,
        model_output_dir=dst_model_path,
        tokenizer_path=src_model_path,
        attention_dropout=src_model_config.attention_dropout,
        max_model_length=max_model_length,
        hidden_size=src_model_config.hidden_size,
        num_attention_heads=src_model_config.num_attention_heads,
        num_kv_heads=src_model_config.num_key_value_heads,
        d_head=src_model_config.head_dim,
        num_hidden_layers=src_model_config.num_hidden_layers,
        dim_feedforward=src_model_config.intermediate_size,
        rope_theta=src_model_config.rope_theta,
        rms_norm_eps=src_model_config.rms_norm_eps,
    )

    logger.debug(pp_config)

    config = env.load_from_ppstring(pp_config).config

    tokenizer = Latent.materialize(config, mtargets="tokenizer")
    logger.debug(tokenizer)

    model_config = Latent.materialize(config, mtargets="model_config")
    logger.info(model_config)

    copy_package_files(dst_model_path, model_config, "ok")

    model_ctor = Latent.materialize(config, mtargets="model")

    print("Constructing destination model")
    model = model_ctor("cpu")
    logger.debug(model)

    if new_dtype:
        print(f"Converting new model dtype to {new_dtype}")
        model.to(dtype=new_dtype)

    print_debug_params(model, "Destination model", args)

    load_state_dict_with_validation(model, mapped_state_dict, strict=False, assign=True)

    # Confirm remapped model produces same logits as original
    compare_model_logits(
        src_model, model, tokenizer, args.prompt, "Source", "Destination"
    )

    test_generation_if_requested(model, tokenizer, args)

    print("Saving model...")
    model_config.save_pretrained(save_directory=dst_model_path)
    tokenizer.save_pretrained(save_directory=dst_model_path)
    save_checkpoint(
        output_dir=dst_model_path,
        module=model,
        # Safetensors can't deal with shared tensors
        safetensors=False,
        include_param_sharing=True,
    )


def convert_forgather_to_hf(args):
    """Convert Forgather Dynamic Llama model to HuggingFace Llama format"""
    src_model_path, dst_model_path, new_dtype = setup_conversion(args)

    # Load the Forgather model configuration to get the original HF config
    src_model_config = AutoConfig.from_pretrained(
        src_model_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(src_model_path)

    print("Loading Forgather model...")
    # Load as meta model first
    with torch.device("meta"):
        src_model = AutoModelForCausalLM.from_config(
            src_model_config, trust_remote_code=True
        )

    # Create sharing metadata and materialize model
    sharing_metadata = create_sharing_metadata(src_model)

    if new_dtype:
        src_model.to(dtype=new_dtype)
    src_model.to_empty(device="cpu")
    retie_parameters(src_model, sharing_metadata)

    # Load the actual weights
    load_checkpoint(src_model_path, src_model, device="cpu", strict=True)

    print_debug_params(src_model, "Source Forgather model", args)

    print("Remapping model weight names to HuggingFace format")
    src_state_dict = src_model.state_dict()
    mapped_state_dict = remap_state_dict(src_state_dict, dllama_to_hflamma)

    max_model_length = src_model_config.max_sequence_length
    if args.max_length:
        max_model_length = args.max_length

    # Create proper HuggingFace Llama config from Forgather config
    print("Creating HuggingFace Llama config...")
    hf_config = LlamaConfig(
        vocab_size=src_model_config.vocab_size,
        hidden_size=src_model_config.hidden_size,
        intermediate_size=src_model_config.dim_feedforward,
        num_hidden_layers=src_model_config.num_hidden_layers,
        num_attention_heads=src_model_config.num_attention_heads,
        num_key_value_heads=src_model_config.num_kv_heads,
        head_dim=src_model_config.d_head,
        max_position_embeddings=max_model_length,
        rms_norm_eps=src_model_config.rms_norm_eps,
        rope_theta=src_model_config.rope_theta,
        attention_dropout=src_model_config.attention_dropout,
        hidden_act="silu",
        mlp_bias=False,
        attention_bias=False,
        tie_word_embeddings=False,
        pad_token_id=src_model_config.pad_token_id,
        bos_token_id=src_model_config.bos_token_id,
        eos_token_id=src_model_config.eos_token_id,
    )

    # Create proper HuggingFace Llama model
    print("Creating HuggingFace Llama model...")
    hf_model = LlamaForCausalLM(hf_config)

    if new_dtype:
        print(f"Converting model dtype to {new_dtype}")
        hf_model.to(dtype=new_dtype)

    print_debug_params(hf_model, "Destination HuggingFace model", args)

    if args.debug_params:
        print("Mapped parameter names:")
        for name in mapped_state_dict.keys():
            print(f"  {name}")

    load_state_dict_with_validation(
        hf_model, mapped_state_dict, strict=False, assign=True, validate_rope_only=True
    )

    # Confirm remapped model produces same logits as original
    compare_model_logits(
        src_model,
        hf_model,
        tokenizer,
        args.prompt,
        "Source Forgather",
        "HuggingFace",
        tolerance=1e-5,
    )

    test_generation_if_requested(hf_model, tokenizer, args)

    print("Saving HuggingFace model...")
    hf_model.save_pretrained(dst_model_path)
    tokenizer.save_pretrained(dst_model_path)


def main():
    args = parse_args()

    if args.reverse:
        print("Converting Forgather Dynamic Llama to HuggingFace Llama")
        convert_forgather_to_hf(args)
    else:
        print("Converting HuggingFace Llama to Forgather Dynamic Llama")
        convert_hf_to_forgather(args)


if __name__ == "__main__":
    main()
