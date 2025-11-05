#!/usr/bin/env python3
import os
import argparse
from argparse import RawTextHelpFormatter
import logging
from contextlib import ExitStack
import yaml

import torch
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig,
    AutoTokenizer,
)
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.mistral import MistralConfig, MistralForCausalLM
from forgather.config import ConfigEnvironment
from forgather.latent import Latent
from forgather.ml.remap_params import remap_state_dict
from forgather.ml.sharded_checkpoint import (
    save_checkpoint,
    load_checkpoint,
    create_sharing_metadata,
    retie_parameters,
    find_latest_checkpoint,
)
from forgather.ml.utils import default_dtype
from forgather.ml.no_init_weights import no_init_weights

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

model_config_template = """
-- extends "models/transformers/dynamic_llama.yaml"

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
    enable_activation_checkpoint: {{ enable_activation_checkpoint }}
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

    # If vocab sizes differ (e.g., added PAD token), compare only the overlapping vocab
    if src_logits.shape != dst_logits.shape:
        min_vocab_size = min(src_logits.shape[-1], dst_logits.shape[-1])
        print(f"Vocab size mismatch: {src_logits.shape[-1]} vs {dst_logits.shape[-1]}")
        print(f"Comparing only first {min_vocab_size} tokens")
        src_logits = src_logits[..., :min_vocab_size]
        dst_logits = dst_logits[..., :min_vocab_size]

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


def create_hf_config_and_model(src_model_config, max_model_length, model_type, new_dtype):
    """Create appropriate HF config and model based on detected type"""
    if model_type == "mistral":
        print("Creating HuggingFace Mistral config...")
        hf_config = MistralConfig(
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
            attention_dropout=getattr(src_model_config, "attention_dropout", 0.0),
            hidden_act="silu",
            tie_word_embeddings=False,
            sliding_window=4096,  # Default Mistral sliding window
            pad_token_id=getattr(src_model_config, "pad_token_id", None),
            bos_token_id=getattr(src_model_config, "bos_token_id", 1),
            eos_token_id=getattr(src_model_config, "eos_token_id", 2),
        )

        print("Creating HuggingFace Mistral model...")
        with ExitStack() as exit_stack:
            if new_dtype:
                exit_stack.enter_context(default_dtype(new_dtype))
            exit_stack.enter_context(torch.device("cpu"))
            exit_stack.enter_context(no_init_weights())
            hf_model = MistralForCausalLM(hf_config)

    else:  # llama
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
            attention_dropout=getattr(src_model_config, "attention_dropout", 0.0),
            hidden_act="silu",
            mlp_bias=False,
            attention_bias=False,
            tie_word_embeddings=False,
            pad_token_id=getattr(src_model_config, "pad_token_id", None),
            bos_token_id=getattr(src_model_config, "bos_token_id", 1),
            eos_token_id=getattr(src_model_config, "eos_token_id", 2),
        )

        print("Creating HuggingFace Llama model...")
        with ExitStack() as exit_stack:
            if new_dtype:
                exit_stack.enter_context(default_dtype(new_dtype))
            exit_stack.enter_context(torch.device("cpu"))
            exit_stack.enter_context(no_init_weights())
            hf_model = LlamaForCausalLM(hf_config)

    return hf_config, hf_model, model_type


def load_additional_tokens(yaml_path):
    """Load additional tokens from YAML file.

    Expected format:
    special_tokens:
      - "<|im_start|>"
      - "<|im_end|>"
    regular_tokens:
      - "custom_token_1"
      - "custom_token_2"
    """
    if not yaml_path:
        return [], []

    with open(yaml_path, 'r') as f:
        token_config = yaml.safe_load(f)

    special_tokens = token_config.get('special_tokens', [])
    regular_tokens = token_config.get('regular_tokens', [])

    return special_tokens, regular_tokens


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
        )
    )
    parser.add_argument(
        "src_model_path",
        type=str,
        help="Path to source model (HF Llama/Mistral or Forgather model)",
    )
    parser.add_argument(
        "dst_model_path",
        help="Output directory",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Convert from Forgather Dynamic Llama to HuggingFace Llama/Mistral (default: HF to Forgather)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["llama", "mistral"],
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
        "--enable-checkpoint",
        action="store_true",
        help="Enable activation checkpointing.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
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
        type=str,
        default=None,
        help="Assign chat template at given path to output tokenizer.",
    )
    parser.add_argument(
        "--add-tokens",
        type=str,
        default=None,
        help="Path to YAML file specifying additional tokens to add to vocabulary.",
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
    """Convert HuggingFace Llama/Mistral model to Forgather Dynamic Llama format"""
    src_model_path, dst_model_path, new_dtype = setup_conversion(args)

    from forgather.ml.construct import copy_package_files

    src_model_config = AutoConfig.from_pretrained(src_model_path)

    # Ensure only supported config values
    assert src_model_config.model_type in [
        "llama",
        "mistral",
    ], f"Unsupported model type: {src_model_config.model_type}"
    assert src_model_config.hidden_act == "silu"
    assert src_model_config.tie_word_embeddings == False

    # Llama-specific checks
    if src_model_config.model_type == "llama":
        assert src_model_config.mlp_bias == False
        assert src_model_config.attention_bias == False
        assert src_model_config.rope_scaling == None

    # Mistral models have these hardcoded to False, so no need to check

    logger.info(src_model_config)
    tokenizer = AutoTokenizer.from_pretrained(src_model_path)

    print("Loading source model...")

    with ExitStack() as exit_stack:
        if new_dtype:
            exit_stack.enter_context(default_dtype(new_dtype))
        src_model = AutoModelForCausalLM.from_pretrained(src_model_path)
    logger.debug(src_model)

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
        attention_dropout=getattr(src_model_config, "attention_dropout", 0.0),
        max_model_length=max_model_length,
        hidden_size=src_model_config.hidden_size,
        num_attention_heads=src_model_config.num_attention_heads,
        num_kv_heads=src_model_config.num_key_value_heads,
        d_head=src_model_config.hidden_size // src_model_config.num_attention_heads,
        num_hidden_layers=src_model_config.num_hidden_layers,
        dim_feedforward=src_model_config.intermediate_size,
        rope_theta=src_model_config.rope_theta,
        rms_norm_eps=src_model_config.rms_norm_eps,
        enable_activation_checkpoint=args.enable_checkpoint,
    )

    logger.debug(pp_config)

    config = env.load_from_ppstring(pp_config).config

    tokenizer = Latent.materialize(config, mtargets="tokenizer")
    logger.debug(tokenizer)
    if args.chat_template_path:
        with open(args.chat_template_path, "r") as f:
            chat_template = f.read()
        logger.info(f"Setting tokenizer chat template to: {chat_template}")
        tokenizer.chat_template = chat_template

    # Check if we need to add a PAD token, but don't do it yet
    # We'll add it after loading the model to avoid vocab size mismatch
    needs_pad_token = tokenizer.pad_token is None

    # Load additional tokens to add
    special_tokens, regular_tokens = load_additional_tokens(args.add_tokens)

    model_config = Latent.materialize(config, mtargets="model_config")
    logger.info(model_config)

    copy_package_files(dst_model_path, model_config, "ok")

    model_ctor = Latent.materialize(config, mtargets="model")

    print("Constructing destination model")
    with ExitStack() as exit_stack:
        if new_dtype:
            exit_stack.enter_context(default_dtype(new_dtype))
        exit_stack.enter_context(no_init_weights())                     
        model = model_ctor()
    logger.debug(model)

    print_debug_params(model, "Destination model", args)

    load_state_dict_with_validation(model, mapped_state_dict, strict=False, assign=True)

    # Add PAD token and additional tokens, then resize embeddings if needed
    tokens_to_add = []
    if needs_pad_token:
        print("No PAD token defined. Adding new PAD token with zero-initialized embedding")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.padding_side = 'right'
        tokens_to_add.append(('pad', tokenizer.pad_token_id))

    # Add additional tokens from YAML file
    if special_tokens or regular_tokens:
        num_added = 0
        if special_tokens:
            print(f"Adding {len(special_tokens)} special tokens: {special_tokens}")
            result = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            num_added += result
        if regular_tokens:
            print(f"Adding {len(regular_tokens)} regular tokens: {regular_tokens}")
            result = tokenizer.add_tokens(regular_tokens)
            num_added += result

        # Track which tokens were added for initialization
        for token in special_tokens + regular_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            tokens_to_add.append(('random', token_id))

        print(f"Successfully added {num_added} new tokens")

    # Resize embeddings if any tokens were added
    if tokens_to_add:
        old_vocab_size = src_model_config.vocab_size
        new_vocab_size = len(tokenizer)

        print(f"Resizing model embeddings from {old_vocab_size} to {new_vocab_size}")
        # Access the input embedding layer
        input_embedding = model.causal_lm.input_encoder.embedding
        output_decoder = model.causal_lm.output_decoder

        # Get dtype and device from existing layers
        embed_dtype = input_embedding.weight.dtype
        embed_device = input_embedding.weight.device

        # Infer initialization std from existing embeddings
        with torch.no_grad():
            embedding_std = input_embedding.weight.std().item()
            print(f"Inferred embedding initialization std: {embedding_std:.6f}")

        # Create new embedding layer with extended size, matching dtype and device
        new_input_embedding = torch.nn.Embedding(
            new_vocab_size,
            input_embedding.weight.shape[1],
            dtype=embed_dtype,
            device=embed_device
        )
        # Copy old embeddings
        with torch.no_grad():
            new_input_embedding.weight[:old_vocab_size] = input_embedding.weight

            # Initialize new token embeddings
            for init_type, token_id in tokens_to_add:
                if init_type == 'pad':
                    # Zero-initialize PAD token
                    new_input_embedding.weight[token_id].zero_()
                    print(f"Initialized token {token_id} (PAD) with zeros")
                elif init_type == 'random':
                    # Random initialize with inferred std
                    new_input_embedding.weight[token_id].normal_(mean=0.0, std=embedding_std)
                    print(f"Initialized token {token_id} with N(0, {embedding_std:.6f})")

        # Replace the embedding layer
        model.causal_lm.input_encoder.embedding = new_input_embedding

        # Also resize output decoder (lm_head)
        new_output_decoder = torch.nn.Linear(
            output_decoder.in_features,
            new_vocab_size,
            bias=output_decoder.bias is not None,
            dtype=embed_dtype,
            device=embed_device
        )
        with torch.no_grad():
            new_output_decoder.weight[:old_vocab_size] = output_decoder.weight

            # Initialize new token output weights
            output_std = output_decoder.weight.std().item()
            print(f"Inferred output weight initialization std: {output_std:.6f}")

            for init_type, token_id in tokens_to_add:
                if init_type == 'pad':
                    # Zero-initialize PAD token
                    new_output_decoder.weight[token_id].zero_()
                elif init_type == 'random':
                    # Random initialize with inferred std
                    new_output_decoder.weight[token_id].normal_(mean=0.0, std=output_std)

            if new_output_decoder.bias is not None:
                new_output_decoder.bias[:old_vocab_size] = output_decoder.bias
                new_output_decoder.bias[old_vocab_size:].zero_()

        model.causal_lm.output_decoder = new_output_decoder

        # Update model config
        model_config.vocab_size = new_vocab_size
        if needs_pad_token:
            model_config.pad_token_id = tokenizer.pad_token_id
            print(f"Set pad_token_id to {tokenizer.pad_token_id}")

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

    if not args.checkpoint_path:
        print(f"Finding latest checkpoint in {src_model_path}")
        latest_checkpoint = find_latest_checkpoint(src_model_path)
        if not latest_checkpoint:
            raise ValueError(
                f"No checkpoints found in {src_model_path}. Please provide a valid Forgather model directory."
            )
    elif not os.path.exists(args.checkpoint_path):
        print(f"Checkpoint path {args.checkpoint_path} does not exist.")
        raise ValueError(f"Checkpoint path {args.checkpoint_path} does not exist.")
    else:
        latest_checkpoint = args.checkpoint_path
    print(f"Using checkpoint: {latest_checkpoint}")
    # Load the Forgather model configuration to get the original HF config
    src_model_config = AutoConfig.from_pretrained(
        src_model_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(src_model_path)

    print("Loading Forgather model...")
    # Create model with no_init_weights() to allow custom initialization (like RoPE)
    # while skipping standard parameter initialization
    with ExitStack() as exit_stack:
        if new_dtype:
            exit_stack.enter_context(default_dtype(new_dtype))
        exit_stack.enter_context(torch.device("cpu"))
        exit_stack.enter_context(no_init_weights())
        src_model = AutoModelForCausalLM.from_config(
            src_model_config, trust_remote_code=True
        )

    # Load the actual weights (RoPE buffers already initialized)
    load_checkpoint(latest_checkpoint, src_model, device="cpu", strict=True)

    print_debug_params(src_model, "Source Forgather model", args)

    print("Remapping model weight names to HuggingFace format")
    src_state_dict = src_model.state_dict()
    mapped_state_dict = remap_state_dict(src_state_dict, dllama_to_hflamma)

    max_model_length = src_model_config.max_sequence_length
    if args.max_length:
        max_model_length = args.max_length

    # Create appropriate HF config and model based on specified type
    hf_config, hf_model, model_type = create_hf_config_and_model(
        src_model_config, max_model_length, args.model_type, new_dtype
    )

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
        f"HuggingFace {model_type.capitalize()}",
        tolerance=1e-5,
    )

    test_generation_if_requested(hf_model, tokenizer, args)

    print(f"Saving HuggingFace {model_type.capitalize()} model...")
    hf_model.save_pretrained(dst_model_path) # Also saves config
    tokenizer.save_pretrained(dst_model_path)


def main():
    args = parse_args()

    # Validate arguments
    if not args.reverse and args.model_type != "llama":
        print(
            "Warning: --model-type is only used for reverse conversion (--reverse). Ignoring."
        )

    if args.reverse:
        print(
            f"Converting Forgather Dynamic Llama to HuggingFace {args.model_type.capitalize()}"
        )
        convert_forgather_to_hf(args)
    else:
        print("Converting HuggingFace Llama/Mistral to Forgather Dynamic Llama")
        convert_hf_to_forgather(args)


if __name__ == "__main__":
    main()
