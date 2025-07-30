#!/usr/bin/env python3
import os
import argparse
from argparse import RawTextHelpFormatter
import logging

from transformers import AutoModelForCausalLM, GenerationConfig, AutoConfig, AutoTokenizer
from forgather.ml.construct import copy_package_files, torch_dtype
from forgather.ml.sharded_checkpoint import load_checkpoint, create_sharing_metadata, retie_parameters
import torch

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Load a model and test text generation",
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Destination model test device",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Torch output dtype",
    )
    parser.add_argument(
        "-c",
        "--as-sharded-checkpoint",
        action='store_true',
        help="Load using sharded_checkpoint.py",
    )
    parser.add_argument(
        "-g",
        "--generation-test",
        action='store_true',
        help="Test model generation with prompt",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The old bookstore at the corner of Elm Street had always held secrets, but today, something unusual caught Emma's eye.",
        help="Destination model test prompt",
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

def main():
    args = parse_args()
    model_path = os.path.abspath(args.model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model...")
    if args.as_sharded_checkpoint:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        sharing_metadata = create_sharing_metadata(model)
        
        model.to(dtype=torch_dtype(args.dtype))
        model.to_empty(device=args.device)
        retie_parameters(model, sharing_metadata)
        load_checkpoint(model_path, model, device=args.device, strict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        from forgather.ml.construct import torch_dtype
        if args.dtype:
            model.to(dtype=torch_dtype(args.dtype))
        model.to(device=args.device)

    if args.generation_test:
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

if __name__ == "__main__":
    main()