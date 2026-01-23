#!/usr/bin/env python3

import argparse
import os
from argparse import RawTextHelpFormatter

import torch
import yaml
from transformers import AutoModel, AutoTokenizer, GenerationConfig


def main(args):
    with open(args.prompts_path, "r") as file:
        prompts = yaml.safe_load(file)
        assert isinstance(prompts, list)

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        device_map=device,
    )

    generation_config = GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )

    tokenizer_outputs = tokenizer(
        prompts,
        truncation=True,
        padding=True,
        return_tensors="pt",
        padding_side="left",
    )

    input_ids = tokenizer_outputs["input_ids"].to(device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            tokenizer=tokenizer,
        )

    output_text = tokenizer.batch_decode(
        outputs.sequences,
        skip_special_tokens=True,
    )

    for prompt, y in zip(prompts, output_text):
        print("-" * 40)
        s = prompt + " [START] " + y[len(prompt) + 1 :]
        print(s)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Test model generation with a list of prompts",
        epilog=(
            "Examples:\n"
            "\n"
            "    %(prog)s ./path/to/model/ ../../prompts/short_stories.yaml\n"
        ),
    )

    parser.add_argument(
        "model", type=os.path.expanduser, help="HuggingFace model path or name"
    )
    parser.add_argument(
        "prompts_path",
        type=os.path.expanduser,
        help="Path to YAML file, with list of prompts.",
    )
    parser.add_argument(
        "--dtype",
        help="Model data type (float32, float16, bfloat16, ...",
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Device to use (cuda, cpu, auto)"
    )
    parser.add_argument(
        "--max-new-tokens", default=512, type=int, help="Maximum new tokens"
    )
    parser.add_argument(
        "--temperature", default=0.7, type=float, help="Generation temperature"
    )
    parser.add_argument(
        "--repetition-penalty", default=1.15, type=float, help="Repetition penalty"
    )
    parser.add_argument("--seed", default=42, type=int, help="Generation seed")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
