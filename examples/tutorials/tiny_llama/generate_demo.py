#!/usr/bin/env python3
"""Standalone text-generation demo for the Tiny Llama tutorial.

Loads the trained model via Hugging Face's ``AutoModelForCausalLM`` and
generates continuations for a few prompts two different ways:

1. ``model.generate(...)`` -- the standard HF entry point, which handles
   sampling, stopping criteria, KV caching, etc. internally.
2. A hand-written forward+sample loop that pulls logits out of the model
   one token at a time and applies temperature / top-p sampling
   explicitly. Useful when you need behaviour ``generate`` doesn't
   expose -- custom stopping conditions, intermediate logit inspection,
   or teaching people how autoregressive sampling actually works.

Run after a Tiny Llama training run has produced at least one checkpoint
under ``output_models/<model_name>/checkpoints/``.

    cd examples/tutorials/tiny_llama
    python generate_demo.py                               # default: output_models/v2
    python generate_demo.py --model output_models/v2      # explicit
    python generate_demo.py --model output_models/v2 --mode manual

See the companion ``project_index.ipynb`` for a notebook version of the
same material.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from forgather.ml.sharded_checkpoint import create_pretrained_symlinks

PROMPTS = [
    "Once upon a time",
    "Alice was so tired when she got back home so she went",
    'Jack and Lily saw a rainbow after a rainy day. They were amazed by the colors. Jack said, "Look, Lily. A rainbow has',
    '"Can cows fly?" Alice asked her mother.',
]


def load_model(model_path: str, device: str, dtype: torch.dtype):
    """Symlink the latest checkpoint into ``model_path`` and load it.

    Forgather writes checkpoints to ``<model_path>/checkpoints/checkpoint-N/``
    and keeps the base model-code + config.json at ``<model_path>/`` itself.
    HF's ``from_pretrained`` only looks in the directory you give it, so we
    symlink the latest checkpoint's weight files up into the base
    directory before loading.  The CLI-equivalent is
    ``forgather -t v2.yaml checkpoint link``.
    """
    create_pretrained_symlinks(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype
    )
    model = model.to(device)  # type: ignore[arg-type]
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_with_hf(model, tokenizer, prompts, device, max_new_tokens=100):
    gen_config = GenerationConfig(
        pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        do_sample=True,
        top_k=20,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.15,
        max_new_tokens=max_new_tokens,
    )

    for prompt in prompts:
        enc = tokenizer([prompt], return_tensors="pt", return_attention_mask=True)
        out = model.generate(
            enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            generation_config=gen_config,
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(
            f"[prompt]\n{prompt}\n[continuation]\n{text[len(prompt):].lstrip()}\n{'-'*40}"
        )


@torch.inference_mode()
def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    prior_ids: torch.Tensor,
) -> int:
    """Return a single next-token id sampled from ``logits`` (shape: vocab).

    This is the piece ``generate`` hides.  Each step:

    1. Apply repetition penalty -- divide (or multiply, for negative
       logits) the logit of any token we've already emitted so the model
       is less likely to loop.
    2. Temperature -- divide logits by T.  T < 1 sharpens the
       distribution, T > 1 flattens it.  T -> 0 is greedy decoding.
    3. Top-p (nucleus) filter -- keep the smallest set of tokens whose
       cumulative probability is >= top_p; mask everything else to -inf.
    4. Softmax + multinomial -- draw one token from the remaining
       distribution.
    """
    # 1. Repetition penalty.
    if repetition_penalty != 1.0 and prior_ids.numel() > 0:
        logits = logits.clone()
        for tid in set(prior_ids.tolist()):
            logits[tid] /= (
                repetition_penalty if logits[tid] > 0 else 1.0 / repetition_penalty
            )

    # 2. Temperature.
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-6)

    # 3. Top-p nucleus.
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        # Keep the nucleus; mask the rest with -inf.
        mask = cum > top_p
        # Shift right so the first token above the threshold is kept.
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        # Unsort.
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)

    # 4. Softmax + multinomial.
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.inference_mode()
def generate_manual(model, tokenizer, prompts, device, max_new_tokens=100):
    """Sample autoregressively without calling ``model.generate``.

    We run ``model(input_ids)`` once per new token and feed the sampled
    id back in as the next input.  No KV cache here for simplicity; a
    production version would either (a) pass ``past_key_values`` through
    and append only the new token each step, or (b) use the model's
    built-in cache API.
    """
    eos = model.config.eos_token_id

    for prompt in prompts:
        input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(device)

        for _ in range(max_new_tokens):
            # return_dict=True ensures we get a ModelOutput with .logits;
            # some custom model classes default to returning a bare tensor.
            out = model(input_ids, return_dict=True)
            logits = out.logits[0, -1, :].float()  # last-position logits
            next_id = sample_token(
                logits,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                prior_ids=input_ids[0],
            )
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)], dim=1
            )
            if eos is not None and next_id == eos:
                break

        text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(
            f"[prompt]\n{prompt}\n[continuation]\n{text[len(prompt):].lstrip()}\n{'-'*40}"
        )


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument(
        "--model", default="output_models/v2", help="Path to Forgather model directory"
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument(
        "--mode",
        choices=["hf", "manual", "both"],
        default="both",
        help="hf = model.generate, manual = hand-rolled sampling, both = run each in turn",
    )
    args = ap.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model, tokenizer = load_model(args.model, args.device, dtype)

    if args.mode in ("hf", "both"):
        print("\n========== model.generate() ==========")
        generate_with_hf(model, tokenizer, PROMPTS, args.device, args.max_new_tokens)

    if args.mode in ("manual", "both"):
        print("\n========== manual autoregressive loop ==========")
        generate_manual(model, tokenizer, PROMPTS, args.device, args.max_new_tokens)


if __name__ == "__main__":
    main()
