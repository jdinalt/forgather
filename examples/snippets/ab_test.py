#!/usr/bin/env python3
"""Interactive blind A/B subjective comparison of two language models.

A companion to ``prompt_test.py``. Given two models and a YAML list of prompts,
it generates a continuation from each model for every prompt across ``--trials``
independent samples, then presents the pairs one at a time -- blind, in a
shuffled sequence, with the left/right order of each pair randomized -- and asks
you which continuation is better. It tabulates the votes and writes a JSON log
designed so multiple participants' runs can be pooled.

Example:

    python ab_test.py  path/to/model_a  path/to/model_b  ../../prompts/tiny_stories.yaml \\
        --trials 3 --seed 42 --no-kv-cache --participant alice

Reproducibility / pooling
-------------------------
``--seed`` drives *everything* (per-trial generation, the prompt-sequence
shuffle, and each pair's left/right order), so a run is fully reproducible. Two
participants who pass the same ``--seed`` and the same two models judge an
identical set of pairs (matched by ``comparison_id``), which is what lets their
JSON files be pooled. The per-pair side shown is recorded, so position bias can
be measured and corrected when aggregating.
"""

import argparse
import datetime
import json
import math
import os
import random
from argparse import RawTextHelpFormatter

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def load_model(path, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, dtype=dtype, device_map=device
    ).eval()
    return tokenizer, model


def generate(model, tokenizer, prompt, args, seed):
    """Generate a single continuation for ``prompt`` with a fixed seed."""
    torch.manual_seed(seed)
    generation_config = GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        # Some models (e.g. the singlehead ALiBi transformer) have no KV cache.
        use_cache=not args.no_kv_cache,
        return_dict_in_generate=True,
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.inference_mode():
        out = model.generate(
            input_ids, generation_config=generation_config, tokenizer=tokenizer
        )
    full = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
    return full[len(prompt) :].strip()


def two_sided_sign_test(wins_a, wins_b):
    """Exact two-sided binomial p-value for wins_a vs wins_b under p=0.5
    (ties excluded). Returns 1.0 for the degenerate empty case."""
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def ask(n, total, prompt, cont_left, cont_right):
    """Render one comparison and return the user's choice."""
    bar = "=" * 70
    print(f"\n{bar}\nComparison {n} / {total}\n{'-' * 70}")
    print(f"PROMPT:\n  {prompt}\n")
    print(f"[1] {cont_left}\n")
    print(f"[2] {cont_right}\n{'-' * 70}")
    while True:
        try:
            choice = input("Better? [1]/[2]/(t)ie/(s)kip/(q)uit: ").strip().lower()
        except EOFError:
            return "q"
        if choice in ("1", "2", "t", "s", "q"):
            return choice
        print("  Please enter 1, 2, t, s, or q.")


def main(args):
    with open(args.prompts_path) as fh:
        prompts = yaml.safe_load(fh)
    assert isinstance(prompts, list), "prompts file must be a YAML list"

    rng = random.Random(args.seed)
    name_a = os.path.basename(os.path.normpath(args.model_a))
    name_b = os.path.basename(os.path.normpath(args.model_b))

    # ---- Generate every (trial, prompt) continuation from both models up front,
    #      then free the GPU before the (slow, human-paced) judging loop. -------
    print(
        f"Generating {args.trials} x {len(prompts)} x 2 continuations "
        f"({name_a} vs {name_b})... this can take a minute."
    )
    comparisons = []
    gens = {"a": {}, "b": {}}
    for path, side in ((args.model_a, "a"), (args.model_b, "b")):
        tokenizer, model = load_model(path, args.device, args.dtype)
        for t in range(args.trials):
            for pi, prompt in enumerate(prompts):
                gseed = args.seed + t * 100003 + pi
                gens[side][(t, pi)] = generate(model, tokenizer, prompt, args, gseed)
        del model
        torch.cuda.empty_cache()
    for t in range(args.trials):
        for pi, prompt in enumerate(prompts):
            comparisons.append(
                {
                    "comparison_id": f"t{t}_p{pi}",
                    "trial": t,
                    "prompt_idx": pi,
                    "prompt": prompt,
                    "cont_a": gens["a"][(t, pi)],
                    "cont_b": gens["b"][(t, pi)],
                }
            )

    # ---- Shuffle the presentation sequence; randomize each pair's side. -------
    order = list(range(len(comparisons)))
    rng.shuffle(order)

    results = []
    tally = {"a": 0, "b": 0, "tie": 0, "skip": 0}
    print(
        f"\n{len(order)} blind comparisons. For each, pick the better continuation.\n"
        "(Neither model's identity is shown; press q to stop early and save.)"
    )
    for n, idx in enumerate(order, 1):
        c = comparisons[idx]
        swap = rng.random() < 0.5
        shown_left, shown_right = ("b", "a") if swap else ("a", "b")
        cont_left = c[f"cont_{shown_left}"]
        cont_right = c[f"cont_{shown_right}"]

        choice = ask(n, len(order), c["prompt"], cont_left, cont_right)
        if choice == "q":
            print("\nStopping early; saving what we have.")
            break
        if choice == "1":
            winner = shown_left
        elif choice == "2":
            winner = shown_right
        elif choice == "t":
            winner = "tie"
        else:  # skip
            winner = "skip"
        tally[winner] += 1
        results.append(
            {
                "comparison_id": c["comparison_id"],
                "trial": c["trial"],
                "prompt_idx": c["prompt_idx"],
                "prompt": c["prompt"],
                "cont_a": c["cont_a"],
                "cont_b": c["cont_b"],
                "shown_left": shown_left,
                "shown_right": shown_right,
                "choice": choice,
                "winner": winner,
            }
        )

    # ---- Write the JSON log. --------------------------------------------------
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.output or f"ab_test_{name_a}_vs_{name_b}_{stamp}.json"
    payload = {
        "meta": {
            "model_a": os.path.abspath(args.model_a),
            "model_b": os.path.abspath(args.model_b),
            "model_a_name": name_a,
            "model_b_name": name_b,
            "seed": args.seed,
            "trials": args.trials,
            "num_prompts": len(prompts),
            "prompts_path": os.path.abspath(args.prompts_path),
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "repetition_penalty": args.repetition_penalty,
                "no_kv_cache": args.no_kv_cache,
            },
            "participant": args.participant,
            "timestamp": stamp,
            "n_comparisons": len(order),
            "n_judged": len(results),
        },
        "results": results,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    # ---- Summary. -------------------------------------------------------------
    a, b, tie, skip = tally["a"], tally["b"], tally["tie"], tally["skip"]
    decided = a + b
    pval = two_sided_sign_test(a, b)
    print(f"\n{'=' * 70}\nResults  ({name_a}  vs  {name_b})\n{'-' * 70}")
    print(
        f"  judged:   {len(results)} of {len(order)}   (ties: {tie}, skipped: {skip})"
    )
    if decided:
        print(f"  {name_a:24s} {a:3d} wins  ({100 * a / decided:.0f}% of decided)")
        print(f"  {name_b:24s} {b:3d} wins  ({100 * b / decided:.0f}% of decided)")
        print(f"  two-sided sign test p = {pval:.4f}")
    print(f"\n  wrote {out_path}")
    print(
        "  Pool multiple participants by summing 'winner' across files that share\n"
        "  the same seed + models (match items by comparison_id)."
    )


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Blind A/B subjective comparison of two models over a prompt set.",
        epilog=(
            "Example:\n"
            "    %(prog)s ./model_a ./model_b ../../prompts/tiny_stories.yaml \\\n"
            "        --trials 3 --seed 42 --no-kv-cache --participant alice\n"
        ),
    )
    p.add_argument("model_a", type=os.path.expanduser, help="First model path/name")
    p.add_argument("model_b", type=os.path.expanduser, help="Second model path/name")
    p.add_argument(
        "prompts_path", type=os.path.expanduser, help="YAML file with a list of prompts"
    )
    p.add_argument("--seed", default=42, type=int, help="Seed (generation + shuffling)")
    p.add_argument(
        "--trials",
        default=1,
        type=int,
        help="Independent samples generated per prompt per model",
    )
    p.add_argument("--dtype", default=None, help="torch dtype (float32, bfloat16, ...)")
    p.add_argument("--device", default="cuda:0", help="Device (cuda, cpu, auto)")
    p.add_argument("--max-new-tokens", default=80, type=int, help="Max new tokens")
    p.add_argument(
        "--temperature", default=0.7, type=float, help="Sampling temperature"
    )
    p.add_argument(
        "--repetition-penalty", default=1.15, type=float, help="Repetition penalty"
    )
    p.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable KV-cache decoding (required by models that lack it, "
        "e.g. the singlehead ALiBi transformer)",
    )
    p.add_argument(
        "--participant",
        default=os.environ.get("USER", "anonymous"),
        help="Participant id",
    )
    p.add_argument("--output", default=None, help="Output JSON path")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
