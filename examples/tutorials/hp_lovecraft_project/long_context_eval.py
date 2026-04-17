"""Long-context generation and perplexity eval for the Lovecraft tutorial variants.

For each model directory passed on the command line, this script:
  1. Loads the latest checkpoint and the model's config.json (so any custom
     rope_parameters you wrote into config.json take effect at load time).
  2. Computes the windowed next-token cross-entropy on the test story at a
     range of context lengths.  This quantifies how the model's perplexity
     behaves as the context grows past the training window.
  3. Generates N continuation tokens from a short seed prompt at a fixed
     temperature / top_p and writes the output to a markdown file.

Usage:
    python long_context_eval.py \
        --model /path/to/fg_mistral_7b_lovecraft_16k_baseline \
        --model /path/to/fg_mistral_7b_lovecraft_16k_fp32rope \
        --model /path/to/fg_mistral_7b_lovecraft_16k_yarn \
        --test-file examples/tutorials/hp_lovecraft_project/hp_lovecraft/at_the_mountains_of_madness.txt \
        --ppl-windows 2048,4096,8192,12288,16384 \
        --gen-tokens 24576 \
        --output-md /tmp/long_context_eval.md

Typically you run this once per variant on a spare GPU while other training
is still in flight.  Pass --device cuda:4 / --device cuda:5 to pin it.
"""

import argparse
import math
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from forgather.ml.sharded_checkpoint import find_latest_checkpoint, load_checkpoint


def load_model(code_dir: str, checkpoint_root: str, device: str, dtype: torch.dtype):
    """Load a Forgather-format model.

    ``code_dir`` is a directory containing the generated Python code and
    ``config.json`` (e.g. ``fg_mistral_7b_v_yarn``).  ``checkpoint_root``
    is the training output directory whose ``checkpoints/`` subdir contains
    the trained weights; the latest checkpoint under it is loaded via
    Forgather's sharded-checkpoint loader.  If ``checkpoint_root == code_dir``
    the original base-model weights in ``code_dir`` are used (useful for
    evaluating the un-fine-tuned model).
    """
    cfg = AutoConfig.from_pretrained(code_dir, trust_remote_code=True)
    print(f"[{code_dir}] rope_parameters: {getattr(cfg, 'rope_parameters', None)}")

    cp_path = find_latest_checkpoint(checkpoint_root)
    if cp_path:
        print(f"  checkpoint: {cp_path}")
        model = AutoModelForCausalLM.from_config(
            cfg, trust_remote_code=True, torch_dtype=dtype
        )
        load_checkpoint(cp_path, model, device=device, strict=True)
        model = model.to(device).eval()
    else:
        print(f"  no checkpoint found under {checkpoint_root}; loading base weights")
        model = (
            AutoModelForCausalLM.from_pretrained(
                code_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            .to(device)
            .eval()
        )

    tok = AutoTokenizer.from_pretrained(code_dir, trust_remote_code=True)
    return model, tok, cfg


@torch.no_grad()
def windowed_perplexity(
    model, tok, text: str, windows: list[int], device: str
) -> dict[int, float]:
    """Compute average next-token cross-entropy on a single long text at
    several truncation lengths.  Returns {ctx_len: avg_loss}."""
    ids = tok(text, return_tensors="pt").input_ids[0].to(device)
    results = {}
    for w in windows:
        if ids.numel() < w + 1:
            results[w] = float("nan")
            continue
        chunk = ids[:w].unsqueeze(0)
        # Compute loss ourselves in fp32 to avoid bf16 quantization masking
        # small differences between variants.
        logits = model(chunk, return_dict=True).logits
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = chunk[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        results[w] = float(loss.item())
        del logits, shift_logits, shift_labels, loss, chunk
        torch.cuda.empty_cache()
    return results


@torch.no_grad()
def generate_long(
    model,
    tok,
    prompt: str,
    n_tokens: int,
    device: str,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> str:
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(
        input_ids,
        max_new_tokens=n_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=(
            tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ),
        use_cache=True,
    )
    return tok.decode(out[0], skip_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="NAME:CODE_DIR:CHECKPOINT_ROOT",
        help=(
            "Evaluate a variant.  NAME is a short tag; CODE_DIR holds the "
            "generated Python + config.json (e.g. fg_mistral_7b_v_yarn); "
            "CHECKPOINT_ROOT is the training output dir whose latest "
            "checkpoint is loaded.  Pass CHECKPOINT_ROOT == CODE_DIR to "
            "evaluate the un-fine-tuned base model.  Repeat for each variant."
        ),
    )
    ap.add_argument("--test-file", required=True, help="Text file for perplexity eval")
    ap.add_argument(
        "--ppl-windows",
        default="2048,4096,8192,12288,16384",
        help="Comma-separated context lengths for perplexity",
    )
    ap.add_argument("--gen-tokens", type=int, default=24576)
    ap.add_argument("--gen-prompt", default="The Stranger (1923)\n\n")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    ap.add_argument("--output-md", default="/tmp/long_context_eval.md")
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()

    windows = [int(w) for w in args.ppl_windows.split(",") if w]
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    with open(args.test_file, "r") as f:
        test_text = f.read()

    report_lines = [
        f"# Long-context eval\n",
        f"- Test file: `{args.test_file}` ({len(test_text)} chars)",
        f"- Perplexity windows: {windows}",
        f"- Generation: prompt={args.gen_prompt!r}, tokens={args.gen_tokens}, "
        f"temperature={args.temperature}, top_p={args.top_p}\n",
    ]

    variants = []
    for spec in args.variant:
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit(
                f"Bad --variant spec {spec!r}; want NAME:CODE_DIR:CHECKPOINT_ROOT"
            )
        variants.append(tuple(parts))

    for tag, code_dir, checkpoint_root in variants:
        print(f"\n=== {tag} ===", flush=True)
        model, tok, cfg = load_model(code_dir, checkpoint_root, args.device, dtype)
        report_lines.append(f"\n## {tag}")
        report_lines.append(
            f"\nCode dir: `{code_dir}`   Checkpoint: `{checkpoint_root}`"
        )
        report_lines.append(
            f"\nrope_parameters: `{getattr(cfg, 'rope_parameters', {})}`\n"
        )

        t0 = time.time()
        ppl = windowed_perplexity(model, tok, test_text, windows, args.device)
        dt = time.time() - t0
        print(f"ppl done in {dt:.1f}s", flush=True)
        report_lines.append("\n| ctx_len | nll | ppl |\n|---------|-----|-----|")
        for w in windows:
            nll = ppl[w]
            p = math.exp(nll) if not math.isnan(nll) else float("nan")
            report_lines.append(f"| {w} | {nll:.4f} | {p:.2f} |")

        if not args.skip_generation:
            t0 = time.time()
            gen = generate_long(
                model,
                tok,
                args.gen_prompt,
                args.gen_tokens,
                args.device,
                args.temperature,
                args.top_p,
            )
            dt = time.time() - t0
            n_out = len(tok(gen).input_ids)
            print(f"generation done in {dt:.1f}s, {n_out} total tokens", flush=True)
            report_lines.append(f"\n### Generation ({n_out} tokens in {dt:.1f}s)\n")
            report_lines.append("```")
            report_lines.append(gen)
            report_lines.append("```\n")

        del model
        torch.cuda.empty_cache()

    with open(args.output_md, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nWrote {args.output_md}")


if __name__ == "__main__":
    main()
