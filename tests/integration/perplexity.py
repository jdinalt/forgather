"""GPT-2 based perplexity scorer for evaluating generated text quality."""

from __future__ import annotations

import math
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_gpt2():
    """Load GPT-2 model and tokenizer once per process."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return model, tokenizer


def compute_perplexity(text: str) -> float:
    """Compute GPT-2 perplexity of a text string.

    Lower perplexity indicates more coherent/natural text.
    Typical ranges:
        - Well-written English prose: 20-60
        - Acceptable generated text: 50-200
        - Poor/incoherent text: 200-1000
        - Random tokens: 1000+

    Args:
        text: The text to evaluate.

    Returns:
        Perplexity score (exp of cross-entropy loss).
    """
    import torch

    model, tokenizer = _load_gpt2()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)

    return math.exp(outputs.loss.item())
