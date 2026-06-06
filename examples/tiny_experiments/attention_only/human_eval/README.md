# Human A/B evaluation

Blind human preference data for the [attention_only](../) experiment, gathered
with [`ab_test.py`](../../../snippets/ab_test.py) and pooled with
[`ab_aggregate.py`](../../../snippets/ab_aggregate.py). The motivating question:
does the eval-loss ranking (and the LLM-judge panel that agrees with it) match
what a *person* actually prefers? (Short answer so far: no — see the experiment
[README](../README.md#blind-ab-evaluation).)

## Contribute a run

Each model must be loadable (co-locate the final checkpoint weights, as in the
experiment README's "Generation" section). Then, from the experiment directory:

```bash
cd examples/tiny_experiments/attention_only

# mlp_small vs the wide attention-only model
python ../../snippets/ab_test.py \
    output_models/mlp_small output_models/attn_only_matched_small \
    ../../../prompts/tiny_stories.yaml \
    --trials 3 --seed 42 --no-kv-cache \
    --participant YOUR_NAME --output human_eval/mlp_vs_wide_YOUR_NAME.json

# mlp_small vs the deep attention-only model
python ../../snippets/ab_test.py \
    output_models/mlp_small output_models/attn_only_deep_small \
    ../../../prompts/tiny_stories.yaml \
    --trials 3 --seed 42 --no-kv-cache \
    --participant YOUR_NAME --output human_eval/mlp_vs_deep_YOUR_NAME.json
```

Keep `--seed 42` (and the same model pair) so everyone judges the **same**
generated continuations — that is what makes the runs poolable. Drop the JSON in
this directory.

## Aggregate

```bash
python ../../snippets/ab_aggregate.py human_eval/
```

It groups by `(model_a, model_b, seed)`, prints each participant's split, the
pooled win counts, a two-sided sign test, and a position-bias check.

## A note on judging

Pairs are often *both* flawed (these are 2–24M-parameter models). The choice is
binary on purpose; when neither is clearly better, a quick gut-call / coin-flip
is the intended behavior — over many comparisons those wash out, and a real
preference (if there is one) still shows through. Don't over-analyze.
