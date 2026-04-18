# H.P. Lovecraft fine-tunes: investigation of the 4K-periodic NLL spike

This document is the investigation log for the 4096-token-periodic per-token
NLL spikes observed in all Lovecraft long-context fine-tunes.  See
`long_context_experiments.md` for the parent experiment.

**TL;DR.**  The spikes were a configuration artefact, not a model phenomenon.
The tutorial's old `base_finetune_proj.yaml` template declared `--window-size`
as a CLI flag but its `dataset_pp_kwargs` block only forwarded `chat_template`
to the dataset project.  The block tokenizer therefore received the default
`max_length=4096` regardless of what was passed on the command line.  Every
"16K training" was 4K-tokenized data padded to 16K by the collator; the model
learned a 4K block structure and reproduced it as periodic NLL spikes on
continuous-text evaluation.  The fix was to migrate to `projects/finetune_v2.yaml`,
which correctly threads `ns.seq_len` through `datasets_preprocessor_args.max_length`.

The surviving reasoning trail below is kept because two earlier root-cause
theories (packed-data document boundaries, RoPE resonance) looked plausible
enough to be written up before the plumbing gap was found.  They are wrong;
what ruled them out and what replaced them is described at the bottom.

## Observation

All five trained Lovecraft variants (Mistral / Llama × default / YaRN ×
sliding / noslide) show **exactly-4096-token-periodic** NLL spikes when
evaluated on a continuous long-context held-out story.  The spike phase
varies by story (4352 in *At the Mountains of Madness*, 2304 in *The Case of
Charles Dexter Ward*), but within any given story the spacing is always
exactly 4096.  Peak values are 2-3 NLL higher than the adjacent low, and
NLL decays smoothly over ~4K tokens back to the low before the next spike.

The untrained base model does *not* show this pattern -- it has a smooth NLL
with gradual degradation past the pretrain context.  So the spike is
introduced by fine-tuning.

## Refuted hypotheses

### Story-content explanation (refuted)

Three spikes at exact 4K multiples can't plausibly be story content
coincidence.  And the phase-vs-content dependence argues against pure
position-locking.  *Partial* evidence: the story text at spike positions is
ordinary prose, not structural markers.  Confirmed refuted by subsequent
experiments.

### Mistral sliding-window hypothesis (refuted)

Initially suspected that Mistral's sliding_window=4096 caused the first spike
via the BOS-drops-out-of-window mechanism (Xiao et al. 2023, *Efficient
Streaming Language Models with Attention Sinks*,
[arxiv:2309.17453](https://arxiv.org/abs/2309.17453)).  Refuted by two
observations:

1. Sliding window is continuous past the first transition; it doesn't predict
   additional spikes at 8K and 12K.
2. Investigation of the Forgather code found that the SDPA path was *ignoring*
   the sliding_window setting entirely -- `causal_mask.py` read
   `config.window_size` but the field is named `sliding_window`, so the
   create_sliding_window_causal_mask branch was never taken.  Both "with
   sliding" and "without sliding" variants were architecturally identical.
   (Separately fixed.)

### RoPE frequency resonance (refuted)

Observed that the geometric distribution of RoPE inv-freqs with
`rope_theta=10000, d_head=128` places many dimensions (roughly dims 28-45) at
periods that are integer-fraction harmonics of 4096.  Hypothesised that
collective phase alignment at 4K multiples might be driving the spikes.

**Refuted by direct experiment.**  Trained four Mistral variants with
rope_theta ∈ {5000, 20000, 40000, 500000} (spanning a 100× range) and one
Llama-2-7B with `rope_type: "llama3"`.  If the hypothesis were correct, spike
spacing should scale with `rope_theta^0.703`:

| Variant | rope_theta | Predicted dim-45 period | **Observed spikes** | Observed spacing |
|---------|-----------:|------------------------:|---------------------|-----------------:|
| theta=5000 | 5000 | 2506 | 4224, 8320, 12672 | 4096 |
| theta=10000 (baseline) | 10000 | 4080 | 4352, 8448, 12544 | 4096 |
| theta=20000 | 20000 | 6643 | 4224, 8320, 12672 | 4096 |
| theta=40000 | 40000 | 10814 | 4224, 8320, 12672 | 4096 |
| theta=500000 | 500000 | 63866 (past eval window) | 4224, 8320, 12672 | 4096 |
| llama3_rope | — | — | 4224, 8320, 12416 | 4096 |

Spike spacing is *independent* of rope_theta across a 100× range.  Even at
theta=500000, where the fundamental RoPE period (63866) is almost 4× the
entire eval window, spikes still land at exactly 4K intervals.  RoPE is not
the cause.

### Packed-data document-boundary distribution (refuted)

A working hypothesis at one point was that the spikes came from the
document-boundary statistics of the packed training data -- the Lovecraft
corpus has a median story length near 4K tokens, so greedy packing into long
blocks would concentrate `document_starts` near 4K multiples.  The model was
assumed to be learning "attention resets every ~4K tokens" from these
boundaries.  Supporting evidence at the time: non-packed training (single
document per block, padded) showed a flat NLL profile with no 4K spikes.

This theory is also wrong -- or rather, it was a correct description of the
*symptom* but not the mechanism.  The real mechanism is below; the
document-boundary statistics of the *4K-tokenized* data happened to align
with the 4K spike pattern because the blocks were literally 4K tokens long.

## Actual cause: the `window_size` plumbing gap

At this point the eval-side hypotheses had been exhausted; further progress
required instrumenting the training-side to confirm what the model was
*actually* being trained on.

A single print in `block_tokenize_fn` settled it:

```
block_tokenize_fn called: max_length=4096 packed=True ...
```

...despite every training command passing `--window-size 16384`.  The
training command line was accepting the flag without error, and the
preprocessed config was showing `window_size: 16384`, but the value was never
reaching the tokenizer.

Tracing back through the templates, the problem was in the old
`templatelib/finetune/projects/base_finetune_proj.yaml`.  That template
declared `window_size` as a dynamic CLI arg, but its `[dataset_pp_kwargs]`
block only forwarded `chat_template` to the dataset project.  Nothing in the
chain fed `ns.window_size` into the dataset's `block_tokenize_fn` call --
so the tokenizer kept its built-in default of `max_length=4096`.

At 16K sequence length, the data collator then padded each real 4K block out
to 16K tokens.  Pre-packed data (most of this tutorial's experiments) was
slightly different: 4K-tokenized blocks got concatenated into 16K blocks at
collator time, but each 4K region came from a single `block_tokenize_fn`
call with `max_length=4096`, which is exactly what creates the 4K structural
cadence the model eventually learned.

The newer `projects/finetune_v2.yaml` + newer dataset API does this
correctly: the training project sets `ns.seq_len`, which feeds into
`[datasets_preprocessor_args]` under the `max_length` key, which the dataset
template unpacks into `block_tokenize_fn`'s `max_length` kwarg.  After
migrating the reference project, instrumenting the tokenizer again shows:

```
block_tokenize_fn called: max_length=16384 packed=True ...
```

which is what it should have been saying all along.

### Why this produces exactly-4K-periodic NLL spikes

With `max_length=4096` at tokenization time, every training block is a
4K-token packed bundle of complete-or-truncated documents.  Whatever
statistical structure the corpus has -- story boundaries, `<BOS>` tokens,
long-range coreference decays -- it *repeats* every 4K tokens across
training.  The model learns the joint of "token at position p within a 4K
bundle" rather than the joint over a true 16K context.

At eval on a continuous 16K story, absolute positions 0..4095 look like
familiar within-bundle positions and NLL is low.  Positions 4096..8191
still look like within-bundle positions to the model's learned weights, but
the *content* continues from the previous 4K smoothly -- which conflicts
with the learned expectation that attention structure resets at each 4K
boundary.  Peak NLL lands a few hundred tokens past each 4K boundary (4352,
8448, 12544) because it takes a little content to wedge against the
model's expectation; NLL then decays over the following 4K.

### Why the story phase varies

The model's learned expectation is "after N tokens of familiar within-bundle
context, something structural happens."  The *exact* NLL peak inside each
4K window is where story content first surprises the learned expectation --
a content-dependent phase, which is why ATMoM peaks at 4352 and Charles
Dexter Ward peaks at 2304.

### Why non-packed training had no spikes

Non-packed training produces one document per 4K block (padded), so the
learned joint is effectively "token position within a single document,
documents up to 4K long."  There is no periodic reset of content structure
at 4K -- the content doesn't continue past the document end, it hits
padding.  Evaluating on a continuous story doesn't trigger the same
expectation-collision.

This is consistent with the earlier observation that non-packed training
had a flat NLL profile.  That observation was correct; the *interpretation*
(packed document boundaries are the cause) was wrong.  The actual cause is
the 4K tokenization window, which non-packed training exposes to the model
as document-length-truncation rather than as a periodic boundary.

### What the parent doc's experiments actually measured

Every result in `long_context_experiments.md` was produced with
`max_length=4096` at tokenization time, not the intended 16K.  What was
being compared across variants (YaRN on/off, sliding-window on/off,
rope_theta sweep, context-length sweep) is still a valid comparison of
*fine-tunes on 4K-tokenized data evaluated at 16K*, because the plumbing
gap affected every variant identically.  But the headline numbers should
not be read as "what long-context fine-tuning buys you"; they are "what a
4K fine-tune buys you when evaluated at 16K, with/without each
intervention."

See the parent doc for updated caveats.

## Lessons

- **Instrument the leaf, not the config.**  The preprocessed config
  (`forgather pp`) looked right -- `window_size: 16384` was everywhere --
  which made the plumbing gap invisible from the template side.  A
  single print in `block_tokenize_fn` closed the investigation that
  multiple months of hypothesis-testing couldn't.
- **Template inheritance hides declarations.**  The old template accepted
  `--window-size` without forwarding it, which is worse than rejecting
  it: the user gets no error, and the config that was "obviously right"
  silently runs on default data.  `finetune_v2.yaml` narrows the
  declaration surface and explicitly passes `ns.seq_len` as
  `max_length` in `datasets_preprocessor_args`, which is harder to
  forget.
- **Believe the unusual but consistent behaviour.**  Three spikes at
  exactly 4096-token spacing, independent of rope_theta, context length,
  model family, and variant, is not a subtle architectural resonance --
  it is 4K shouting its existence at the evaluator.  The early "4K is
  weirdly special" framing in the parent doc should have been read as a
  hint that 4K was literally the block size, not an emergent property.

## Infrastructure (archived)

- `/tmp/run_theta_sweep_eval.sh` -- parallel eval across 5 GPUs
- `/tmp/analyze_spikes.py` -- extract spike positions from eval reports
- `/tmp/rope_phase_analysis.py`, `/tmp/rope_similarity_to_origin.py` -- RoPE
  phase analysis that partially invalidated the resonance hypothesis before
  the training sweep sealed it
