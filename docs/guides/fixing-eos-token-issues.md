# Fixing End-of-Sequence (EOS) Token Issues

This guide covers a common class of bug that shows up after a model
has had new special tokens added -- most often a ChatML pair
(`<|im_start|>` / `<|im_end|>`) grafted onto a base model via
`forgather convert --add-tokens` or `forgather update-vocab`. The
symptom is simple: **the model appears to generate forever**, running
right through what should be the end of an assistant turn and only
stopping when the request's `max_tokens` budget runs out.

The root cause is almost always the same: the model's new EOS token
is configured in `tokenizer_config.json` but *not* in
`generation_config.json`. HuggingFace's `model.generate()` reads its
stopping criterion from `generation_config.json`, so if the two files
disagree, the tokenizer's EOS setting is ignored and generation never
halts.

## Recognising the symptom

Things that look like an EOS issue:

- A chat client (`forgather inf client --message ...`) emits a
  reasonable-looking response, followed by a long tail of repeated
  stop-like tokens or apparently unrelated text, until `max_tokens`.
- Completion requests return `finish_reason: "length"` where you
  expected `"stop"`.
- The inference server logs `Stop token IDs: [128256]` (or similar)
  at startup, but generation still doesn't stop at that token.
- The raw decoded model output contains the expected turn-boundary
  token (e.g. `<|im_end|>`) somewhere in the middle, not at the end.

If you see any of these, check the three files below before assuming
the model is untrained, mis-trained, or using the wrong chat
template.

## The three files that matter

After a token-addition pass, there are three places a model carries
EOS information:

| File | Field | Who reads it |
|---|---|---|
| `tokenizer_config.json` | `eos_token` (a string) | Tokenizer encode/decode, `tokenizer.eos_token_id` |
| `config.json` | `eos_token_id` (int or list) | The model's own forward pass (mostly informational) |
| `generation_config.json` | `eos_token_id` (int or list) | **`model.generate()` stopping criterion** |

HuggingFace's generation loop only looks at
`generation_config.json`. If that file still carries the base
model's original scalar EOS after you've added ChatML tokens, the
added tokens will not terminate generation -- regardless of what
`tokenizer_config.json` says.

## Diagnosing

Open the three files and compare.

```bash
cd /path/to/your/model

# What does the tokenizer think is EOS?
python -c "
import json
tc = json.load(open('tokenizer_config.json'))
print('tokenizer_config.json eos_token:', tc.get('eos_token'))
"

# What does config.json say?
python -c "
import json
c = json.load(open('config.json'))
print('config.json eos_token_id:', c.get('eos_token_id'))
"

# What does generation_config.json say? THIS is what model.generate()
# actually uses at inference time.
python -c "
import json
g = json.load(open('generation_config.json'))
print('generation_config.json eos_token_id:', g.get('eos_token_id'))
"
```

**The bug fingerprint** is: the tokenizer reports the added token
(e.g. `<|im_end|>`) as `eos_token`, `config.json` reports a list
like `[<added_id>, <orig_eos_id>]`, but `generation_config.json`
still holds the scalar `<orig_eos_id>` from the source model.

## Fixing an existing model by hand

`generation_config.json` accepts a list of token IDs. HuggingFace's
`EosTokenCriteria` stops on *any* listed token (via `torch.isin`),
so the fix is to widen the scalar into a list that includes both
the original EOS and any added stop tokens.

```python
import json

GEN_CONFIG = "/path/to/your/model/generation_config.json"
NEW_EOS = [128256, 128001]   # <|im_end|>, <|end_of_text|> for Llama3

with open(GEN_CONFIG) as f:
    cfg = json.load(f)
print("old:", cfg.get("eos_token_id"))

cfg["eos_token_id"] = NEW_EOS

with open(GEN_CONFIG, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
print("new:", cfg["eos_token_id"])
```

The token IDs are specific to the tokenizer -- look them up:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/path/to/your/model", trust_remote_code=True)
print("im_end  ->", tok.convert_tokens_to_ids("<|im_end|>"))
print("eos_token ->", tok.eos_token_id)
```

A useful shape to match: the order in `generation_config.json`
should match whatever `config.json` has. That's purely cosmetic --
order doesn't affect the stopping criterion -- but keeps a casual
diff between the two files easy to read.

**No retraining is required.** `eos_token_id` only affects
`generate()`'s stopping criterion. It is never read during the
training forward pass or the loss computation, so fixing the file
after training is safe and complete.

## Verifying the fix

Start the inference server and send a plain chat request with **no
explicit `--stop`**. If the fix is in place, the client will return
a short, self-terminating response instead of running to
`--max-tokens`.

```bash
# In one terminal:
forgather inf server -c -m /path/to/your/model

# In another:
forgather inf client --message "What is the capital of Japan? Answer in one word."
```

You should see something like `The capital of Japan is Tokyo.` and
the client should exit immediately, not stall until `max_tokens`.

Also check the server startup log. For a correctly-configured
ChatML model, `Final default generation config:` should show a list
for `eos_token_id`, e.g.:

```
eos_token_id: [
    128256,
    128001
],
```

A scalar value there is the bug fingerprint.

## Preventing future recurrences

`forgather convert --add-tokens` and `forgather update-vocab` both
write `generation_config.json` with the merged EOS set automatically
-- they load the destination file, update `eos_token_id` to match
`config.json`'s merged list, and save it back. If you're converting
a model using an up-to-date Forgather checkout, you should not see
this problem on new conversions.

The bug shows up most often in older Forgather-converted models
that were built before the converter started syncing
`generation_config.json`, or in models converted with other tools
that don't touch generation config at all. For older Forgather
conversions the simplest fix is to re-run the conversion in place:

```bash
forgather convert --add-tokens tokens.yaml /path/to/model /path/to/output
```

The re-run will produce the correct `generation_config.json`. You
can also apply the in-place hand-edit from the previous section --
both are valid, and the one-off JSON edit is faster when you only
want to patch the final artefact.

## Reference: models that do it right

For cross-checking, several HuggingFace and Forgather models
already ship with list-valued `eos_token_id` in
`generation_config.json`. Instruction-tuned Llama 3 variants use
`[128001, 128008, 128009]`; Qwen3 chat models use
`[151645, 151643]`. If you're converting from one of these and the
list form doesn't survive round-trip, the conversion tool is the
bug, not the source model.

## Related docs

- [Model Conversion](model-conversion.md) -- `forgather convert`
  reference
- [Vocabulary and Chat Template Update](update-vocab.md) --
  `forgather update-vocab` reference, the other tool most likely to
  leave a model in a state that triggers this bug
