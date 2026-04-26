# EOS Tokens and `generate()` Stopping Criteria

How HuggingFace's `model.generate()` decides when a sequence is finished,
and how the multiple files that carry "end of sequence" information
relate to each other. This isn't pulled together cleanly anywhere in the
HF docs, but the resolution is mechanical and worth understanding once.

For tool-driven provisioning of correct EOS configuration, see
[Finalize Model](finalize-model.md) and [Model Conversion](model-conversion.md).

## How `generate()` decides to stop

After every sampled token, `generate()` runs a `StoppingCriteriaList`. The
list is assembled by `GenerationMixin._get_stopping_criteria` based on the
effective `GenerationConfig` for that call, plus any user-supplied
criteria. As of Transformers 4.x:

| Criterion | Active when | What it checks |
|---|---|---|
| `MaxLengthCriteria` | `generation_config.max_length` is set (almost always) | `input_ids.shape[1] >= max_length` |
| `MaxTimeCriteria` | `generation_config.max_time` is set | wall-clock |
| `StopStringCriteria` | `stop_strings` set on the gen config or kwarg (requires `tokenizer=`) | substring match in decoded output so far |
| `EosTokenCriteria` | `generation_config.eos_token_id` is not None | the criterion this guide is about |
| `ConfidenceCriteria` | speculative-decoding assistant runs only | assistant-confidence threshold |
| User-supplied | passed via `stopping_criteria=` | anything you implement |

Each criterion returns a per-sequence boolean tensor; a sequence is
finished as soon as **any** criterion returns `True` for it. In batched
generation, individual sequences finish independently and stay frozen
once finished.

## What `EosTokenCriteria` actually does

The implementation is short:

```python
class EosTokenCriteria(StoppingCriteria):
    def __init__(self, eos_token_id):
        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id)
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores, **kwargs):
        self.eos_token_id = self.eos_token_id.to(input_ids.device)
        return torch.isin(input_ids[:, -1], self.eos_token_id)
```

Two things to take from this:

1. **It's a per-sequence equality test against the most recently sampled
   token.** No fuzzy matching, no sub-token matching, no substring
   matching. The sampled ID either is in the list or it is not.
2. **A list of IDs is treated as a set.** `torch.isin` returns true for
   any matching member. Multi-EOS models (Llama 3 instruct's
   `[128001, 128008, 128009]`) work because all three IDs are in the
   tensor and a hit on any of them ends the sequence.

`EosTokenCriteria` is appended to the list iff
`generation_config._eos_token_tensor is not None` -- which is set by the
GenerationConfig from `eos_token_id` at construction time. If that field
is `None`, no EOS-based stopping happens at all and the sequence runs to
`max_length`.

## The four EOS-bearing fields

After conversion or finalize, a model directory typically carries EOS
information in three on-disk files. A fourth source -- per-call
overrides -- only exists at `generate()` time.

| Source | Type | Loaded into | Read by `generate()`? |
|---|---|---|---|
| `tokenizer_config.json` `eos_token` | string | `tokenizer.eos_token` (and via vocab lookup, `tokenizer.eos_token_id`) | **No.** Used by tokenizer encode/decode, chat-template rendering, and special-token handling. |
| `config.json` `eos_token_id` | int or list | `model.config.eos_token_id` | **Indirectly.** Only when no `generation_config.json` exists, in which case it seeds the GenerationConfig built from the model config. |
| `generation_config.json` `eos_token_id` | int or list | `model.generation_config.eos_token_id` | **Yes.** This is the source of truth at inference time. |
| `generate(eos_token_id=...)` kwarg | int or list | overrides for one call | **Yes.** Highest precedence; replaces the field on the effective GenerationConfig for this call. |

### Resolution at `generate()` time

```
effective_generation_config = (
    user_kwargs (eos_token_id=..., generation_config=...)
    overlaid on
    model.generation_config
)

stopping_criteria includes EosTokenCriteria(
    effective_generation_config.eos_token_id
)
```

`tokenizer.eos_token_id` is **never read directly** by `generate()`. The
tokenizer's view of EOS only matters for things like
`tokenizer.apply_chat_template(...)` (which may emit the EOS string at
turn boundaries) and for downstream code that decides what string to
append manually.

### Why three on-disk files exist

Historical layering. `tokenizer_config.json` and `config.json` predate
the `GenerationConfig` abstraction, which was introduced in
Transformers 4.26 to separate inference-time generation parameters from
model architecture. Before that split, `generate()` read its EOS from
`model.config.eos_token_id`. The older fields live on for backward
compat; the new field is now authoritative. When a model is saved with
`save_pretrained`, all three files are written and can drift apart if
edited piecemeal.

## What `from_pretrained()` actually loads

```python
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
```

does, in order:

1. Reads `config.json` into `model.config`.
2. Tries `GenerationConfig.from_pretrained(path)`:
   - If `generation_config.json` is present, loads it into `model.generation_config`.
   - If absent, builds a GenerationConfig from `model.config` (a
     `_from_model_config=True` instance), inheriting `eos_token_id`,
     `bos_token_id`, `pad_token_id` from there.
3. The tokenizer (loaded separately via `AutoTokenizer`) reads
   `tokenizer_config.json` for its own `eos_token`.

So the single line that tells you what the model will stop on is:

```python
print(model.generation_config.eos_token_id)
```

Anything else (the tokenizer's eos, `model.config.eos_token_id`) is
informational with respect to `generate()`.

## Lists, scalars, and turn-marker stop tokens

`generation_config.eos_token_id` accepts either a scalar int or a list
of ints; `EosTokenCriteria` handles both transparently. Modern instruct
models commonly ship a list:

| Model | `eos_token_id` | Meaning |
|---|---|---|
| Llama 3.x Instruct | `[128001, 128008, 128009]` | `<\|end_of_text\|>`, `<\|eom_id\|>`, `<\|eot_id\|>` |
| Qwen 3 chat | `[151645, 151643]` | `<\|im_end\|>`, `<\|endoftext\|>` |
| Mistral / Llama 2 base | `2` | `</s>` only |

When you graft a chat template (e.g. ChatML) onto a base model that
originally had a single scalar EOS, the resulting model needs to stop
on **both** the original EOS (which it learned to emit during
pretraining) and the new turn-marker (which the chat template emits at
assistant turn ends). The right shape is a list:

```json
"eos_token_id": [<original_eos>, <new_turn_marker>]
```

`forgather finalize` and `forgather convert` produce exactly this shape
automatically; the original EOS is preserved at index 0 and any added
ChatML / EOT / end-of-turn tokens are appended.

## Per-call overrides

Three ways to override at call time, in increasing precedence:

```python
# 1. Replace the whole GenerationConfig for this call.
from transformers import GenerationConfig
gc = GenerationConfig(eos_token_id=[2, 32000], max_new_tokens=200)
model.generate(input_ids, generation_config=gc)

# 2. Override one field of the effective GenerationConfig.
model.generate(input_ids, eos_token_id=[2, 32000], max_new_tokens=200)

# 3. Add a custom stopping criterion -- runs alongside, doesn't replace.
from transformers import StoppingCriteriaList
extra = StoppingCriteriaList([MyCustomCriterion(...)])
model.generate(input_ids, stopping_criteria=extra)
```

`stop_strings=` is a fourth path; it adds a `StopStringCriteria` that
decodes the running output and substring-matches. Use it when no
single-token stopper is convenient. It requires `tokenizer=` because
it has to decode.

## Diagnosing a model that doesn't stop

The fastest single-line check:

```python
m = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
print(m.generation_config.eos_token_id)
```

If that doesn't include the token IDs you expect to terminate
generation, `generation_config.json` is the file to fix. Common shapes:

- **`None`** -- no EOS criterion will be installed; only `max_length`
  stops generation. Usually means `generation_config.json` is absent
  *and* `model.config.eos_token_id` is unset.
- **Wrong scalar** -- the most common bug after manually adding ChatML
  tokens. The tokenizer reports `<|im_end|>` as `eos_token`,
  `config.json` may report a list, but `generation_config.json` still
  carries the source model's pre-graft scalar EOS.
- **Correct value, but generation still doesn't stop** -- check whether
  caller code is overriding via `generate(eos_token_id=...)` (a custom
  inference server might pass an explicit list). Also verify the chat
  template actually emits the token you're trying to stop on; if the
  template never produces `<|im_end|>` at turn ends, no amount of EOS
  configuration will help.

To inspect all three files at once:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("/path/to/your/model")
print("tokenizer_config.json eos_token   ->",
      json.load(open(root / "tokenizer_config.json")).get("eos_token"))
print("config.json           eos_token_id ->",
      json.load(open(root / "config.json")).get("eos_token_id"))
gc = root / "generation_config.json"
if gc.is_file():
    print("generation_config.json eos_token_id ->",
          json.load(open(gc)).get("eos_token_id"))
else:
    print("generation_config.json (absent; generate() will fall back to model.config)")
PY
```

If the three files disagree on which token marks end-of-sequence, the
**third** value is what `generate()` will use.

## Patching a model by hand

Editing `generation_config.json` is safe -- `eos_token_id` only affects
the stopping criterion. It is never read during the training forward
pass or loss computation, so no retraining is needed.

```python
import json

GEN_CONFIG = "/path/to/your/model/generation_config.json"
NEW_EOS = [128001, 128256]   # original EOS first, new turn-marker second

with open(GEN_CONFIG) as f:
    cfg = json.load(f)
cfg["eos_token_id"] = NEW_EOS
with open(GEN_CONFIG, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
```

Token IDs are tokenizer-specific:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/path/to/your/model", trust_remote_code=True)
print("im_end ->", tok.convert_tokens_to_ids("<|im_end|>"))
print("eos    ->", tok.eos_token_id)
```

## How Forgather tools handle this

`forgather convert` and `forgather finalize` both write a merged
`generation_config.eos_token_id` automatically:

- The original EOS (from the source `config.json`) is preserved at
  index 0.
- Any wider list already present in the source `generation_config.json`
  is folded in (so Llama 3 instruct's `[128001, 128008, 128009]` is not
  silently narrowed during a round-trip).
- With auto-stop-token detection on (default), any added token whose
  name matches `<|im_end|>`, `<|eot|>`, or `<|end_of_turn|>` -- and
  whatever was added under the `--add-tokens` YAML's `eos_token` key --
  is appended to the list.

If your model came out of one of these tools recently, you should not
need to hand-patch `generation_config.json`. The hand-patch above is
for older artefacts or models produced by third-party tools that don't
sync the file.

## Related docs

- [Finalize Model](finalize-model.md) -- the recommended way to add
  stop tokens and synthesize a matching `generation_config.json` in
  one step.
- [Model Conversion](model-conversion.md) -- `forgather convert`
  reference, including its EOS-merge behaviour during HF→Forgather.
- [Add-Tokens Config](add-tokens-config.md) -- YAML format for
  `--add-tokens`.
