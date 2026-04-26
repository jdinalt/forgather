# Add-Tokens Configuration

`forgather finalize --add-tokens` and `forgather convert --add-tokens` accept
a YAML file describing tokens to add or replace on an existing tokenizer.
This guide covers the YAML format, when each shape is appropriate, and how
to author a config for your own use case.

The bundled, ready-to-use configurations live in the repo's
[`add_tokens_config/`](../../add_tokens_config/) directory.

## Where this fits

After pre-training a base model from scratch you typically need to graft on
the tokens a chat template will reference (an `<|im_end|>` to mark turn
boundaries, an `<|im_start|>` to mark roles), and you may need to add a pad
token if the tokenizer was trained without one. An add-tokens config makes
this declarative: the YAML lists the tokens, and `forgather finalize` does
the embedding resize, the EOS-merge logic, and the
`generation_config.json` synthesis in one pass.

The same YAML format is also accepted by `forgather convert --add-tokens`
when converting between HuggingFace and Forgather formats.

## File structure

The top-level keys are:

| Key | Purpose |
|-----|---------|
| `bos_token` | Begin-of-sequence token. |
| `eos_token` | End-of-sequence token. |
| `pad_token` | Padding token. |
| `unk_token` | Unknown-token marker. |
| `special_tokens` | List of additional special tokens (e.g. turn markers). |
| `regular_tokens` | List of plain vocabulary additions. |

All keys are optional. Omit the keys you don't need.

### Named token form (`bos_token` / `eos_token` / `pad_token` / `unk_token`)

Each named token may be either a plain string (uses the default init) or a
dict with explicit options:

```yaml
# Plain string — uses default init for that role
unk_token: "<|unknown|>"

# Dict with init strategy
eos_token:
  token: "<|end_of_text|>"
  init: "mean"
  if_missing: false   # (default) always set, even if it overrides an existing value
```

Dict keys:

| Key | Required | Description |
|-----|----------|-------------|
| `token` | yes | The literal token string. |
| `init` | no | Embedding init for the new ID. See [Init strategies](#init-strategies). Defaults to `"mean"` for `bos_token` / `eos_token` / `unk_token`, and `"zero"` for `pad_token`. |
| `if_missing` | no | When `true`, only add/set the token if the tokenizer doesn't already define one for this role. Defaults to `false`. |

#### Replacing an existing role

If you set `eos_token: "<|im_end|>"` on a tokenizer whose EOS is already
defined as something else, two things happen:

1. The tokenizer's `eos_token` pointer is moved to `<|im_end|>`. (If
   `<|im_end|>` is not yet in the vocabulary, it is added.)
2. The new EOS embedding row is initialized by **copying the original EOS
   row's weights** (`init = "copy:OLD_EOS_ID"` is selected automatically).
   This preserves the model's existing "stop here" behavior so the new
   token starts off acting like the old one.

`forgather finalize` then ensures both the original and the new EOS IDs end
up in the destination's `generation_config.eos_token_id` list, so
generation halts on either token. The model's `config.json` `eos_token_id`
is unchanged.

### Special tokens (`special_tokens`)

```yaml
special_tokens:
  - "<|im_start|>"
  - "<|im_end|>"
```

Each entry is registered with `tokenizer.add_special_tokens` under
`additional_special_tokens`. Special-token registration matters for
tokenization: special tokens are not split by the BPE / unigram model and
are excluded from `decode(skip_special_tokens=True)`.

If a string already exists in `additional_special_tokens`, it is skipped.

### Regular tokens (`regular_tokens`)

```yaml
regular_tokens:
  - "domain_term"
  - "another_term"
```

Each entry is added via `tokenizer.add_tokens` and gets a fresh embedding
row. Use this when extending vocabulary for a domain that doesn't have
turn-marker semantics.

If a string already exists in the vocab, it is skipped.

## Init strategies

| Strategy | Effect |
|---|---|
| `"mean"` | Initialize the new row to the mean of the existing token embeddings. HuggingFace's `resize_token_embeddings(mean_resizing=True)` does this for newly-added rows. Default for `bos`/`eos`/`unk`. |
| `"zero"` | Zero-fill the row. Recommended for pad tokens (a pad row with non-zero weights can leak into attention if the mask is mis-set; zero-init avoids that). Default for `pad_token`. |
| `"copy:ID"` | Copy embedding values from the existing token at integer ID `ID`. Selected automatically when replacing an existing named role; you typically don't need to write this by hand. |

Output and input embeddings are kept in sync: when embeddings are tied, the
single shared row is initialized once; otherwise both `embed_tokens` and
`lm_head` rows are initialized identically.

## Authoring tips

- **Use `if_missing: true` for pad tokens** when the upstream tokenizer
  *might* already define one. This makes the same config safe to reuse
  across different base models.
- **Don't put your turn markers in `special_tokens` *and* set them as
  `eos_token`.** Pick one. Setting `eos_token: "<|im_end|>"` is enough on
  its own; finalize's auto-stop-token detection still merges any matching
  named special tokens into the generation config's stop list.
- **Keep `regular_tokens` short.** Adding many embedding rows you don't
  intend to train is wasteful. Use `special_tokens` for things the chat
  template emits and prefer specifying real new vocabulary rather than
  ad-hoc strings.

## Worked example

`add_tokens_config/chatml.yaml`:

```yaml
eos_token: "<|im_end|>"

special_tokens:
  - "<|im_start|>"

pad_token:
  token: "<|pad|>"
  init: "zero"
  if_missing: true
```

Apply it with finalize:

```bash
forgather finalize examples/pretrain/small-llm/output_models/wds out/wds_chatml \
    --add-tokens add_tokens_config/chatml.yaml \
    -t chat_templates/chatml.jinja
```

This produces a destination directory where:

- `tokenizer.eos_token` is `<|im_end|>`. If the source already had
  `<|im_end|>` in the vocabulary (some BPE tokenizers do), its existing
  embedding is reused; otherwise a new row is added and copy-initialized
  from the original EOS row's weights.
- `<|im_start|>` is registered as an additional special token.
- A pad token is present (the original if there was one; `<|pad|>` if not).
- `generation_config.eos_token_id` contains both the original EOS ID and
  the `<|im_end|>` ID, so generation stops on either.
- `tokenizer.chat_template` is the ChatML template from
  `chat_templates/chatml.jinja`.

## See also

- [Finalize Model](finalize-model.md) — `forgather finalize` reference
- [Model Conversion](model-conversion.md) — `forgather convert` also
  accepts `--add-tokens` with this same YAML format
- [EOS Tokens and `generate()` Stopping Criteria](eos-and-generate-stopping.md) —
  background on how the merged `eos_token_id` list is actually consumed by
  HuggingFace's `model.generate()` at inference time
