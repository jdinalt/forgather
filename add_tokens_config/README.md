# add_tokens_config

YAML configurations for `forgather finalize --add-tokens` (and
`forgather convert --add-tokens`). Each file describes a set of tokens to
add or replace on an existing tokenizer, plus how to initialize the
corresponding embedding rows.

## Available configs

- **chatml.yaml** — Base setup for a Forgather model pre-trained from
  scratch that you want to fine-tune for chat using the
  [ChatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md)
  dialogue format. Adds `<|im_start|>` as a special token, promotes
  `<|im_end|>` to the tokenizer's EOS, and adds a pad token if missing.
  Pair with `chat_templates/chatml.jinja`.

## Usage

```bash
forgather finalize SOURCE DEST \
    --add-tokens add_tokens_config/chatml.yaml \
    -t chat_templates/chatml.jinja
```

For the full YAML format reference, init strategies, and authoring guide,
see [docs/guides/add-tokens-config.md](../docs/guides/add-tokens-config.md).
