# finalize_model

Finalize a trained Forgather model into a clean handoff directory ready for
fine-tuning, sharing, or inference.

For full documentation, see
[docs/guides/finalize-model.md](../../docs/guides/finalize-model.md).

## Quick examples

The examples below assume your shell is at the repo root. They reference
the bundled add-tokens configs in [`add_tokens_config/`](../../add_tokens_config/)
and chat templates in [`chat_templates/`](../../chat_templates/).

### Duplicate a trained model into a clean directory

```bash
# Latest checkpoint, no vocab change, no optimizer.
forgather finalize \
    examples/pretrain/small-llm/output_models/wds \
    out/wds_final
```

### Preserve optimizer state for warm-start fine-tuning

```bash
forgather finalize \
    examples/pretrain/small-llm/output_models/wds \
    out/wds_final_warm \
    --keep-optimizer
```

### Set up a from-scratch model for ChatML fine-tuning

This is the typical post-pre-training step: graft on `<|im_start|>` /
`<|im_end|>`, install a chat template, and synthesize a
`generation_config.json` whose `eos_token_id` lists both the original EOS
and the new `<|im_end|>` so generation halts on either.

```bash
forgather finalize \
    examples/pretrain/small-llm/output_models/wds \
    out/wds_chatml \
    --add-tokens add_tokens_config/chatml.yaml \
    -t chat_templates/chatml.jinja
```

What this does:

- Adds `<|im_start|>` as a special token; promotes `<|im_end|>` to the
  tokenizer's EOS (its embedding is copied from the original EOS row).
- Adds `<|pad|>` only if the source had no pad token (`if_missing: true`).
- Installs `chat_templates/chatml.jinja` on the tokenizer.
- Writes `generation_config.json` with
  `eos_token_id: [orig_eos_id, im_end_id]`.

### Use a non-latest checkpoint

```bash
forgather finalize \
    examples/pretrain/small-llm/output_models/wds \
    out/wds_old \
    -c examples/pretrain/small-llm/output_models/wds/checkpoints/checkpoint-385440
```

### Single-copy layout (no `checkpoints/` subdirectory)

```bash
forgather finalize \
    examples/pretrain/small-llm/output_models/wds \
    out/wds_flat \
    --root-copy
```

## See also

- [Finalize Model guide](../../docs/guides/finalize-model.md) — full
  reference for every flag and the destination layout
- [Add-Tokens Config guide](../../docs/guides/add-tokens-config.md) —
  YAML format for `--add-tokens`
- [chat_templates/](../../chat_templates/) — bundled Jinja chat templates
- [add_tokens_config/](../../add_tokens_config/) — bundled `--add-tokens`
  YAML configs
