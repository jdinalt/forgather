# OpenThoughts3-1.2M Dataset

[OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) is a reasoning-focused instruction dataset of ~1.2M multi-turn conversations covering math (~850K), code (~250K) and science (~100K), annotated by QwQ-32B (16x annotations per unique question).

The dataset stores each example as a ShareGPT-style `conversations` list of `{from, value}` turns where `from` is `human` or `assistant`. The map functions in `src/openthoughts.py` map those role names to chat-template canonical roles (`user`/`assistant`), render the turns through a Jinja chat template (default ChatML), and either tokenize the result directly or hand it off to `block_tokenize_fn` for sequence packing.

## Configurations

- **`openthoughts.yaml`** -- Renders conversations through the chat template and applies best-fit sequence packing: multiple conversations are concatenated into fixed-length training samples for higher GPU utilisation. Use `--chat-template` to override the default ChatML template, and `--max-length` / `--stride` to control the packing window.

## Splits

The HuggingFace dataset only ships a single `train` split. The configs slice it into:
- `train_dataset_split` -- `train[10000:]`
- `validation_dataset_split` -- `train[:1000]`
- `test_dataset_split` -- `train[1000:10000]`

## Notes

- The first time the configs are run, the underlying dataset (~28GB across 120 parquet files) is downloaded and indexed; subsequent runs use the cache and are fast.
- The dataset has no dedicated `system` role, so the chat template is rendered with only `user`/`assistant` turns.
