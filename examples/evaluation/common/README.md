# Evaluation configs

Named evaluation configurations discoverable by `forgather eval`.

Each config is a metadata-only Forgather project that selects a dataset
project + config and supplies sensible defaults (batch size, max length).
The actual trainer, model, and tokenizer are constructed directly in Python
by `scripts/eval_script.py`; this project only supplies the metadata and
forwards preprocessing kwargs (`max_length`, `stride`) to the dataset
project when the eval script materializes its `test_dataset` target.

See [`docs/guides/evaluating-models.md`](../../../docs/guides/evaluating-models.md)
for full usage.

## Available configs

| Name | Dataset project | Dataset config |
|------|-----------------|----------------|
| `tinystories` | `examples/datasets/roneneldan` | `tinystories-packed.yaml` |
| `c4` | `examples/datasets/allenai` | `en-packed.yaml` |
| `openorca` | `examples/datasets/Open-Orca` | `openorca-packed.yaml` |
| `openassistant` | `examples/datasets/OpenAssistant` | `openassistant_packed.yaml` |
| `fineweb-edu-dedup` | `examples/datasets/HuggingFaceTB` | `smollm-corpus/fineweb-edu-packed.yaml` |

## Usage

```bash
forgather eval list
forgather eval show c4
forgather eval test tinystories -M /path/to/model
forgather eval test fineweb-edu-dedup -M /path/to/model --trainer pipeline
```

## How configs are structured

Each config extends the base eval template at
`templatelib/base/test/test_type.yaml`, which stamps
`config_class = "type.evaluation"` into the meta block (this is the tag the
CLI uses to discover eval configs). A config supplies:

- `ns.eval_name` — short identifier (e.g. `c4`)
- `ns.dataset_proj` — absolute path to the dataset project
- `ns.dataset_config` — dataset config template to load
- `ns.default_batch_size`, `ns.default_max_length` — fallback defaults when
  the CLI does not pass `--batch-size` / `--max-length`

See `templates/configs/tinystories.yaml` for a minimal example.
