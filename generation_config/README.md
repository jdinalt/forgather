# Generation Config Presets

Reference presets shipped with Forgather. The forgather-server webui's
Inference panel exposes these alongside user-saved presets from
`~/.forgather/generation_config/`.

Files here are read-only from the UI: they cannot be deleted or overwritten.
To customize one, copy it to `~/.forgather/generation_config/` under a new
name (or edit it in place with a text editor).

Each file is a JSON object whose keys are forwarded verbatim to the
inference server's `GenerationConfig`. See
`tools/inference_server/models/chat.py` for the accepted fields.

## Bundled presets

- `greedy.json` — deterministic decoding, no sampling.
- `precise.json` — low temperature, mild nucleus filter.
- `balanced.json` — general-purpose chat defaults.
- `creative.json` — higher temperature / top_p for exploratory prose.
- `beam_search.json` — 4-way beam search with n-gram block.
- `contrastive.json` — contrastive search (penalty_alpha + small top_k).
