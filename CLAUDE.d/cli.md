# Forgather CLI quickref

The `forgather` command is the primary interface. Every subcommand
supports `--help` — prefer `forgather <sub> --help` over reading this
file when you just need flag syntax.

```bash
forgather [-p PROJECT_DIR] [-t CONFIG_TEMPLATE] <subcommand>
```

## Project exploration

```bash
forgather index              # project overview as markdown
forgather ls                 # name, description, configs (this dir)
forgather ls -r              # recurse into subdirs
forgather tlist              # list all template files
forgather tlist --format md  # inheritance hierarchy
forgather -t cfg.yaml pp     # preprocessed config (DEBUG configs here)
forgather -t cfg.yaml trefs  # template inheritance for this config
forgather -t cfg.yaml targets
```

`forgather ls` is also the canonical "did my edits parse" check —
failed configs show as `PARSE ERROR` instead of their description.

## Training

```bash
forgather -t cfg.yaml train                # local
forgather -t cfg.yaml train -d 0,1         # specific GPUs
forgather -t cfg.yaml train --dry-run      # print command only
forgather -t cfg.yaml train --enqueue      # send to forgather server
```

## Job control (server-managed jobs)

```bash
forgather control list                     # discoverable jobs
forgather control status JOB_ID
forgather control save JOB_ID              # checkpoint now
forgather control stop JOB_ID              # graceful stop + save
forgather control save-stop JOB_ID
forgather control abort JOB_ID             # no save (failed experiments)
forgather control cleanup [--force]
```

Job control requires `TrainerControlCallback` in the trainer; see
`examples/trainer_control/` and `docs/trainers/trainer-control.md`.

## Log analysis

```bash
forgather logs list
forgather logs summary [path/to/trainer_logs.json]
forgather logs summary --format one-line --all
forgather logs plot [--loss-curves|--grad-norm|--perplexity]
forgather logs plot --compare run1/... run2/... --loss-curves
```

Plots default to outlier-aware y-axis scaling; pass
`--no-ignore-outliers` to disable. Full reference:
`docs/guides/logs-analysis.md`.

## Server / cluster / GPU

```bash
forgather sched   status | list | pause | resume | cancel <id> | cleanup
forgather job     status | save | stop | abort | tail | dump <id>
forgather gpu     status | disable | enable | priority | kill <idx>
forgather cluster nodes | jobs [<id>] | submit | cancel <id>
forgather tls     init | status | renew | export-ca | import-ca | mint | install
```

All accept `--server URL` or `$FORGATHER_SERVER_URL` (default
`http://127.0.0.1:8765`). Cluster + TLS detail in
`docs/operations/tls.md`. Server architecture and CLI mapping to webui
modals: `tools/forgather_server/README.md`.

## Inference

```bash
forgather inf server -m MODEL_PATH         # bf16, cuda:0
forgather inf server -c -m MODEL_PATH      # from Forgather checkpoint
forgather inf client                        # interactive chat
forgather inf client --message "..."        # single message
forgather inf client --completion "..."     # completion mode
```

Detail: `tools/inference_server/README.md`.

## Webui build

Do **not** invoke `npm run build` directly:

```bash
./build-webui.sh             # incremental
./build-webui.sh --watch     # live reload
./build-webui.sh --clean
./build-webui.sh --install
```

Before editing webui, read `tools/forgather_server/README.md`.

## vLLM

> **2026-03 note:** vLLM integration is broken — Forgather moved to
> Transformers v5 and vLLM does not yet support that. Commands below
> are kept for reference.

```bash
vllm serve output_models/my_model --trust-remote-code
vllm serve output_models/my_model --trust-remote-code \
    --tensor-parallel-size 4 --pipeline-parallel-size 2 \
    --dtype bfloat16 --max-model-len 8192
```

Custom models need `tp_plan` / `pp_plan` in
`[model_code_generator]`; reference template:
`templatelib/examples/models/transformers/dynamic_llama.yaml`
(`base_model_tp_plan` / `base_model_pp_plan` blocks). Detail:
`docs/inference/vllm_integration.md`.

## Workspace + project creation

Prefer the CLI over copying template directories:

```bash
forgather ws create --name "..." --description "..." \
    --forgather-dir /path/to/forgather

forgather project create --name "..." --description "..." \
    [--config-prefix experiments] [--default-config baseline.yaml]
```

Cross-project model inheritance (experiments extending model
projects with `modelsrc/`) needs a separate `models/` sub-project to
keep template search paths clean. See `CLAUDE.d/gotchas.md` for the
`ModuleNotFoundError` trap and `examples/tiny_experiments/canon/` for
a worked example.
