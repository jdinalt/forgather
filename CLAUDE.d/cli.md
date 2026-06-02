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
forgather -t cfg.yaml train                # local, foreground
forgather -t cfg.yaml train -d 0,1         # specific GPUs
forgather -t cfg.yaml train --dry-run      # print command only
forgather -t cfg.yaml train --schedule     # submit to the scheduler (background)
forgather -t cfg.yaml train --schedule --foreground   # ...and attach
```

### Running via the scheduler

```bash
forgather -t cfg.yaml submit                # shorthand for `train --schedule` (single-node)
forgather -p <abs-path> submit --global \
    --member HOST:GPUS[:IFACE] ...          # multi-node fan-out
```

`forgather submit` is the canonical scheduler entry point.
`forgather cluster submit ...` is a deprecated alias of
`forgather submit --global ...`.

## Job control (server-managed jobs)

```bash
forgather job list                     # queued + active jobs
forgather job status JOB_ID
forgather job save JOB_ID              # checkpoint now
forgather job stop JOB_ID              # graceful stop + save
forgather job save-stop JOB_ID
forgather job abort JOB_ID             # no save (failed experiments)
forgather job cleanup [JOB_ID]
```

Save/stop/abort require `TrainerControlCallback` in the trainer; see
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
forgather job     list | status | save | stop | abort | tail | dump <id>
forgather job     cancel <id> | cleanup [<id>] | gc
forgather job     scheduler status | pause | resume
forgather gpu     status | disable | enable | priority | kill <idx>
forgather cluster nodes | jobs [<id>] | cancel <id>
forgather tls     init | status | renew | export-ca | import-ca | mint | install
```

`forgather job` is the merged queue + job-control surface.
`forgather sched ...` is a deprecated alias (`sched status/pause/resume`
now live under `job scheduler`).

All accept `--server URL` or `$FORGATHER_SERVER_URL` (default
`http://127.0.0.1:8765`). Cluster + TLS detail in
`docs/operations/tls.md`. Server architecture and CLI mapping to webui
modals: `tools/forgather_server/README.md`.

## Inference

```bash
forgather inf server -m MODEL_PATH         # bf16, cuda:0 (scheduler, background)
forgather inf server -c -m MODEL_PATH      # from Forgather checkpoint
forgather inf server --local-only -m MODEL_PATH   # foreground (old default)
forgather inf client                        # interactive chat
forgather inf client --message "..."        # single message
forgather inf client --completion "..."     # completion mode
```

`inf server` is a long-running service: it submits to the scheduler
(background) by default. `--local-only` runs it in the foreground;
`--local-fallback` foregrounds only when the server is unreachable.

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
