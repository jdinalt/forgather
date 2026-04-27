# Forgather Server from the CLI

The `forgather` CLI can talk to a running Forgather server directly, no
browser required. This is the "I'm in tmux over SSH and don't want to
keep a browser tab open" path: queue training runs, inspect what's
running, follow live logs, toggle GPU eligibility, and control jobs —
all from the terminal you're already in.

This guide walks through the common workflows. For the API reference,
see [forgather-server.md](../forgather-server.md); for a full UI
walkthrough, see [forgather-server-walkthrough.md](forgather-server-walkthrough.md).

## What you need

A running server on the same host (or reachable over the network):

```bash
forgather server                       # default: 127.0.0.1:8765
```

Every CLI command below that talks to the server accepts:

- `--server URL` — point at a non-default server
- `$FORGATHER_SERVER_URL` — same thing as an environment variable

Defaults to `http://127.0.0.1:8765`. If the server isn't running, every
CLI command prints

```
could not reach forgather-server at http://127.0.0.1:8765; is it running? (start with: forgather server)
```

and exits 1. No traceback, no five-line stack from `requests`.

## Submitting jobs: `--enqueue`

Most existing CLI commands grow an `--enqueue` flag that submits the
job to the server's queue instead of running it locally. The same
project / config / arguments you'd use for a local run work for a
queued run — the server picks the GPUs.

### Training

```bash
# Inside a Forgather project directory
forgather -t train.yaml train --enqueue
forgather -t train.yaml train --enqueue --priority 5
forgather -t train.yaml train --enqueue --requested-gpus 2
```

By default, `--requested-gpus` follows the config's `nproc_per_node`.
The server's scheduler picks which physical GPU indices to use; trying
to combine `--enqueue` with the local `-d / --devices` flag is rejected
with a clear error.

The command prints `queued: q_<id> (priority=N, gpus=M)` and exits — no
streaming, no waiting. To follow the run, use `forgather job tail` (see
below).

### Eval

```bash
forgather eval test c4 -M output_models/my_model --enqueue
forgather eval test c4 -M output_models/my_model --enqueue --priority 3 \
  --batch-size 8 --max-length 2048
```

All the local flags (`--trainer`, `--dtype`, `--attn-implementation`,
`--batch-size`, `--max-length`, `--max-steps`, `--checkpoint`,
`--compile`, `--output-dir`) carry through to the queued run.

### TensorBoard / MkDocs / Inference

These default `--requested-gpus` to the right value (0 for
TensorBoard / MkDocs, 1 for inference) and pick the canonical port for
each tool.

```bash
forgather tb --enqueue                              # logdir = project's output_dir, port 6006
forgather tb --enqueue --port 6007 --bind-all       # explicit port + listen on all interfaces

forgather mkdocs -f docs/mkdocs.yml --enqueue       # default port 8000
forgather mkdocs -f docs/mkdocs.yml --enqueue --port 8001 --strict

forgather inf server --enqueue -m output_models/my_model           # default port 8137
forgather inf server --enqueue -m output_models/my_model -p 8138 --dtype float16
```

`forgather inf server`, `convert`, and `finalize` use a different
parsing trick: when `--enqueue` is anywhere on the command line, the
remaining arguments are parsed as structured flags; otherwise they're
forwarded verbatim to the underlying script (the existing local
behavior). To see the structured flag list, just ask for help:

```bash
forgather inf server --enqueue --help
forgather convert --enqueue --help
forgather finalize --enqueue --help
```

### Convert / Finalize

```bash
forgather convert --enqueue --src output_models/my_model --dst /tmp/hf_export
forgather finalize --enqueue --source output_models/my_model --dest /tmp/final \
  --safetensors --dry-run
```

These are CPU jobs (`requested_gpus=0`), so they don't compete with
training for cards.

### MkDocs (new)

`forgather mkdocs` is a new wrapper around `mkdocs serve` that works
both locally (the default) and through the queue:

```bash
# Local
forgather mkdocs -f docs/mkdocs.yml                 # serves on 127.0.0.1:8000
forgather mkdocs -f docs/mkdocs.yml --port 8001 --strict
forgather mkdocs -f docs/mkdocs.yml --no-livereload --watch src/

# Queued
forgather mkdocs -f docs/mkdocs.yml --enqueue
```

When run locally it just shells out to `mkdocs serve --dev-addr
host:port` with the right flags. When queued, it sends the same
parameters as the webui's MkDocs modal.

## Watching the queue: `forgather sched`

```bash
forgather sched status
# enabled=True  queued=2  running=3  last_tick=1s ago

forgather sched list
# Status     ID                              Type        Priority  GPUs  Project/Config              Time
# running    q_1738012345_a1b2c3d4           training    0         2     tiny_llama:train.yaml       2m ago
# running    q_1738012360_e5f6g7h8           inference   0         1     my_model                    1m ago
# running    q_1738012398_i9j0k1l2           tensorboard 0         0     tensorboard:6006            30s ago
# queued     q_1738012412_m3n4o5p6           eval        5         1     my_model:c4.yaml            12s ago
# queued     q_1738012420_q7r8s9t0           training    0         1     small_llm:train.yaml        4s ago
```

`status` is one line; `list` is a wide table that shows queued items
first (priority-ordered), then active jobs, then the most recent
terminal records.

### Pausing / resuming dispatch

Useful when you need to stop the queue from chewing through pending
work — for example, before plugging in a GPU, swapping a cable, or
running an interactive debug session that needs the cards yourself.

```bash
forgather sched pause                    # stop dispatching new jobs
forgather sched resume                   # resume

forgather sched cancel q_1738012412_m3n4o5p6     # remove a queued or running job
forgather sched cleanup                          # remove all terminal records
forgather sched cleanup q_1738012345_a1b2c3d4    # remove one specific terminal record
```

`pause` only affects the dispatcher — running jobs continue, and you
can still enqueue new ones (they just sit until you resume).

## Per-job control: `forgather job`

```bash
forgather job status q_1738012345_a1b2c3d4
# Trainer status (proxied):
#   global_step: 1234
#   loss: 1.872
#   ...

forgather job tail  q_1738012345_a1b2c3d4    # stream live TTY; Ctrl-C exits cleanly
forgather job dump  q_1738012345_a1b2c3d4    # full captured log to stdout
forgather job dump  q_1738012345_a1b2c3d4 > training.log
```

For training jobs the trainer exposes a control endpoint that the
server proxies — `forgather job` wraps it:

```bash
forgather job save        <id>     # trigger an out-of-band checkpoint save
forgather job stop        <id>     # graceful: save final checkpoint, then exit
forgather job save-stop   <id>     # save now, then exit
forgather job abort       <id>     # immediate stop, no checkpoint
forgather job kill        <id>     # SIGTERM the process group (server-launched only)
forgather job force-kill --yes <id>   # SIGKILL the process group
```

Use the trainer-level actions (`save` / `stop` / `save-stop` / `abort`)
when the trainer is responsive — they go through the trainer's
checkpoint pathway. Drop to `kill` / `force-kill` when the trainer is
hung or unresponsive (e.g. a wedged dataloader, NCCL deadlock).

`force-kill` requires `--yes` to make accidental SIGKILLs harder.

### Workflow: spot-check then commit

```bash
# In one terminal
forgather -t train.yaml train --enqueue --priority 5

# In another, watch it land
forgather sched list
forgather job tail q_1738012345_a1b2c3d4
# ... see weird loss spike, decide to abort
^C                                            # leaves tail (job keeps running)
forgather job abort q_1738012345_a1b2c3d4
```

Or, if it's behaving:

```bash
forgather job save q_1738012345_a1b2c3d4      # checkpoint without stopping
forgather job stop q_1738012345_a1b2c3d4      # save final, then exit
```

## GPU control: `forgather gpu`

```bash
forgather gpu status
# Idx  Name              Util%  Mem (GB)      Temp  Power  Fan%  Disabled  MinPri  PIDs
# 0    NVIDIA RTX 4090   87     21.5/24.0     72    420W   45    -         0       2
# 1    NVIDIA RTX 4090   91     22.1/24.0     74    430W   47    -         0       2
# 2    NVIDIA RTX 4090   0      0.0/24.0      35    18W    0     (excluded)  0     0
# ...
```

`(excluded)` means the operator started the server with
`CUDA_VISIBLE_DEVICES` filtering this index out — the scheduler will
never assign it.

### Scheduling policy

Every GPU has two persistent dials:

- `disabled` (bool) — temporarily pull a card out of rotation without
  restarting the server.
- `min_priority` (int) — only assign jobs whose priority is `>=` this
  value. Useful for reserving high-end cards for big runs.

```bash
forgather gpu disable 3                      # take GPU 3 out of rotation
forgather gpu enable  3                      # put it back

forgather gpu priority 0 10                  # only priority>=10 jobs get GPU 0
forgather gpu priority 0 0                   # back to default
```

Both settings persist across server restarts (`~/.forgather/server/gpu_policy.json`).

### Emergency: clear a wedged GPU

When a process refuses to die — wedged trainer, leaked CUDA context,
zombie torchrun — and the card stays at 100% util with stale memory:

```bash
forgather gpu kill --yes 0
# gpu 0: killed=[12345, 12346], failed=[]
```

This SIGKILLs **every** compute process on the card, including ones
the server didn't launch. The `--yes` is mandatory because of that
blast radius. The webui shows the same "Kill all processes on this
GPU…" button with a confirm dialog.

## Recipes

### Submit a sweep, then tail one of them

```bash
for lr in 1e-4 3e-4 1e-3; do
  forgather -t train.yaml train --enqueue \
    --dynamic-arg learning_rate=$lr
done

forgather sched list
forgather job tail q_<one of the ids printed above>
```

### Reserve GPUs for an interactive session

```bash
forgather sched pause
forgather gpu disable 0
forgather gpu disable 1
# ... do interactive work on GPU 0 + 1 ...
forgather gpu enable 0
forgather gpu enable 1
forgather sched resume
```

### Save+stop a slow run, free the slot for the next sweep

```bash
forgather job save-stop q_<long_running_id>
# Wait for it to land in 'done' state, then:
forgather sched cleanup            # tidy the record list
```

### Connect to a remote forgather-server over SSH

The server is single-host today, but the CLI works fine through an SSH
port-forward:

```bash
# On the laptop
ssh -L 8765:localhost:8765 user@training-host

# In another terminal, also on the laptop
export FORGATHER_SERVER_URL=http://127.0.0.1:8765
forgather sched status                       # talks to the remote server
forgather job tail q_<id>                    # stream remote logs
```

Or pass `--server http://127.0.0.1:8765` to one-shot commands.

## How `--enqueue` chooses GPU count

| Job type      | Default `requested_gpus` | Override |
| ------------- | ------------------------ | --- |
| `train`       | `nproc_per_node` from config | `--requested-gpus N` |
| `eval`        | 1                        | (always 1) |
| `inf server`  | 1                        | (always 1) |
| `tb`          | 0                        | (always 0) |
| `mkdocs`      | 0                        | (always 0) |
| `convert`     | 0                        | (always 0) |
| `finalize`    | 0                        | (always 0) |

Zero-GPU jobs run regardless of card availability. GPU-bound jobs wait
for the scheduler to find a free card whose `disabled=False` and whose
`min_priority <=` the job's priority.

## Differences from `forgather control`

`forgather control` (separate, older) talks directly to a trainer's
HTTP control endpoint by discovering it from
`~/.forgather/jobs/<job_id>/endpoint.json`. It works for any job that
has the `TrainerControlCallback` enabled — including jobs you started
yourself via `forgather train` (no server involved).

`forgather job` only works for jobs the server knows about (queued or
launched through it), but it adds queue-aware operations (cancel,
log dump, status while still queued, server-side SIGTERM/SIGKILL).

The two coexist; pick whichever matches the job's discovery model.

## Troubleshooting

**"could not reach forgather-server at …"** — server isn't running, or
isn't on the URL you're pointing at. Start it with `forgather server`,
or fix `--server` / `FORGATHER_SERVER_URL`.

**`forgather job status <id>` says "still starting"** — the trainer
process exists (its `JobRecord` is on the server) but it hasn't yet
written its endpoint file. This is normal in the first few seconds
after dispatch. Try again, or use `forgather job tail` to watch it
start.

**"requested_gpus must be >= 1 for training jobs"** — the server
requires at least one GPU for training. Drop `--requested-gpus 0`, or
queue a different job type.

**`forgather job tail` exits immediately** — either the job has
already terminated (the stream closes 3 s after the terminal status),
or the job wasn't server-launched (TTY capture only exists for jobs
the server spawned). Use `forgather job dump` for a one-shot pull, or
`forgather control` if it's an externally-launched run.

**`forgather inf server --help` shows the inference server's flags,
not mine** — that's intentional: without `--enqueue`, the wrapper
forwards everything to `tools/inference_server/server.py`. To see the
enqueue flags, type `forgather inf server --enqueue --help`.
