# DiLoCo Features

This project exists to **exercise the less-travelled DiLoCo features** and prove
they still run, end-to-end, through the orchestrated (scheduler) path:

- **Streaming sync** (`--num-fragments N > 1`) — split each sync round into N
  fragments synced in the background, overlapping the next steps' compute.
- **Asynchronous mode** (`--async`) — the server applies each worker's
  pseudo-gradient as it arrives, with no cross-worker barrier.
- **Delayed Nesterov buffer** (`--dn-buffer-size N`, async-only) — buffer N
  async submissions before applying the momentum outer step; plain GD between.
- **DyLU / Dynamic Local Updates** (`--dylu`, async-only) — adapt each worker's
  `sync_every` to its own throughput, so a slow worker syncs less often.
- The **transport × wire-format matrix**: `{http, gRPC} × {pickle, safetensors}`
  for the bulk legs.

These are all features of the **HTTP sync backend** (streaming in particular is
*only* implemented there — the shared-memory and collective backends raise
`NotImplementedError` on fragment sync). They are server-authoritative: you
select them with flags on `forgather diloco server`, and every worker adopts
them from the server's `/info`. The same worker config (`default.yaml`) runs all
of them — the feature under test is chosen at the **server**, not in the config.

It is a sibling of [`../diloco`](../diloco), the end-to-end DiLoCo walkthrough —
start there if you have never run DiLoCo here. This README assumes those
mechanics (server / worker / forgather-server roles, the inner/outer optimizer
split, work-unit dispatch) and focuses on the feature matrix. The authoritative
reference is [`docs/trainers/diloco.md`](../../../docs/trainers/diloco.md).

All commands assume you are in this project directory:

```bash
cd examples/tiny_experiments/diloco_features
```

---

## What it runs

| Run | Server flags (beyond `-o <master> -n 2`) | What it proves |
|---|---|---|
| `http-pickle`  | `--sync-every 20 --wire-format pickle` | HTTP bulk legs, pickle codec |
| `http-st`      | `--sync-every 20 --wire-format safetensors` | HTTP bulk legs, safetensors codec |
| `grpc-pickle`  | `--sync-every 20 --grpc --wire-format pickle` | gRPC bulk legs, pickle codec |
| `grpc-st`      | `--sync-every 20 --grpc --wire-format safetensors` | gRPC bulk legs, safetensors codec |
| `baseline`     | `--grpc --wire-format safetensors --sync-every 20` | sync DiLoCo on the through-line transport |
| `streaming`    | `… --num-fragments 2 --verbose-sync` | per-fragment streaming sync |
| `async`        | `… --async --verbose-sync` | barrier-free async apply |
| `async-dn`     | `… --async --dn-buffer-size 4 --verbose-sync` | delayed-Nesterov buffering |
| `async-dylu`   | `… --async --dylu --dylu-base-sync-every 20` | per-worker adaptive `sync_every` |

The first four are quick functional smokes (`--compile no`, ~60 steps, ~3 sync
rounds) covering the transport matrix. The rest are real runs on the
**gRPC + safetensors** through-line (`--compile` on for the feature runs).

Everything runs through the forgather **scheduler** — `forgather diloco server`
(scheduled, not `--local-only`) plus `forgather submit --diloco` — because that
orchestrated path is itself part of what's being validated.

---

## Running it

A driver script orchestrates each run (start a scheduled server with the right
flags, submit workers, wait, capture logs under `runs/<name>/`, shut down):

```bash
./harness.sh all          # every run, in order
./harness.sh streaming    # just one
```

Prerequisites:

- A running **forgather server** (`forgather server`) and **dataset server** —
  the standard cluster setup the sibling `../diloco` walkthrough establishes.
- A **master model** the param server initialises from, built once:

  ```bash
  forgather -p ../../models/llama -t small.yaml \
      model --device cpu --save-checkpoint --safetensors \
      --output-dir ../../../models/small_llama_features_master \
      construct
  ```

The harness copies that master to a fresh `models/.feat_runs/<name>/` per run, so
the param server's checkpoint / shutdown saves never mutate the reference and
every run starts from identical init.

### Simulating heterogeneous workers (DyLU)

DyLU only does anything when workers run at **different speeds**, but the GPUs
here are identical. The worker therefore honours a debug-only throttle,
`DILOCO_DEBUG_STEP_DELAY` (seconds of `sleep` per local step), delivered to a
specific worker via `submit --env`:

```bash
# one slow worker, one fast worker, same server -> a real speed gradient
forgather submit --diloco --diloco-worker-count 1 --worker-id feat-slow \
    --env DILOCO_DEBUG_STEP_DELAY=0.10 --heartbeat-interval 5 ...
forgather submit --diloco --diloco-worker-count 1 --worker-id feat-fast \
    --heartbeat-interval 5 ...
```

`DILOCO_DEBUG_STEP_DELAY` is debug-only — it throttles real training and is never
set in production. `submit --env KEY=VALUE` (repeatable) forwards env to the
scheduled worker process(es) via `job_params.extra_env`; it is honoured on plain
`--diloco` worker submits only (not the `--global` compose or collective paths,
which reject it rather than silently dropping it).

---

## Results: a real-budget comparison

The functional checks above prove each feature *engages*. To confirm each one
actually *trains* — and to see how it bends the loss trajectory — the five
configs were run to a real budget and harvested into loss curves
([`analysis/harvest.py`](analysis/harvest.py) →
[`assets/curves.csv`](assets/curves.csv) →
[`analysis/plot_experiment.py`](analysis/plot_experiment.py)).

**Setup.** Small Llama (34.4M) on Fineweb-Edu, 2 workers, **H = 100**,
**~1B tokens** (2× Chinchilla; 520M/worker × 2), gRPC + safetensors,
`torch.compile` on. Identical pristine init and the same seed/data across every
run — only the feature flag varies. The whole sweep runs through the scheduler
via [`experiment.sh`](experiment.sh) (`./experiment.sh run`, then
`./experiment.sh dylu` for the DN-buffered DyLU run).

| Run | final train | final eval | sync rounds | outcome |
|---|---|---|---|---|
| **Baseline** (sync, H=100) | 2.903 | **2.918** | 160 | healthy — the reference |
| **Async + DN buffer** | 2.994 | **2.997** | 320 | healthy — async, stabilized |
| **Streaming** (2 fragments) | 3.028 | **3.049** | 320 | healthy — small convergence cost |
| **Async + DN + DyLU** | 3.267 | **3.225** | 347 | healthy — adaptive H, more variance |
| **Async (no DN buffer)** | 7.786 | **7.169** | 123 | **diverged** — aborted at step 6400 |

![Feature comparison: train loss, eval loss, grad norm](assets/loss_comparison.png)
![Eval-loss endgame (converged runs)](assets/eval_tail.png)

### Healthy vs. not — the divergence the functional test missed

Four of the five configs descend cleanly (monotonic loss, stable grad norm). The
fifth, **pure `--async` with no DN buffer, diverges**: the loss never drops below
~6.9, the `DivergenceDetector` trips (`abs-divergence 1.04 > 1.0`) and aborts the
run at step 6400. This is the expected failure mode, not a bug — the docs
prescribe `--dn-buffer-size N` (N = num_workers) for async, because applying each
worker's pseudo-gradient with full-LR Nesterov momentum and no cross-worker
averaging is unstable. Adding the DN buffer (`Async + DN`) recovers a healthy
2.997, right behind the sync baseline. **A functional "it engages" test passes
this config; only the loss curve catches that it never learns** — which is the
whole reason the curves exist.

### Feature impact, relative to baseline

The converged ordering (see the endgame zoom) is
**baseline (2.918) < async+DN (2.997) < streaming (3.049) < async+DN+DyLU (3.225)**:

- **Async + DN** costs only ~0.08 eval loss vs the synchronous baseline — the
  barrier-free schedule is nearly free here, once stabilized.
- **Streaming** (2 fragments) costs ~0.13 — fragmenting the sync into staggered
  partial updates measurably perturbs convergence at H=100.
- **DyLU** lands highest of the converged runs and noisier — per-worker adaptive
  `sync_every` on deliberately heterogeneous workers (one throttled) trades some
  convergence for schedule adaptivity; it stays stable on the DN-buffered base.

### Baseline check: gRPC + safetensors is lossless

The baseline differs from the sibling [`diloco`](../diloco) project's 1B
`h100` run (DiLoCo 2w, H=100, same model/seed/data/LR, same `torch.optim.AdamW`)
in exactly two things: the **transport** (gRPC vs the historical HTTP) and the
**wire codec** (safetensors vs pickle). Both are lossless — safetensors carries
the identical bf16 bytes, gRPC is pure plumbing — so the trajectories should
coincide. They do
([`analysis/verify_baseline.py`](analysis/verify_baseline.py)):

![Baseline vs original h100](assets/baseline_vs_h100.png)

The two curves overlay across the full 1B-token run; final eval **2.918 vs
2.936** (−0.018, i.e. *better* by a hair). A lossy transport would raise loss or
diverge (cf. the async run at 7.2); a sub-noise delta on the good side is
run-to-run / CUDA nondeterminism, confirming the switch is lossless.

---

## Verification signals

"Runs without crashing" is not enough — each run should show the feature
actually **engaged**. The negotiated bulk transport is echoed once per worker at
startup:

```
DiLoCoCallback: using server settings sync_every=20 up=bf16 down=fp32 \
    dylu=False num_fragments=1 transport=grpc(127.0.0.1:NNNNN) wire=safetensors
```

| Run | Look for (worker log unless noted) |
|---|---|
| transport matrix | `transport=<http\|grpc>(…) wire=<pickle\|safetensors>` matches the flags; `stopped after N sync rounds` with N ≥ 1 (the wire codec actually moved bytes — the `up_mb`/`dn_mb` columns are non-zero). Server log: `grpc_enabled`, `wire_format`, and (gRPC) `gRPC bulk listener on …`. |
| streaming | `num_fragments=2` in the settings echo; the worker performs per-fragment syncs and finishes without `NotImplementedError`; server (`--verbose-sync`) advances per-fragment rounds. |
| async | server log shows async mode (no barrier); sync round advances as submissions arrive; the two workers can reach **different** rounds. |
| async-dn | server applies the momentum (outer) step only every `dn-buffer-size` submissions, simple GD between (visible with `--verbose-sync`). |
| async-dylu | server emits a per-worker `recommended_sync_every`; the slow worker's effective `sync_every` drops below the fast worker's (`H_w = floor((v_w/v_max)·H_base)`); the worker logs a DyLU adjustment. |

Because the worker has **no silent HTTP fallback** when the server advertises
gRPC (a gRPC failure raises rather than downgrading), a gRPC run that *completes
its sync rounds* has provably used the gRPC bulk path.

---

## Files

- `templates/configs/default.yaml` — the single small-Llama DiLoCo worker config
  (extends `projects/small.yaml`, `enable_diloco=True`).
- `harness.sh` — the run driver (server lifecycle + submit + capture per recipe).
- `runs/` — captured per-run logs (gitignored scratch).
