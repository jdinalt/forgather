# DiLoCo: Distributed Low-Communication Training

DiLoCo (Distributed Low-Communication) is a **two-level optimization scheme**. An
ordinary *inner* optimizer (e.g. AdamW) trains each worker locally for `H` steps
(default 500), exactly as in non-distributed training; then, at the sync
boundary, each worker's net weight change over those H steps — its
**pseudo-gradient** — is averaged across workers and fed to an *outer* optimizer
(SGD with Nesterov momentum, `lr=0.7`, `momentum=0.9`) on a parameter server,
which produces the new global weights everyone adopts. Syncing every ~500 steps
instead of every step is the whole idea: it is a general **low-communication**
training method.

There are two independent reasons to use it — and they are easy to conflate:

- **Bandwidth / scaling.** Communicating ~500x less often makes data-parallel
  training practical over slow or heterogeneous links (commodity Ethernet,
  cross-host, WAN), where DDP/FSDP stall the GPUs all-reducing every step.
- **Quality / generalization.** The outer step is a slow-momentum
  (SlowMo / Lookahead) update that finds flatter minima, and on eval loss DiLoCo
  can **overtake** an all-reduce-every-step baseline — how early depends on the
  sync interval `H` (in the reference sweep, ~21% of a 1B run at `H=20`), not on
  needing an enormous budget. The win holds even with a **single worker on one
  GPU**, where it is a pure regularizer with no data parallelism at all. See
  [When to use DiLoCo](#when-to-use-diloco) and the empirical
  [sweep](../../examples/tiny_experiments/diloco/README.md#extended-sweep-budget-sync-interval-and-single-worker-local-sgd).

> **Running DiLoCo from the CLI?** The canonical, end-to-end, verified
> walkthrough is
> [`examples/tiny_experiments/diloco/`](../../examples/tiny_experiments/diloco/README.md)
> — start there for a guided run (build the model, start the servers, launch
> workers, monitor, stop, resume). This page is the reference hub — concepts,
> when to use it, how it works, and quick start; the protocol, every setting,
> and the advanced modes the example doesn't cover live in the linked
> [reference chapters](#diloco-documentation-map) below.

The system supports two operating modes:
- **Synchronous** (default): all workers must submit before the server applies
  the outer optimizer. Simple and deterministic.
- **Asynchronous** (`--async`): workers submit independently without waiting,
  for heterogeneous fleets where fast workers shouldn't idle on slow ones;
  Delayed Nesterov (DN) momentum and Dynamic Local Updates (DyLU) keep stale
  updates from destabilizing training.

## DiLoCo documentation map

This page is the **conceptual + quick-start hub**: what DiLoCo is, when to use
it, how it works, and how to run it through the Forgather server. The detailed
CLI, backend, and programmatic/advanced reference lives in companion chapters
so this page stays readable in one sitting.

**Reference chapters (this directory):**

| Chapter | What it covers |
|---|---|
| [CLI reference](diloco-cli.md) | Manual/low-level invocation, the server-coordinated workflow in depth, monitoring & control, network configuration |
| [Sync backends](diloco-backends.md) | HTTP-star, shared-memory, and collective backends; group agreement; pipeline-parallel composition |
| [Programmatic API & advanced modes](diloco-advanced.md) | Python API, server configuration, unified statistics, async mode, streaming/fragments, fault tolerance, HTTP API, Forgather integration, work-unit dispatch |

**Deeper / related:**

| Document | What it covers |
|---|---|
| [Architecture & Maintainer Guide](diloco-architecture.md) | Internals: wire protocol, server/worker classes, checkpoint + meta-init, threading model |
| [Work-Unit Dispatch](../design/diloco-work-unit-dispatch.md) | How workers shard the training set via server-issued row ranges |
| [Pipeline Groups](../design/diloco-pipeline-groups.md) | DiLoCo + pipeline parallel: per-rank workers and server-aware groups |
| [Security Model](../design/diloco-security.md) | Auth, mTLS, the endpoint trust split, audit log |
| Example — [`tiny_experiments/diloco`](../../examples/tiny_experiments/diloco/README.md) | Canonical end-to-end CLI walkthrough; DiLoCo vs DDP / PostLocalSGD sweep |
| Example — [`tiny_experiments/diloco_lowprec`](../../examples/tiny_experiments/diloco_lowprec/README.md) | Low-precision wire transport (bf16 ± stochastic rounding) experiment sweep |
| Example — [`tiny_experiments/diloco_features`](../../examples/tiny_experiments/diloco_features/README.md) | Exercises streaming / async / DN-buffer / DyLU and the transport×wire matrix through the scheduler |

## When to use DiLoCo

DiLoCo is a good fit when you want any of:

- **Multi-node / slow-network scaling.** Training spans machines on commodity
  Ethernet, different rooms/buildings, or a WAN — anywhere per-step all-reduce
  would stall the GPUs on the network. DiLoCo syncs ~500x less often and keeps
  them busy.
- **A DDP alternative for better final quality.** The outer step is not just a
  bandwidth tradeoff — it converges to a *better* optimum than an
  all-reduce-every-step baseline. In the reference
  [sweep](../../examples/tiny_experiments/diloco/README.md#extended-sweep-budget-sync-interval-and-single-worker-local-sgd),
  2-worker DiLoCo at a quality-tuned `H=20` overtakes the 2-GPU DDP baseline on
  eval loss at **~21% of a 1B-token run** (and finishes 2.887 vs 3.005); how
  early the crossover comes is set by the sync interval `H`, not by needing a
  huge budget (the bandwidth-frugal default `H=500` is the slowest to cross).
- **Single-node / single-worker regularization.** Even one worker on one GPU,
  no data parallelism, benefits: the outer SGD-with-Nesterov wrapped around the
  inner trajectory is a SlowMo/Lookahead-style slow-weight update that finds
  flatter minima and generalizes better. *"A DiLoCo-style outer step is worth
  folding into ordinary single-node training, no parameter server required."*
- **Heterogeneous fleets.** Workers of different speeds / GPU counts, via
  asynchronous mode (Delayed Nesterov + DyLU), so fast workers don't idle on
  slow ones.

### Myths to ignore

A reader who skims the bandwidth pitch tends to conclude DiLoCo is *only* for
big models on slow multi-machine links — and, in particular, that you should
**always prefer DDP on a single host with a fast, low-latency interconnect.**
That is wrong, and it is contradicted by both Forgather's own
[sweep](../../examples/tiny_experiments/diloco/README.md#extended-sweep-budget-sync-interval-and-single-worker-local-sgd)
and the local-SGD literature ([References](#references)). None of the following
is a contraindication:

- **"Small models don't benefit."** False — the reference sweep uses a 34.4M
  Llama and shows the full set of gains. Model size is not a barrier.
- **"You need multiple machines / multiple GPUs."** False — a single worker on
  a single GPU already wins on eval loss; the Lookahead/SlowMo effect needs no
  data parallelism.
- **"On a fast / single-host link, always prefer DDP."** False — this conflates
  the two benefits. The *throughput* win needs a slow link to matter; the
  *quality/generalization* win is **independent of link speed** and shows up on
  one box with a fast bus, so single-host DiLoCo can still finish at a better
  optimum. On throughput the picture depends on the backend: the HTTP
  parameter-server path adds some per-sync overhead (the single-host reference
  run measured ~310K vs ~360K tok/s, ~13%, though it syncs only 16 times), while
  the **shared-memory / collective** backends average locally every `H` steps
  instead of all-reducing every step, keeping single-host throughput close to
  DDP. So on a single host the case for DDP is weak: comparable throughput
  (shared-memory) and a better final optimum.
- **"Don't use it with per-step gradient sync / a large learning rate."** There
  is no such contraindication (it is confabulated). DiLoCo *replaces* per-step
  sync by design; the literature's generalization edge actually wants a *small*
  LR and a *long* run, not a large LR.

### Honest tradeoffs

- **Extra moving parts.** All paths run a parameter server alongside the workers
  (cheap, CPU-only, but another process to operate). The HTTP path also moves the
  sync tensors over the wire; the single-host **shared-memory / collective**
  backends keep the tensor traffic on-host — shared-memory through an mmap region
  the co-located server aggregates in place, collective through an all-reduce
  among the workers — so only the lightweight coordination stays on HTTP.
- **Token-budget caveat.** A worker runs the *full* schedule as if standalone —
  it does not know it is one of N — so N workers process N× the intended tokens
  unless you give **each worker `1/N` of the budget**.
- **The crossover point is set by `H`, not just the budget.** It's tempting to
  read the results as "DiLoCo only wins at the very end of a huge run," but
  that's the **`H=500` default** talking — the bandwidth-frugal setting is the
  *slowest* to cross (it doesn't pass the baseline within 1B tokens). Tighten the
  sync interval and the crossover moves early: at `H=20`, 2-worker DiLoCo passes
  DDP at ~21% of a 1B run and a single worker passes its baseline almost
  immediately. The honest caveat is narrower than "short runs lose": at the
  *shortest* budget (~1× Chinchilla, ~500M tokens) with the *default* `H=500`,
  DiLoCo can finish behind (eval 3.343 vs ~3.15). If you want the quality win
  sooner, run a shorter `H` rather than a longer budget.
- **`H` trades bandwidth for convergence.** This is the same dial from the other
  side: smaller `H` (more frequent sync) converges better but communicates
  proportionally more (H=20 syncs 25× more than H=500); larger `H` is
  bandwidth-frugal but drifts more and crosses later. Tune it to your
  interconnect — a fast/single-host bus can afford a short `H` and the early
  crossover; a slow link wants a larger `H` and accepts a later one.

## Quick Start (recommended): through the Forgather server

The recommended way to run DiLoCo is through the **Forgather server**, which
schedules the parameter server and workers as managed jobs, captures their logs,
provisions auth tokens and TLS, and lets workers auto-discover the dataset
server — so you don't wire up discovery, tokens, or certificates by hand. For a
verified, end-to-end walkthrough (build the model, start the servers, launch
workers, monitor, stop, resume) follow the canonical CLI example:
[`examples/tiny_experiments/diloco/`](../../examples/tiny_experiments/diloco/README.md).

The short version, with a `forgather server` already running:

```bash
# 1. Start the parameter server (CPU-only) as a scheduled job, seeded from a
#    model you built first (see the example's "Construct the Model" step).
forgather diloco server --output-dir path/to/model --num-workers 2 -H 0.0.0.0

# 2. Launch N workers in one command — each a scheduled training job. The single
#    running server is auto-picked and --dataset defaults to cluster routing in
#    cluster mode (the in-process loader otherwise), so neither needs naming.
forgather submit --diloco --diloco-worker-count 2

# 3. Monitor, then stop cleanly (save every worker, checkpoint the server).
forgather diloco status --queues --watch
forgather diloco shutdown
```

Full flag detail and the discovery / locality rules are in the
[CLI reference](diloco-cli.md#running-through-the-forgather-server-detailed-reference).
The lower-level foreground path — starting each process by hand — is under
[Manual / low-level invocation](diloco-cli.md#manual--low-level-invocation-development--debugging),
useful for development, debugging, or when no Forgather server is available.

## How It Works

Each machine runs any existing Forgather trainer (single GPU, DDP, or pipeline)
as an independent "worker." Workers train locally for H steps using their inner
optimizer (e.g., AdamW), then synchronize with a central parameter server. The
server averages the workers' updates and applies an outer optimizer (SGD with
Nesterov momentum) to produce new global parameters that all workers adopt.

```
                    +-------------------+
                    |   DiLoCo Server   |
                    | (standalone proc) |
                    |                   |
                    | - Global params   |
                    | - Outer optimizer |
                    | - Worker registry |
                    +--------+----------+
                             |
                 HTTP over 1G Ethernet
                             |
         +-------------------+-------------------+
         |                   |                   |
   +-----+-----+      +-----+-----+      +-----+-----+
   |  Worker 0  |      |  Worker 1  |      |  Worker 2  |
   | (Machine A)|      | (Machine B)|      | (Machine C)|
   |            |      |            |      |            |
   | Pipeline   |      | Single GPU |      | DDP        |
   | Trainer    |      | Trainer    |      | Trainer    |
   | (4x 3090)  |      | (1x 4090)  |      | (2x A6000) |
   +------------+      +------------+      +------------+
```

### Synchronous Protocol

In the default synchronous mode, each round follows these steps:

1. Workers train locally for `sync_every` optimizer steps (the "inner loop")
2. Each worker computes pseudo-gradients: `global_params - local_params`
3. Workers submit pseudo-gradients to the server over HTTP
4. Server waits until all workers have submitted (synchronous barrier)
5. Server averages the pseudo-gradients across all workers
6. Server applies the outer optimizer step (SGD with Nesterov momentum)
7. Updated global parameters are returned to all workers
8. Workers load the new parameters and begin the next inner loop

### Asynchronous Protocol

In async mode (`--async`), the barrier is removed. Each worker submits
pseudo-gradients and receives updated global params immediately without waiting
for other workers. This is essential for heterogeneous clusters where machines
have different training speeds.

The server applies each worker's pseudo-gradients as they arrive. To mitigate
the momentum amplification problem caused by stale gradients, the server
supports **Delayed Nesterov (DN)** momentum and **Dynamic Local Updates (DyLU)**.

See [Async Mode](diloco-advanced.md#async-mode) for configuration details.

### Bandwidth Efficiency

Each sync round moves the full model twice: workers send their **pseudo-gradient**
up to the server (upload), and the server sends the new averaged **parameters**
back down (download). Either leg can be transported in bfloat16 — halving that
leg's bandwidth — and the fp32→bf16 cast can use **stochastic rounding (SR)** to
remain unbiased in expectation. This is governed by four server-authoritative
knobs (see [Wire precision](#wire-precision)); by default the upload is bf16 and
the download is fp32. With `sync_every=500`, a 1B parameter model transfers ~2 GB
every 500 training steps, achieving >97% compute utilization on 1 Gig Ethernet.

| Model Size | BF16 Size | Transfer Time (1 Gbps) | H=500 steps @ 1s/step | Utilization |
|------------|-----------|------------------------|----------------------|-------------|
| 150M       | 300 MB    | 2.4s                   | 500s compute         | 99.5%       |
| 1B         | 2 GB      | 16s                    | 500s compute         | 97%         |
| 7B         | 14 GB     | 112s                   | 500s compute         | 82%         |

### Wire precision

Both transport legs are controlled by four **server-authoritative** knobs (set on
`forgather diloco server`; every worker adopts them from `/info` at registration,
so the whole group shares one wire format). The defaults reproduce the historical
behavior — bf16 upload, fp32 download.

| Knob | Server flag | Default | Effect |
|---|---|---|---|
| `upload_dtype` | `--upload-dtype {fp32,bf16}` | `bf16` | worker→server pseudo-gradient dtype |
| `upload_sr` | `--upload-sr` | off | stochastic-round the fp32→bf16 upload cast |
| `download_dtype` | `--download-dtype {fp32,bf16}` | `fp32` | server→worker averaged-params dtype (`bf16` halves the return leg) |
| `download_sr` | `--download-sr` | off | stochastic-round the fp32→bf16 download cast |

`--no-bf16` is a deprecated alias for `--upload-dtype fp32`. **Stochastic
rounding** (SR) routes the narrowing cast through the same
`fp32_to_bf16_stochastic_round` the bf16 optimizers use, keeping it unbiased in
expectation so sub-ULP signal survives across many rounds (it only applies to an
fp32→bf16 cast; it is a no-op when the source is already bf16).

**Lineage.** Low-precision communication of the **upload** leg (the
pseudo-gradient / outer gradient) is established prior art: *OpenDiLoCo* first
all-reduced it in FP16 "without noticeable performance hit," and *Streaming
DiLoCo* swept the outer-gradient precision through **bf16/fp8/fp4** with "no sign
of performance regression … even at the billion scale." Both compress only the
upload. The **download** leg — broadcasting the *averaged parameters* back in
bf16 — is not covered by that work; the
[`diloco_lowprec`](../../examples/tiny_experiments/diloco_lowprec/README.md)
experiment finds bf16 download (± SR) essentially lossless on a small Llama at
~1B tokens. See [References](#references).

### Bulk transport

How the bulk legs (pseudo-gradients up, averaged weights down) are serialized and
moved is independent of the wire precision above, and likewise server-authoritative
(advertised via `/info`, adopted by every worker).

| Knob | Server flag | Default | Effect |
|---|---|---|---|
| wire codec | `--wire-format {pickle,safetensors}` | `pickle` | `safetensors` drops pickle for an explicit typed, zero-copy frame; same format as on-disk checkpoints |
| transport | `--grpc` | off (HTTP) | serve the bulk legs over a streaming gRPC listener instead of the HTTP control port |

- **`--wire-format safetensors`** removes pickle from the wire (no arbitrary-code
  deserialization) and makes every tensor's dtype/shape explicit. The codec is
  negotiated, so a mixed old/new fleet stays interoperable; the upload also stamps
  the codec per request.
- **`--grpc`** moves the bulk legs onto an HTTP/2 streaming listener (chunked, with
  backpressure), advertised via `/info`; workers negotiate it and fall back to HTTP
  if a server doesn't offer it. It **supersedes** `--bulk-cleartext` (gRPC is the
  single bulk fast-path). The control plane (register / heartbeat / `/info`) stays
  on HTTP. The gRPC listener **follows the control-plane TLS posture**: a TLS server
  (a CA-provisioned cluster) runs gRPC over TLS too, with the worker authenticating
  by **bearer over the encrypted channel**; a cleartext server runs gRPC cleartext
  (trusted-LAN). gRPC TLS has no `CERT_OPTIONAL` equivalent, so the bulk plane
  authenticates by bearer rather than the control plane's mTLS-or-bearer — the
  worker always holds the token, and TLS still provides encryption + server
  authentication. Best paid off on large models / slow links, where the streaming +
  framing wins matter; for tiny experiments the HTTP default is fine.

Both knobs are available wherever a DiLoCo server is launched: the direct
`forgather diloco server` CLI, a scheduled server job (`forgather diloco server`
through the forgather server, which threads them onto the spawned argv), and the
webui's **DiLoCo Server** modal (a wire-format selector + a gRPC toggle in its
security section). A running server's negotiated transport is surfaced in the
webui DiLoCo panel.

## References

**DiLoCo and direct lineage**

- Douillard et al., "DiLoCo: Distributed Low-Communication Training of Language Models" ([arXiv:2311.08105](https://arxiv.org/abs/2311.08105))
- Douillard et al., "DiPaCo: Distributed Path Composition" (2024)
- Liu et al., "Asynchronous Local-SGD Training for Language Modeling" (2024) — Async DiLoCo, Delayed Nesterov, DyLU
- Jaghouar, Ong & Hagemann, "OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training" (2024, [arXiv:2407.07852](https://arxiv.org/abs/2407.07852)) — first FP16 all-reduce of the pseudo-gradient (the origin of low-precision *upload* communication in the DiLoCo family)
- Douillard et al., "Streaming DiLoCo with Overlapping Communication" (2025, [arXiv:2501.18512](https://arxiv.org/abs/2501.18512)) — fragment-based staggered sync; §2.4 sweeps the outer-gradient (upload) communication precision through bf16/fp8/fp4 with no observed regression. (Neither paper compresses the server→worker *download* of averaged weights — that is what `download_dtype=bf16` adds.)
- Charles et al., "Communication-Efficient Language Model Training Scales Reliably and Robustly: Scaling Laws for DiLoCo" ([arXiv:2503.09799](https://arxiv.org/abs/2503.09799))
- TorchFt (Meta) — fault-tolerant distributed training library

**Local SGD, slow momentum, and the outer optimizer**

- Wang, Tantia, Ballas & Rabbat, "SlowMo: Improving Communication-Efficient Distributed SGD with Slow Momentum" (ICLR 2020, [arXiv:1910.00643](https://arxiv.org/abs/1910.00643)) — the slow/outer-momentum update DiLoCo's outer optimizer generalizes
- Lin, Stich, Patel & Jaggi, "Don't Use Large Mini-Batches, Use Local SGD" (ICLR 2020, [arXiv:1808.07217](https://arxiv.org/abs/1808.07217))
- Zhang, Lucas, Ba & Hinton, "Lookahead Optimizer: k steps forward, 1 step back" (NeurIPS 2019, [arXiv:1907.08610](https://arxiv.org/abs/1907.08610)) — the single-worker local-SGD analog

**Generalization / flat minima** (why local SGD can train *better*, not just cheaper)

- Gu, Lyu, Huang & Arora, "Why (and When) does Local SGD Generalize Better than SGD?" (ICLR 2023, [arXiv:2303.01215](https://arxiv.org/abs/2303.01215)) — sharpness-reduction drift; needs small LR + long training
- Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization" (SWA, [arXiv:1803.05407](https://arxiv.org/abs/1803.05407))
- Keskar et al., "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima" (ICLR 2017, [arXiv:1609.04836](https://arxiv.org/abs/1609.04836))

A worked, reproducible illustration of these effects (DiLoCo overtaking a DDP
baseline at a longer budget; single-worker local-SGD generalizing better) is in
the [canonical example](../../examples/tiny_experiments/diloco/README.md#extended-sweep-budget-sync-interval-and-single-worker-local-sgd).
