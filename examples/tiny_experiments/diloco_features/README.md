# Validating Async and Streaming DiLoCo at Small Scale

A controlled reproduction study of two communication-efficient extensions to
DiLoCo — **asynchronous local-SGD** (Delayed Nesterov + Dynamic Local Updates +
a server-side grace period; [arXiv:2401.09135](https://arxiv.org/abs/2401.09135))
and **Streaming DiLoCo** (block-boundary fragment sync;
[arXiv:2501.18512](https://arxiv.org/abs/2501.18512)) — measured against
synchronous DiLoCo as the reference, on a single small Llama (34.4M) trained from
scratch on Fineweb-Edu across 4 workers.

> **Status.** The experimental *design* below is final, the harness is built
> ([`experiment.sh`](experiment.sh), [`analysis/`](analysis)), and the
> from-scratch run matrix (§3.7, including the DN-buffer depth sweep) has been
> run — the **Results** tables and figures are filled. The design was fixed as a
> pre-registration *before* the GPU time was spent: methodology, controls, and the
> pass/fail gates predate the numbers. One run per arm at a single small scale:
> **suggestive, not a benchmark.** A warm-start / pretrain-budget sweep (§6), to
> test whether a pretrained start closes the from-scratch async gap, is pending.

All commands assume you are in this project directory:

```bash
cd examples/tiny_experiments/diloco_features
```

---

## Abstract

DiLoCo cuts inter-worker communication by synchronizing only every `H` local
steps instead of every step. Three further extensions reduce or *restructure*
that communication: **streaming** spreads each sync across the local-training
window as block-boundary fragments instead of one all-at-once burst; **async**
drops the cross-worker barrier (so fast workers never wait on a straggler), made
stable by a **Delayed-Nesterov (DN)** outer-optimizer buffer, optionally tightened
by a **grace period** (a soft barrier that coalesces near-simultaneous
submissions) and **Dynamic Local Updates (DyLU)** (per-worker sync rate matched to
throughput). Each buys something — smoother bandwidth, no straggler waits,
tolerance for uneven hardware — at some cost in convergence. This study measures
those costs at small scale, reproducing the two source papers' qualitative
*convergence* trends under a real worker count and synthetically-induced staleness,
from a random initialization.

**Scope, stated plainly.** On 4 identical 4090s, synchronous DiLoCo is genuinely
fine — there is no straggler tail, no mixed-speed GPU pool, no bandwidth wall. The
hardware is a test rig, not a deployment. Async, grace, and DyLU exist to solve
problems this rig *does not have* (heterogeneous "slow + fast" GPU populations,
large worker counts, constrained WAN links). So this study **validates that the
mechanisms function and reproduce the papers' convergence trends** — it does **not**
demonstrate their real-world *payoff*, which lives on the **wall-clock** axis under
conditions absent here (the async paper's own headline, Fig 2: "matches sync per
update step, **significantly surpasses it in wall-clock time**"). We do not
manufacture synthetic stragglers to fake a benefit we can't measure; that
demonstration is deferred to a two-population (e.g. RTX 4090 + 3090) / WAN setup
(Future Directions). Every arm below is tagged a convergence-trend reproduction or
a mechanism-validation under a synthetic-but-measurable condition.

**Practitioner TL;DR.** What each knob buys, what it costs, and what we measured
(the "Measured here" column is the from-scratch run matrix, §3.7; from a random
init at this small scale — suggestive, single seed):

| Knob | Flag (on `diloco server`) | Buys you | Costs | Measured here |
|---|---|---|---|---|
| **Streaming** | `--num-fragments N` `--fragment-assignment {strided,sequential}` | smooths bandwidth — fragments sent across the compute window, no all-at-once burst | a small, steady convergence cost; strided ≥ sequential, gap grows at finer grain | small per-token gap (+0.06–0.14 eval); **faster wall-clock**; strided N=5 closest + fastest |
| **Async + DN** | `--async --dn-buffer-size N` | drops the barrier — fast workers don't wait on stragglers | needs the DN buffer (required with `--async`); some convergence cost under staleness | **does not reach sync from scratch**; depth is a non-monotonic lever — **N=8 optimum** (+0.25), N=4 (+0.80) under-buffered, N=16 (+0.51) over |
| **Grace period** | `--grace-period S` | a brief, opportunistic wait so a finished worker can coalesce with a near-simultaneous finisher — cuts the slow-worker *tail* in a heterogeneous / large-N pool | only pays off when the tail is real (large N, mixed speeds); its benefit is **wall-clock**, not convergence | **not a study arm** — the rig has no tail to cut; validated functional only, real benefit → Future Directions |
| **DyLU** | `--dylu` | matches each worker's sync rate to its throughput, cutting staleness | only helps heterogeneous / unevenly-loaded workers | **neutral here** (Δ +0.02, within noise) — the rig's induced spread is too mild; mechanism functional, payoff → Future Directions |

These are all **server-authoritative**: you select them with flags on `forgather
diloco server`, and every worker adopts them from the server's `/info`. The same
worker config (`default.yaml`) runs all of them. They are features of the **HTTP
sync backend** — streaming in particular is only implemented there (the
shared-memory and collective backends raise `NotImplementedError` on fragment
sync).

---

## 1. Introduction

DiLoCo (Distributed Low-Communication training) replaces the per-step all-reduce
of data-parallel training with an inner/outer optimization: each worker trains a
local replica with AdamW for `H` steps, then the workers' *pseudo-gradients*
(local drift, `global − local`) are averaged and applied by an outer
SGD-with-Nesterov-momentum step. Communication drops by a factor of `H`. The
sibling [`../diloco`](../diloco) project walks through the mechanism and shows the
headline result at a longer budget — DiLoCo's infrequent sync acts as a
*regularizer* and can match or beat an all-reduce baseline outright.

That baseline is **synchronous**: every worker waits at a barrier each sync round.
Two lines of follow-up work attack the two remaining costs:

1. **The barrier.** On heterogeneous hardware or a busy cluster, a synchronous
   round runs at the speed of the slowest worker. *Asynchronous* local-SGD
   ([arXiv:2401.09135](https://arxiv.org/abs/2401.09135)) drops the barrier:
   each worker submits its pseudo-gradient when ready and immediately pulls the
   latest global weights. The hazard is *staleness* — a submission computed
   against weights that have since moved — which a naive outer step handles
   badly. The paper's fixes are the **Delayed-Nesterov** outer optimizer, a
   **grace period**, and **Dynamic Local Updates**.

2. **The burst.** A synchronous round transfers the whole model at once; the link
   sits idle through the `H`-step window, then is slammed at sync time.
   *Streaming DiLoCo* ([arXiv:2501.18512](https://arxiv.org/abs/2501.18512))
   partitions the model into block-boundary **fragments** and overlaps each
   fragment's transfer with ongoing compute, so the link is used steadily.

Both papers validate at scale and (for async) largely in a *finetuning* regime
warm-started from a pretrained checkpoint. **This study asks the narrower
question:** do the qualitative findings hold at small scale, at a real worker
count, under real staleness, **trained from scratch**? It is deliberately scoped
to *synchronous DiLoCo as the reference* versus *async + streaming*; the
DiLoCo-vs-DDP token-efficiency comparison is out of scope here (covered at larger
scale in [`../../pretrain/small-llm`](../../pretrain/small-llm): 4×DDP vs 4-worker
sync DiLoCo, 150M, ~11× Chinchilla — cross-referenced in §3).

Like the other [Tiny Experiments](..), this does double duty: every arm drives
the full orchestrated path (scheduler → param server → workers), so the same
matrix that *measures* what each knob costs also serves as an end-to-end exercise
of it.

---

## 2. Background

### 2.1 Synchronous DiLoCo

Each of `k` workers holds a replica. For `H` inner steps it trains locally
(AdamW). At the sync round, worker *i* computes its pseudo-gradient `Δᵢ = θ_global
− θ_local,i`; the server averages them, `Δ̄ = (1/k) Σ Δᵢ`, and applies one outer
step with SGD + Nesterov momentum (here outer LR 0.7, momentum 0.9). Workers then
pull the updated `θ_global` and continue. The barrier makes every outer step an
average over *exactly* `k` fresh pseudo-gradients.

### 2.2 Asynchronous local-SGD and the Delayed-Nesterov buffer

Drop the barrier and each pseudo-gradient arrives *alone* and possibly *stale*
(computed against an older `θ_global`). Applying full-LR Nesterov momentum to each
lone submission the instant it arrives is unstable — pseudo-gradients are applied
sequentially rather than aggregated, and under real staleness the run can diverge.

The **Delayed-Nesterov (DN)** outer optimizer (paper Algorithm 3, with `c = 0`)
decouples the momentum update from the per-submission update. Maintaining a
running accumulator `Δ` and a momentum buffer `m`, for each submission `g`:

- apply a small immediate gradient step: `θ ← θ − ε·g/N`, and accumulate `Δ ← Δ + g`;
- every `N`-th submission, fold the accumulator into momentum and apply it:
  `m ← β·m + Δ/N`, `θ ← θ − ε·β·m`, then reset `Δ ← 0`.

`N` is the **DN buffer size**. At `N = 1` this reduces to plain Nesterov; as `N`
grows, each momentum step aggregates more submissions, behaving more like a
synchronous "aggregate, then one step." Crucially the buffer is **O(model)
memory** — a single running `Δ` and `m`, not `N` retained pseudo-gradients — so
its memory cost is independent of `N` (this corrects an earlier implementation
that retained an `N`-deep list; the faithful Algorithm-3 form holds only the
accumulator). With the memory penalty removed, the expectation under real
staleness at `N = k` is **async + DN ≈ sync**, not the "N must be several×
workers" behavior an O(N·model) list-based buffer exhibited.

### 2.3 Grace period (soft barrier)

A pure async server applies one outer step per submission. The **grace period**
is a wall-clock soft barrier: when a submission arrives, the server waits up to
`S` seconds and *coalesces* any other submissions that land in the window into a
**single** outer step (one DN tick over their average). The window is anchored to
the *first* arrival. This reduces the number of stale, lonely outer steps without
reintroducing a hard barrier — provided `S` is tuned so it coalesces a *few* near-
simultaneous stragglers (2–3 of `k`), **not** all `k` every round (which would be
synchronous DiLoCo in disguise — see the methodology in §3.3).

### 2.4 Dynamic Local Updates (DyLU)

Under heterogeneous worker speeds, a fast worker completes many more `H`-step
cycles than a slow one between the slow worker's submissions, so the slow worker's
pseudo-gradient is very stale. **DyLU** sets each worker's `sync_every` inversely
to its measured throughput, so all workers return pseudo-gradients at roughly the
same wall-clock cadence — reducing staleness at the source instead of buffering it
away. It only helps when worker speeds genuinely differ; on identical workers it
has nothing to adapt.

### 2.5 Streaming fragments

Streaming partitions the model into `N` fragments **at transformer-block
boundaries** (derived from the model's `_no_split_modules`, so a fragment is a
whole number of blocks — never a split inside attention/MLP) and syncs one
fragment per portion of the window, overlapping transfer with compute. Blocks are
assigned to fragments either **strided** (round-robin: fragment *f* gets blocks
*f, f+N, f+2N, …*) or **sequentially** (contiguous slabs). The paper reports
strided slightly better, with the gap growing at finer grain / smaller fragments.
Our 34.4M model has **10 transformer blocks**, giving faithful (block-divisible)
fragment counts `N ∈ {2, 5}` (5 blocks/fragment and 2 blocks/fragment
respectively); embeddings attach to the first fragment and the LM head to the
last.

### 2.6 Why from scratch

The source papers mostly study a *finetuning* regime warm-started from a
pretrained checkpoint, and the async paper inherits the original DiLoCo's caution
of pretraining ~24K steps before the local-SGD phase. We train **everything from a
random initialization**, for three reasons:

1. **The original DiLoCo sweep favors it.** DiLoCo swept the pretrain budget
   `{0, 12K, 24K, 48K}` steps and found performance grew *monotonically worse*
   with more pretraining — `0` (from scratch) was best, and "from scratch" not
   only matched but led the other curves.
2. **The 24K figure is inherited caution, not optimization.** Post-local-SGD
   (PyTorch) defaults to a 500-step warmup; the async paper's 24K is in that
   lineage. At 150M with seq 256 / batch 512 it is ~1× Chinchilla — not the
   4×/12.5-billion-token checkpoint a seq-1024 reading would imply.
3. **From scratch *is* the test.** Plain local-SGD is known to struggle from a
   cold start; DiLoCo does not. Whether async preserves that random-init
   robustness is itself a small contribution — and the cleaner, more reproducible
   protocol.

**Pre-registration (regime confound).** For *streaming* the from-scratch choice is
clean — the Streaming paper is itself from-scratch, Chinchilla-optimal. For *async*
it is a genuine hazard: the async paper's tiny ≈sync gaps are measured *late in
finetuning from a 24K-step checkpoint*, where pseudo-gradients are small and
well-aligned. From scratch, early pseudo-gradients are large and poorly-aligned —
exactly where sequentially-applied stale gradients do the most damage. So a
from-scratch async **≈sync *failure* on the DN arm is INCONCLUSIVE** (a regime
confound), **not** a refutation of DN. We pre-register that reading now, and a
reactive fallback: if the from-scratch async headline shows a surprising gap, add a
warm-start diagnostic for the DN arm and report it (a dedicated async warm-start /
pretrain-budget sweep is in Future Directions).

---

## 3. Experiments

### 3.1 Fixed setup (all arms, from scratch)

All 10 arms use **4 workers** (= 4 GPUs/run, so arms run **serially** on one param
server, `:8512`), from a fresh pristine master copy and a fixed seed; only the
server flags (+ the async jitter / DyLU spread worker env) differ. Training stops
on the server's **`--token-budget`** for every arm (not a per-worker `--max-steps`),
so all arms train to equal total tokens — the basis of the total-tokens axis. The
heartbeat-driven one-shot stop may slightly overshoot, so analysis aligns the axis
to the **actual** harvested `total_tokens`.

**Hyperparameters, and why these values** (every number has a reason; cross-check
against [`experiment.sh`](experiment.sh)):

| Quantity | Value | Why this value |
|---|---|---|
| Model | small Llama, **34.4M** (26.2M non-embedding), **10 blocks** | small enough for a 4-GPU study; 10 blocks ⇒ block-divisible fragment counts `N ∈ {2,5}`. Chinchilla-optimal ≈ **525M tokens** (20 × 26.2M non-embedding; reported by `forgather -t small.yaml model construct`) |
| Workers `k` | **4** | the async paper's main worker count (`k = 4`); enough to produce staleness ≈ `k−1` ≈ 3 |
| Sync interval `H` | **100** | matches the sibling [`../diloco`](../diloco) project's `h100` reference; DiLoCo's regime (sync every ~100 local steps) |
| Outer optimizer | SGD-Nesterov, **LR 0.7, mom 0.9** | DiLoCo's outer-optimizer settings (the `diloco server` defaults) |
| Inner optimizer | AdamW, **lr 2.07e-4** (peak) | per-worker local optimizer. The lr is `base_lr` 1.5e-4 scaled for the 32,768-tok batch by the **sqrt rule** (`lr_alpha 0.5`, ref batch 16,384) → 2.07e-4; **WSD** schedule (warmup **1606** steps, `min_lr` 2.07e-5) |
| Batch × seq | **8 × 4096** = 32,768 tok/step/worker | `per_device_train_batch_size = 8` (set in `small.yaml`), packed 4096-token sequences; note this is **2× the 16,384-token LR reference**, which is what upscales the inner lr by √2 (next row) |
| Token budget | **2B total** | **~4× Chinchilla** (the model's Chinchilla-optimal is 525M tokens, from its 26.2M non-embedding params) — chosen to capture async's **longer-term dynamics** (DiLoCo-family benefits emerge over a longer budget), which the measured runtime makes affordable. At 4 workers ≈ 500M tok/worker ≈ 150 sync rounds |
| DN buffer `N` | **4** (= `k`) | the async paper's main config; tests the derived `N = k ≈ sync` prediction (§2.2) |
| Fragments `N` | **{2, 5}** | the only block-divisible counts ≤ 10 blocks (5 / 2 layers per fragment); N=5 is the fine grain where striding should show |
| Seed | **42** | fixed; one run/arm; varies data-order/dropout, not init (§3.5) |
| Jitter `J` | **0.15 s** | tuned so the *measured* staleness lands ≈ `k−1` — the staleness **gate** (§3.3) is the check, not the exact value |
| DyLU spread | **0 / .06 / .12 / .18 s** | calibrated to a **~2×** slowest/fastest ratio (RTX 4090 + 3090-style): max delay ≈ the measured ~0.18s steady-state step time, so the slowest worker (0.18+0.18) ≈ 2× the fastest |
| Transport | gRPC + safetensors | the fast/safe path; lossless (already validated against HTTP+pickle in the sibling project — not re-tested here) |
| `torch.compile` | on | real-run default (`--compile no` is smoke-only) |

### 3.2 The primary axis: total tokens = "Total Local Updates"

The async paper plots convergence against **Total Local Updates** — local
optimizer steps, which (at fixed batch/sequence) are directly proportional to
tokens consumed. The token-budget global stop trains every arm to equal total
tokens, so **eval loss vs total tokens** is the faithful primary axis. The
outer/global-step count appears only as an optional secondary diagnostic (how much
each arm coalesces), never as a competing axis.

### 3.3 Methodology: inducing staleness vs inducing heterogeneity

The GPUs here are identical, so we synthesize the two conditions async and DyLU
respectively target, using debug-only per-step worker throttles (delivered via
`submit --env`; see Appendix B):

- **Phase jitter** (`DILOCO_DEBUG_STEP_JITTER`, the *same* value on every worker,
  seeded *per worker* so they differ): a random `uniform(0, J)` s sleep per step.
  The randomness decorrelates the workers' phase so they drift out of lock-step
  and the server sees genuine **staleness ≈ k − 1**, while every worker keeps the
  same *average* speed — so there is no slow-worker solo tail confounding the
  result. This is how the async staleness arms (5–6) are run (`J = 0.15`). It
  *measures async's impact*; it is not a faithful real deployment (which would also
  have real device-timing variance).

- **Speed spread** (`DILOCO_DEBUG_STEP_DELAY`, a *fixed* per-worker delay): creates
  genuine *average-speed* differences. The DyLU arms (7–8) use a spread calibrated
  from the measured base step time to a **~2× slowest/fastest ratio** — a realistic
  mixed-GPU cluster (e.g. RTX 4090 + RTX 3090), which is DyLU's real target. (Jitter,
  equal average speed, would give DyLU nothing to adapt to.) This spread is
  *synthetic* — it validates DyLU's mechanism, it is not the real heterogeneity
  payoff (that's Future Directions).

**Staleness is a pass/fail gate.** The async conclusions *depend on* the workers
actually running out of lock-step, so we verify it from the captured `/status`
snapshot (`server sync_round − worker.last_sync_server_round`,
[`analysis/staleness.py`](analysis/staleness.py)) — with a round-robin subtlety
worth stating precisely. For `k` **equal-average-speed** workers (the jitter arms),
at any instant the workers occupy staleness `{0, 1, …, k−1}`, one per phase of the
`k`-step sync cycle. So the **snapshot mean is `(k−1)/2`** (= 1.5 at k=4) — the
signature of *full* decorrelation (a synchronized run collapses it toward 0) —
while the async paper's "staleness ≈ `k−1`" is the **per-submission** staleness, i.e.
how stale a gradient is *when applied*, which is the **max** of the cycle (= 3 at
k=4). We therefore gate the jitter arms (`async_nodn`, `async_dn4`) on **snapshot
mean ≈ (k−1)/2** *before* the headline tier and report the per-submission **max ≈
k−1**. (More jitter can't raise the snapshot mean above `(k−1)/2` at equal average
speed — the workers stay round-robin — so a *low* mean means weak jitter, not a low
cap.) The DyLU arms use a *delay spread* (not jitter), so they're **not**
round-robin; `dylu_off` isn't gated at a fixed value — it's the control, and the
reducer `dylu_on` succeeds by landing at **lower** mean staleness than `dylu_off`.

**Grace is validated, not measured.** Grace is **not a study arm** (see §3.5 and
the scope note in the Abstract) — its payoff is wall-clock tail-reduction in a
heterogeneous / large-N pool, which 4×4090 doesn't have and loopback can't measure.
The `validate` run includes a short async+grace probe, and
[`analysis/grace_batches.py`](analysis/grace_batches.py) confirms the **mechanism
works to the paper's spec** — it *coalesces* near-simultaneous finishers (a
grace-batch histogram with mass at 2+) **and** *proceeds immediately* when none
arrive within `S` (mass at 1). That is the whole grace check here; the real
demonstration is Future Directions.

### 3.4 Run matrix (10 arms, single run each)

**Every arm shares one reference command** — a 4-worker sync DiLoCo run to the
token budget — and differs *only* in the `diloco server` flags (and, for the async
arms, a worker `--env` throttle). The reference:

```bash
# Server (one per arm), started fresh from a pristine master copy:
forgather diloco server -o <fresh master> -n 4 --save-every 0 \
    --grpc --wire-format safetensors --sync-every 100 --token-budget 1B \
    --run-name <arm>  [+ the arm's server flag(s)]

# 4 workers against it (the same default.yaml worker, max_steps=-1 so the
# server's budget is the sole stop):
forgather -t default.yaml submit --diloco --diloco-server 127.0.0.1:8512 \
    --diloco-worker-count 4 --seed 42  [+ the arm's --env, for async/DyLU]
```

The matrix is the deltas on that base. Each arm is tagged **[trend]** (a measurable
convergence-trend reproduction) or **[mech]** (a mechanism-validation under a
synthetic-but-measurable condition).

| # | Arm | Kind | `diloco server` delta | worker `--env` | Tests |
|---|---|---|---|---|---|
| 1 | `baseline` | — | *(none — the reference)* | — | reference |
| 2 | `stream_str2` | trend | `--num-fragments 2 --fragment-assignment strided` | — | streaming, coarse |
| 3 | `stream_seq2` | trend | `--num-fragments 2 --fragment-assignment sequential` | — | assignment A/B vs #2 |
| 4 | `stream_str5` | trend | `--num-fragments 5 --fragment-assignment strided` | — | finer grain (strided edge) |
| 5 | `async_nodn` | trend | `--async` | jitter 0.15 | no-DN control (expect divergence) |
| 6 | `async_dn4` | trend | `--async --dn-buffer-size 4` | jitter 0.15 | async + DN, paper default N=k=4 |
| 7 | `async_dn8` | trend | `--async --dn-buffer-size 8` | jitter 0.15 | **DN-depth sweep** (optimum) |
| 8 | `async_dn16` | trend | `--async --dn-buffer-size 16` | jitter 0.15 | DN-depth sweep (over-buffered) |
| 9 | `dylu_off` | mech | `--async --dn-buffer-size 4` | delay spread (~2×) | DyLU-off control |
| 10 | `dylu_on` | mech | `--async --dylu --dylu-base-sync-every 100 --dn-buffer-size 4` | delay spread (~2×) | DyLU cuts staleness (A/B vs #9) |

Async arms (5–10) all run to the token budget. Staleness arms (5–8) use **jitter**
(equal average speed, staleness ≈ 3, no speed-spread confound); DyLU arms (9–10)
use the **synthetic delay spread** (average-speed heterogeneity is the point).
Arms 6–8 form a **DN-buffer depth sweep** at fixed staleness (N ∈ {4, 8, 16}),
isolating buffer depth from every other knob — the paper's main config is N = k =
4, and the sweep tests whether that default is well-tuned for the from-scratch +
staleness regime (§3.7 finding 2: it is not — N=8 is the empirical optimum).

**Deliberately not in the matrix** (kept the study focused on the two papers'
communication mechanisms): grace (validate-only — §3.5); a 2-vs-4-worker
token-efficiency scaling arm (DiLoCo's own batch-scaling penalty is covered in the
sibling [`../diloco`](../diloco) / [`../../pretrain/small-llm`](../../pretrain/small-llm)
projects, not here); and a wire/transport lossless check (gRPC+safetensors was
already validated against HTTP+pickle in the sibling project — it was never an
experiment, just a transport-correctness validation).

### 3.5 Metrics, controls, statistics

- **Metrics.** Eval loss and **perplexity** = `exp(loss)`; eval loss vs total
  tokens (primary). Per-arm mean **staleness** (the gate; reducer semantics per
  §3.3). Throughput / wall-clock from the server JSONL. (Grace's batch histogram is
  a `validate`-only mechanism check, §3.3 — not a study metric.)
- **Controls (each isolates one knob, everything else fixed).** DN on/off (5 vs 6);
  DyLU on/off (7 vs 8, same spread + N=4); streaming assignment (2 vs 3); grain
  (2 vs 4).
- **Statistics.** **One run per arm, no multi-seed.** Re-running is a poor use of
  GPU time: seed noise is typically small and the effects we want are larger than
  it. An effect visible only by averaging seeds is, by construction, a *small-effect
  finding* (flag and understand it), not a re-run trigger; a large effect won't
  change with more seeds. Reported framing throughout: **"suggestive at small
  scale."**

### 3.6 What this can and cannot validate

**Can** (relative *convergence* trends, single small scale): *whether* async + DN
reaches sync at N = k *per token*, and how it moves with buffer depth (with the
regime caveat in §2.6 — a from-scratch *failure* is inconclusive); streaming's
small steady cost and strided ≥ sequential; *whether* DyLU reduces staleness under
a synthetic spread (A/B, mechanism-validation); and, suggestively, whether a
from-scratch async≈sync result would extend DiLoCo's random-init robustness to
async. (Answers: §3.7.)

**Cannot** (stated plainly): **any wall-clock / scheduling *benefit*** — grace's
tail-reduction, async's no-barrier win, streaming's peak-smoothing — because the rig
is homogeneous and loopback (the async paper's win is *per wall-clock*, Fig 2; we
can only show the convergence *tie*, not the wall-clock win); the fp4 / "400× bits"
claim (no fp4 codec; wire dtypes fp32/bf16 only); exact peak-bandwidth absolute
numbers (structural ~1/N argument only); the papers' finetuning regime (we train
from scratch, §2.6); deterministic device-trace heterogeneity levels (synthetic
jitter/delay model); and large scale. These benefits are Future Directions.

### 3.7 Results

Single run per arm (4 workers, 2B tokens, H=100, from scratch). Eval loss is the
final eval; perplexity is the best (lowest) eval perplexity; "vs sync" is the
final-eval delta from the sync baseline. The harvest + plot scripts regenerate
the table and figures into `assets/` ([`analysis/harvest.py`](analysis/harvest.py)).

| Configuration | eval loss | perplexity | vs sync | mean staleness | notes |
|---|---|---|---|---|---|
| Sync DiLoCo (baseline) | 2.859 | 17.4 | — | — | reference |
| + Streaming (strided N=2) | 2.992 | 19.9 | +0.134 | — | |
| + Streaming (sequential N=2) | 3.001 | 20.1 | +0.142 | — | assignment A/B |
| + Streaming (strided N=5) | 2.919 | 18.5 | +0.060 | — | finer grain |
| Async without DN | 8.323 | 4118 | +5.46 | 3.0 | diverged (control) |
| Async + DN (N=4) | 3.661 | 38.9 | +0.802 | 1.5 | paper default (N=k) |
| Async + DN (N=8) | **3.107** | **22.4** | **+0.249** | 1.5 | **DN-sweep optimum** |
| Async + DN (N=16) | 3.373 | 29.2 | +0.514 | 1.3 | over-buffered |
| Async + DN, spread, DyLU off | 3.559 | 35.1 | +0.700 | 1.5 | control |
| Async + DN + DyLU, spread | 3.583 | 36.0 | +0.724 | 2.5 | A/B vs control |

Three findings:

**1. Streaming trades per-token quality for wall-clock — and at fine grain nearly
erases the trade.** All streaming arms are slightly worse *per token* than the
sync baseline (+0.06 to +0.14 eval loss), as expected: fragment-wise outer steps
mean each parameter is updated against a slightly staler global view. But they all
finish *faster in wall-clock* (44.5–45.1 min vs the baseline's 50.5) because the
per-fragment barriers overlap communication with compute. On the fair
loss-vs-wall-clock axis (`walltime_comparison.png`) the finest grain, strided
N=5, both lands closest per token (+0.060) *and* finishes first — it essentially
matches the baseline's loss trajectory in real time. The strided-vs-sequential
assignment A/B is a wash at N=2 (+0.009 in favor of strided), so fragment *grain*
matters more than assignment here.

**2. The DN buffer stabilizes async, and its depth is a real, non-monotonic
lever.** Without the Delayed-Nesterov buffer, async with induced staleness ~3
diverges (eval 8.32, the run trips the divergence detector ~3% in) — the control
that motivates DN. With DN the run is stable, but depth matters more than the
paper's `N = k` default suggests: sweeping N ∈ {4, 8, 16} (`dn_sweep.png`) is
**non-monotonic**, with **N=8 the optimum** — it closes roughly half the
from-scratch gap to the baseline (+0.249 vs N=4's +0.802), while N=16 *regresses*
(+0.514). The sweet spot sits near ~2× the mean staleness: too shallow
under-absorbs stale pseudo-gradients, too deep over-delays the Nesterov momentum.
The paper's `N = k = 4` is thus *under*-buffered for the from-scratch + staleness
regime, but "deeper is always better" is false. (Single seed; the N=8 < N=16
ordering carries a noise caveat, but the N=4 → N=8 improvement is large.)

**3. Async from scratch does not reach the sync baseline — the central open
question.** Even the best async arm (N=8, +0.249) trails the sync baseline, and
the heterogeneity arms (DyLU off/on) sit at +0.70–0.72. This is consistent with
the from-scratch premise being the hard part for async specifically (§2.6): sync
DiLoCo reaches the baseline from scratch, but async — even DN-stabilized and
depth-tuned — does not, in this budget. The source papers warm-start their async
runs from a pretrained checkpoint; whether that closes the gap is the natural next
experiment (a warm-start / pretrain-budget sweep, §6).

**On DyLU:** at this induced speed spread DyLU was **neutral** — dylu_on (+0.724)
vs its dylu_off control (+0.700) differ by 0.024, within single-seed noise, and
the snapshot staleness ordering even inverts. This matches the design's own caveat
(§3.5–3.6): DyLU's payoff is tail-latency reduction under genuine, large-scale
worker heterogeneity, which a homogeneous 4×4090 loopback rig with a small
injected delay spread barely exercises. We report the mechanism as *functional*
(adaptive per-worker `sync_every`; dylu_on takes more sync rounds, 760 vs 616) but
its quality benefit is not measurable here.

Figures (regenerated by the named scripts):

- **`loss_comparison.png`** / **`training_health.png`** — the headline eval-loss
  comparison and the train-loss/grad-norm health check
  ([`analysis/plot_experiment.py`](analysis/plot_experiment.py)).
- **`streaming.png`** — fragment count + assignment, strided vs sequential
  ([`analysis/streaming.py`](analysis/streaming.py)).
- **`dn_sweep.png`** — DN-buffer depth sweep (N=4/8/16 vs baseline), showing the
  non-monotonic optimum at N=8
  ([`analysis/dn_sweep.py`](analysis/dn_sweep.py)).
- **`walltime_comparison.png`** — eval loss vs *relative wall-clock time*, the
  fair axis for the streaming comm/compute-overlap trade
  ([`analysis/plot_walltime.py`](analysis/plot_walltime.py)).
- **`grace_hist.png`** — *`validate`-only*: the grace coalescing histogram, used
  to confirm the mechanism works to spec (coalesces near-simultaneous finishers,
  proceeds immediately when alone). Not a results figure
  ([`analysis/grace_batches.py`](analysis/grace_batches.py)).
- **`dylu_control.png`** — DyLU off vs on at a fixed speed spread
  ([`analysis/dylu_control.py`](analysis/dylu_control.py)).

---

## 4. Related Work

**DiLoCo** (Douillard et al., 2023) introduced the inner-AdamW / outer-SGD-Nesterov
split with periodic pseudo-gradient averaging; the sibling [`../diloco`](../diloco)
project reproduces its regularization-at-longer-budget result. **Asynchronous
Local-SGD** (Liu et al., 2024;
[arXiv:2401.09135](https://arxiv.org/abs/2401.09135)) identifies staleness as the
core obstacle to dropping the barrier and contributes the Delayed-Nesterov outer
optimizer, DyLU, and the grace period. We evaluate DN (arms 5–6) and DyLU (arms
7–8); the grace period — an Algorithm-2 input the paper never ablates, whose payoff
is wall-clock tail-reduction — is validated functional only and deferred (§3.5).
**Streaming DiLoCo** (Douillard et al., 2025;
[arXiv:2501.18512](https://arxiv.org/abs/2501.18512)) overlaps fragment
communication with compute and additionally quantizes the wire transfer (fp4/E3M0);
we evaluate the fragmentation (arms 2–4) but not the quantization codec. The
DiLoCo-vs-DDP token-efficiency question — orthogonal to the communication-schedule
knobs studied here — is covered at larger scale in
[`../../pretrain/small-llm`](../../pretrain/small-llm).

---

## 5. Conclusions

*Filled after the run matrix.* **Scope boundary (restated):** these are *convergence*
conclusions on a homogeneous test rig — they validate that the mechanisms function
and reproduce the papers' per-token trends; the wall-clock *benefits* (grace's tail,
async's no-barrier, streaming's peak-smoothing) are out of reach here and are
Future Directions. The study is structured to support (or refute) a small set of
pre-registered hypotheses, each with a fixed control:

1. **Async + DN matches sync at N = k = 4** *per token* under induced staleness
   (arm 6 vs 1), with the no-DN control (arm 5) diverging.
   *Result: partially refuted.* The no-DN control diverges as predicted (DN is
   necessary), but the Algorithm-3 N = k = 4 form is **not** sufficient from
   scratch — it trails sync by +0.80 eval. Buffer depth turned out to be a
   non-monotonic lever (the §3.4 sweep): **N=8 is the optimum** (+0.25, ~half the
   gap closed), N=16 regresses. Per the §2.6 regime caveat, the from-scratch gap
   is inconclusive as a refutation of the paper (which warm-starts), and motivates
   the warm-start sweep (§6).
2. **Streaming costs little and strided ≥ sequential**, the assignment gap visible
   at the finer N=5 grain (arms 2–4 vs 1).
   *Result: supported.* Small per-token cost (+0.06–0.14 eval), strided N=5
   closest (+0.060); on the wall-clock axis streaming finishes faster and N=5
   nearly matches the baseline trajectory. The N=2 assignment A/B is a wash
   (+0.009), so grain dominates assignment at this scale.
3. **DyLU cuts staleness under a synthetic speed spread** (arm 10 vs the DyLU-off
   control arm 9) — a mechanism-validation; the real mixed-GPU benefit is future work.
   *Result: not measurable here.* DyLU was neutral (Δ +0.02, within seed noise);
   the rig's mild induced spread doesn't create enough heterogeneity for the
   adaptive `sync_every` to pay off. Mechanism functional, benefit → §6.
4. **Suggestively:** a from-scratch async≈sync result would extend DiLoCo's
   random-init robustness to async (the whole study trains from scratch).
   *Result: open.* We did **not** see async≈sync from scratch; whether a
   pretrained warm start recovers it is the headline open question (§6).

Any headline result ambiguous within plausible seed noise is reported as a
documented small-effect finding, not re-run.

---

## 6. Future Directions

- **Real WAN, two-population run (headline) — and grace's true home.** The
  synthetic knobs here become real at once on a genuinely distributed, heterogeneous
  cluster: the author has RTX 3090 machines at a separate physical site (the "slow"
  population to the local 4090s' "fast"). Running this matrix over a WAN turns the
  synthetic conditions real simultaneously — real 3090-vs-4090 heterogeneity (DyLU's
  actual target), a real bandwidth-constrained link (streaming's peak-smoothing),
  real device-timing staleness, and a real slow-worker **tail** — and the tail is
  exactly what the **grace period** cuts (a finished worker briefly coalesces with a
  near-simultaneous finisher, else proceeds immediately). On this setup the payoff is
  measurable where it isn't on loopback: **eval loss / perplexity vs wall-clock
  time** (the async paper's Fig-2 axis), where sync pays the tail and async+grace
  do not. This is DiLoCo's actual deployment scenario.
- **DN-buffer sweep** (N vs k): is N = workers genuinely sufficient post-fix, and
  how does convergence move as N → ∞ (toward synchronous)?
- **Async warm-start / pretrain-budget sweep:** test the from-scratch premise (§2.6)
  directly for the async path.
- **Heterogeneity-level sweep:** beyond the single ~2× spread, map DyLU's benefit
  across spreads.
- **Bandwidth-constrained link simulation:** measure real token throughput / time
  saved (not just the structural ~1/N peak argument) under a throttled link.
- **fp4 / E3M0 streaming codec** and **larger scale.**

---

## 7. References

- Douillard et al., *"DiLoCo: Distributed Low-Communication Training of Language
  Models"* (2023). [arXiv:2311.08105](https://arxiv.org/abs/2311.08105)
- Liu et al., *"Asynchronous Local-SGD Training for Language Modeling"* (2024).
  [arXiv:2401.09135](https://arxiv.org/abs/2401.09135) — async, the
  Delayed-Nesterov buffer, the grace period, and DyLU.
- Douillard et al., *"Streaming DiLoCo with overlapping communication: Towards a
  Distributed Free Lunch"* (2025).
  [arXiv:2501.18512](https://arxiv.org/abs/2501.18512) — fragmented overlapped sync.

The authoritative Forgather reference for the trainer is
[`docs/trainers/diloco.md`](../../../docs/trainers/diloco.md).

---

## Appendix A — Reproducing it

Prerequisites: a running **forgather server** + **dataset server** (the standard
cluster setup the `../diloco` walkthrough establishes), and a **master model** the
param server initialises from, built once:

```bash
forgather -p ../../models/llama -t small.yaml \
    model --device cpu --save-checkpoint --safetensors \
    --output-dir ../../../models/small_llama_features_master \
    construct
```

Then run the matrix through the scheduler — [`experiment.sh`](experiment.sh) starts
each param server with the right flags, submits the workers, waits, captures the
worker + server logs and a live `/status` snapshot under `runs/<arm>/`, and tears
down, copying the master fresh per run (identical init). 4 workers = 4 GPUs/run, so
runs are **serial**:

```bash
./experiment.sh validate          # short plumbing check (each feature FIRES)
./experiment.sh run               # the full 10-arm matrix (~9-11 h on 4x4090s)
./experiment.sh run async_dn8     # or a single arm by name
./experiment.sh run <arm-name>    # re-run a single arm by name
python analysis/harvest.py && python analysis/plot_experiment.py  # main comparison
python analysis/staleness.py           # the async staleness gate (should-be-stale vs reducer)
python analysis/streaming.py           # fragment count + assignment (strided vs sequential)
python analysis/dylu_control.py        # the DyLU off-vs-on eval overlay
python analysis/grace_batches.py       # VALIDATE-only: grace mechanism check (on v_grace)
```

**Order of execution (do not skip the gates).** `validate` (each feature fires:
fragments engaged with no `NoBlockPlanError` fallback; the token-budget
`save_and_stop` relay; and the **grace mechanism check** on the `v_grace` run —
`grace_batches.py` confirms it coalesces near-simultaneous finishers *and* proceeds
immediately when alone, all-k fraction below the guardrail) → **staleness gate**
(`staleness.py`: should-be-stale arms mean ≈ workers − 1; the reducer `dylu_on`
below its control) → then the headline tier. **No long GPU runs until validate +
the staleness gate pass.** (Grace is validated here only — not a study arm; its
real demonstration is the two-population / WAN future work.)

For a fast smoke (does each feature engage, in minutes rather than hours) without
the convergence study, [`harness.sh`](harness.sh) runs short versions:
`./harness.sh all` or `./harness.sh streaming`.

## Appendix B — Inducing async timing and heterogeneity (debug knobs)

The GPUs here are identical, but the async arms need workers to submit at
*different times* (real staleness) and DyLU needs workers at different *speeds*.
Two debug-only worker env knobs provide this, delivered via `submit --env`:

- **`DILOCO_DEBUG_STEP_JITTER=J`** — sleep a random `uniform(0, J)` s per step, from
  a *per-worker-seeded* RNG. Set the **same** `J` on every worker (the async arms
  use `0.15`): the randomness decorrelates the workers' phase so they drift out of
  lock-step (staleness ~ workers − 1) while keeping the same *average* speed — no
  slow-worker solo tail. This is what makes the async arms actually async.
- **`DILOCO_DEBUG_STEP_DELAY=D`** — a *fixed* per-step delay, set **differently per
  worker** (the DyLU arms use a spread calibrated to ~2× slowest/fastest) to create
  the average-speed differences DyLU adapts to.

Both are debug-only — they throttle real training and are never set in production.
`submit --env KEY=VALUE` (repeatable) forwards env to the scheduled worker process
via `job_params.extra_env`; honoured on plain `--diloco` worker submits only (the
`--global` compose and collective paths reject it rather than silently dropping it).
The per-worker RNG seed matters: a *shared* seed would give every worker the
identical jitter sequence — they'd stay phase-locked and the jitter would do
nothing.

## Appendix C — Confirming a feature is active

Each feature is selected at the server, so the worker echoes the settings it
adopted from `/info` once at startup — a quick way to confirm your config took:

```
DiLoCoCallback: using server settings sync_every=100 up=bf16 down=fp32 \
    dylu=False num_fragments=2 transport=grpc(127.0.0.1:NNNNN) wire=safetensors
```

| Feature | Where to see it |
|---|---|
| wire / transport | the `transport=…(…) wire=…` echo matches your flags; server log: `grpc_enabled`, `wire_format`, `gRPC bulk listener on …` |
| streaming | `num_fragments=N` in the echo; the server (`--verbose-sync`) advances per-fragment rounds; **no** `NoBlockPlanError` fallback in the server log (= block-faithful fragmentation engaged) |
| async | server log shows async mode; sync round advances as submissions arrive, so workers can reach **different** rounds |
| DN buffer | the server applies the momentum outer step only every `dn-buffer-size` submissions (visible with `--verbose-sync`) |
| grace | the server logs a grace flush coalescing *k* submissions into one outer step; the `/status` snapshot carries `grace_batch_histogram` |
| token budget | the server relays `save_and_stop` once the aggregated `total_tokens` crosses `--token-budget` |
| DyLU | the server emits a per-worker `recommended_sync_every`; the slow worker logs a `DyLU adjusted sync_every …` line below the fast worker's |

## Appendix D — Files

- `templates/configs/default.yaml` — the single small-Llama DiLoCo worker config
  (extends `projects/small.yaml`, `enable_diloco=True`); every feature is chosen by
  server flags, not here.
- `experiment.sh` — the run-matrix driver (`validate` / `run` / `run <arm>`),
  token-budget global stop, `GRACE_S` env knob for the grace window.
- `harness.sh` — the fast functional smoke driver (incl. a `token-budget` recipe).
- `analysis/` — `harvest.py`, `plot_experiment.py`, `streaming.py`,
  `dylu_control.py`, `staleness.py` (the gate), `grace_batches.py` (validate-only).
- `assets/` — `curves.csv` (the committed source of truth) + the plots
  (`loss_comparison.png`, `training_health.png`, `streaming.png`,
  `dylu_control.png`); `grace_hist.png`/`grace_hist.csv` are `validate`-only.
- `runs/` — captured per-run logs + `status.json` (gitignored scratch).
