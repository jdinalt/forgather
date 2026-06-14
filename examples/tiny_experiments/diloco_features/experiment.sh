#!/usr/bin/env bash
#
# experiment.sh — the reproduction-study run matrix for diloco_features.
#
# Each arm is a from-scratch DiLoCo run trained to the SAME total-token budget
# (the server's --token-budget global stop, not a per-worker --max-steps) at a
# realistic sync interval (H=100). Equal total tokens across arms makes them
# comparable on the total-tokens axis (= the async paper's "Total Local
# Updates") even under a per-worker speed spread. Every arm keeps its worker +
# server logs and a /status snapshot under runs/<arm>/ so analysis/ can harvest
# the loss trajectory, the actual total_tokens, the grace-batch histogram and
# the staleness distribution.
#
# Async staleness is induced with a small per-step JITTER
# (DILOCO_DEBUG_STEP_JITTER, the same value on every worker, seeded per-worker
# so they differ): the randomness decorrelates the workers' phase so they drift
# out of lock-step and produce real async staleness (~workers-1) while keeping
# the same *average* speed (no slow-worker solo tail). DyLU instead uses a fixed
# per-worker speed SPREAD (DILOCO_DEBUG_STEP_DELAY) calibrated to a realistic
# mixed-GPU cluster (~2x slowest/fastest, 4090+3090-style) — DyLU adapts to
# average-speed differences, which jitter (equal average speed) wouldn't show.
#
# gRPC + safetensors, torch.compile on by the config default. Every run starts
# from an identical pristine master copy and a fixed seed (--seed); only the
# feature flag (+ the jitter/spread debug throttle) varies. 4 workers = 4
# GPUs/run, so arms run SERIALLY on one server (:8512).
#
# Usage:
#   ./experiment.sh validate         # short plumbing check — each feature FIRES
#   ./experiment.sh run              # the full 10-arm matrix (~9-12 h, 4x4090)
#   ./experiment.sh run <arm-name>   # re-run a single arm by name
#
# Then: python analysis/harvest.py && python analysis/plot_experiment.py
#       python analysis/staleness.py && python analysis/streaming.py
#       python analysis/dylu_control.py && python analysis/worker_scaling.py
#       python analysis/verify_baseline.py
#       (python analysis/grace_batches.py is a VALIDATE-only mechanism check)
#
# Before any long batch: run `./experiment.sh validate` (each feature fires, incl.
# the grace MECHANISM check via analysis/grace_batches.py — grace is validate-only,
# NOT a study arm), then confirm mean staleness ~= workers-1 on the async arms
# (analysis/staleness.py). Do NOT start the headline tier until both pass.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PRISTINE="$REPO/models/small_llama_features_master"
RESULTS="$HERE/runs"
CONFIG="default.yaml"

H=100                          # inner sync interval (steps between outer steps)
SEED=42                        # fixed; matches the config default (single run/arm)
JITTER=0.15                    # per-step jitter (s) for async phase-decorrelation (-> staleness ~3)
GRACE_S="${GRACE_S:-0.5}"      # grace window (s) — VALIDATE ONLY (v_grace mechanism check; grace is not a study arm)

# Per-worker fixed delays (s) for the DyLU speed-spread arms. Calibrate from the
# MEASURED base step time (from validate) so the slowest worker's step time is
# ~2x the fastest (a realistic 4090+3090 mix). e.g. base~0.10s -> 0/.03/.06/.10.
DYLU_SPREAD=(0 0.03 0.06 0.10)

# Token budgets (global stop). 4w = 520M tok/worker x4; the 2w arms run to the
# matched-total half-budget so the 2w-vs-4w scaling compares like total tokens.
BUDGET_4W=2080000000
BUDGET_2W=1040000000

MODE="${1:-run}"
ONLY="${2:-}"                  # optional: run a single arm by name

case "$MODE" in
  validate) MAXSTEPS_ARGS=(--max-steps 120); COMPILE_ARGS=(--compile no) ;;
  run)      MAXSTEPS_ARGS=();                COMPILE_ARGS=() ;;
  *) echo "usage: $0 {validate|run} [arm-name]" >&2; exit 2 ;;
esac

mkdir -p "$RESULTS"

log()  { printf '\033[1;36m[exp]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[exp]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[exp]\033[0m %s\n' "$*" >&2; }

server_alive_on() { forgather diloco status --diloco-server "127.0.0.1:$1" 2>/dev/null | grep -qiE 'Status:\s*running'; }
wait_server_ready() { for ((i=0; i<60; i++)); do server_alive_on "$1" && return 0; sleep 3; done; return 1; }
shutdown_on() { forgather diloco shutdown --diloco-server "127.0.0.1:$1" >/dev/null 2>&1; }

wait_no_servers() {
  for ((i=0; i<40; i++)); do
    forgather diloco servers 2>/dev/null | grep -qiE 'alive' || return 0
    forgather diloco shutdown --force >/dev/null 2>&1; sleep 3
  done; return 1
}

# wait_jobs_terminal <max_polls> <job-id...>
wait_jobs_terminal() {
  local tries=$1; shift
  for ((i=0; i<tries; i++)); do
    local list pending=0; list="$(forgather job list 2>/dev/null)"
    for id in "$@"; do
      echo "$list" | grep -F -- "$id" | grep -qiE 'done|failed|error|cancelled|aborted' || pending=1
    done
    [[ $pending -eq 0 ]] && return 0
    sleep 20
  done; return 1
}

# start_one <name> <port> <nw> <budget> -- <server-flags...>
# Fresh pristine master per run; --token-budget is the global stop (budget 0 =>
# open-ended, used by validate). gRPC+safetensors is the default transport; an
# arm that passes its own --wire-format (the transport check) overrides it.
start_one() {
  local name="$1" port="$2" nw="$3" budget="$4"; shift 4
  [[ "${1:-}" == "--" ]] && shift
  local out="$REPO/models/small_llama_feat_$name"
  rm -rf "$out"; cp -r "$PRISTINE" "$out"
  local budget_args=(); [[ "$budget" -gt 0 ]] && budget_args=(--token-budget "$budget")
  # Default transport unless the arm specifies its own wire format.
  local have_transport=0 a
  for a in "$@"; do [[ "$a" == "--wire-format" ]] && have_transport=1; done
  local transport_args=(--grpc --wire-format safetensors)
  [[ $have_transport -eq 1 ]] && transport_args=()
  log "server '$name' on :$port (-n $nw, budget=$budget) — $*"
  forgather diloco server -o "$out" --port "$port" -n "$nw" --save-every 0 \
    "${transport_args[@]}" --sync-every "$H" --run-name "$name" \
    "${budget_args[@]}" "$@" >/dev/null 2>&1
  wait_server_ready "$port" || { err "server '$name' (:$port) never ready"; return 1; }
  log "server '$name' ready on :$port"
}

# submit_sync <port> <nw> -> nw workers in one submit, no throttle
submit_sync() {
  forgather -t "$CONFIG" submit --diloco --diloco-server "127.0.0.1:$1" \
    --diloco-worker-count "$2" --seed "$SEED" \
    "${MAXSTEPS_ARGS[@]}" "${COMPILE_ARGS[@]}" 2>/dev/null \
    | grep -oE 'q_[0-9]+_[0-9a-f]+'
}

# submit_async <port> <nw> -> nw single-worker submits (fixed ids w0..) + jitter.
# Fixed ids keep the per-worker jitter seed (hence the phase-decorrelation
# pattern) identical across async arms, so they differ only in the feature under
# test. The jitter is seeded per worker id, so the workers still decorrelate.
submit_async() {
  local port="$1" nw="$2" k
  for ((k=0; k<nw; k++)); do
    forgather -t "$CONFIG" submit --diloco --diloco-server "127.0.0.1:$port" \
      --diloco-worker-count 1 --worker-id "w$k" --heartbeat-interval 5 --seed "$SEED" \
      "${MAXSTEPS_ARGS[@]}" "${COMPILE_ARGS[@]}" \
      --env "DILOCO_DEBUG_STEP_JITTER=$JITTER" 2>/dev/null | grep -oE 'q_[0-9]+_[0-9a-f]+'
  done
}

# submit_dylu <name> <port> <nw> -> nw single-worker submits with a fixed speed spread
submit_dylu() {
  local name="$1" port="$2" nw="$3" k
  for ((k=0; k<nw; k++)); do
    forgather -t "$CONFIG" submit --diloco --diloco-server "127.0.0.1:$port" \
      --diloco-worker-count 1 --worker-id "${name}-w${k}" --heartbeat-interval 5 --seed "$SEED" \
      "${MAXSTEPS_ARGS[@]}" "${COMPILE_ARGS[@]}" \
      --env "DILOCO_DEBUG_STEP_DELAY=${DYLU_SPREAD[$k]}" 2>/dev/null | grep -oE 'q_[0-9]+_[0-9a-f]+'
  done
}

# sample_status_live <out_dir> <port> — write <out_dir>/status.json IFF the
# snapshot has registered (live) workers. Per-worker staleness + the grace
# histogram only exist while workers are running; once the token budget stops them
# they deregister (workers:{}), so a post-terminal capture loses them. We keep the
# LAST live sample (steady-state staleness, which is what we want).
sample_status_live() {
  local out="$1" port="$2" s
  s="$(forgather diloco status --diloco-server "127.0.0.1:$port" --json 2>/dev/null)"
  [[ -n "$s" ]] || return 1
  printf '%s' "$s" | python3 -c '
import sys, json
try: d = json.load(sys.stdin)
except Exception: sys.exit(1)
st = d["status"] if isinstance(d.get("status"), dict) else d  # orchestrator wraps under "status"
sys.exit(0 if (st.get("workers") or {}) else 1)
' 2>/dev/null || return 1
  printf '%s' "$s" > "$out/status.json"
}

# wait_terminal_sampling <name> <port> <tries> <wid...> — poll jobs to terminal
# while sampling /status, so runs/<name>/status.json ends up holding the last LIVE
# snapshot (workers still registered) rather than a post-terminal empty one.
wait_terminal_sampling() {
  local name="$1" port="$2" tries="$3"; shift 3
  local out="$RESULTS/$name"; mkdir -p "$out"
  local i
  for ((i=0; i<tries; i++)); do
    sample_status_live "$out" "$port"
    local list pending=0; list="$(forgather job list 2>/dev/null)"
    for id in "$@"; do
      echo "$list" | grep -F -- "$id" | grep -qiE 'done|failed|error|cancelled|aborted' || pending=1
    done
    [[ $pending -eq 0 ]] && return 0
    sleep 12
  done; return 1
}

# capture_one <name> <port> <worker-id...> — logs + (fallback) status snapshot.
# status.json is normally the last LIVE sample wait_terminal_sampling already
# wrote; we only take a post-terminal snapshot if no live sample was captured
# (total_tokens + grace histogram still survive there; per-worker staleness won't).
capture_one() {
  local name="$1" port="$2"; shift 2
  local out="$RESULTS/$name" n=0
  mkdir -p "$out"
  [[ -f "$out/status.json" ]] || forgather diloco status --diloco-server "127.0.0.1:$port" --json > "$out/status.json" 2>/dev/null \
    || warn "status snapshot for '$name' failed (no live sample, server may have stopped)"
  local sid; sid="$(forgather diloco servers 2>/dev/null | grep -F ":$port" | grep -oE 'q_[0-9]+_[0-9a-f]+' | head -1)"
  [[ -n "$sid" ]] && forgather diloco logs "$sid" > "$out/server.log" 2>&1
  for wid in "$@"; do forgather job dump "$wid" > "$out/worker$n.log" 2>&1; n=$((n+1)); done
  log "captured runs/$name/ ($# worker log(s) + status.json)"
}

# do_one <name> <kind> <nw> <budget> -- <server-flags...>   kind: sync|async|dylu
do_one() {
  local name="$1"
  if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then return 0; fi
  local kind="$2" nw="$3" budget="$4"; shift 4
  [[ "${1:-}" == "--" ]] && shift
  local port=8512
  wait_no_servers || { err "a prior server won't clear; skipping '$name'"; return 1; }
  start_one "$name" "$port" "$nw" "$budget" -- "$@" || return 1
  local ids
  case "$kind" in
    sync)  ids="$(submit_sync "$port" "$nw")" ;;
    async) ids="$(submit_async "$port" "$nw")" ;;
    dylu)  ids="$(submit_dylu "$name" "$port" "$nw")" ;;
  esac
  local wids=($ids)
  if [[ ${#wids[@]} -ne $nw ]]; then
    err "exp '$name': expected $nw worker ids, got ${#wids[@]}: ${wids[*]}"
    shutdown_on "$port"; return 1
  fi
  log "submitted $nw workers for '$name'"
  wait_terminal_sampling "$name" "$port" 2000 "${wids[@]}" || warn "exp '$name' not all terminal in time"
  capture_one "$name" "$port" "${wids[@]}"
  shutdown_on "$port"; sleep 6
}

trap 'echo; warn "interrupted — shutting down :8512"; shutdown_on 8512; exit 130' INT TERM

log "MODE=$MODE  H=$H  seed=$SEED  jitter=$JITTER  grace_s=$GRACE_S  ${ONLY:+arm=$ONLY}"
log "budgets: 4w=$BUDGET_4W  2w=$BUDGET_2W  dylu_spread=(${DYLU_SPREAD[*]})"

# ---------------------------------------------------------------------------
# validate — short plumbing check (max-steps 120, compile off). Confirms each
# feature FIRES (not "no crash"): block-faithful streaming fragmentation, the
# grace flush, and the token-budget save_and_stop relay. Grep the captured logs
# per README.md (e.g. NoBlockPlanError must NOT appear; "grace" flush lines must;
# the budget arm must show the save_and_stop relay before step 120).
# ---------------------------------------------------------------------------
if [[ "$MODE" == validate ]]; then
  do_one v_stream  sync  4 0          -- --num-fragments 5 --fragment-assignment strided
  do_one v_grace   async 4 0          -- --async --dn-buffer-size 4 --grace-period "$GRACE_S"
  # Small budget + a generous step cap so the budget (not max-steps) stops it,
  # exercising the save_and_stop relay. Lower the budget if it doesn't fire.
  MAXSTEPS_ARGS=(--max-steps 5000)
  do_one v_budget  sync  4 3000000    -- --sync-every "$H"
  log "VALIDATE DONE — grep runs/v_*/ : fragments engaged, grace flush, budget relay."
  exit 0
fi

# ---------------------------------------------------------------------------
# run — the 10-arm matrix (single run per arm, from scratch, token-budget stop)
# ---------------------------------------------------------------------------
# 1  Sync baseline (4w) — reference.
do_one baseline        sync  4 "$BUDGET_4W" -- --sync-every "$H"

# 2-4  Streaming: block-boundary fragments, assignment A/B at two grains.
do_one stream_str2     sync  4 "$BUDGET_4W" -- --sync-every "$H" --num-fragments 2 --fragment-assignment strided
do_one stream_seq2     sync  4 "$BUDGET_4W" -- --sync-every "$H" --num-fragments 2 --fragment-assignment sequential
do_one stream_str5     sync  4 "$BUDGET_4W" -- --sync-every "$H" --num-fragments 5 --fragment-assignment strided

# 5  Async no-DN control (jitter) — expect divergence (DN off).
do_one async_nodn      async 4 "$BUDGET_4W" -- --sync-every "$H" --async

# 6  Async + DN N=k=4 (jitter) — the headline: async+DN ~ sync (per token)?
do_one async_dn4       async 4 "$BUDGET_4W" -- --sync-every "$H" --async --dn-buffer-size 4

# NOTE: grace is NOT a study arm — its payoff is wall-clock tail-reduction in a
# heterogeneous/large-N pool, which this homogeneous rig doesn't have and loopback
# can't measure. It is validated functional in `validate` (v_grace) only; the real
# demonstration is the two-population/WAN future work. See README §3.5.

# 7  Async + DN, speed spread, DyLU OFF (control) — average-speed heterogeneity.
do_one dylu_off        dylu  4 "$BUDGET_4W" -- --sync-every "$H" --async --dn-buffer-size 4

# 8  Async + DN + DyLU, same speed spread — DyLU cuts staleness (A/B vs #7).
do_one dylu_on         dylu  4 "$BUDGET_4W" -- --async --dylu --dylu-base-sync-every "$H" --dn-buffer-size 4

# 9  Sync 2 workers — token-efficiency scaling vs 4w sync (half-budget total).
#    Also serves as the gRPC+safetensors side of the transport check (#10).
do_one baseline_2w     sync  2 "$BUDGET_2W" -- --sync-every "$H"

# 10 Wire/transport check (2w): HTTP+pickle vs the gRPC+safetensors baseline_2w.
#    Internal lossless-transport check (UNRELATED to the paper's fp4 claim).
do_one wire_http_pk    sync  2 "$BUDGET_2W" -- --sync-every "$H" --wire-format pickle

log "MATRIX DONE — harvest: python analysis/harvest.py && python analysis/plot_experiment.py"
log "  then: grace_batches.py, staleness.py, streaming.py, dylu_control.py, worker_scaling.py, verify_baseline.py"
