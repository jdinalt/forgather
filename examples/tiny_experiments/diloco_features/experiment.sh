#!/usr/bin/env bash
#
# experiment.sh — real-budget feature-comparison sweep for diloco_features.
#
# Runs each feature config to a REAL token budget at a realistic sync interval
# (H=100) and keeps every run's logs so the loss trajectory can be harvested and
# compared against the synchronous baseline. **4 workers** — async stress scales
# with worker count, and 2 workers can't produce meaningful staleness.
#
# Async is exercised by adding a small per-step JITTER (DILOCO_DEBUG_STEP_JITTER,
# the SAME on every worker, seeded per-worker so they differ): the randomness
# decorrelates the workers' phase so they drift out of lock-step and produce real
# async staleness (~N-1), while keeping the same *average* speed — so there is no
# slow-worker solo tail. This is a controlled way to *measure async's impact*,
# not a faithful real-deployment async (which would also want real device-timing
# variance + the server-side grace period, issue #221). DyLU instead uses a fixed
# per-worker speed SPREAD (DILOCO_DEBUG_STEP_DELAY), since DyLU adapts to
# average-speed differences and co-terminates the workers.
#
# Budget: config default (~16k steps/worker). gRPC + safetensors, torch.compile
# on by the config default. Every run starts from an identical pristine master
# copy and the config's fixed seed; only the feature flag (+ the jitter/spread
# debug throttle) varies. 4 workers = 4 GPUs/run, so runs are SERIAL.
#
# Usage:
#   ./experiment.sh validate   # short (max-steps 120, compile off) — plumbing check
#   ./experiment.sh run        # the real sweep (~8 h on 4 idle 4090s)
#
# Then: python analysis/harvest.py && python analysis/plot_experiment.py
#       python analysis/dn_sweep.py && python analysis/verify_baseline.py

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PRISTINE="$REPO/models/small_llama_features_master"
RESULTS="$HERE/runs"
CONFIG="default.yaml"
H=100
NW=4                 # workers per run (= GPUs/run)
JITTER=0.15          # per-step jitter (s) for async phase-decorrelation (-> staleness ~3)
DYLU_SPREAD=(0 0.05 0.10 0.15)   # per-worker fixed delays (s) for the DyLU run (speed spread)

MODE="${1:-run}"
case "$MODE" in
  validate) MAXSTEPS_ARGS=(--max-steps 120); COMPILE_ARGS=(--compile no) ;;
  run)      MAXSTEPS_ARGS=();                COMPILE_ARGS=() ;;
  *) echo "usage: $0 {validate|run}" >&2; exit 2 ;;
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

# start_one <name> <port> <server-flags...> -> fresh-master server with -n NW
start_one() {
  local name="$1" port="$2"; shift 2
  local out="$REPO/models/small_llama_feat_$name"
  rm -rf "$out"; cp -r "$PRISTINE" "$out"
  log "server '$name' on :$port (-n $NW) — $*"
  forgather diloco server -o "$out" --port "$port" -n "$NW" --save-every 0 \
    --grpc --wire-format safetensors --sync-every "$H" --run-name "$name" "$@" >/dev/null 2>&1
  wait_server_ready "$port" || { err "server '$name' (:$port) never ready"; return 1; }
  log "server '$name' ready on :$port"
}

# submit_sync <port> -> NW workers, no throttle
submit_sync() {
  forgather -t "$CONFIG" submit --diloco --diloco-server "127.0.0.1:$1" \
    --diloco-worker-count "$NW" "${MAXSTEPS_ARGS[@]}" "${COMPILE_ARGS[@]}" 2>/dev/null \
    | grep -oE 'q_[0-9]+_[0-9a-f]+'
}

# submit_async <port> -> NW workers with fixed ids w0..w(N-1) + the same jitter.
# Fixed ids make the per-worker jitter seed (hence the phase-decorrelation
# pattern) identical across the async/DN-sweep runs, so they differ only in the
# DN buffer size — a clean comparison. The jitter is seeded *per worker id*, so
# the four workers within a run still decorrelate from each other.
submit_async() {
  local port="$1" k
  for ((k=0; k<NW; k++)); do
    forgather -t "$CONFIG" submit --diloco --diloco-server "127.0.0.1:$port" \
      --diloco-worker-count 1 --worker-id "w$k" --heartbeat-interval 5 \
      "${MAXSTEPS_ARGS[@]}" "${COMPILE_ARGS[@]}" \
      --env "DILOCO_DEBUG_STEP_JITTER=$JITTER" 2>/dev/null | grep -oE 'q_[0-9]+_[0-9a-f]+'
  done
}

# submit_dylu <name> <port> -> NW workers individually with a fixed speed spread
submit_dylu() {
  local name="$1" port="$2" k
  for k in "${!DYLU_SPREAD[@]}"; do
    forgather -t "$CONFIG" submit --diloco --diloco-server "127.0.0.1:$port" \
      --diloco-worker-count 1 --worker-id "${name}-w${k}" --heartbeat-interval 5 \
      "${MAXSTEPS_ARGS[@]}" "${COMPILE_ARGS[@]}" \
      --env "DILOCO_DEBUG_STEP_DELAY=${DYLU_SPREAD[$k]}" 2>/dev/null | grep -oE 'q_[0-9]+_[0-9a-f]+'
  done
}

capture_one() {  # <name> <port> <worker-id...>
  local name="$1" port="$2"; shift 2
  local out="$RESULTS/$name" n=0
  mkdir -p "$out"
  local sid; sid="$(forgather diloco servers 2>/dev/null | grep -F ":$port" | grep -oE 'q_[0-9]+_[0-9a-f]+' | head -1)"
  [[ -n "$sid" ]] && forgather diloco logs "$sid" > "$out/server.log" 2>&1
  for wid in "$@"; do forgather job dump "$wid" > "$out/worker$n.log" 2>&1; n=$((n+1)); done
  log "captured runs/$name/ ($# worker log(s))"
}

# do_one <name> <kind> <server-flags...>   kind: sync|async|dylu  (all on port 8512, serial)
do_one() {
  local name="$1" kind="$2"; shift 2
  local port=8512
  wait_no_servers || { err "a prior server won't clear; skipping '$name'"; return 1; }
  start_one "$name" "$port" "$@" || return 1
  local ids
  case "$kind" in
    sync)  ids="$(submit_sync "$port")" ;;
    async) ids="$(submit_async "$port")" ;;
    dylu)  ids="$(submit_dylu "$name" "$port")" ;;
  esac
  local wids=($ids)
  if [[ ${#wids[@]} -ne $NW ]]; then
    err "exp '$name': expected $NW worker ids, got ${#wids[@]}: ${wids[*]}"
    shutdown_on "$port"; return 1
  fi
  log "submitted $NW workers for '$name'"
  wait_jobs_terminal 2000 "${wids[@]}" || warn "exp '$name' not all terminal in time"
  capture_one "$name" "$port" "${wids[@]}"
  shutdown_on "$port"; sleep 6
}

trap 'echo; warn "interrupted — shutting down :8512"; shutdown_on 8512; exit 130' INT TERM

log "MODE=$MODE  NW=$NW  H=$H  jitter=$JITTER  budget=$([[ ${#MAXSTEPS_ARGS[@]} -gt 0 ]] && echo "${MAXSTEPS_ARGS[*]}" || echo 'config default (~16k/worker)')"

do_one baseline    sync  --sync-every "$H"
do_one streaming   sync  --sync-every "$H" --num-fragments 2
do_one async       async --sync-every "$H" --async
do_one async_dn_b4 async --sync-every "$H" --async --dn-buffer-size 4
do_one async_dn_b8 async --sync-every "$H" --async --dn-buffer-size 8
do_one async_dn_b16 async --sync-every "$H" --async --dn-buffer-size 16
do_one dylu        dylu  --async --dylu --dylu-base-sync-every "$H" --dn-buffer-size 4

log "SWEEP DONE — harvest: python analysis/harvest.py && python analysis/plot_experiment.py && python analysis/dn_sweep.py"
