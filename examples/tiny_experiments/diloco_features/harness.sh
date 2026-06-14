#!/usr/bin/env bash
#
# diloco_features test harness — exercises the less-travelled DiLoCo features
# (streaming, async, DN-buffer, DyLU) and the transport x wire-format matrix,
# all through the forgather scheduler (the orchestrated path is itself part of
# the test). Each run:
#
#   1. copies the pristine master to a fresh per-run scratch dir (so the diloco
#      server's checkpoint / shutdown saves never mutate the reference model and
#      every run starts from identical init);
#   2. starts a scheduled `forgather diloco server` with the run's feature flags;
#   3. submits worker(s) via `forgather submit --diloco` (scheduled);
#   4. waits for the worker job(s) to reach a terminal state;
#   5. captures the worker + server logs under runs/<name>/;
#   6. shuts the server down.
#
# Verification (asserting the feature ENGAGED, not just "no crash") is done by
# grepping the captured logs — see README.md for the per-feature signals.
#
# Usage:
#   ./harness.sh <recipe>     # one recipe (see the case dispatch at the bottom)
#   ./harness.sh all          # every recipe, in order
#
# Debug/validation harness, not a product surface. The DyLU run leans on
# DILOCO_DEBUG_STEP_DELAY (a debug-only per-step throttle in the worker, set via
# `submit --env`) to simulate heterogeneous worker speeds on identical GPUs.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PRISTINE="$REPO/models/small_llama_features_master"
SCRATCH_BASE="$REPO/models/.feat_runs"
RESULTS="$HERE/runs"
CONFIG="default.yaml"

mkdir -p "$RESULTS" "$SCRATCH_BASE"

log()  { printf '\033[1;36m[harness]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[harness]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[harness]\033[0m %s\n' "$*" >&2; }

# On Ctrl-C / kill, tear the current server (and its save-stopped workers) down
# so an interrupted run doesn't leave an orphaned server holding port 8512.
trap 'echo; warn "interrupted — shutting down server + workers"; shutdown_server; exit 130' INT TERM

wait_server_alive() {
  # Two-stage readiness: the registry reports STATE=alive as soon as the server
  # process starts, but its HTTP listener isn't bound until after the model
  # loads — so gate on `diloco status` actually answering (a live /info probe),
  # else the scheduler's backend-derivation query races the listener and fails
  # the worker job with "connection refused".
  for ((i=0; i<40; i++)); do
    if forgather diloco servers 2>/dev/null | grep -qiE 'alive' \
       && forgather diloco status 2>/dev/null | grep -qiE 'Status:\s*running'; then
      return 0
    fi
    sleep 3
  done
  return 1
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
    sleep 6
  done
  return 1
}

server_id() { forgather diloco servers 2>/dev/null | grep -oE 'q_[0-9]+_[0-9a-f]+' | head -1; }

# Block until the registry reports no live diloco server (port 8512 freed,
# registry settled). Each run reuses the default port, so consecutive runs must
# not overlap — an overlap makes `submit` ambiguous ("2 servers running") and
# the second server collide on the port.
wait_no_servers() {
  for ((i=0; i<40; i++)); do
    forgather diloco servers 2>/dev/null | grep -qiE 'alive' || return 0
    forgather diloco shutdown --force >/dev/null 2>&1
    sleep 3
  done
  return 1
}

shutdown_server() { forgather diloco shutdown >/dev/null 2>&1; wait_no_servers; }

fresh_scratch() {
  local dir="$SCRATCH_BASE/$1"
  rm -rf "$dir"; cp -r "$PRISTINE" "$dir"; echo "$dir"
}

# start_server <name> <num_workers> <server-flags...>
start_server() {
  local name="$1" nw="$2"; shift 2
  wait_no_servers || { err "a prior diloco server won't clear"; return 1; }
  local scratch; scratch="$(fresh_scratch "$name")"
  log "server '$name' (-n $nw): $*"
  forgather diloco server -o "$scratch" -n "$nw" --save-every 0 \
    --run-name "$name" "$@" 2>&1 | tail -2
  wait_server_alive || { err "server '$name' never alive"; return 1; }
  log "alive: $(forgather diloco servers 2>/dev/null | grep alive)"
}

# capture <name> <server-job-id> <worker-job-id...>
capture() {
  local name="$1" sid="$2"; shift 2
  local out="$RESULTS/$name" n=0
  mkdir -p "$out"
  forgather diloco logs "$sid" > "$out/server.log" 2>&1 || \
    forgather job dump "$sid" > "$out/server.log" 2>&1
  for wid in "$@"; do
    forgather job dump "$wid" > "$out/worker$n.log" 2>&1; n=$((n+1))
  done
  log "captured -> runs/$name/ (server.log + $# worker log(s))"
}

# Standard run: 2 workers in one submit.
# do_run <name> <max_steps> <compile yes|no> -- <server-flags...>
do_run() {
  local name="$1" max_steps="$2" compile="$3"; shift 3
  [[ "$1" == "--" ]] && shift
  start_server "$name" 2 "$@" || return 1
  local sid; sid="$(server_id)"
  # No --diloco-server: the harness guarantees exactly one live server
  # (wait_no_servers), so submit auto-resolves it. (Passing the bare job id
  # q_... here would be misread as a hostname -> name-resolution failure; the
  # registry id is local:q_..., but auto-resolution sidesteps the footgun.)
  log "submit 2 workers (max_steps=$max_steps compile=$compile) -> $sid"
  local out; out="$(forgather -t "$CONFIG" submit --diloco \
      --diloco-worker-count 2 --max-steps "$max_steps" --compile "$compile" 2>&1)"; echo "$out"
  local wids=($(echo "$out" | grep -oE 'q_[0-9]+_[0-9a-f]+'))
  [[ ${#wids[@]} -gt 0 ]] || { err "no worker jobs"; shutdown_server; return 1; }
  wait_jobs_terminal 70 "${wids[@]}" || warn "workers not terminal in time"
  capture "$name" "$sid" "${wids[@]}"
  shutdown_server
}

# DyLU: heterogeneous worker speeds via two single-worker submits, one throttled.
run_async_dylu() {
  local name=async_dylu
  start_server "$name" 2 --grpc --wire-format safetensors \
    --async --dylu --dylu-base-sync-every 20 --heartbeat-timeout 120 || return 1
  local sid; sid="$(server_id)"
  log "submit SLOW worker (DILOCO_DEBUG_STEP_DELAY=0.10) -> $sid"
  local s; s="$(forgather -t "$CONFIG" submit --diloco \
      --diloco-worker-count 1 --worker-id feat-slow --heartbeat-interval 5 \
      --max-steps 400 --compile no --env DILOCO_DEBUG_STEP_DELAY=0.10 2>&1)"; echo "$s"
  log "submit FAST worker (no delay) -> $sid"
  local f; f="$(forgather -t "$CONFIG" submit --diloco \
      --diloco-worker-count 1 --worker-id feat-fast --heartbeat-interval 5 \
      --max-steps 400 --compile no 2>&1)"; echo "$f"
  local wids=($(echo "$s$f" | grep -oE 'q_[0-9]+_[0-9a-f]+'))
  [[ ${#wids[@]} -ge 2 ]] || { err "expected 2 worker jobs"; shutdown_server; return 1; }
  wait_jobs_terminal 90 "${wids[@]}" || warn "workers not terminal in time"
  capture "$name" "$sid" "${wids[@]}"
  shutdown_server
}

# Token budget: the server's --token-budget is the sole stop authority. Workers
# submit with NO --max-steps (the config defaults max_steps=-1), so a small budget
# (2M tokens) is what stops them — confirm via the server log
# ("Token budget reached … relaying save_and_stop") and the worker's clean
# TrainOutput well before any step cap. --token-budget takes a K/M/B suffix.
run_token_budget() {
  local name=token_budget
  start_server "$name" 2 --grpc --wire-format safetensors --sync-every 20 \
    --token-budget 2M --heartbeat-timeout 120 || return 1
  local sid; sid="$(server_id)"
  log "submit 2 workers (compile off, NO --max-steps; budget 2M stops them) -> $sid"
  local out; out="$(forgather -t "$CONFIG" submit --diloco \
      --diloco-worker-count 2 --compile no 2>&1)"; echo "$out"
  local wids=($(echo "$out" | grep -oE 'q_[0-9]+_[0-9a-f]+'))
  [[ ${#wids[@]} -ge 2 ]] || { err "expected 2 worker jobs"; shutdown_server; return 1; }
  wait_jobs_terminal 90 "${wids[@]}" || warn "workers not terminal in time"
  capture "$name" "$sid" "${wids[@]}"
  shutdown_server
}

case "${1:-}" in
  # transport x wire matrix — quick functional smokes, baseline sync, compile off.
  # --sync-every 20 so 60 steps yield ~3 real sync rounds (each round exercises
  # the bulk wire codec end-to-end; without it the default H=500 > max_steps and
  # nothing is ever transmitted).
  http-pickle) do_run http_pickle 60 no -- --sync-every 20 --wire-format pickle ;;
  http-st)     do_run http_st     60 no -- --sync-every 20 --wire-format safetensors ;;
  grpc-pickle) do_run grpc_pickle 60 no -- --sync-every 20 --grpc --wire-format pickle ;;
  grpc-st)     do_run grpc_st     60 no -- --sync-every 20 --grpc --wire-format safetensors ;;

  # feature tests — real, gRPC+safetensors, compile on
  baseline)  do_run baseline  120 yes -- --grpc --wire-format safetensors --sync-every 20 ;;
  streaming) do_run streaming 120 yes -- --grpc --wire-format safetensors --sync-every 20 --num-fragments 2 --verbose-sync ;;
  async)     do_run async     120 yes -- --grpc --wire-format safetensors --sync-every 20 --async --verbose-sync ;;
  async-dn)  do_run async_dn  160 yes -- --grpc --wire-format safetensors --sync-every 20 --async --dn-buffer-size 4 --verbose-sync ;;
  async-dylu) run_async_dylu ;;
  token-budget) run_token_budget ;;

  all)
    for r in http-pickle http-st grpc-pickle grpc-st baseline streaming async async-dn async-dylu token-budget; do
      log "==================== RECIPE: $r ===================="; "$0" "$r"
    done ;;
  *) err "unknown recipe '${1:-}'"; exit 2 ;;
esac
