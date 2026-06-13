#!/usr/bin/env bash
#
# experiment.sh — real-budget feature-comparison sweep for diloco_features.
#
# Unlike harness.sh (quick functional checks), this runs the 5 feature configs
# to a REAL token budget at a realistic sync interval and keeps every run's logs
# so the loss trajectory can be harvested and plotted against the baseline:
#
#   baseline   sync DiLoCo            (the reference trajectory)
#   streaming  --num-fragments 2
#   async      --async
#   async_dn   --async --dn-buffer-size 4
#   async_dylu --async --dylu         (+ one throttled worker, heterogeneous)
#
# Budget: the config default (small.yaml total_tokens=500 -> ~16k steps/worker,
# ~1B total across 2 workers = 2x Chinchilla). H=100. gRPC + safetensors,
# torch.compile on. Each run starts from an identical pristine copy of the master
# and shares the config's fixed seed, so only the feature flag varies.
#
# Runs 2 experiments at a time across GPUs 1-4 (each experiment = 2 workers =
# 2 GPUs); the two members of a batch use distinct param-server ports so their
# workers never cross-register, and the pair is awaited together so neither
# server stalls at a sync barrier waiting on a not-yet-placed worker.
#
# Usage:
#   ./experiment.sh validate   # short (max-steps 120, compile off) — proves the
#                              # 2-server-port mechanic + harvest end-to-end
#   ./experiment.sh run        # the real sweep (~3.5h on 4 idle 4090s)
#
# After it finishes, harvest + plot:
#   python analysis/harvest.py && python analysis/plot_experiment.py

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PRISTINE="$REPO/models/small_llama_features_master"
RESULTS="$HERE/runs"
CONFIG="default.yaml"
H=100

MODE="${1:-run}"
case "$MODE" in
  validate) MAXSTEPS_ARGS=(--max-steps 120); COMPILE=no;  DYLU_DELAY=0.05 ;;
  run)      MAXSTEPS_ARGS=();                COMPILE=yes; DYLU_DELAY=0.03 ;;
  dylu)     MAXSTEPS_ARGS=();                COMPILE=yes; DYLU_DELAY=0.03 ;;
  *) echo "usage: $0 {validate|run|dylu}" >&2; exit 2 ;;
esac

mkdir -p "$RESULTS"

log()  { printf '\033[1;36m[exp]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[exp]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[exp]\033[0m %s\n' "$*" >&2; }

# A diloco server is identified to `submit` by host:port (loopback-matched);
# bare job ids are misread as hostnames, and the registry "local:q_…" id also
# works but host:port is unambiguous across two concurrent servers.
server_alive_on() { forgather diloco status --diloco-server "127.0.0.1:$1" 2>/dev/null | grep -qiE 'Status:\s*running'; }

wait_server_ready() {  # <port>
  for ((i=0; i<60; i++)); do server_alive_on "$1" && return 0; sleep 3; done
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
    sleep 20
  done
  return 1
}

shutdown_on() { forgather diloco shutdown --diloco-server "127.0.0.1:$1" >/dev/null 2>&1; }

# start_one <name> <port> <server-flags...> -> starts a fresh-master server, echoes nothing
start_one() {
  local name="$1" port="$2"; shift 2
  local out="$REPO/models/small_llama_feat_$name"
  rm -rf "$out"; cp -r "$PRISTINE" "$out"
  log "server '$name' on :$port — $*"
  forgather diloco server -o "$out" --port "$port" -n 2 --save-every 0 \
    --grpc --wire-format safetensors --sync-every "$H" --run-name "$name" "$@" \
    >/dev/null 2>&1
  wait_server_ready "$port" || { err "server '$name' (:$port) never ready"; return 1; }
  log "server '$name' ready on :$port"
}

# submit_workers <name> <port>  -> echoes the 2 worker job ids
submit_workers() {
  local name="$1" port="$2"
  forgather -t "$CONFIG" submit --diloco --diloco-server "127.0.0.1:$port" \
    --diloco-worker-count 2 "${MAXSTEPS_ARGS[@]}" --compile "$COMPILE" 2>/dev/null \
    | grep -oE 'q_[0-9]+_[0-9a-f]+'
}

# submit_workers_dylu <name> <port> -> echoes 2 worker job ids (one throttled)
submit_workers_dylu() {
  local name="$1" port="$2"
  forgather -t "$CONFIG" submit --diloco --diloco-server "127.0.0.1:$port" \
    --diloco-worker-count 1 --worker-id "${name}-slow" --heartbeat-interval 5 \
    "${MAXSTEPS_ARGS[@]}" --compile "$COMPILE" \
    --env "DILOCO_DEBUG_STEP_DELAY=$DYLU_DELAY" 2>/dev/null | grep -oE 'q_[0-9]+_[0-9a-f]+'
  forgather -t "$CONFIG" submit --diloco --diloco-server "127.0.0.1:$port" \
    --diloco-worker-count 1 --worker-id "${name}-fast" --heartbeat-interval 5 \
    "${MAXSTEPS_ARGS[@]}" --compile "$COMPILE" 2>/dev/null | grep -oE 'q_[0-9]+_[0-9a-f]+'
}

capture_one() {  # <name> <port> <worker-id...>
  local name="$1" port="$2"; shift 2
  local out="$RESULTS/$name" n=0
  mkdir -p "$out"
  local sid; sid="$(forgather diloco servers 2>/dev/null | grep -F ":$port" | grep -oE 'q_[0-9]+_[0-9a-f]+' | head -1)"
  [[ -n "$sid" ]] && forgather diloco logs "$sid" > "$out/server.log" 2>&1
  for wid in "$@"; do forgather job dump "$wid" > "$out/worker$n.log" 2>&1; n=$((n+1)); done
  log "captured runs/$name/"
}

# A batch entry is "name:port:kind:flags" where kind is std|dylu.
run_batch() {
  local entries=("$@")
  declare -A WIDS PORTS
  # start servers + submit workers for every entry, then await them all together
  for e in "${entries[@]}"; do
    IFS='|' read -r name port kind flags <<<"$e"
    # shellcheck disable=SC2086
    start_one "$name" "$port" $flags || { err "skip '$name' (server failed)"; continue; }
    PORTS[$name]="$port"
    local ids
    if [[ "$kind" == dylu ]]; then ids="$(submit_workers_dylu "$name" "$port")";
    else ids="$(submit_workers "$name" "$port")"; fi
    WIDS[$name]="$ids"
    log "submitted workers for '$name': $(echo "$ids" | tr '\n' ' ')"
  done
  # wait for every worker job in the batch
  local all=()
  for name in "${!WIDS[@]}"; do for id in ${WIDS[$name]}; do all+=("$id"); done; done
  log "awaiting ${#all[@]} worker jobs: ${all[*]}"
  wait_jobs_terminal 1200 "${all[@]}" || warn "batch not all terminal in time"
  # capture + tear down
  for name in "${!WIDS[@]}"; do
    # shellcheck disable=SC2086
    capture_one "$name" "${PORTS[$name]}" ${WIDS[$name]}
    shutdown_on "${PORTS[$name]}"
  done
  sleep 8
}

log "MODE=$MODE  H=$H  compile=$COMPILE  budget=$([[ ${#MAXSTEPS_ARGS[@]} -gt 0 ]] && echo "${MAXSTEPS_ARGS[*]}" || echo 'config default (~16k steps/worker)')"

# The meaningful DyLU run needs a stable (DN-buffered) base — pure async + DyLU
# diverges like pure async. DN buffer size = num_workers (2), per the docs.
if [[ "$MODE" == dylu ]]; then
  run_batch \
    "async_dn_dylu|8512|dylu|--async --dylu --dylu-base-sync-every $H --dn-buffer-size 2"
  log "DYLU RUN DONE — re-harvest with: python analysis/harvest.py && python analysis/plot_experiment.py"
else
  run_batch \
    "baseline|8512|std|--sync-every $H" \
    "streaming|8513|std|--sync-every $H --num-fragments 2"

  run_batch \
    "async|8512|std|--sync-every $H --async" \
    "async_dn|8513|std|--sync-every $H --async --dn-buffer-size 4"

  # Pure async + DyLU (no DN) — diverges like pure async; kept as a cautionary
  # point. The stable, meaningful DyLU run is `./experiment.sh dylu` (DN-buffered).
  run_batch \
    "async_dylu|8512|dylu|--async --dylu --dylu-base-sync-every $H"

  log "SWEEP DONE — harvest with: python analysis/harvest.py && python analysis/plot_experiment.py"
fi
