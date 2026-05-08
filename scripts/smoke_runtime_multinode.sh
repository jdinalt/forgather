#!/bin/bash
# Smoke test the runtime image's multi-node workflow end-to-end.
#
# Steps:
#   1. Build the runtime image on this host.
#   2. Save the image and ssh-load it on the remote host.
#   3. Start a server container on each host with --cluster and --no-auth.
#   4. Wait for the cluster to form (both members reachable).
#   5. Submit a Tiny Llama v2 training run via `forgather cluster submit`
#      across every reachable GPU.
#   6. Wait for the run to finish; assert it reached "done".
#   7. Verify a checkpoint landed on the shared FS.
#   8. Cleanup (unless --keep): stop + remove containers on both hosts.
#
# Designed to be runnable autonomously: every step has loud diagnostics
# on failure, and a single log-dump action prints everything you'd
# need to triage from one place.
#
# Prerequisites:
#   - This host can `ssh <REMOTE>` passwordless to the remote.
#   - Both hosts have docker (with --gpus support, nvidia-container-toolkit).
#   - /mnt/rust/aiassets is mounted at the same path on both hosts (NFS or
#     equivalent) — needed because Tiny Llama v2's project_dir must
#     resolve identically on every peer.
#   - The remote host has a working forgather clone (any branch — the
#     runtime image's source tree is baked from the image we ship).
#
# Usage:
#   scripts/smoke_runtime_multinode.sh                # default: muthur
#   scripts/smoke_runtime_multinode.sh --remote box2  # custom remote
#   scripts/smoke_runtime_multinode.sh --keep         # leave containers running on success
#   scripts/smoke_runtime_multinode.sh --diagnose-on-failure
#                                                     # default; explicit toggle
#   scripts/smoke_runtime_multinode.sh --no-build     # skip rebuild + redeploy
#
# Exit codes: 0 on success, non-zero on any failure (and diagnose runs).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- args ----
REMOTE="${REMOTE:-muthur}"
CLUSTER_NAME="${CLUSTER_NAME:-smoke-$$}"
PROJECT_DIR="${PROJECT_DIR:-${REPO_ROOT}/examples/tutorials/tiny_llama}"
CONFIG="${CONFIG:-v2.yaml}"
LOCAL_CONTAINER="${LOCAL_CONTAINER:-forgather-smoke-local}"
REMOTE_CONTAINER="${REMOTE_CONTAINER:-forgather-smoke-remote}"
IMAGE_TAG="${IMAGE_TAG:-forgather:smoke}"
IMAGE_TAR="${IMAGE_TAR:-/mnt/rust/aiassets/.tmp/${IMAGE_TAG//[:\/]/_}.tar}"
SUBMIT_TIMEOUT="${SUBMIT_TIMEOUT:-1800}"
CLUSTER_FORM_TIMEOUT="${CLUSTER_FORM_TIMEOUT:-90}"
# Non-default port so the smoke test doesn't collide with any
# pre-existing forgather server (dev or runtime) that's already
# bound 8765 under host networking. The port flows through to
# both containers' --port arg and to the cluster CLI's --server.
SMOKE_PORT="${SMOKE_PORT:-18765}"

# Parse flags
KEEP=0
NO_BUILD=0
DIAGNOSE=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote) REMOTE="$2"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        --no-build) NO_BUILD=1; shift ;;
        --diagnose-on-failure) DIAGNOSE=1; shift ;;
        --no-diagnose) DIAGNOSE=0; shift ;;
        --cluster-name) CLUSTER_NAME="$2"; shift 2 ;;
        --project-dir) PROJECT_DIR="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "unknown flag: $1 (try --help)" >&2
            exit 2 ;;
    esac
done

# ---- log helpers ----
ts() { date -u +%H:%M:%S; }
log() { echo "[$(ts)] $*" >&2; }
fail() { echo "[$(ts)] FAIL: $*" >&2; }

# ---- diagnostics on failure ----
DIAGNOSTIC_DUMP_PATH=""

dump_diagnostics() {
    [[ "${DIAGNOSE}" -eq 0 ]] && return 0
    DIAGNOSTIC_DUMP_PATH="${IMAGE_TAR%/*}/smoke-${CLUSTER_NAME}-failure-$(date +%Y%m%dT%H%M%S).log"
    mkdir -p "$(dirname "${DIAGNOSTIC_DUMP_PATH}")"
    {
        echo "=== smoke test failure dump @ $(date -u) ==="
        echo
        echo "=== docker ps -a (local) ==="
        docker ps -a --filter "name=${LOCAL_CONTAINER}" --format \
            "{{.Names}}\t{{.Status}}\t{{.Image}}" 2>&1 || true
        echo
        echo "=== docker logs (local) ==="
        docker logs "${LOCAL_CONTAINER}" 2>&1 | tail -200 || true
        echo
        echo "=== docker ps -a (${REMOTE}) ==="
        ssh "${REMOTE}" "docker ps -a --filter 'name=${REMOTE_CONTAINER}' --format '{{.Names}}\t{{.Status}}\t{{.Image}}'" 2>&1 || true
        echo
        echo "=== docker logs (${REMOTE}) ==="
        ssh "${REMOTE}" "docker logs ${REMOTE_CONTAINER} 2>&1 | tail -200" 2>&1 || true
        echo
        echo "=== cluster nodes (local) ==="
        docker exec -e "FORGATHER_SERVER_URL=${SERVER_URL:-http://127.0.0.1:8765}" \
            "${LOCAL_CONTAINER}" forgather cluster nodes --json 2>&1 || true
        echo
        echo "=== cluster jobs (local) ==="
        docker exec -e "FORGATHER_SERVER_URL=${SERVER_URL:-http://127.0.0.1:8765}" \
            "${LOCAL_CONTAINER}" forgather cluster jobs --json 2>&1 || true
        if [[ -n "${CLUSTER_JOB_ID:-}" ]]; then
            echo
            echo "=== bundle detail: ${CLUSTER_JOB_ID} ==="
            docker exec -e "FORGATHER_SERVER_URL=${SERVER_URL:-http://127.0.0.1:8765}" \
                "${LOCAL_CONTAINER}" forgather cluster jobs "${CLUSTER_JOB_ID}" 2>&1 || true
            # Per-rank tty logs (read each peer's tty file)
            for peer_container in "${LOCAL_CONTAINER}" "remote:${REMOTE_CONTAINER}"; do
                if [[ "${peer_container}" == remote:* ]]; then
                    name="${peer_container#remote:}"
                    runner="ssh ${REMOTE} docker exec ${name}"
                else
                    name="${peer_container}"
                    runner="docker exec ${name}"
                fi
                echo
                echo "=== peer ${name}: latest tty.log ==="
                ${runner} bash -lc 'ls -lt ~/.forgather/server/jobs/*.tty 2>/dev/null | head -3; for f in $(ls -t ~/.forgather/server/jobs/*.tty 2>/dev/null | head -1); do echo "--- $f ---"; tail -200 "$f"; done' 2>&1 || true
            done
        fi
        echo
        echo "=== nvidia-smi (local) ==="
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader 2>&1 || true
        echo
        echo "=== nvidia-smi (${REMOTE}) ==="
        ssh "${REMOTE}" 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader' 2>&1 || true
    } > "${DIAGNOSTIC_DUMP_PATH}" 2>&1 || true
    log "diagnostic dump: ${DIAGNOSTIC_DUMP_PATH}"
}

cleanup() {
    rc=$?
    if [[ "${rc}" -ne 0 ]]; then
        fail "smoke test failed (rc=${rc})"
        dump_diagnostics
    else
        log "smoke test passed"
    fi
    if [[ "${KEEP}" -eq 1 && "${rc}" -eq 0 ]]; then
        log "--keep set; leaving containers running on local + ${REMOTE}"
        log "  local : docker stop ${LOCAL_CONTAINER} && docker rm ${LOCAL_CONTAINER}"
        log "  remote: ssh ${REMOTE} 'docker stop ${REMOTE_CONTAINER} && docker rm ${REMOTE_CONTAINER}'"
    else
        log "tearing down containers"
        docker rm -f "${LOCAL_CONTAINER}" 2>/dev/null || true
        ssh "${REMOTE}" "docker rm -f ${REMOTE_CONTAINER}" 2>/dev/null || true
    fi
    exit "${rc}"
}
trap cleanup EXIT

# ---- step 1: build runtime image locally ----
if [[ "${NO_BUILD}" -eq 0 ]]; then
    # Air-gap source: bake the local working tree into the image
    # (instead of git-cloning ${FORGATHER_GIT_REF} at build time).
    # That's the right default for smoke testing: we want to test
    # what's in this tree, not whatever's on the upstream branch.
    # Set FORGATHER_GIT_REF=<branch> to switch back to the
    # git-clone path.
    log "step 1/8: building runtime image (${IMAGE_TAG}) from local tree"
    docker build -t "${IMAGE_TAG}" \
        --build-arg FORGATHER_SOURCE_DIR=. \
        -f "${REPO_ROOT}/Dockerfile.runtime" "${REPO_ROOT}" >/dev/null
else
    log "step 1/8: --no-build set; assuming ${IMAGE_TAG} already present locally"
    docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1 || {
        fail "image ${IMAGE_TAG} not present locally; drop --no-build for first run"
        exit 1
    }
fi

# ---- step 2: deploy to remote via NFS-shared tarball ----
if [[ "${NO_BUILD}" -eq 0 ]]; then
    log "step 2/8: saving image to ${IMAGE_TAR} for shared-FS deploy"
    mkdir -p "$(dirname "${IMAGE_TAR}")"
    docker save -o "${IMAGE_TAR}" "${IMAGE_TAG}"
    log "step 2/8: loading image on ${REMOTE} from ${IMAGE_TAR}"
    ssh "${REMOTE}" "docker load -i ${IMAGE_TAR}" >/dev/null
    rm -f "${IMAGE_TAR}"
else
    log "step 2/8: --no-build set; skipping deploy (assuming remote has ${IMAGE_TAG})"
fi

# ---- step 3: start containers on both hosts ----
log "step 3/8: starting cluster containers (cluster name: ${CLUSTER_NAME})"
# Belt-and-suspenders: scrub any pre-existing containers with our names.
docker rm -f "${LOCAL_CONTAINER}" 2>/dev/null || true
ssh "${REMOTE}" "docker rm -f ${REMOTE_CONTAINER}" 2>/dev/null || true

# Both containers run with --network host for mDNS multicast, --no-auth
# for token-free CLI access, and a bind mount for the shared NFS path
# so the project_dir resolves identically on both peers. The state
# volume is ephemeral (no persistent named volume) since this is a
# smoke test; --rm-on-stop is achieved via the cleanup trap.
docker run -d \
    --init \
    --name "${LOCAL_CONTAINER}" \
    --gpus all \
    --network host \
    --shm-size=8g \
    --ipc=host \
    -e "PUID=$(id -u)" \
    -e "PGID=$(id -g)" \
    -v /mnt/rust/aiassets:/mnt/rust/aiassets \
    "${IMAGE_TAG}" \
    forgather server -H 0.0.0.0 -p "${SMOKE_PORT}" --cluster "${CLUSTER_NAME}" --no-auth \
    >/dev/null

ssh "${REMOTE}" "docker run -d \
    --init \
    --name ${REMOTE_CONTAINER} \
    --gpus all \
    --network host \
    --shm-size=8g \
    --ipc=host \
    -e PUID=\$(id -u) \
    -e PGID=\$(id -g) \
    -v /mnt/rust/aiassets:/mnt/rust/aiassets \
    ${IMAGE_TAG} \
    forgather server -H 0.0.0.0 -p ${SMOKE_PORT} --cluster ${CLUSTER_NAME} --no-auth" \
    >/dev/null

# Cluster CLI inside the smoke containers needs to point at the
# non-standard port. Set FORGATHER_SERVER_URL via env so every
# `forgather cluster ...` invocation picks it up automatically.
SERVER_URL="http://127.0.0.1:${SMOKE_PORT}"

# ---- step 4: wait for cluster to form ----
log "step 4/8: waiting for cluster to form (timeout: ${CLUSTER_FORM_TIMEOUT}s)"

# Helper: how many reachable members does the local server see?
# Returns a single integer; defaults to 0 on any error so the
# bash arithmetic comparison below never sees a non-numeric value.
reachable_count() {
    local raw
    raw="$(docker exec -e "FORGATHER_SERVER_URL=${SERVER_URL}" \
        "${LOCAL_CONTAINER}" \
        forgather cluster nodes --json 2>/dev/null || true)"
    [[ -z "${raw}" ]] && { echo 0; return; }
    echo "${raw}" | python3 -c '
import json, sys
try:
    d = json.loads(sys.stdin.read())
    members = d.get("members") or []
    print(sum(1 for m in members if m.get("reachable")))
except Exception:
    print(0)
' 2>/dev/null || echo 0
}

deadline=$(( $(date +%s) + CLUSTER_FORM_TIMEOUT ))
n=0
while [[ $(date +%s) -lt ${deadline} ]]; do
    sleep 3
    n="$(reachable_count)"
    n="${n//[^0-9]/}"
    [[ -z "${n}" ]] && n=0
    if [[ "${n}" -ge 2 ]]; then
        log "step 4/8: cluster has ${n} reachable members"
        break
    fi
done
if [[ "${n:-0}" -lt 2 ]]; then
    fail "cluster did not form within ${CLUSTER_FORM_TIMEOUT}s (saw ${n:-0} reachable members)"
    exit 1
fi

# ---- step 5: submit Tiny Llama v2 across every reachable GPU ----
# Resolve the project dir to its canonical (NFS) path. The smoke
# test only bind-mounts /mnt/rust/aiassets, so any host-side
# symlink (e.g. ~/ai_assets/forgather → /mnt/rust/...) needs to
# be followed before we hand the path to the in-container CLI —
# the symlink itself isn't visible inside the container.
PROJECT_DIR_CANONICAL="$(readlink -f -- "${PROJECT_DIR}")"
log "step 5/8: submitting ${CONFIG} from ${PROJECT_DIR_CANONICAL}"

# ``-p`` / ``-t`` are GLOBAL forgather flags, so they go before
# ``cluster submit``, not after. The cluster-submit-specific flags
# (--allow-version-mismatch, --json) follow the subcommand.
SUBMIT_OUT="$(docker exec -e "FORGATHER_SERVER_URL=${SERVER_URL}" \
    "${LOCAL_CONTAINER}" \
    forgather \
        -p "${PROJECT_DIR_CANONICAL}" \
        -t "${CONFIG}" \
        cluster submit \
        --allow-version-mismatch \
        --json 2>&1)" || {
    fail "submit failed: ${SUBMIT_OUT}"
    exit 1
}
CLUSTER_JOB_ID="$(echo "${SUBMIT_OUT}" \
    | python3 -c '
import json, sys
try:
    d = json.loads(sys.stdin.read())
    print((d.get("cluster_job") or {}).get("cluster_job_id") or "")
except Exception:
    pass
' 2>/dev/null || echo '')"
CLUSTER_JOB_ID="${CLUSTER_JOB_ID//[^a-zA-Z0-9_-]/}"
if [[ -z "${CLUSTER_JOB_ID}" ]]; then
    fail "no cluster_job_id in submit response: ${SUBMIT_OUT}"
    exit 1
fi
log "step 5/8: bundle ${CLUSTER_JOB_ID}"

# ---- step 6: wait for bundle to terminate ----
log "step 6/8: waiting for bundle to finish (timeout: ${SUBMIT_TIMEOUT}s)"
deadline=$(( $(date +%s) + SUBMIT_TIMEOUT ))
last_status=""
status=""
while [[ $(date +%s) -lt ${deadline} ]]; do
    sleep 10
    bundle_json="$(docker exec -e "FORGATHER_SERVER_URL=${SERVER_URL}" \
        "${LOCAL_CONTAINER}" \
        forgather cluster jobs "${CLUSTER_JOB_ID}" --json 2>/dev/null || echo '{}')"
    status="$(echo "${bundle_json}" | python3 -c '
import json, sys
try:
    d = json.loads(sys.stdin.read()) or {}
    print(d.get("rolled_up_status") or d.get("status") or "")
except Exception:
    pass
' 2>/dev/null || echo '')"
    status="${status//[^a-z]/}"  # only the alpha word
    if [[ "${status}" != "${last_status}" ]]; then
        log "step 6/8: status -> ${status:-(unknown)}"
        last_status="${status}"
    fi
    case "${status}" in
        done) break ;;
        failed|cancelled)
            fail "bundle reached terminal status: ${status}"
            exit 1 ;;
    esac
done
if [[ "${status}" != "done" ]]; then
    fail "bundle did not reach 'done' within ${SUBMIT_TIMEOUT}s (last status: ${status:-(unknown)})"
    exit 1
fi
log "step 6/8: bundle ${CLUSTER_JOB_ID} status=done"

# ---- step 7: verify a checkpoint landed ----
# This block disables ``set -e`` locally because the various ``ls`` /
# glob lookups below fail (rc != 0) when the checkpoint dir doesn't
# exist or the glob doesn't match — and with ``set -euo pipefail``
# at the top of the script those failures abort before we get to
# print a useful message. Wrap in a subshell-style "guard, check,
# report" pattern instead.
log "step 7/8: verifying checkpoint(s) on shared FS"
OUTPUT_DIR="$(readlink -f -- "${PROJECT_DIR}")/output_models/v2"

set +e
ckpt_dirs=( "${OUTPUT_DIR}/checkpoints"/checkpoint-* )
# When the glob doesn't match, bash leaves the pattern literal in
# the array. Detect that case explicitly.
if [[ ! -d "${ckpt_dirs[0]}" ]]; then
    set -e
    fail "no checkpoint directory under ${OUTPUT_DIR}/checkpoints"
    exit 1
fi
# Sort newest-first (mtime) so we report the latest checkpoint —
# v2.yaml may write multiple if save_steps fires more than once.
latest=""
for d in "${ckpt_dirs[@]}"; do
    if [[ -z "${latest}" || "${d}" -nt "${latest}" ]]; then
        latest="${d}"
    fi
done
shards=( "${latest}"/*.safetensors "${latest}"/*.bin )
shard=""
for s in "${shards[@]}"; do
    if [[ -f "${s}" ]]; then
        shard="${s}"
        break
    fi
done
set -e

if [[ -z "${shard}" ]]; then
    fail "checkpoint dir ${latest} has no shard file (.safetensors / .bin)"
    exit 1
fi
log "step 7/8: ${#ckpt_dirs[@]} checkpoint(s); latest=${latest##*/} shard=$(basename "${shard}")"

# ---- step 8: cleanup is done by the EXIT trap ----
log "step 8/8: smoke test complete"
exit 0
