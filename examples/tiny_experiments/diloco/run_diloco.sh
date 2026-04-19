#!/usr/bin/env bash
#
# Launch a DiLoCo training run with N workers on localhost.
#
# Usage:
#   ./run_diloco.sh                   # 2 workers, default settings
#   ./run_diloco.sh -n 4              # 4 workers
#   ./run_diloco.sh -s 100            # sync every 100 steps
#   ./run_diloco.sh -p 8600           # use port 8600
#
# The script:
#   1. Checks for model weights, constructs them if missing
#   2. Starts the DiLoCo server
#   3. Waits for the server to be ready
#   4. Starts N workers with correct shard indices
#   5. Waits for all workers to finish
#   6. Stops the server
#
# All processes are cleaned up on Ctrl-C.

set -euo pipefail

# Defaults
NUM_WORKERS=2
SYNC_EVERY=500
PORT=8512
CONFIG=default.yaml
MODEL_DIR=output_models/default_model
MODEL_PROJECT=../../models/causal_lm
MODEL_TEMPLATE=4M.yaml

# Parse arguments
while getopts "n:s:p:c:h" opt; do
    case $opt in
        n) NUM_WORKERS=$OPTARG ;;
        s) SYNC_EVERY=$OPTARG ;;
        p) PORT=$OPTARG ;;
        c) CONFIG=$OPTARG ;;
        h)
            echo "Usage: $0 [-n num_workers] [-s sync_every] [-p port] [-c config.yaml]"
            exit 0
            ;;
        *) echo "Unknown option: -$opt" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Track child PIDs for cleanup
PIDS=()
cleanup() {
    echo ""
    echo "Stopping all processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Give processes a moment to exit, then force kill
    sleep 1
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    echo "Done."
}
trap cleanup EXIT INT TERM

# Step 1: Check/construct model weights
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "==> Model weights not found at $MODEL_DIR, constructing..."
    forgather -p "$MODEL_PROJECT" -t "$MODEL_TEMPLATE" \
        model --device cpu --save-checkpoint --safetensors \
        --output-dir "$MODEL_DIR" \
        construct
    echo "==> Model constructed."
else
    echo "==> Model weights found at $MODEL_DIR"
fi

# Step 2: Start server
echo "==> Starting DiLoCo server on port $PORT with $NUM_WORKERS workers..."
forgather diloco server \
    -m "$MODEL_DIR" \
    -n "$NUM_WORKERS" \
    --port "$PORT" \
    > >(sed 's/^/[server] /') 2>&1 &
SERVER_PID=$!
PIDS+=("$SERVER_PID")

# Step 3: Wait for server to be ready
echo "==> Waiting for server..."
for i in $(seq 1 30); do
    if curl -s "http://localhost:$PORT/status" > /dev/null 2>&1; then
        echo "==> Server is ready."
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server process died." >&2
        exit 1
    fi
    sleep 0.5
done

if ! curl -s "http://localhost:$PORT/status" > /dev/null 2>&1; then
    echo "ERROR: Server did not start within 15 seconds." >&2
    exit 1
fi

# Step 4: Start workers
echo "==> Starting $NUM_WORKERS workers (sync_every=$SYNC_EVERY, config=$CONFIG)..."
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "    Worker $i starting..."
    forgather diloco worker \
        --server "localhost:$PORT" \
        --sync-every "$SYNC_EVERY" \
        -p . \
        -t "$CONFIG" \
        train --num-shards "$NUM_WORKERS" --shard-index "$i" \
        > >(sed "s/^/[worker$i] /") 2>&1 &
    WORKER_PID=$!
    PIDS+=("$WORKER_PID")
    # Small delay to avoid race conditions in log output
    sleep 0.5
done

echo "==> All workers started. Training in progress..."
echo "    Monitor: forgather diloco status --server localhost:$PORT"
echo "    Dashboard: http://localhost:$PORT/dashboard"
echo "    Press Ctrl-C to stop."

# Step 5: Wait for workers to finish
# Wait for any worker to exit first, then wait for all
wait_for_workers() {
    local worker_pids=("${PIDS[@]:1}")  # Skip server PID
    for pid in "${worker_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
        fi
    done
}

wait_for_workers
echo "==> All workers finished."
