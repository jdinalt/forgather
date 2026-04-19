#!/bin/bash
#
# Test driver for DataloaderDispatcher multi-dimensional parallelism tests.
#
# Uses gloo backend by default for CPU-only testing (no GPU dependencies).
# Can be run with nccl backend if GPUs are available.
#
# Usage:
#   ./run_dataloader_dispatcher_tests.sh           # Run all tests with gloo
#   ./run_dataloader_dispatcher_tests.sh nccl      # Run all tests with nccl (requires GPUs)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test_dataloader_dispatcher_multidim.py"

BACKEND="${1:-gloo}"

echo "============================================================"
echo "DataloaderDispatcher Test Suite"
echo "Backend: ${BACKEND}"
echo "============================================================"
echo

run_test() {
    local nproc=$1
    local dp=$2
    local mp=$3
    local description=$4
    local extra_args="${5:-}"

    echo "------------------------------------------------------------"
    echo "Test: ${description}"
    echo "  nproc=${nproc}, dp=${dp}, mp=${mp}, backend=${BACKEND} ${extra_args}"
    echo "------------------------------------------------------------"

    torchrun --standalone --nproc_per_node "${nproc}" \
        "${TEST_SCRIPT}" \
        --dp "${dp}" \
        --mp "${mp}" \
        --backend "${BACKEND}" \
        ${extra_args} \
        2>&1 | grep -v "^\[Gloo\]" | grep -v "^W[0-9]" | grep -v "OMP_NUM_THREADS"

    echo
}

# =============================================================================
# 1D Mesh Tests
# =============================================================================

echo "=== 1D Mesh Tests ==="
echo

# 1D Pure DP tests (each rank gets different batch)
run_test 2 2 1 "1D Pure DP (2 ranks)"
run_test 4 4 1 "1D Pure DP (4 ranks)"

# 1D Pure MP tests (all ranks get same batch)
run_test 2 1 2 "1D Pure MP (2 ranks)"
run_test 4 1 4 "1D Pure MP (4 ranks)"

# Single rank (edge case)
run_test 1 1 1 "Single rank (dp=1, mp=1)"

# =============================================================================
# 2D Mesh Tests
# =============================================================================

echo "=== 2D Mesh Tests ==="
echo

# 2D Hybrid tests with dp_dim=0 (default)
run_test 4 2 2 "2D Hybrid 2x2 (dp_dim=0)"
run_test 6 2 3 "2D Hybrid 2x3 (dp_dim=0)"
run_test 6 3 2 "2D Hybrid 3x2 (dp_dim=0)"

# 2D Hybrid tests with dp_dim=1 (swapped dimensions)
run_test 4 2 2 "2D Hybrid 2x2 (dp_dim=1)" "--dp-dim 1"
run_test 6 2 3 "2D Hybrid 2x3 (dp_dim=1)" "--dp-dim 1"

echo "============================================================"
echo "All tests passed!"
echo "============================================================"
