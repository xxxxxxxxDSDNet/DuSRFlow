#!/usr/bin/env bash

set -euo pipefail

# Purpose:
#   Provide one public entry point for all released DuFlowNet and DuSRFlow
#   evaluation commands on the four supported datasets.
#
# Supported datasets:
#   1. DuSR-RealV2-Paired
#   2. CameraFusion-Real
#   3. DuSR-Real
#   4. RealMCVSR-Real
#
# Usage:
#   bash run_all.sh [MODE] [GPU_IDS]
#
# Arguments:
#   MODE             Operation to run. The default is test-all.
#   GPU_IDS          Comma-separated CUDA device IDs. Default: 0,1,2,3.
# Evaluation modes (use the released checkpoints):
#   test-all         Run Flow, L1 SR, and GAN SR evaluation on all datasets.
#   test-flow        Run DuFlowNet evaluation on all datasets.
#   test-sr-l1       Run L1 DuSRFlow evaluation on all datasets.
#   test-sr-gan      Run GAN DuSRFlow evaluation on all datasets.
#
# Examples:
#   bash run_all.sh test-all 0,1,2,3
#   bash run_all.sh test-flow 0,1,2,3
#
# Output directories:
#   Flow evaluation:  flow_results/
#   SR evaluation:    results/
#   Launcher logs:    launcher_logs/
#
# Notes:
#   - Evaluation always uses the released checkpoints under weights/.
#   - The four dataset jobs are distributed over GPU_IDS and run in parallel.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATH_CONFIG="$SCRIPT_DIR/configs/paths.yaml"
mapfile -t CONFIGURED_PATHS < <(
    /usr/bin/python3 "$SCRIPT_DIR/scripts/read_path_config.py" "$PATH_CONFIG" \
        python_bin dataset_root flow_foreground_root
)
export PYTHON_BIN="${PYTHON_BIN:-${CONFIGURED_PATHS[0]}}"
export DATASET_ROOT="${DATASET_ROOT:-${CONFIGURED_PATHS[1]}}"
export FLOW_FOREGROUND_ROOT="${FLOW_FOREGROUND_ROOT:-${CONFIGURED_PATHS[2]}}"

MODE="${1:-test-all}"
GPU_IDS="${2:-0,1,2,3}"
LAUNCHER="$SCRIPT_DIR/scripts/launchers/run_four_datasets.sh"

cd "$SCRIPT_DIR"

run_flow_test() {
    echo "============================================================"
    echo "Evaluating released DuFlowNet checkpoints on four datasets"
    echo "============================================================"
    bash "$LAUNCHER" flow test "$GPU_IDS"
}

run_sr_l1_test() {
    echo "============================================================"
    echo "Evaluating released L1 DuSRFlow checkpoints on four datasets"
    echo "============================================================"
    bash "$LAUNCHER" sr test "$GPU_IDS" l1
}

run_sr_gan_test() {
    echo "============================================================"
    echo "Evaluating released GAN DuSRFlow checkpoints on four datasets"
    echo "============================================================"
    bash "$LAUNCHER" sr test "$GPU_IDS" gan
}

case "$MODE" in
    test-all)
        # This is the recommended command for checking every released result.
        # It runs 4 Flow jobs, 4 L1 SR jobs, and 4 GAN SR jobs.
        run_flow_test
        run_sr_l1_test
        run_sr_gan_test
        ;;
    test-flow)
        run_flow_test
        ;;
    test-sr-l1)
        run_sr_l1_test
        ;;
    test-sr-gan)
        run_sr_gan_test
        ;;
    help|-h|--help)
        sed -n '5,/^$/p' "$0" | sed 's/^# \{0,1\}//'
        ;;
    *)
        echo "Error: unknown mode '$MODE'." >&2
        echo "Run 'bash run_all.sh help' to list supported modes." >&2
        exit 2
        ;;
esac
