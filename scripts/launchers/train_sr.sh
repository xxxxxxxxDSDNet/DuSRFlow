#!/usr/bin/env bash

set -euo pipefail

# Purpose:
#   Launch one SR training configuration through the unified YAML runner.
#   Patch size, offset range, loss, Flow checkpoint, and training settings are
#   all read from the selected YAML file.
#
# Usage:
#   bash scripts/launchers/train_sr.sh DATASET LOSS GPU [OPTIONS]
#
# Arguments:
#   DATASET  DuSR-RealV2-Paired | CameraFusion-Real | DuSR-Real | RealMCVSR-Real
#   LOSS     l1 | gan
#   GPU      CUDA device ID
#
# Examples:
#   bash scripts/launchers/train_sr.sh DuSR-RealV2-Paired l1 0
#   bash scripts/launchers/train_sr.sh DuSR-Real l1 1 --check-only

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PATH_CONFIG="$PROJECT_ROOT/configs/paths.yaml"
mapfile -t CONFIGURED_PATHS < <(
    /usr/bin/python3 "$PROJECT_ROOT/scripts/read_path_config.py" "$PATH_CONFIG" \
        python_bin dataset_root flow_foreground_root
)
export PYTHON_BIN="${PYTHON_BIN:-${CONFIGURED_PATHS[0]}}"
export DATASET_ROOT="${DATASET_ROOT:-${CONFIGURED_PATHS[1]}}"
export FLOW_FOREGROUND_ROOT="${FLOW_FOREGROUND_ROOT:-${CONFIGURED_PATHS[2]}}"

usage() {
    printf '%s\n' \
        'Usage: bash train.sh DATASET LOSS GPU [OPTIONS]' \
        '' \
        'DATASET: DuSR-RealV2-Paired | CameraFusion-Real | DuSR-Real | RealMCVSR-Real' \
        'LOSS:    l1 | gan' \
        'GPU:     CUDA device ID' \
        '' \
        'Example: bash train.sh DuSR-RealV2-Paired l1 0'
}

if (($# < 3)); then
    usage >&2
    exit 2
fi

DATASET="$1"
LOSS="$2"
GPU="$3"
shift 3

case "$DATASET" in
    DuSR-RealV2-Paired)
        experiment_prefix="dusr_realv2_paired"
        ;;
    CameraFusion-Real)
        experiment_prefix="camera_fusion_real"
        ;;
    DuSR-Real)
        experiment_prefix="dusr_real"
        ;;
    RealMCVSR-Real)
        experiment_prefix="realmcvsr_real"
        ;;
    *)
        echo "Error: unsupported dataset: $DATASET" >&2
        exit 2
        ;;
esac

if [[ "$LOSS" != "l1" && "$LOSS" != "gan" ]]; then
    echo "Error: LOSS must be 'l1' or 'gan'." >&2
    exit 2
fi

exec bash "$PROJECT_ROOT/scripts/run.sh" \
    "${experiment_prefix}_${LOSS}" train "$GPU" "$@"
