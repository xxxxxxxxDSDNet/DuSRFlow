#!/usr/bin/env bash

set -euo pipefail

# Purpose:
#   Train or evaluate DuFlowNet on one of the four released datasets.
#
# Usage:
#   bash flow.sh train DATASET GPU [FOREGROUND_ROOT] [--debug|--check-only]
#   bash flow.sh test  DATASET GPU [CHECKPOINT] [--debug|--check-only]
#
# Examples:
#   bash flow.sh train DuSR-RealV2-Paired 0
#   bash flow.sh test DuSR-RealV2-Paired 0

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

ACTION="${1:-}"
DATASET="${2:-}"
GPU="${3:-0}"

if (($# >= 3)); then
    shift 3
else
    shift "$#"
fi

if [[ "$ACTION" != "train" && "$ACTION" != "test" ]]; then
    echo "Usage: bash flow.sh {train|test} DATASET GPU [FOREGROUND_ROOT|CHECKPOINT] [options]" >&2
    exit 2
fi

case "$DATASET" in
    DuSR-RealV2-Paired|CameraFusion-Real|DuSR-Real|RealMCVSR-Real) ;;
    *)
        echo "Unsupported dataset: $DATASET" >&2
        exit 2
        ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "$PYTHON_BIN" == */* ]]; then
    export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
fi
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/datasets}"
FLOW_WEIGHT_ROOT="${FLOW_WEIGHT_ROOT:-$PROJECT_ROOT/weights/pretrained}"
FLOW_OUTPUT_ROOT="${FLOW_OUTPUT_ROOT:-$PROJECT_ROOT/flow_training_outputs}"
FLOW_RESULT_ROOT="${FLOW_RESULT_ROOT:-$PROJECT_ROOT/flow_results}"
CONFIG="$PROJECT_ROOT/configs/flow/$DATASET.yaml"
export DATASET_ROOT FLOW_WEIGHT_ROOT CUDA_VISIBLE_DEVICES="$GPU"
cd "$PROJECT_ROOT"

if [[ "$ACTION" == "train" ]]; then
    FOREGROUND_ROOT="${1:-${FLOW_FOREGROUND_ROOT:-}}"
    if [[ -z "$FOREGROUND_ROOT" ]]; then
        echo "Flow training requires RGBA foreground patches." >&2
        echo "Pass FOREGROUND_ROOT as argument 4 or set FLOW_FOREGROUND_ROOT." >&2
        exit 2
    fi
    if (($# >= 1)) && [[ "$1" != --* ]]; then shift 1; fi
    exec "$PYTHON_BIN" -m tools.train_flow \
        --config "$CONFIG" \
        --dataset-root "$DATASET_ROOT" \
        --foreground-root "$FOREGROUND_ROOT" \
        --output-root "$FLOW_OUTPUT_ROOT" \
        "$@"
fi

CHECKPOINT=""
if (($# >= 1)) && [[ "$1" != --* ]]; then
    CHECKPOINT="$1"
    shift 1
fi

COMMAND=(
    "$PYTHON_BIN" -m tools.test_flow
    --config "$CONFIG"
    --dataset-root "$DATASET_ROOT"
    --output-root "$FLOW_RESULT_ROOT"
)
if [[ -n "$CHECKPOINT" ]]; then
    COMMAND+=(--checkpoint "$CHECKPOINT")
fi
exec "${COMMAND[@]}" "$@"
