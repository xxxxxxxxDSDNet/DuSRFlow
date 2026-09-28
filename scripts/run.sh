#!/usr/bin/env bash

set -euo pipefail

# Purpose: launch one self-contained SR experiment YAML for training or testing.
#
# Usage:
#   bash scripts/run.sh EXPERIMENT train|test [GPU] [EXTRA_ARGS...]
#
# Examples:
#   bash scripts/run.sh dusr_realv2_paired_l1 train 2
#   bash scripts/run.sh dusr_real_l1 train 3 --check-only
#   bash scripts/run.sh camera_fusion_real_gan test 0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT="${1:?Usage: bash scripts/run.sh EXPERIMENT train|test [GPU] [EXTRA_ARGS...]}"
STAGE="${2:?Usage: bash scripts/run.sh EXPERIMENT train|test [GPU] [EXTRA_ARGS...]}"
GPU="${3:-0}"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PATH_CONFIG="$PROJECT_ROOT/configs/paths.yaml"
mapfile -t CONFIGURED_PATHS < <(
    /usr/bin/python3 "$PROJECT_ROOT/scripts/read_path_config.py" "$PATH_CONFIG" \
        python_bin dataset_root flow_foreground_root
)
export PYTHON_BIN="${PYTHON_BIN:-${CONFIGURED_PATHS[0]}}"
export DATASET_ROOT="${DATASET_ROOT:-${CONFIGURED_PATHS[1]}}"
export FLOW_FOREGROUND_ROOT="${FLOW_FOREGROUND_ROOT:-${CONFIGURED_PATHS[2]}}"

CONFIG="$PROJECT_ROOT/configs/sr/${EXPERIMENT%.yaml}.yaml"

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: experiment YAML not found: $CONFIG" >&2
    exit 2
fi

if (($# >= 3)); then
    shift 3
else
    shift "$#"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "$PYTHON_BIN" == */* ]]; then
    export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
fi
cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m tools.run_experiment \
    "$CONFIG" "$STAGE" --gpu "$GPU" "$@"
