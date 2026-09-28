#!/usr/bin/env bash

set -euo pipefail

# Purpose:
#   Distribute the four released dataset jobs across a comma-separated GPU list.
#
# Usage:
#   bash scripts/launchers/run_four_datasets.sh flow train GPU_IDS FOREGROUND_ROOT
#   bash scripts/launchers/run_four_datasets.sh flow test  GPU_IDS
#   bash scripts/launchers/run_four_datasets.sh sr   train GPU_IDS LOSS
#   bash scripts/launchers/run_four_datasets.sh sr   test  GPU_IDS LOSS
#
# Examples:
#   bash scripts/launchers/run_four_datasets.sh flow train 0,1,2,3 /path/to/rgba_foregrounds
#   bash scripts/launchers/run_four_datasets.sh flow test 0,1,2,3
#   bash scripts/launchers/run_four_datasets.sh sr train 0,1,2,3 l1
#   bash scripts/launchers/run_four_datasets.sh sr test 0,1,2,3 l1

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

COMPONENT="${1:-}"
STAGE="${2:-}"
GPU_IDS="${3:-0}"
OPTION="${4:-}"

if [[ "$COMPONENT" != "flow" && "$COMPONENT" != "sr" ]]; then
    echo "COMPONENT must be flow or sr." >&2
    exit 2
fi
if [[ "$STAGE" != "train" && "$STAGE" != "test" ]]; then
    echo "STAGE must be train or test." >&2
    exit 2
fi

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if ((${#GPUS[@]} == 0)); then
    echo "No GPU IDs supplied." >&2
    exit 2
fi

DATASETS=(DuSR-RealV2-Paired CameraFusion-Real DuSR-Real RealMCVSR-Real)
LOG_ROOT="${LAUNCHER_LOG_ROOT:-$PROJECT_ROOT/launcher_logs/${COMPONENT}_${STAGE}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_ROOT"
PIDS=()

for index in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$index]}"
    gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
    log="$LOG_ROOT/${dataset}.log"
    echo "[$dataset] GPU=$gpu log=$log"
    if [[ "$COMPONENT" == "flow" ]]; then
        if [[ "$STAGE" == "train" ]]; then
            foreground_root="${OPTION:-${FLOW_FOREGROUND_ROOT:-}}"
            if [[ -z "$foreground_root" ]]; then
                echo "Flow training requires FOREGROUND_ROOT as argument 4." >&2
                exit 2
            fi
            bash "$SCRIPT_DIR/flow.sh" train "$dataset" "$gpu" "$foreground_root" >"$log" 2>&1 &
        else
            bash "$SCRIPT_DIR/flow.sh" test "$dataset" "$gpu" >"$log" 2>&1 &
        fi
    else
        loss="${OPTION:-l1}"
        case "$dataset" in
            DuSR-RealV2-Paired) prefix="dusr_realv2_paired" ;;
            CameraFusion-Real) prefix="camera_fusion_real" ;;
            DuSR-Real) prefix="dusr_real" ;;
            RealMCVSR-Real) prefix="realmcvsr_real" ;;
        esac
        if [[ "$loss" != "l1" && "$loss" != "gan" ]]; then
            echo "LOSS must be l1 or gan." >&2
            exit 2
        fi
        if [[ "$STAGE" == "train" ]]; then
            bash "$SCRIPT_DIR/train_sr.sh" "$dataset" "$loss" "$gpu" >"$log" 2>&1 &
        else
            bash "$PROJECT_ROOT/scripts/run.sh" "${prefix}_${loss}" test "$gpu" >"$log" 2>&1 &
        fi
    fi
    PIDS+=("$!")
done

STATUS=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        STATUS=1
    fi
done

if ((STATUS != 0)); then
    echo "One or more jobs failed. Check: $LOG_ROOT" >&2
    exit "$STATUS"
fi
echo "All jobs finished. Logs: $LOG_ROOT"
