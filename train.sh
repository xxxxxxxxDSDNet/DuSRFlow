#!/usr/bin/env bash

set -euo pipefail

# Purpose:
#   Train exactly one DuFlowNet or DuSRFlow model. Each experiment name selects
#   one dataset and its corresponding release configuration.
#
# Available commands (run from the repository root):
#
# Train Flow models:
#   bash train.sh flow_dusr_realv2_paired 0
#   bash train.sh flow_camera_fusion_real 0
#   bash train.sh flow_dusr_real 0
#   bash train.sh flow_realmcvsr_real 0
#
# Train SR models with L1 loss:
#   bash train.sh dusr_realv2_paired_l1 0
#   bash train.sh camera_fusion_real_l1 0
#   bash train.sh dusr_real_l1 0
#   bash train.sh realmcvsr_real_l1 0
#
# Train SR models with GAN loss:
#   bash train.sh dusr_realv2_paired_gan 0
#   bash train.sh camera_fusion_real_gan 0
#   bash train.sh dusr_real_gan 0
#   bash train.sh realmcvsr_real_gan 0
#
# The final argument is the GPU ID. Optional runner flags may be appended:
#   bash train.sh dusr_realv2_paired_l1 0 --check-only
#   bash train.sh dusr_realv2_paired_l1 0 --dry-run
#   bash train.sh dusr_realv2_paired_l1 0 \
#       --dataset-root /path/to/datasets \
#       --flow-checkpoint /path/to/DuFlowNet_DuSR-RealV2-Paired.pth
#
# Usage:
#   bash train.sh EXPERIMENT GPU [OPTIONS]

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATH_CONFIG="$PROJECT_ROOT/configs/paths.yaml"
mapfile -t CONFIGURED_PATHS < <(
    /usr/bin/python3 "$PROJECT_ROOT/scripts/read_path_config.py" "$PATH_CONFIG" \
        python_bin dataset_root flow_foreground_root
)
export PYTHON_BIN="${PYTHON_BIN:-${CONFIGURED_PATHS[0]}}"
export DATASET_ROOT="${DATASET_ROOT:-${CONFIGURED_PATHS[1]}}"
export FLOW_FOREGROUND_ROOT="${FLOW_FOREGROUND_ROOT:-${CONFIGURED_PATHS[2]}}"

usage() {
    sed -n '5,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

if (($# < 2)); then
    usage >&2
    exit 2
fi

EXPERIMENT="${1%.yaml}"
GPU="$2"
shift 2

case "$EXPERIMENT" in
    flow_dusr_realv2_paired|flow_camera_fusion_real|\
    flow_dusr_real|flow_realmcvsr_real|\
    dusr_realv2_paired_l1|dusr_realv2_paired_gan|\
    camera_fusion_real_l1|camera_fusion_real_gan|\
    dusr_real_l1|dusr_real_gan|\
    realmcvsr_real_l1|realmcvsr_real_gan)
        ;;
    *)
        echo "Error: unsupported training experiment: $EXPERIMENT" >&2
        usage >&2
        exit 2
        ;;
esac

case "$EXPERIMENT" in
    flow_dusr_realv2_paired)
        flow_dataset="DuSR-RealV2-Paired"
        ;;
    flow_camera_fusion_real)
        flow_dataset="CameraFusion-Real"
        ;;
    flow_dusr_real)
        flow_dataset="DuSR-Real"
        ;;
    flow_realmcvsr_real)
        flow_dataset="RealMCVSR-Real"
        ;;
    *)
        flow_dataset=""
        ;;
esac

if [[ -n "$flow_dataset" ]]; then
    if [[ -z "${FLOW_FOREGROUND_ROOT:-}" ]]; then
        if [[ -d "$PROJECT_ROOT/foregrounds" ]]; then
            export FLOW_FOREGROUND_ROOT="$PROJECT_ROOT/foregrounds"
        else
            echo "Error: RGBA foreground directory was not found." >&2
            echo "Set FLOW_FOREGROUND_ROOT before starting Flow training." >&2
            exit 2
        fi
    fi
    exec bash "$PROJECT_ROOT/scripts/launchers/flow.sh" \
        train "$flow_dataset" "$GPU" "$FLOW_FOREGROUND_ROOT" "$@"
fi

exec bash "$PROJECT_ROOT/scripts/run.sh" "$EXPERIMENT" train "$GPU" "$@"
