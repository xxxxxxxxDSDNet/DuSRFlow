#!/usr/bin/env bash

set -euo pipefail

# Purpose:
#   Test exactly one released DuFlowNet or DuSRFlow checkpoint. Each experiment
#   name loads the matching dataset, inference settings, and checkpoint.
#
# Available commands (run from the repository root):
#
# Test Flow checkpoints:
#   bash test.sh flow_dusr_realv2_paired 0
#   bash test.sh flow_camera_fusion_real 0
#   bash test.sh flow_dusr_real 0
#   bash test.sh flow_realmcvsr_real 0
#
# Test SR checkpoints trained with L1 loss:
#   bash test.sh dusr_realv2_paired_l1 0
#   bash test.sh camera_fusion_real_l1 0
#   bash test.sh dusr_real_l1 0
#   bash test.sh realmcvsr_real_l1 0
#
# Test SR checkpoints trained with GAN loss:
#   bash test.sh dusr_realv2_paired_gan 0
#   bash test.sh camera_fusion_real_gan 0
#   bash test.sh dusr_real_gan 0
#   bash test.sh realmcvsr_real_gan 0
#
# Unpaired inference and no-reference IQA (uses the Paired-L1 checkpoint):
#   bash test.sh dusr_realv2_unpaired 0
#
# The final argument is the GPU ID. Use --debug to evaluate one image:
#   bash test.sh dusr_realv2_paired_l1 0 --debug
#   bash test.sh dusr_realv2_paired_l1 0 \
#       --dataset-root /path/to/datasets \
#       --checkpoint /path/to/DuSRFlow_DuSR-RealV2-Paired_L1.pth
#
# Usage:
#   bash test.sh EXPERIMENT GPU [OPTIONS]

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
    realmcvsr_real_l1|realmcvsr_real_gan|\
    dusr_realv2_unpaired)
        ;;
    *)
        echo "Error: unsupported test experiment: $EXPERIMENT" >&2
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
    exec bash "$PROJECT_ROOT/scripts/launchers/flow.sh" \
        test "$flow_dataset" "$GPU" "$@"
fi

if [[ "$EXPERIMENT" == "dusr_realv2_unpaired" ]]; then
    dry_run=false
    checkpoint_stem="DuSRFlow_DuSR-RealV2-Paired_L1"
    read_checkpoint_path=false
    for option in "$@"; do
        if [[ "$read_checkpoint_path" == true ]]; then
            checkpoint_stem="$(basename -- "${option%.pth}")"
            read_checkpoint_path=false
        elif [[ "$option" == "--checkpoint" ]]; then
            read_checkpoint_path=true
        elif [[ "$option" == --checkpoint=* ]]; then
            checkpoint_path="${option#--checkpoint=}"
            checkpoint_stem="$(basename -- "${checkpoint_path%.pth}")"
        elif [[ "$option" == "--dry-run" ]]; then
            dry_run=true
        fi
    done

    bash "$PROJECT_ROOT/scripts/run.sh" "$EXPERIMENT" test "$GPU" "$@"
    if [[ "$dry_run" == true ]]; then
        exit 0
    fi

    PYTHON_BIN="${PYTHON_BIN:-python}"
    RESULT_ROOT_PATH="${RESULT_ROOT:-$PROJECT_ROOT/results}"
    RESULT_DIR="$RESULT_ROOT_PATH/DuSR-RealV2-Unpaired/Paired_L1/$checkpoint_stem"
    export CUDA_VISIBLE_DEVICES="$GPU"
    exec "$PYTHON_BIN" -m tools.evaluate_nr_iqa \
        --input-dir "$RESULT_DIR" \
        --output-dir "$RESULT_DIR/nr_iqa" \
        --device cuda
fi

exec bash "$PROJECT_ROOT/scripts/run.sh" "$EXPERIMENT" test "$GPU" "$@"
