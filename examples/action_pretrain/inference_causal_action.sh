#!/usr/bin/env bash
# SPDX-License-Identifier: OpenMDW-1.1
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

: "${CHECKPOINT_ROOT:=/path/to/trained/checkpoint}"
: "${ACTION_SOURCES_FILE:=$SCRIPT_DIR/sources/mixed.json}"
: "${COSMOS3_EDGE_PROCESSOR_PATH:=/path/to/Cosmos3-Edge}"
: "${WAN_VAE_PATH:=/path/to/Wan2.2_VAE.pth}"
: "${OUTPUT_ROOT:=outputs/causal_action/inference}"

for path_var in CHECKPOINT_ROOT ACTION_SOURCES_FILE COSMOS3_EDGE_PROCESSOR_PATH WAN_VAE_PATH OUTPUT_ROOT; do
    [[ "${!path_var}" = /* ]] || printf -v "$path_var" '%s/%s' "$PWD" "${!path_var}"
done
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export COSMOS_DEVICE="${COSMOS_DEVICE:-npu}"

CMD=(
    torchrun --standalone --nproc_per_node=1
    -m cosmos_framework.inference.causal_action.cli
    --checkpoint "$CHECKPOINT_ROOT"
    --sources-file "$ACTION_SOURCES_FILE"
    --source-index "${SOURCE_INDEX:-0}"
    --processor "$COSMOS3_EDGE_PROCESSOR_PATH"
    --vae "$WAN_VAE_PATH"
    --output "$OUTPUT_ROOT"
    --mode "${MODE:-policy}"
    --video-stride "${VIDEO_STRIDE:-4}"
    --index "${SAMPLE_INDEX:-0}"
    --steps "${NUM_STEPS:-20}"
    --guidance "${GUIDANCE:-3}"
    --actions-per-block "${ACTIONS_PER_BLOCK:-32}"
    --max-action-steps "${MAX_ACTION_STEPS:-96}"
    --overlap-action-steps "${OVERLAP_ACTION_STEPS:-16}"
    --history "${HISTORY_BLOCKS:-8}"
    "$@"
)

printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'
[[ "${PRINT_ONLY:-0}" == 1 ]] && exit 0

cd "$REPO_ROOT"
exec "${CMD[@]}"
