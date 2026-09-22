#!/usr/bin/env bash
# SPDX-License-Identifier: OpenMDW-1.1
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

: "${CHECKPOINT_ROOT:=/path/to/trained/checkpoint}"
: "${DROID_ROOT:=/path/to/Cosmos3-DROID/success}"
: "${COSMOS3_EDGE_PROCESSOR_PATH:=/path/to/Cosmos3-Edge}"
: "${WAN_VAE_PATH:=/path/to/Wan2.2_VAE.pth}"
: "${ACTION_STATISTICS_PATH:?Set matching state/delta statistics}"
: "${OUTPUT_ROOT:=outputs/causal_action/inference}"

for path_var in CHECKPOINT_ROOT DROID_ROOT COSMOS3_EDGE_PROCESSOR_PATH WAN_VAE_PATH ACTION_STATISTICS_PATH OUTPUT_ROOT; do
    [[ "${!path_var}" = /* ]] || printf -v "$path_var" '%s/%s' "$PWD" "${!path_var}"
done
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export COSMOS_DEVICE="${COSMOS_DEVICE:-npu}"

CMD=(
    torchrun --standalone --nproc_per_node=1
    -m cosmos_framework.inference.causal_action.cli
    --checkpoint "$CHECKPOINT_ROOT"
    --dataset-root "$DROID_ROOT"
    --processor "$COSMOS3_EDGE_PROCESSOR_PATH"
    --vae "$WAN_VAE_PATH"
    --statistics "$ACTION_STATISTICS_PATH"
    --output "$OUTPUT_ROOT"
    --mode "${MODE:-policy}"
    --video-stride 1
    --index "${SAMPLE_INDEX:-0}"
    --steps "${NUM_STEPS:-20}"
    --guidance "${GUIDANCE:-3}"
    --block "${BLOCK_SIZE:-1}"
    --history "${HISTORY_BLOCKS:-8}"
    "$@"
)

printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'
[[ "${PRINT_ONLY:-0}" == 1 ]] && exit 0

cd "$REPO_ROOT"
exec "${CMD[@]}"
