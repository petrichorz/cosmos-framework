#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Portable torchrun template for Cosmos3-Edge causal TND pre-training.
# Activate a torch_npu/CANN environment before running this script, or copy and
# edit the paired environment template first.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

: "${DATASET_PATH:=/path/to/lerobot_v3_dataset_or_parent}"
: "${BASE_CHECKPOINT_PATH:=/path/to/Cosmos3-Edge-DCP}"
: "${COSMOS3_EDGE_PROCESSOR_PATH:=/path/to/Cosmos3-Edge}"
: "${WAN_VAE_PATH:=/path/to/Wan2.2_VAE.pth}"
: "${OUTPUT_ROOT:=/path/to/output/vision_causal_edge_tnd_pretrain}"
: "${MROPE_BASE_FPS:=24}"
TOML_PATH="${TOML_PATH:-$SCRIPT_DIR/toml/sft_config/vision_pretrain_edge_causal_tnd.toml}"

for path_var in DATASET_PATH BASE_CHECKPOINT_PATH COSMOS3_EDGE_PROCESSOR_PATH WAN_VAE_PATH OUTPUT_ROOT TOML_PATH; do
    [[ "${!path_var}" = /* ]] || printf -v "$path_var" '%s/%s' "$PWD" "${!path_var}"
done

export DATASET_PATH BASE_CHECKPOINT_PATH COSMOS3_EDGE_PROCESSOR_PATH WAN_VAE_PATH
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export COSMOS_DEVICE="${COSMOS_DEVICE:-npu}"

TORCHRUN_ARGS=(
    --nproc_per_node="${NPROC_PER_NODE:-4}"
    --master_port="${MASTER_PORT:-50012}"
)
[[ -n "${NNODES:-}" ]] && TORCHRUN_ARGS+=(--nnodes="$NNODES")
[[ -n "${NODE_RANK:-}" ]] && TORCHRUN_ARGS+=(--node_rank="$NODE_RANK")
[[ -n "${MASTER_ADDR:-}" ]] && TORCHRUN_ARGS+=(--master_addr="$MASTER_ADDR")

TRAIN_ARGS=(-m cosmos_framework.scripts.train --sft-toml="$TOML_PATH")
[[ "${DRY_RUN:-0}" == 1 ]] && TRAIN_ARGS+=(--dryrun)

CMD=(
    torchrun "${TORCHRUN_ARGS[@]}" "${TRAIN_ARGS[@]}" --
    model=mot_causal_fsdp
    "model.config.diffusion_expert_config.base_fps=$MROPE_BASE_FPS"
    model.config.vlm_config.tokenizer.repository=null
    model.config.vlm_config.tokenizer.revision=null
    "+model.config.vlm_config.tokenizer.tokenizer_type=$COSMOS3_EDGE_PROCESSOR_PATH"
    '~dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:0.7,1:0.2,2:0.1}'
    '+dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:1.0}'
    "$@"
)

printf 'Repository: %s\nTOML: %s\nOutput: %s\n' "$REPO_ROOT" "$TOML_PATH" "$OUTPUT_ROOT"
printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'
[[ "${PRINT_ONLY:-0}" == 1 ]] && exit 0

for path_var in DATASET_PATH BASE_CHECKPOINT_PATH COSMOS3_EDGE_PROCESSOR_PATH; do
    [[ -d "${!path_var}" ]] || { echo "ERROR: missing directory: $path_var=${!path_var}" >&2; exit 1; }
done
for path_var in WAN_VAE_PATH TOML_PATH; do
    [[ -f "${!path_var}" ]] || { echo "ERROR: missing file: $path_var=${!path_var}" >&2; exit 1; }
done

mkdir -p "$OUTPUT_ROOT"
cd "$REPO_ROOT"
set +e
"${CMD[@]}" 2>&1 | tee -a "$OUTPUT_ROOT/launcher_rank${NODE_RANK:-0}.log"
exit_code=${PIPESTATUS[0]}
set -e
exit "$exit_code"
