#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Portable torchrun launcher for Cosmos3 Edge/Nano causal action TND mid-training.
# Activate a torch_npu/CANN environment before running this script, or copy and
# edit the paired environment template first.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

: "${SOURCE_SET:=mixed}"
: "${MODEL_VARIANT:=edge}"
case "$MODEL_VARIANT" in
    edge|nano) ;;
    *) printf 'Unsupported MODEL_VARIANT: %s (expected edge or nano)\n' "$MODEL_VARIANT" >&2; exit 1 ;;
esac
: "${AGIBOT_ROOT:=/path/to/agibot_processed}"
: "${EGOSUITE_ROOT:=/path/to/egosuite_processed}"
: "${AGIBOT_GROUP_STATS_PATH:=/path/to/agibot_group_stats.json}"
: "${EGOSUITE_GROUP_STATS_PATH:=/path/to/egosuite_group_stats.json}"
: "${ACTION_SOURCES_FILE:=$SCRIPT_DIR/sources/$SOURCE_SET.json}"
if [[ "$MODEL_VARIANT" == nano ]]; then
    : "${BASE_CHECKPOINT_PATH:=/path/to/Cosmos3-Nano-DCP}"
else
    : "${BASE_CHECKPOINT_PATH:=/path/to/Cosmos3-Edge-DCP}"
fi
: "${COSMOS3_EDGE_PROCESSOR_PATH:=/path/to/Cosmos3-Edge}"
: "${NANO_TOKENIZER_PATH:=/path/to/Qwen3-VL-8B-Instruct}"
: "${WAN_VAE_PATH:=/path/to/Wan2.2_VAE.pth}"
: "${OUTPUT_ROOT:=outputs/causal_action/midtrain}"
: "${MROPE_BASE_FPS:=24}"
TOML_PATH="${TOML_PATH:-$SCRIPT_DIR/action_midtrain_${MODEL_VARIANT}_causal_tnd.toml}"

for path_var in AGIBOT_ROOT EGOSUITE_ROOT AGIBOT_GROUP_STATS_PATH EGOSUITE_GROUP_STATS_PATH ACTION_SOURCES_FILE BASE_CHECKPOINT_PATH COSMOS3_EDGE_PROCESSOR_PATH NANO_TOKENIZER_PATH WAN_VAE_PATH OUTPUT_ROOT TOML_PATH; do
    [[ "${!path_var}" = /* ]] || printf -v "$path_var" '%s/%s' "$PWD" "${!path_var}"
done

export AGIBOT_GROUP_STATS_PATH EGOSUITE_GROUP_STATS_PATH
export AGIBOT_ROOT EGOSUITE_ROOT ACTION_SOURCES_FILE BASE_CHECKPOINT_PATH COSMOS3_EDGE_PROCESSOR_PATH WAN_VAE_PATH
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export COSMOS_DEVICE="${COSMOS_DEVICE:-npu}"
if [[ "$COSMOS_DEVICE" == npu && "$MODEL_VARIANT" == nano ]]; then
    export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
fi
# Release freed CPU tensor pages promptly in mimalloc-backed PyTorch builds.
export MIMALLOC_PURGE_DELAY="${MIMALLOC_PURGE_DELAY:-0}"
# Bound glibc arena retention in long-lived workers; callers can override the cap.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

TORCHRUN_ARGS=(
    --nproc_per_node="${NPROC_PER_NODE:-4}"
    --master_port="${MASTER_PORT:-50142}"
)
[[ -n "${NNODES:-}" ]] && TORCHRUN_ARGS+=(--nnodes="$NNODES")
[[ -n "${NODE_RANK:-}" ]] && TORCHRUN_ARGS+=(--node_rank="$NODE_RANK")
[[ -n "${MASTER_ADDR:-}" ]] && TORCHRUN_ARGS+=(--master_addr="$MASTER_ADDR")

TRAIN_ARGS=(-m cosmos_framework.scripts.train --sft-toml="$TOML_PATH")
[[ "${DRY_RUN:-0}" == 1 ]] && TRAIN_ARGS+=(--dryrun)

CMD=(
    torchrun "${TORCHRUN_ARGS[@]}" "${TRAIN_ARGS[@]}" --
    "model.config.diffusion_expert_config.base_fps=$MROPE_BASE_FPS"
    # 长片段按 token 预算 packing；沿用昨晚验证过的 lookahead=1。
    "dataloader_train.lookahead_limit=${LOOKAHEAD_LIMIT:-1}"
    dataloader_train.max_samples_per_batch=null
)
if [[ "$MODEL_VARIANT" == nano ]]; then
    CMD+=(
        model.config.vlm_config.tokenizer.config_variant=hf
        "model.config.vlm_config.tokenizer.pretrained_model_name=$NANO_TOKENIZER_PATH"
    )
else
    CMD+=(
        model.config.vlm_config.tokenizer.repository=null
        model.config.vlm_config.tokenizer.revision=null
        "+model.config.vlm_config.tokenizer.tokenizer_type=$COSMOS3_EDGE_PROCESSOR_PATH"
    )
fi
# 只覆盖用户显式提供的选项；默认沿用 TOML，命令行 overrides 最后生效。
[[ -n "${MODE:-}" ]] && CMD+=("data_setting.action.mode=$MODE")
[[ -n "${ALLOW_MOCK_STATISTICS:-}" ]] && CMD+=("data_setting.action.allow_mock_statistics=$ALLOW_MOCK_STATISTICS")
CMD+=("$@")

printf 'Repository: %s\nTOML: %s\nOutput: %s\n' "$REPO_ROOT" "$TOML_PATH" "$OUTPUT_ROOT"
printf 'Sources: %s\nAgiBot statistics: %s\nEgoSuite statistics: %s\n' \
    "$ACTION_SOURCES_FILE" "$AGIBOT_GROUP_STATS_PATH" "$EGOSUITE_GROUP_STATS_PATH"
printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'
[[ "${PRINT_ONLY:-0}" == 1 ]] && exit 0

mkdir -p "$OUTPUT_ROOT"
cd "$REPO_ROOT"
set +e
"${CMD[@]}" 2>&1 | tee -a "$OUTPUT_ROOT/launcher_rank${NODE_RANK:-0}.log"
exit_code=${PIPESTATUS[0]}
set -e
exit "$exit_code"
