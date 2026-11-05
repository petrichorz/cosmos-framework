#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# 环境模板：将 /path/to 占位路径替换为实际路径，或通过环境变量覆盖。
# 回放期间只生成 latent；片段结束后整段解码，MP4 保存方式与 EveryNDrawSample 一致。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
MODEL_VARIANT="${MODEL_VARIANT:-edge}"
case "$MODEL_VARIANT" in
    edge|nano) ;;
    *) printf 'Unsupported MODEL_VARIANT: %s (expected edge or nano)\n' "$MODEL_VARIANT" >&2; exit 1 ;;
esac
ASCEND_ENV="${ASCEND_ENV:-/path/to/ascend-toolkit/set_env.sh}"
CONDA_ROOT="${CONDA_ROOT:-/path/to/miniforge3}"
CONDA_ENV="${CONDA_ENV:-cosmos-framework}"
# shellcheck disable=SC1090
source "$ASCEND_ENV"
set +u
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export COSMOS_DEVICE="${COSMOS_DEVICE:-npu}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ROBOTWIN_GROUP_STATS_PATH="${ROBOTWIN_GROUP_STATS_PATH:-/path/to/robotwin_group_stats.json}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/path/to/trained/checkpoint}"
COSMOS3_EDGE_PROCESSOR_PATH="${COSMOS3_EDGE_PROCESSOR_PATH:-/path/to/Cosmos3-Edge}"
NANO_TOKENIZER_PATH="${NANO_TOKENIZER_PATH:-/path/to/Qwen3-VL-8B-Instruct}"
WAN_VAE_PATH="${WAN_VAE_PATH:-/path/to/Wan2.2_VAE.pth}"
if [[ "$MODEL_VARIANT" == nano ]]; then
    PROCESSOR_PATH="$NANO_TOKENIZER_PATH"
else
    PROCESSOR_PATH="$COSMOS3_EDGE_PROCESSOR_PATH"
fi

CMD=(torchrun --standalone --nproc_per_node=1
    -m cosmos_framework.inference.causal_action.server
    --model-variant "$MODEL_VARIANT"
    --checkpoint "$CHECKPOINT_ROOT" --processor "$PROCESSOR_PATH" --vae "$WAN_VAE_PATH"
    --statistics "$ROBOTWIN_GROUP_STATS_PATH"
    --actions-per-block 32 --execute-steps 32 --video-stride 4
    --steps "${NUM_STEPS:-10}" --guidance "${GUIDANCE:-1}" --history "${HISTORY_BLOCKS:-32}"
    --fps "${FPS:-30}" --seed "${SEED:-42}" --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
    "$@")
printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'
[[ "${PRINT_ONLY:-0}" == 1 ]] && exit 0
cd "$REPO_ROOT"
exec "${CMD[@]}"
