#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# 环境模板：将 /path/to 占位路径替换为实际路径，或通过环境变量覆盖。

# RoboTwin policy 全段视频预测模板。
# 每轮预测 32 步，保存完整 32 步并用下一组真实观测重新预测，
# 首个真实 latent 与所有保留的预测 latent 拼接后一次解码为 video.mp4。
# MP4 沿用 EveryNDrawSample 的保存函数、默认质量和 16 FPS；动作写成 action.npy。
# 使用与 server 相同的推理引擎，每 32 步接收数据集真实观测；预测画面不进入历史。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

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

export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/path/to/trained/checkpoint}"
export ACTION_SOURCES_FILE="${ACTION_SOURCES_FILE:-$SCRIPT_DIR/sources/robotwin.json}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/path/to/robotwin_processed}"
export ROBOTWIN_GROUP_STATS_PATH="${ROBOTWIN_GROUP_STATS_PATH:-/path/to/robotwin_group_stats.json}"
export COSMOS3_EDGE_PROCESSOR_PATH="${COSMOS3_EDGE_PROCESSOR_PATH:-/path/to/Cosmos3-Edge}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-/path/to/Wan2.2_VAE.pth}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/causal_action/inference_robotwin}"

# 与训练配方对齐：896 action、重叠 161、历史 32 个 block。一次只取一个片段。
export MODE=policy
export SOURCE_INDEX="${SOURCE_INDEX:-0}"
export SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
export DATA_SPLIT_SEED="${DATA_SPLIT_SEED:-42}"
export MAX_ACTION_STEPS="${MAX_ACTION_STEPS:-896}"
export OVERLAP_ACTION_STEPS="${OVERLAP_ACTION_STEPS:-161}"
export HISTORY_BLOCKS="${HISTORY_BLOCKS:-32}"
export ACTIONS_PER_BLOCK="${ACTIONS_PER_BLOCK:-32}"
export VIDEO_STRIDE="${VIDEO_STRIDE:-4}"
export NUM_STEPS="${NUM_STEPS:-10}"
export GUIDANCE="${GUIDANCE:-1}"

mkdir -p "$OUTPUT_ROOT"

bash "$SCRIPT_DIR/inference_causal_action.sh" \
    --execute-steps "${EXECUTE_STEPS:-$ACTIONS_PER_BLOCK}" \
    --decode-video \
    "$@"

[[ "${PRINT_ONLY:-0}" == 1 ]] && exit 0

python - <<'PY'
import os
from pathlib import Path

import numpy as np
import torch

output = Path(os.environ["OUTPUT_ROOT"])
payload = torch.load(output / "sample.pt", map_location="cpu", weights_only=True)
action = payload["action"][0].detach().float().cpu().numpy()
path = output / "action.npy"
np.save(path, action)
print(f"Saved {path} shape={tuple(action.shape)} dtype={action.dtype}")
PY
