#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# 通用环境模板：将 /path/to 占位路径替换为实际路径，或通过环境变量覆盖。
# 本机配置见 launch_midtrain_local.sh。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ASCEND_ENV="${ASCEND_ENV:-/path/to/Ascend/ascend-toolkit/set_env.sh}"
CONDA_ROOT="${CONDA_ROOT:-/path/to/miniforge3}"
CONDA_ENV="${CONDA_ENV:-cosmos-framework-py312}"

set +u
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
source "$ASCEND_ENV"
set -u
# 环境激活脚本可能遗留其他 Python 版本的 Torch 路径；优先使用当前解释器的库。
TORCH_LIB="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/torch/lib")')"
export LD_LIBRARY_PATH="$TORCH_LIB:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

unset TORCH_DEVICE_BACKEND_AUTOLOAD

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-50142}"

export AGIBOT_ROOT="${AGIBOT_ROOT:-/path/to/agibot_processed}"
export EGOSUITE_ROOT="${EGOSUITE_ROOT:-/path/to/egosuite_processed}"
export AGIBOT_GROUP_STATS_PATH="${AGIBOT_GROUP_STATS_PATH:-/path/to/agibot_group_stats.json}"
export EGOSUITE_GROUP_STATS_PATH="${EGOSUITE_GROUP_STATS_PATH:-/path/to/egosuite_group_stats.json}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/path/to/Cosmos3-Edge-DCP}"
export COSMOS3_EDGE_PROCESSOR_PATH="${COSMOS3_EDGE_PROCESSOR_PATH:-/path/to/Cosmos3-Edge}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-/path/to/Wan2.2_VAE.pth}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/causal_action/midtrain}"
export MROPE_BASE_FPS="${MROPE_BASE_FPS:-24}"

exec bash "$SCRIPT_DIR/launch_midtrain_action_causal.sh" "$@"
