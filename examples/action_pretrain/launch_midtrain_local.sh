#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# 本机 Ascend 训练入口：复制通用模板并填入本机路径，可通过环境变量覆盖。

# 启动前请在外部环境中准备 CANN 和所需动态库搜索路径。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# 数据来源：agibot / egosuite / mixed；显式 ACTION_SOURCES_FILE 优先。
# 自定义 mixed 权重时，将 ACTION_SOURCES_FILE 指向更新后的来源清单。
export SOURCE_SET="${SOURCE_SET:-mixed}"
export ACTION_SOURCES_FILE="${ACTION_SOURCES_FILE:-$SCRIPT_DIR/sources/$SOURCE_SET.json}"
# 与 README 中统计命令使用同一个 RUN_DIR；新实验请换目录。
export RUN_DIR="${RUN_DIR:-${REPO_ROOT}/outputs/group_long_20260927}"
export AGIBOT_GROUP_STATS_PATH="${AGIBOT_GROUP_STATS_PATH:-$RUN_DIR/stats/agibot_group_stats.json}"
export EGOSUITE_GROUP_STATS_PATH="${EGOSUITE_GROUP_STATS_PATH:-$RUN_DIR/stats/egosuite_group_stats.json}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_DIR/train_$SOURCE_SET}"
export TOML_PATH="${TOML_PATH:-$SCRIPT_DIR/action_midtrain_edge_causal_tnd.toml}"
# joint 为 FD / ID / Policy 联训；与数据来源的 mixed 是两个独立设置。
export MODE="${MODE:-joint}"
export ALLOW_MOCK_STATISTICS="${ALLOW_MOCK_STATISTICS:-false}"
export LOOKAHEAD_LIMIT="${LOOKAHEAD_LIMIT:-1}"

CONDA_ROOT="${CONDA_ROOT:-/mnt/sfs_turbo/public/apps/miniforge3}"
CONDA_ENV="${CONDA_ENV:-cosmos-framework-py312}"

set +u
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

unset TORCH_DEVICE_BACKEND_AUTOLOAD

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-50142}"

export AGIBOT_ROOT="${AGIBOT_ROOT:-/mnt/sfs_turbo/public/datasets/agibot_processed/v0}"
export EGOSUITE_ROOT="${EGOSUITE_ROOT:-/mnt/sfs_turbo/public/datasets/egosuite_processed}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge-DCP}"
export COSMOS3_EDGE_PROCESSOR_PATH="${COSMOS3_EDGE_PROCESSOR_PATH:-/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-/mnt/sfs_turbo/public/ckpts/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
export MROPE_BASE_FPS="${MROPE_BASE_FPS:-24}"

exec bash "$SCRIPT_DIR/launch_midtrain_action_causal.sh" "$@"
