#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Copy this file, replace the /path/to placeholders, and use it as the
# machine-specific entrypoint for causal TND pre-training.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ASCEND_ENV="${ASCEND_ENV:-/path/to/Ascend/ascend-toolkit/set_env.sh}"
CONDA_ROOT="${CONDA_ROOT:-/path/to/miniforge3}"
CONDA_ENV="${CONDA_ENV:-cosmos-framework}"

[[ -f "$ASCEND_ENV" ]] || { echo "ERROR: missing CANN environment: $ASCEND_ENV" >&2; exit 1; }
[[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]] || {
    echo "ERROR: missing conda initialization under $CONDA_ROOT" >&2
    exit 1
}

source "$ASCEND_ENV"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-50012}"

export DATASET_PATH="${DATASET_PATH:-/path/to/lerobot_v3_dataset_or_parent}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/path/to/Cosmos3-Edge-DCP}"
export COSMOS3_EDGE_PROCESSOR_PATH="${COSMOS3_EDGE_PROCESSOR_PATH:-/path/to/Cosmos3-Edge}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-/path/to/Wan2.2_VAE.pth}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/output/vision_causal_edge_tnd_pretrain}"
export MROPE_BASE_FPS="${MROPE_BASE_FPS:-24}"

exec bash "$SCRIPT_DIR/launch_pretrain_vision_causal.sh" "$@"
