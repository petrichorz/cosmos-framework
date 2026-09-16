#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 先在当前机器激活 Python/torch_npu 与 CANN 环境；不依赖固定 conda 路径。
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export COSMOS_DEVICE=npu
export COSMOS_GEN_BSA64=0
export COSMOS_BSA64_BUCKETS="${COSMOS_BSA64_BUCKETS:-0}"

# 与 profile_local 相同的路径接口，按目标机器修改或通过环境变量覆盖。
export DATASET_DIR="${DATASET_DIR:-$SCRIPT_DIR/data/Cosmos3-DROID/success}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$SCRIPT_DIR/checkpoints/Cosmos3-Edge-DCP}"
export COSMOS3_EDGE_PROCESSOR_PATH="${COSMOS3_EDGE_PROCESSOR_PATH:-$SCRIPT_DIR/checkpoints/Cosmos3-Edge}"
export VAE_PATH="${VAE_PATH:-$SCRIPT_DIR/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/../outputs/vision_edge_tnd_profile}"

# 单机保持 NNODES=1、NODE_RANK=0；多机共用 master 地址及端口。
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-50012}"

bash "$SCRIPT_DIR/launch_causal_edge_tnd_profiling.sh" "$@"
