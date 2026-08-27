#!/usr/bin/env bash
set -euo pipefail

# 当前机器的运行环境
export HF_HUB_OFFLINE=1   # 关闭huggingface联网，只用本地缓存模型
export COSMOS_DEVICE=npu  # Cosmos 模型强制在NPU芯片上运行

# 当前的数据集、权重和输出路径
# 【LeRobot 3.x 适配】DATASET_PATH 改为 LeRobot 数据集根目录（含 meta/info.json）
export DATASET_PATH="/mi/data2T/liujin/dataset/toy_lerobot3_multi_with_caption"
# 【留档】原 JSONL 数据集路径（改用 LeRobot 后注释掉，未删除）
# export DATASET_PATH="/mi/data2T/Embodied-AI/datasets/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge"
export BASE_CHECKPOINT_PATH="/mi/data2T/Embodied-AI/ckpts/Cosmos/Cosmos3-Edge-DCP"
export COSMOS3_EDGE_PROCESSOR_PATH="/mi/data2T/Embodied-AI/ckpts/Cosmos/Cosmos3-Edge"
export WAN_VAE_PATH="/mi/data2T/Embodied-AI/ckpts/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
export OUTPUT_ROOT="/mi/data2T/liujin/code/cosmos_ascend/cosmos_trainging_logs"

# 配置Huggingface缓存
if [ ! -e ~/.cache/huggingface ]; then
    mkdir -p ~/.cache
    ln -s /mi/data2T/liujin/ckpts/huggingface ~/.cache/huggingface
fi

# torchrun 单机单卡设置
export ASCEND_RT_VISIBLE_DEVICES="10"  
export NPROC_PER_NODE=1
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=50012

# # torchrun 多机多卡 云道配置
# export NPROC_PER_NODE="$MA_NUM_GPUS"
# export NNODES="$MA_NUM_HOSTS"
# export NODE_RANK="$VC_TASK_INDEX"          # 当前机器序号 VC_TASK_INDEX
# export MASTER_ADDR="${VC_WORKER_HOSTS%%,*}" # 主节点（Rank 0）的内网IP
# export MASTER_PORT="${MASTER_PORT:-50012}"

# 切换conda环境
CONDA_HOME="/mi/sfs_turbo/lilin_v1/anaconda3"
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate cosmos-framework


# 安装当前cosmos-framework包
cd /mi/data2T/liujin/code/cosmos_ascend/cosmos-framework
pip install -e .
cd ..

# 补充torchcodec需要的库
# 编译 torchcodec 全过程（Python 解释器、pybind11、FFmpeg、cmake 依赖、运行时动态库）全部使用指定的 conda 虚拟环境，隔绝系统环境的库，避免版本冲突。
export PYTHON_BIN=$CONDA_PREFIX/bin/python
export pybind11_DIR=$($PYTHON_BIN -m pybind11 --cmakedir)
export FFMPEG_ROOT=$CONDA_PREFIX
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for vision_sft_edge (T2V / I2V / V2V vision-only
# SFT on Nemotron-2B-Dense-VL / Cosmos3-Edge, 8-GPU FSDP). Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/vision_sft_edge.toml.
#
# Optional env vars (defaults below point under examples/; override to put
# data or checkpoints on a different filesystem):
#   DATASET_PATH          default: examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge
#                         (must contain train/video_dataset_file.jsonl)
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Edge
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   HF_TOKEN              not needed for nvidia/Cosmos3-Edge (the repo is
#                         ungated); set only if another download requires it
#   OUTPUT_ROOT           default: outputs/train
#
# Usage (8-GPU allocation, inside the training container, from the repo root):
#   bash examples/launch_sft_vision_edge.sh

TOML_FILE="examples/toml/sft_config/vision_sft_edge.toml"
: "${DATASET_PATH:=examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"

# 【LeRobot 3.x 适配】校验 LeRobot 数据集：$DATASET_PATH 下任意深度存在 meta/info.json（支持父目录下多个数据集）
EXTRA_DATASET_CHECK='[[ -n "$(find "$DATASET_PATH" -path "*/meta/info.json" -print -quit)" ]] || { echo "ERROR: no meta/info.json found under $DATASET_PATH" >&2; exit 1; }'
# 【留档】原 JSONL 校验（改用 LeRobot 后注释掉，未删除）
# EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }'
TAIL_OVERRIDES=(
      "model.config.vlm_config.tokenizer.repository=null"
      "model.config.vlm_config.tokenizer.revision=null"
      "+model.config.vlm_config.tokenizer.tokenizer_type=$COSMOS3_EDGE_PROCESSOR_PATH"
  )

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
