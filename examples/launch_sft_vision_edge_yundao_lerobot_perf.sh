#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

# 当前机器的运行环境
export HF_HUB_OFFLINE=1   # 关闭huggingface联网，只用本地缓存模型
export COSMOS_DEVICE=npu  # Cosmos 模型强制在NPU芯片上运行

# 当前的数据集、权重和输出路径
# 【LeRobot 3.x 适配】DATASET_PATH 改为 LeRobot 数据集根目录（含 meta/info.json）
export DATASET_PATH="/mnt/sfs_turbo/public/datasets/egosuite_demo_v1"
# 【留档】原 JSONL 数据集路径（改用 LeRobot 后注释掉，未删除）
# export DATASET_PATH="/mi/data2T/Embodied-AI/datasets/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge"
export BASE_CHECKPOINT_PATH="/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge-DCP"
export COSMOS3_EDGE_PROCESSOR_PATH="/mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge"
export WAN_VAE_PATH="/mnt/sfs_turbo/public/ckpts/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
export OUTPUT_ROOT="/data5T/liujin/training_logs"

# ============ 数据加载性能评测（默认关闭，按需打开） ============
export PERFORMANCE_ANALYSIS_FILE_ROOT="${OUTPUT_ROOT}/performance_analysis_file"

# 1) worker 内细粒度打点（视频解码/后处理/caption 分词耗时）
#    开启后各进程往该目录写 events_*.jsonl，训练完用 summarize_performance_events 聚合
export COSMOS_PERF_OUTPUT_DIR="${PERFORMANCE_ANALYSIS_FILE_ROOT}/perf/events"

# 2) 训练迭代级 benchmark（单 iter 耗时 + 多卡同步等待）
#    开启后各 rank 写 benchmark_rank*.jsonl，rank0 训练结束自动聚合出 iterations.csv/summary.json
export COSMOS_BENCHMARK_DIR="${PERFORMANCE_ANALYSIS_FILE_ROOT}/benchmark"

# 3) 聚合输出目录（仅用于 Layer1 事后聚合脚本，非运行期必需）
export PERF_SUMMARY_DIR="${PERFORMANCE_ANALYSIS_FILE_ROOT}/perf/summary"



# 配置Huggingface缓存
if [ ! -e ~/.cache/huggingface ]; then
    mkdir -p ~/.cache
    ln -s /data5T/liujin/ckpts/huggingface ~/.cache/huggingface
fi

# torchrun 单机单卡设置
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"
export NPROC_PER_NODE=4
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
CONDA_HOME="/data5T/liujin/software/miniconda3"
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate cosmos-framework


# # 安装当前cosmos-framework包
# cd /data5T/liujin/code/cosmos-framework
# pip install -e .
# cd ..

# 补充torchcodec需要的库
# 编译 torchcodec 全过程（Python 解释器、pybind11、FFmpeg、cmake 依赖、运行时动态库）全部使用指定的 conda 虚拟环境，隔绝系统环境的库，避免版本冲突。
export PYTHON_BIN=$CONDA_PREFIX/bin/python
export pybind11_DIR=$($PYTHON_BIN -m pybind11 --cmakedir)
export FFMPEG_ROOT=$CONDA_PREFIX
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH



TOML_FILE="/data5T/liujin/code/cosmos-framework/examples/toml/sft_config/vision_pretrain_edge_causal_tnd.toml"
: "${DATASET_PATH:=examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"

# 【LeRobot 3.x 适配】_sft_launcher_common.sh 对 DATASET_PATH 做 -d（目录）硬检查，
# manifest 的 .jsonl 文件过不了。这里保存原始值，若为文件则临时指向其父目录通过检查。
_DATASET_ORIGINAL="$DATASET_PATH"
if [[ -f "$DATASET_PATH" && ! -d "$DATASET_PATH" ]]; then
    DATASET_PATH="$(dirname "$DATASET_PATH")"
fi

# EXTRA_DATASET_CHECK：校验原始路径存在 + 恢复 DATASET_PATH（供 config 的 ${oc.env:DATASET_PATH} 读取）
EXTRA_DATASET_CHECK="[[ -e \"$_DATASET_ORIGINAL\" ]] || { echo \"ERROR: dataset not found: $_DATASET_ORIGINAL\" >&2; exit 1; }; export DATASET_PATH=\"$_DATASET_ORIGINAL\";"
TAIL_OVERRIDES=(
      "model=mot_causal_fsdp"    # ← fsdp 版；若用 ddp 则写 model=mot_causal_ddp
      '~dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:0.7,1:0.2,2:0.1}'
      '+dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:1.0}'
      "model.config.vlm_config.tokenizer.repository=null"
      "model.config.vlm_config.tokenizer.revision=null"
      "+model.config.vlm_config.tokenizer.tokenizer_type=$COSMOS3_EDGE_PROCESSOR_PATH"
  )

# ============ 训练结束后自动聚合（EXIT trap 触发） ============
# 说明：_sft_launcher_common.sh 末尾有 exit，会让本脚本直接退出，source 之后的代码不会执行；
# 因此用 EXIT trap 在「训练进程结束、脚本退出前」自动触发聚合。
# Layer1 细粒度打点由这里聚合；Layer3 benchmark 已由 rank0 在训练末尾自动写 summary，无需处理。
_perf_aggregate_on_exit() {
    local events_dir="${COSMOS_PERF_OUTPUT_DIR:-}"
    local summary_dir="${PERF_SUMMARY_DIR:-}"
    if [ -z "$events_dir" ] || [ ! -d "$events_dir" ]; then
        return
    fi
    echo ">>> $(date '+%H:%M:%S') 聚合数据加载细粒度打点：$events_dir -> $summary_dir"
    if python -c "
from cosmos_framework.utils.performance import summarize_performance_events
rows = summarize_performance_events('$events_dir', '$summary_dir')
print(f'聚合完成，共 {len(rows)} 个打点阶段')
"; then
        echo ">>> 聚合完成：$summary_dir/performance_summary.{json,csv,md}"
    else
        echo ">>> 警告：性能聚合失败，请检查 $events_dir 是否有 events_*.jsonl" >&2
    fi
}
trap _perf_aggregate_on_exit EXIT

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
