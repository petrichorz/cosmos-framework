#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/path/to/checkpoints/iter_000000400}"
export CONFIG_FILE="${CONFIG_FILE:-/path/to/run/config.yaml}"
export IMAGE_PATH="${IMAGE_PATH:-/path/to/image.jpg}"
export PROMPT="${PROMPT:-A person picks up the object on the table.}"
export PROMPT_FILE="${PROMPT_FILE:-}" # 非空时从文件读取文本
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/causal_ti2v/$(date +%Y%m%d_%H%M%S)}"

export RESOLUTION="${RESOLUTION:-480}"
export ASPECT_RATIO="${ASPECT_RATIO:-4,3}"
export FPS="${FPS:-15}"
export NUM_BLOCKS="${NUM_BLOCKS:-56}" # Wan VAE: 1+4*56*2=449 帧，约 30 秒
export BLOCK_SIZE="${BLOCK_SIZE:-2}"
export HISTORY_BLOCKS="${HISTORY_BLOCKS:-16}"
export NUM_STEPS="${NUM_STEPS:-35}"
export GUIDANCE="${GUIDANCE:-1.0}"
export SEED="${SEED:-1}"

export COSMOS_DEVICE=npu
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID:-0}"
conda run --no-capture-output -n "${ENV_NAME:-cosmos-framework-py312}" \
  python -m cosmos_framework.inference.causal_ti2v
