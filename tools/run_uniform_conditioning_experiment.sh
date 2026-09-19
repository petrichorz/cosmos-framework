#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Activate cosmos-framework-py312 and CANN before running.
export DATASET_PATH="${DATASET_PATH:-/path/to/egosuite_demo_v1}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/path/to/Cosmos3-Edge-DCP}"
export COSMOS3_EDGE_PROCESSOR_PATH="${COSMOS3_EDGE_PROCESSOR_PATH:-/path/to/Cosmos3-Edge}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-/path/to/Wan2.2_VAE.pth}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/uniform_conditioning/train}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}" MASTER_PORT="${MASTER_PORT:-50128}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-52300}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-54200-54300}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
bash examples/launch_pretrain_vision_causal.sh \
 trainer.max_iter=60 checkpoint.save_iter=60 model.config.ema.enabled=false \
 'dataloader_train.dataloader.datasets.video.dataset.resolution_tiers=["480"]' \
 dataloader_train.dataloader.datasets.video.dataset.max_video_duration_s=30.0 \
 dataloader_train.max_sequence_length=null +dataloader_train.max_samples_per_batch=1 \
 dataloader_train.dataloader.datasets.video.dataset.use_multi_fps=false \
 dataloader_train.dataloader.num_workers=4 \
 "$@"
