#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Single-Ascend-NPU functional smoke for Scheme-B causal teacher forcing.
# This deliberately uses T2V-only 17-frame clips and runs three optimizer steps.

TOML_FILE="examples/toml/sft_config/vision_causal_smoke_edge.toml"
: "${DATASET_PATH:=examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
: "${NPROC_PER_NODE:=1}"

TAIL_OVERRIDES=(
    "model=mot_causal_ddp"
    "dataloader_train.max_sequence_length=null"
    "~dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:0.7,1:0.2,2:0.1}"
    "+dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:1.0,1:0.0,2:0.0}"
    "dataloader_train.dataloader.datasets.video.dataset.num_video_frames=17"
)

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }'

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
