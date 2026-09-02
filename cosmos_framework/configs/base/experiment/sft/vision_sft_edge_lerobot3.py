# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Edge causal vision SFT recipe backed by full LeRobot v3 episodes."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.vision_sft_edge import vision_sft_edge
from cosmos_framework.data.generator.local_datasets.lerobot_episode_sft_dataset import (
    get_lerobot_episode_sft_dataset,
)
from cosmos_framework.utils.lazy_config import LazyCall as L

vision_sft_edge_lerobot3 = copy.deepcopy(vision_sft_edge)
vision_sft_edge_lerobot3.job.name = "vision_sft_edge_lerobot3"
# Unlike the generic Edge SFT recipe, this experiment is directly runnable as
# causal teacher forcing and does not rely on a launch-script model override.
vision_sft_edge_lerobot3.defaults[0] = {"override /model": "mot_causal_fsdp"}

# Full episodes are large. Keep worker-side in-flight memory bounded by default;
# users can still override these loader settings explicitly for their hardware.
vision_sft_edge_lerobot3.dataloader_train.dataloader.num_workers = 2
vision_sft_edge_lerobot3.dataloader_train.dataloader.prefetch_factor = 1
vision_sft_edge_lerobot3.dataloader_train.dataloader.pin_memory = False
vision_sft_edge_lerobot3.dataloader_train.dataloader.datasets.video.dataset = L(get_lerobot_episode_sft_dataset)(
    roots="${oc.env:DATASET_PATH}",
    metadata_load_workers=8,
    video_view="head",
    video_view_aliases=None,
    resolution="256",
    max_video_fps=15.0,
    caption_key=None,
    min_short_edge=0,
    temporal_compression_factor=4,
    num_ffmpeg_threads=1,
    cfg_dropout_rate=0.1,
    use_system_prompt=False,
    max_caption_tokens=2048,
    append_duration_fps_timestamps=True,
    append_resolution_info=True,
    cfg_dropout_keep_metadata=False,
    caption_suffix="",
    conditioning_fps=-1,
    conditioning_fps_noise_std=0.0,
    # The causal teacher-forcing workflow trains the T2V geometry only.
    conditioning_config={0: 1.0},
    tokenizer_config="${model.config.vlm_config.tokenizer}",
)


ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="vision_sft_edge_lerobot3",
    node=vision_sft_edge_lerobot3,
)
