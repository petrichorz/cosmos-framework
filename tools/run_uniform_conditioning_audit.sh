#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Same dataset/checkpoint/processor/VAE environment variables as the training launcher.
export COSMOS_DEVICE=npu
export IMAGINAIRE_OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/uniform_conditioning/online_audit_fp32}"
export CACHE_PAIR_RTOL="${CACHE_PAIR_RTOL-0.00003}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-53500}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-54000-54100}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node="${NPROC_PER_NODE:-4}" --master_port="${MASTER_PORT:-50139}" -m tools.train_uniform_conditioning_audit \
 --sft-toml examples/toml/sft_config/vision_pretrain_edge_causal_tnd.toml -- \
 model=mot_causal_fsdp model.config.vlm_config.tokenizer.repository=null model.config.vlm_config.tokenizer.revision=null \
 "+model.config.vlm_config.tokenizer.tokenizer_type=$COSMOS3_EDGE_PROCESSOR_PATH" \
 model.config.precision=float32 model.config.parallelism.fsdp_mixed_precision_enabled=false trainer.max_iter=1 checkpoint.save_iter=1000 model.config.ema.enabled=true \
 'dataloader_train.dataloader.datasets.video.dataset.resolution_tiers=["256"]' \
 dataloader_train.dataloader.datasets.video.dataset.max_video_duration_s=2.0 \
 dataloader_train.dataloader.datasets.video.dataset.min_video_frames=17 \
 dataloader_train.dataloader.datasets.video.dataset.video_window_overlap_s=0.0 \
 dataloader_train.dataloader.datasets.video.dataset.use_multi_fps=false \
 'dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:0.0,1:0.0,2:1.0}' \
 dataloader_train.max_sequence_length=null +dataloader_train.max_samples_per_batch=1 \
 trainer.callbacks.every_n_sample_reg.every_n=1 trainer.callbacks.every_n_sample_ema.every_n=1 \
 trainer.callbacks.every_n_sample_reg.n_viz_sample=1 trainer.callbacks.every_n_sample_ema.n_viz_sample=1 \
 trainer.callbacks.every_n_sample_reg.num_sampling_step=2 trainer.callbacks.every_n_sample_ema.num_sampling_step=2 \
 'trainer.callbacks.every_n_sample_reg.guidance=[1.0,3.0]' 'trainer.callbacks.every_n_sample_ema.guidance=[1.0]' \
 trainer.callbacks.every_n_sample_reg.causal_num_blocks=3 trainer.callbacks.every_n_sample_ema.causal_num_blocks=3 \
 trainer.callbacks.every_n_sample_reg.causal_block_size=2 trainer.callbacks.every_n_sample_ema.causal_block_size=2 \
 trainer.callbacks.every_n_sample_reg.causal_history_blocks=1 trainer.callbacks.every_n_sample_ema.causal_history_blocks=1 \
 trainer.callbacks.every_n_sample_reg.causal_condition_frames=2 trainer.callbacks.every_n_sample_ema.causal_condition_frames=0 \
 trainer.callbacks.every_n_sample_reg.causal_use_kv_cache=false trainer.callbacks.every_n_sample_ema.causal_use_kv_cache=true \
 trainer.callbacks.device_monitor.every_n=1 "$@"
