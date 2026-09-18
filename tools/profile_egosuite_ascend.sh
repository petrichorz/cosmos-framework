#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Four-rank Ascend profiling with Python stacks. Activate the CANN/torch_npu
# environment and configure the paths documented by launch_pretrain_template.sh.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -n "${PROFILE_RUN_DIR:-}" ]] && export OUTPUT_ROOT="$PROFILE_RUN_DIR"

exec bash "$REPO_ROOT/examples/launch_pretrain_vision_causal.sh" \
  trainer.max_iter=9 trainer.profiling.enable_profiling=true \
  trainer.profiling.profile_freq=8 trainer.profiling.profile_warmup=2 \
  trainer.profiling.profile_active=2 \
  'trainer.profiling.target_ranks=[0,1,2,3]' \
  trainer.profiling.record_shape=true trainer.profiling.profile_memory=true \
  trainer.profiling.with_stack=true trainer.profiling.with_modules=true \
  model.config.ema.enabled=false \
  dataloader_train.dataloader.num_workers=8 \
  'dataloader_train.dataloader.datasets.video.dataset.resolution="256"' \
  "$@"
