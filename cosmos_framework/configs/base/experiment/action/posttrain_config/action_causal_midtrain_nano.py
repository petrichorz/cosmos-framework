# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Nano multi-source causal action mid-training with the shared action pipeline."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_causal_midtrain_edge import cfg as edge_cfg
from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.utils.lazy_config import LazyDict

# Keep causal geometry, data loading and training policy shared with Edge.
# Replace entire subtrees so Edge-only backbone and loss fields do not leak in.
cfg = copy.deepcopy(edge_cfg)
cfg.job.name = "action_causal_midtrain_nano"
for name in ("vlm_config", "rectified_flow_training_config"):
    cfg.model.config[name] = LazyDict(copy.deepcopy(NANO_MODEL_CONFIG[name]), flags={"allow_objects": True})

ConfigStore.instance().store(group="experiment", package="_global_", name="action_causal_midtrain_nano", node=cfg)
