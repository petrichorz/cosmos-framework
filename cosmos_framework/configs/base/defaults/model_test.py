# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.configs.base.defaults.model import (
    MOT_CAUSAL_DDP_CONFIG,
    MOT_CAUSAL_FSDP_CONFIG,
    MOT_DDP_CONFIG,
    MOT_FSDP_CONFIG,
)
from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


def test_causal_model_groups_select_causal_subclass_without_changing_base_groups():
    assert MOT_DDP_CONFIG["model"]["_target_"] is OmniMoTModel
    assert MOT_FSDP_CONFIG["model"]["_target_"] is OmniMoTModel
    assert MOT_CAUSAL_DDP_CONFIG["model"]["_target_"] is OmniMoTCausalModel
    assert MOT_CAUSAL_FSDP_CONFIG["model"]["_target_"] is OmniMoTCausalModel


def test_causal_model_groups_enable_teacher_forcing():
    for group in (MOT_CAUSAL_DDP_CONFIG, MOT_CAUSAL_FSDP_CONFIG):
        config = group["model"]["config"]
        assert config.causal_training_strategy == "teacher_forcing"
        assert config.joint_attn_implementation == "teacher_forcing"
        assert config.teacher_forcing_block_size_min == 1
        assert config.teacher_forcing_block_size_max == 4
        assert config.teacher_forcing_history_blocks_min == 1
        assert config.teacher_forcing_history_blocks_max == 32
        assert config.teacher_forcing_dense_mode == "global"
        assert config.teacher_forcing_visualize_sdpa_mask is False
