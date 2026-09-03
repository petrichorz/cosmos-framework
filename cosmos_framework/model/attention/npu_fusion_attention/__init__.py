# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Ascend TND variable-length fused-attention backend."""

from importlib.util import find_spec

from cosmos_framework.model.attention.npu_fusion_attention.functions import npu_fusion_attention


# Avoid importing torch_npu while importing the attention package. Training
# entrypoints initialise it separately, and CPU/CUDA environments must remain
# able to import this module without the optional Ascend dependency installed.
NPU_FUSION_ATTENTION_SUPPORTED: bool = find_spec("torch_npu") is not None

__all__ = ["NPU_FUSION_ATTENTION_SUPPORTED", "npu_fusion_attention"]
