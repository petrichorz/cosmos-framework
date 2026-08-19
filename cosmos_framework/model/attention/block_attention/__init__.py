# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Ascend NPU block-sparse attention backend."""

from importlib.util import find_spec

from cosmos_framework.model.attention.block_attention.functions import block_attention


# Keep torch_npu optional at import time so CPU/CUDA environments can still
# import the attention package without Ascend dependencies.
BLOCK_ATTENTION_SUPPORTED: bool = find_spec("torch_npu") is not None

__all__ = ["BLOCK_ATTENTION_SUPPORTED", "block_attention"]
