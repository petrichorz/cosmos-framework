# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

SDPA Backend: device-agnostic fallback built on
``torch.nn.functional.scaled_dot_product_attention``.

Unlike the flash2 / flash3 / natten backends (which require a CUDA arch >= 75),
the SDPA backend runs on any device that PyTorch supports - including Ascend NPU,
where ``torch_npu`` registers a fused SDPA kernel. It is selected automatically
when ``get_arch_tag(device)`` returns 0 (i.e. not a CUDA device); see
``backends.get_backend_list``.
"""

from cosmos_framework.model.attention.sdpa.functions import sdpa_attention

# SDPA is torch-native; always importable. (Kept for parity with the
# ``FLASH2_SUPPORTED`` / ``NATTEN_SUPPORTED`` flags consumed by checks.)
SDPA_SUPPORTED: bool = True

__all__ = ["sdpa_attention", "SDPA_SUPPORTED"]
