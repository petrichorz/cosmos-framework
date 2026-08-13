# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dense SDPA backend with an explicit arbitrary boolean visibility mask."""

from cosmos_framework.model.attention.masked_sdpa.functions import masked_sdpa_attention

__all__ = ["masked_sdpa_attention"]
