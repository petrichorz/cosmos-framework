# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Teacher-forcing adapter for the registered explicit-mask attention backend."""

import torch

from cosmos_framework.model.attention import attention


def teacher_forcing_dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed_mask: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Attend GEN queries once over unified UND/clean/noisy keys.

    ``allowed_mask[q, k] == True`` means key ``k`` is visible to query ``q``.
    The implementation uses one SDPA softmax and never exposes or merges LSE.
    """

    output = attention(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        backend="masked_sdpa",
        backend_kwargs={"allowed_mask": allowed_mask},
        scale=scale,
    )
    return output.squeeze(0)
