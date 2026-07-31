# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Single-softmax Dense attention for causal teacher-forcing GEN queries."""

import torch


def _validate_dense_attention_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed_mask: torch.Tensor,
) -> None:
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError(
            "query, key, and value must use heads-last [tokens, heads, head_dim] tensors, "
            f"got {tuple(query.shape)}, {tuple(key.shape)}, and {tuple(value.shape)}"
        )
    if key.shape[0] != value.shape[0] or key.shape[1] != value.shape[1]:
        raise ValueError(
            "key and value must have matching token and head dimensions, "
            f"got {tuple(key.shape)} and {tuple(value.shape)}"
        )
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(f"query and key head dimension must match, got {query.shape[-1]} and {key.shape[-1]}")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError(
            f"query head count must be divisible by KV head count, got {query.shape[1]} and {key.shape[1]}"
        )
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise TypeError(
            f"query, key, and value must have the same dtype, got {query.dtype}, {key.dtype}, and {value.dtype}"
        )
    if query.device != key.device or query.device != value.device:
        raise ValueError(
            f"query, key, and value must be on the same device, got {query.device}, {key.device}, and {value.device}"
        )
    if allowed_mask.dtype != torch.bool:
        raise TypeError(f"allowed_mask must use bool dtype, got {allowed_mask.dtype}")
    expected_mask_shape = (query.shape[0], key.shape[0])
    if allowed_mask.shape != expected_mask_shape:
        raise ValueError(f"allowed_mask must have shape {expected_mask_shape}, got {tuple(allowed_mask.shape)}")
    if allowed_mask.device != query.device:
        raise ValueError(
            f"allowed_mask must be on the same device as query, got {allowed_mask.device} and {query.device}"
        )
    if not bool(allowed_mask.any(dim=-1).all()):
        raise ValueError("every teacher-forcing query must have at least one visible key")


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

    _validate_dense_attention_inputs(query, key, value, allowed_mask)

    repeats = query.shape[1] // key.shape[1]
    if repeats > 1:
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)

    query_heads_first = query.transpose(0, 1).unsqueeze(0)
    key_heads_first = key.transpose(0, 1).unsqueeze(0)
    value_heads_first = value.transpose(0, 1).unsqueeze(0)
    broadcast_mask = allowed_mask.unsqueeze(0).unsqueeze(0)
    output = torch.nn.functional.scaled_dot_product_attention(
        query_heads_first,
        key_heads_first,
        value_heads_first,
        attn_mask=broadcast_mask,
        dropout_p=0.0,
        scale=scale,
    )
    return output.squeeze(0).transpose(0, 1)
