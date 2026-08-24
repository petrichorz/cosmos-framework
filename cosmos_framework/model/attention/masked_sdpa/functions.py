# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Explicit-mask dense attention implemented with PyTorch SDPA."""

import torch
from torch import Tensor

from cosmos_framework.model.attention.checks import assert_universal_tensor_checks
from cosmos_framework.model.attention.masked_sdpa.checks import masked_sdpa_attention_check
from cosmos_framework.model.attention.masks import CausalType


def masked_sdpa_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    causal_type: CausalType | None = None,
    scale: float | None = None,
    cumulative_seqlen_Q: Tensor | None = None,
    cumulative_seqlen_KV: Tensor | None = None,
    max_seqlen_Q: int | None = None,
    max_seqlen_KV: int | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor:
    """Run one dense SDPA call using ``allowed_mask`` (``True`` means visible)."""
    del max_seqlen_Q, max_seqlen_KV
    assert_universal_tensor_checks(query, key, value)
    is_varlen = cumulative_seqlen_Q is not None or cumulative_seqlen_KV is not None
    assert masked_sdpa_attention_check(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        dtype=query.dtype,
        device=query.device,
        requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
        is_causal=is_causal,
        causal_type=causal_type,
        is_varlen=is_varlen,
        deterministic=deterministic,
        raise_error=True,
    )
    if return_lse:
        raise NotImplementedError("masked_sdpa does not expose logsumexp.")

    kwargs = backend_kwargs.copy() if backend_kwargs is not None else {}
    allowed_mask = kwargs.pop("allowed_mask", None)
    validate_allowed_mask = kwargs.pop("validate_allowed_mask", True)
    if kwargs:
        raise ValueError(f"Unsupported masked_sdpa backend kwargs: {sorted(kwargs)}")
    if not isinstance(validate_allowed_mask, bool):
        raise TypeError(f"validate_allowed_mask must be bool, got {type(validate_allowed_mask).__name__}")
    if allowed_mask is None:
        raise ValueError("masked_sdpa requires backend_kwargs['allowed_mask'].")
    if allowed_mask.dtype != torch.bool:
        raise TypeError(f"allowed_mask must use bool dtype, got {allowed_mask.dtype}")
    expected_mask_shape = (query.shape[1], key.shape[1])
    if tuple(allowed_mask.shape) != expected_mask_shape:
        raise ValueError(f"allowed_mask must have shape {expected_mask_shape}, got {tuple(allowed_mask.shape)}")
    if allowed_mask.device != query.device:
        raise ValueError(
            f"allowed_mask must be on the same device as query, got {allowed_mask.device} and {query.device}"
        )
    # Reducing a device mask and converting the result to Python bool synchronizes
    # the host with the accelerator. Keep validation enabled for generic callers,
    # but allow trusted mask builders to disable this repeated hot-path check.
    if validate_allowed_mask and not bool(allowed_mask.any(dim=-1).all()):
        raise ValueError("every masked_sdpa query must have at least one visible key")

    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    repeats = q.shape[1] // k.shape[1]
    if repeats > 1:
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    output = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=allowed_mask.unsqueeze(0).unsqueeze(0),
        dropout_p=0.0,
        scale=scale,
    )
    return output.transpose(1, 2)
