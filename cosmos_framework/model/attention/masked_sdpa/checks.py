# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compatibility checks for the explicit-mask SDPA backend."""

from functools import partial

import torch

from cosmos_framework.model.attention.checks import attention_param_checks, attention_tensor_checks
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.utils import log_or_raise_error

MASKED_SDPA_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def masked_sdpa_attention_check(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    is_causal: bool,
    causal_type: CausalType | None,
    is_varlen: bool,
    deterministic: bool = False,
    raise_error: bool = False,
) -> bool:
    """Return whether explicit-mask dense SDPA can handle the requested tensors."""
    del deterministic
    target_fn = partial(log_or_raise_error, raise_error=raise_error)
    supported_dtypes = MASKED_SDPA_DTYPES + ([torch.float64] if device.type == "cpu" else [])

    if is_varlen:
        target_fn(
            "masked_sdpa accepts one dense packed sequence; sample isolation must be encoded in allowed_mask.",
            exception=ValueError,
        )
        return False
    if is_causal:
        target_fn(
            "masked_sdpa does not combine is_causal with allowed_mask; encode all visibility in allowed_mask.",
            exception=ValueError,
        )
        return False
    if not attention_tensor_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        requires_grad=requires_grad,
        supported_dtypes_forward=supported_dtypes,
        supported_dtypes_backward=supported_dtypes,
        supports_mla=False,
        supports_gqa_mqa=True,
        raise_error=raise_error,
        backend_name="Explicit-mask SDPA (masked_sdpa)",
    ):
        return False

    attention_param_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        is_causal=is_causal,
        causal_type=causal_type,
    )
    return True
