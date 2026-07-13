# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

SDPA backend checks.

Unlike flash2 / flash3 / natten, the SDPA backend is device-agnostic: it does
NOT gate on a CUDA ``arch_tag``. It is compatible whenever the tensors are 4-D,
heads-last, and of a dtype ``scaled_dot_product_attention`` supports
(fp16 / bf16 / fp32). This is what makes it usable as the Ascend NPU fallback.
"""

from functools import partial

import torch

from cosmos_framework.model.attention.checks import attention_param_checks, attention_tensor_checks
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.utils import log_or_raise_error


# scaled_dot_product_attention accepts fp16 / bf16 / fp32 on CUDA and (via
# torch_npu) on Ascend. fp8 is intentionally excluded - it is backend-specific.
SDPA_SUPPORTED_DTYPES: list[torch.dtype] = [torch.float16, torch.bfloat16, torch.float32]


def sdpa_attention_check(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    is_causal: bool,
    causal_type: CausalType,
    is_varlen: bool,
    deterministic: bool = False,
    raise_error: bool = False,
) -> bool:
    """
    Input validation function for the SDPA backend.

    Device-agnostic: no CUDA ``arch_tag`` requirement (unlike flash2 / flash3 /
    natten). Returns ``True`` whenever the universal tensor/dtype/shape checks
    pass and the causal mode is one SDPA can express.

    Parameters:
        query_shape (torch.Size): Shape of 4-D query tensor (`[batch, seqlen, heads, head_dim]`).

        key_shape (torch.Size): Shape of 4-D key tensor (`[batch, seqlen_kv, heads_kv, head_dim]`).

        value_shape (torch.Size): Shape of 4-D value tensor (`[batch, seqlen_kv, heads_kv, head_dim_v]`).

        dtype (torch.dtype): Data type of tensors.

        device (torch.device): Device of tensors (unused - SDPA is device-agnostic).

        requires_grad (bool): Whether tensors require gradients (training vs inference).

        is_causal (bool): whether or not causal masking is enabled.

        causal_type (CausalType): causal masking mode. SDPA's ``is_causal=True``
            implements TopLeft causal; ``DontCare`` (seqlen_q == seqlen_kv) is
            also expressible. ``BottomRight`` is NOT yet implemented.

        is_varlen (bool): whether or not a variable length (varlen) use case.

        deterministic (bool): Deterministic backward pass required. (Accepted
            but not enforced - SDPA's backward determinism is backend-specific.)

        raise_error (bool): whether to raise an error if any checks fail,
            instead of just returning False. Default is False.

    Returns:
        success (bool): whether use case is compatible with the SDPA backend.
    """
    target_fn = partial(log_or_raise_error, raise_error=raise_error)

    if not attention_tensor_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        requires_grad=requires_grad,
        supported_dtypes_forward=SDPA_SUPPORTED_DTYPES,
        supported_dtypes_backward=SDPA_SUPPORTED_DTYPES,
        supports_mla=False,
        supports_gqa_mqa=True,
        raise_error=raise_error,
        backend_name="SDPA (scaled_dot_product_attention)",
    ):
        target_fn("SDPA backend does not support the given inputs.", exception=RuntimeError)
        return False

    # Verifies causal_type is a CausalType instance when is_causal, and that
    # DontCare is only used when seqlen_q == seqlen_kv.
    attention_param_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        is_causal=is_causal,
        causal_type=causal_type,
    )

    # SDPA's is_causal=True is TopLeft (lower-triangular). BottomRight needs a
    # shifted additive mask, which is not yet wired up. MoT only uses
    # TopLeft / DontCare, so this does not block current paths.
    if is_causal and causal_type == CausalType.BottomRight:
        target_fn(
            "SDPA backend does not yet support CausalType.BottomRight. "
            "Use TopLeft / DontCare, or another backend.",
            exception=NotImplementedError,
        )
        return False

    return True
