# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compatibility checks for Ascend TND ``npu_fusion_attention``."""

from functools import partial

import torch

from cosmos_framework.model.attention.checks import attention_param_checks, attention_tensor_checks
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.npu_fusion_attention import NPU_FUSION_ATTENTION_SUPPORTED
from cosmos_framework.model.attention.utils import log_or_raise_error


NPU_FUSION_ATTENTION_DTYPES: list[torch.dtype] = [torch.float16, torch.bfloat16, torch.float32]


def npu_fusion_attention_check(
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
    """Return whether the inputs can use the Ascend TND fused operator."""
    target_fn = partial(log_or_raise_error, raise_error=raise_error)

    if not NPU_FUSION_ATTENTION_SUPPORTED or device.type != "npu":
        target_fn(
            "npu_fusion_attention requires torch_npu and Ascend NPU tensors.",
            exception=RuntimeError,
        )
        return False

    # This backend deliberately covers only the packed TND path. Dense calls
    # continue to use SDPA, which keeps this backend's semantics unambiguous.
    if not is_varlen:
        target_fn(
            "npu_fusion_attention backend currently supports only TND variable-length attention.",
            exception=ValueError,
        )
        return False

    if deterministic:
        target_fn(
            "npu_fusion_attention backend does not guarantee deterministic backward.",
            exception=ValueError,
        )
        return False

    if not attention_tensor_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        requires_grad=requires_grad,
        supported_dtypes_forward=NPU_FUSION_ATTENTION_DTYPES,
        supported_dtypes_backward=NPU_FUSION_ATTENTION_DTYPES,
        supports_mla=False,
        supports_gqa_mqa=True,
        raise_error=raise_error,
        backend_name="Ascend npu_fusion_attention (TND)",
    ):
        target_fn("npu_fusion_attention does not support the given inputs.", exception=RuntimeError)
        return False

    attention_param_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        is_causal=is_causal,
        causal_type=causal_type,
    )

    if is_causal and causal_type not in (CausalType.TopLeft, CausalType.DontCare):
        target_fn(
            "npu_fusion_attention TND backend currently supports TopLeft/DontCare causal masking only.",
            exception=ValueError,
        )
        return False

    return True
