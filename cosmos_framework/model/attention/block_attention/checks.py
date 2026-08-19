# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compatibility checks for the Ascend NPU block-sparse attention backend."""

from functools import partial

import torch

from cosmos_framework.model.attention.block_attention import BLOCK_ATTENTION_SUPPORTED
from cosmos_framework.model.attention.checks import attention_param_checks, attention_tensor_checks
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.utils import log_or_raise_error


BLOCK_ATTENTION_DTYPES: list[torch.dtype] = [torch.float16, torch.bfloat16]
BLOCK_ATTENTION_FORWARD_HEAD_DIMS: tuple[int, ...] = (64, 128)
BLOCK_ATTENTION_BACKWARD_HEAD_DIMS: tuple[int, ...] = (128,)


def block_attention_check(
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
    """Return whether the inputs can use ``torch_npu.npu_block_sparse_attention``."""

    target_fn = partial(log_or_raise_error, raise_error=raise_error)

    if not BLOCK_ATTENTION_SUPPORTED or device.type != "npu":
        target_fn(
            "block_attention requires torch_npu and Ascend NPU tensors.",
            exception=RuntimeError,
        )
        return False

    if deterministic:
        target_fn(
            "block_attention backend does not guarantee deterministic backward.",
            exception=ValueError,
        )
        return False

    # The block-sparse operator expresses visibility entirely through
    # ``block_sparse_mask``. It has no equivalent of PyTorch's ``is_causal``.
    if is_causal:
        target_fn(
            "block_attention does not accept is_causal=True; encode all visibility in block_sparse_mask.",
            exception=ValueError,
        )
        return False

    head_dim = query_shape[-1]
    if head_dim not in BLOCK_ATTENTION_FORWARD_HEAD_DIMS:
        target_fn(
            f"block_attention forward supports head_dim in {BLOCK_ATTENTION_FORWARD_HEAD_DIMS}, got {head_dim}.",
            exception=ValueError,
        )
        return False

    if requires_grad and head_dim not in BLOCK_ATTENTION_BACKWARD_HEAD_DIMS:
        target_fn(
            f"block_attention backward supports head_dim in {BLOCK_ATTENTION_BACKWARD_HEAD_DIMS}, got {head_dim}.",
            exception=ValueError,
        )
        return False

    if not attention_tensor_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        requires_grad=requires_grad,
        supported_dtypes_forward=BLOCK_ATTENTION_DTYPES,
        supported_dtypes_backward=BLOCK_ATTENTION_DTYPES,
        supports_mla=False,
        supports_gqa_mqa=True,
        raise_error=raise_error,
        backend_name="Ascend block_attention (npu_block_sparse_attention)",
    ):
        target_fn("block_attention does not support the given inputs.", exception=RuntimeError)
        return False

    attention_param_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        is_causal=is_causal,
        causal_type=causal_type,
    )

    return True
