# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Ascend TND variable-length fused-attention implementation."""

from functools import lru_cache

import torch
from torch import Tensor

from cosmos_framework.model.attention.checks import assert_universal_tensor_checks
from cosmos_framework.model.attention.masks import CausalType


_COMPRESSED_CAUSAL_MASK_SIZE = 2048


@lru_cache(maxsize=None)
def _compressed_causal_mask(device_type: str, device_index: int | None) -> Tensor:
    """Create Ascend's reusable 2048x2048 left-up causal compressed mask."""
    device = (
        torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)
    )
    return torch.triu(
        torch.ones(
            (_COMPRESSED_CAUSAL_MASK_SIZE, _COMPRESSED_CAUSAL_MASK_SIZE),
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )


def _ascend_actual_seq_lengths(cumulative_seqlen: Tensor) -> list[int]:
    """Convert Cosmos ``[0, ...]`` cumulative offsets to Ascend's ``[...]`` list."""
    values = cumulative_seqlen.tolist()
    if not values or values[0] != 0:
        raise ValueError("cumulative sequence lengths must start with 0")
    actual_seq_lengths = values[1:]
    if not actual_seq_lengths:
        raise ValueError("npu_fusion_attention requires at least one packed sequence")
    # Ascend supports zero-length Q batches, represented by repeated cumulative
    # offsets. Cosmos does not use the trailing-zero "batch not participating"
    # convention, so offsets must remain monotonically non-decreasing here.
    if any(end < start for start, end in zip(values[:-1], actual_seq_lengths, strict=True)):
        raise ValueError("cumulative sequence lengths must be monotonically non-decreasing")
    if len(actual_seq_lengths) > 1024:
        raise ValueError("npu_fusion_attention TND supports at most 1024 packed sequences")
    return actual_seq_lengths


def npu_fusion_attention(
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
    """Run one packed TND attention call instead of looping over samples."""
    # Imported only on the selected NPU path so CPU/CUDA imports remain optional.
    import torch_npu

    from cosmos_framework.model.attention.npu_fusion_attention.checks import npu_fusion_attention_check

    assert_universal_tensor_checks(query, key, value)
    if cumulative_seqlen_Q is None or cumulative_seqlen_KV is None:
        raise ValueError("npu_fusion_attention requires cumulative TND sequence lengths")
    if return_lse:
        raise NotImplementedError("npu_fusion_attention backend does not currently expose merge-compatible LSE")

    backend_kwargs = backend_kwargs.copy() if backend_kwargs is not None else {}
    if backend_kwargs:
        raise ValueError(f"Unsupported npu_fusion_attention backend kwargs: {sorted(backend_kwargs)}")

    assert npu_fusion_attention_check(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        dtype=query.dtype,
        device=query.device,
        requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
        is_causal=is_causal,
        causal_type=causal_type,
        is_varlen=True,
        deterministic=deterministic,
        raise_error=True,
    )
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("TND variable-length attention requires singleton packed batch dimension")

    actual_seq_qlen = _ascend_actual_seq_lengths(cumulative_seqlen_Q)
    actual_seq_kvlen = _ascend_actual_seq_lengths(cumulative_seqlen_KV)
    if len(actual_seq_qlen) != len(actual_seq_kvlen):
        raise ValueError("query and key/value sequence metadata must contain the same number of samples")
    if actual_seq_qlen[-1] != query.shape[1]:
        raise ValueError("final query sequence offset must equal the packed query token count")
    if actual_seq_kvlen[-1] != key.shape[1]:
        raise ValueError("final key/value sequence offset must equal the packed key/value token count")

    q = query.squeeze(0).contiguous()
    k = key.squeeze(0).contiguous()
    v = value.squeeze(0).contiguous()
    scale = scale if scale is not None else query.shape[-1] ** -0.5

    if is_causal:
        atten_mask = _compressed_causal_mask(query.device.type, query.device.index)
        sparse_mode = 2  # left-up causal; DontCare is equivalent when Sq == Skv.
    else:
        atten_mask = None
        sparse_mode = 0  # full attention; actual_seq_lengths isolates samples.

    output = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num=query.shape[2],
        input_layout="TND",
        atten_mask=atten_mask,
        scale=scale,
        keep_prob=1.0,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=sparse_mode,
    )[0]
    if output.shape != (q.shape[0], q.shape[1], v.shape[2]):
        raise RuntimeError(f"Unexpected npu_fusion_attention TND output shape: {output.shape}")
    return output.unsqueeze(0)
