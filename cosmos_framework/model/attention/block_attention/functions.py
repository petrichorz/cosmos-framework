# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Ascend NPU block-sparse attention implementation."""

from __future__ import annotations

import torch
from torch import Tensor

from cosmos_framework.model.attention.checks import assert_universal_tensor_checks
from cosmos_framework.model.attention.masks import CausalType


def _as_list_of_positive_ints(value: object, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(value)}")
    values = list(value)
    if len(values) < 2 or any(not isinstance(x, int) or x <= 0 for x in values):
        raise ValueError(f"{name} must contain at least two positive integers, got {value}")
    return values


def _validate_block_shape(block_shape: list[int]) -> None:
    block_size_y = block_shape[1]
    if block_size_y % 128 != 0:
        raise ValueError(
            "block_attention requires block_shape[1] to be a multiple of 128, "
            f"got {block_size_y}"
        )


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _cumulative_to_actual(cumulative_seqlen: Tensor) -> list[int]:
    """Convert Cosmos ``[0, s1, s1+s2, ...]`` offsets to Ascend actual lengths."""

    values = cumulative_seqlen.tolist()
    if not values or values[0] != 0:
        raise ValueError("cumulative sequence lengths must start with 0")
    actual = [values[i] - values[i - 1] for i in range(1, len(values))]
    if not actual or any(x < 0 for x in actual):
        raise ValueError("cumulative sequence lengths must be monotonically non-decreasing")
    return actual


def _validate_mask(
    mask: Tensor,
    query: Tensor,
    *,
    batch: int,
    heads: int,
    max_q_len: int,
    max_kv_len: int,
    block_shape: list[int],
) -> None:
    if mask.dtype != torch.int8:
        raise TypeError(f"block_sparse_mask must be int8, got {mask.dtype}")
    if mask.device != query.device:
        raise ValueError(
            f"block_sparse_mask must be on the same device as query, got {mask.device} and {query.device}"
        )

    expected_shape = (
        batch,
        heads,
        _ceil_div(max_q_len, block_shape[0]),
        _ceil_div(max_kv_len, block_shape[1]),
    )
    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            f"block_sparse_mask must have shape {expected_shape}, got {tuple(mask.shape)}"
        )


def _resolve_inner_precise(dtype: torch.dtype, inner_precise: int | None) -> int:
    if inner_precise is None:
        return 1 if dtype == torch.float16 else 0
    if inner_precise not in (0, 1):
        raise ValueError(f"inner_precise must be 0 or 1, got {inner_precise}")
    if dtype == torch.bfloat16 and inner_precise != 0:
        raise ValueError("bfloat16 inputs require inner_precise=0")
    return inner_precise


def block_attention(
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
    """Run ``torch_npu.npu_block_sparse_attention`` through the unified backend API.

    Required ``backend_kwargs``:
        block_sparse_mask: int8 tensor ``[batch, heads, ceil_q, ceil_kv]``.
        block_shape: ``[block_shape_x, block_shape_y]``. ``block_shape_y`` must be
            a multiple of 128.

    Optional ``backend_kwargs``:
        inner_precise: 0 (float32 softmax) or 1 (float16 softmax).
        actual_seq_lengths / actual_seq_lengths_kv: per-batch lengths for BNSD.
            For the TND varlen path they are derived from ``cumulative_seqlen_*``.
    """

    del max_seqlen_Q, max_seqlen_KV
    assert_universal_tensor_checks(query, key, value)
    is_varlen = cumulative_seqlen_Q is not None or cumulative_seqlen_KV is not None

    # Import only after selection so CPU/CUDA import paths do not need torch_npu.
    import torch_npu  # noqa: F401

    from cosmos_framework.model.attention.block_attention.checks import block_attention_check

    assert block_attention_check(
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
        raise NotImplementedError("block_attention backend does not currently expose logsumexp.")

    kwargs = backend_kwargs.copy() if backend_kwargs is not None else {}
    block_sparse_mask = kwargs.pop("block_sparse_mask", None)
    block_shape = kwargs.pop("block_shape", None)
    inner_precise = kwargs.pop("inner_precise", None)
    actual_seq_lengths = kwargs.pop("actual_seq_lengths", None)
    actual_seq_lengths_kv = kwargs.pop("actual_seq_lengths_kv", None)
    if kwargs:
        raise ValueError(f"Unsupported block_attention backend kwargs: {sorted(kwargs)}")

    if block_sparse_mask is None:
        raise ValueError("block_attention requires backend_kwargs['block_sparse_mask'].")
    if block_shape is None:
        raise ValueError("block_attention requires backend_kwargs['block_shape'].")

    block_shape = _as_list_of_positive_ints(block_shape, "block_shape")
    _validate_block_shape(block_shape)

    scale = scale if scale is not None else query.shape[-1] ** -0.5
    inner_precise = _resolve_inner_precise(query.dtype, inner_precise)
    num_q_heads = query.shape[2]
    num_kv_heads = key.shape[2]

    if not is_varlen:
        # Generic API layout is [B, S, H, D]. Ascend BNSD layout is [B, H, S, D].
        q = query.permute(0, 2, 1, 3).contiguous()
        k = key.permute(0, 2, 1, 3).contiguous()
        v = value.permute(0, 2, 1, 3).contiguous()

        _validate_mask(
            block_sparse_mask,
            query,
            batch=query.shape[0],
            heads=num_q_heads,
            max_q_len=query.shape[1],
            max_kv_len=key.shape[1],
            block_shape=block_shape,
        )

        output, _ = torch_npu.npu_block_sparse_attention(
            q,
            k,
            v,
            block_sparse_mask,
            block_shape,
            q_input_layout="BNSD",
            kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads,
            scale_value=scale,
            inner_precise=inner_precise,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            softmax_lse_flag=0,
        )
        return output.permute(0, 2, 1, 3)

    # Varlen path: generic packed API uses batch=1.
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("block_attention varlen path requires singleton packed batch dimension")
    if cumulative_seqlen_Q is None or cumulative_seqlen_KV is None:
        raise ValueError("block_attention varlen path requires cumulative_seqlen_Q and cumulative_seqlen_KV")

    actual_seq_lengths = _cumulative_to_actual(cumulative_seqlen_Q)
    actual_seq_lengths_kv = _cumulative_to_actual(cumulative_seqlen_KV)
    if len(actual_seq_lengths) != len(actual_seq_lengths_kv):
        raise ValueError("query and key/value sequence metadata must contain the same number of samples")
    if actual_seq_lengths[-1] != query.shape[1]:
        raise ValueError("final query sequence offset must equal the packed query token count")
    if actual_seq_lengths_kv[-1] != key.shape[1]:
        raise ValueError("final key/value sequence offset must equal the packed key/value token count")

    q = query.squeeze(0).contiguous()
    k = key.squeeze(0).contiguous()
    v = value.squeeze(0).contiguous()

    _validate_mask(
        block_sparse_mask,
        query,
        batch=len(actual_seq_lengths),
        heads=num_q_heads,
        max_q_len=max(actual_seq_lengths),
        max_kv_len=max(actual_seq_lengths_kv),
        block_shape=block_shape,
    )

    output, _ = torch_npu.npu_block_sparse_attention(
        q,
        k,
        v,
        block_sparse_mask,
        block_shape,
        q_input_layout="TND",
        kv_input_layout="TND",
        num_key_value_heads=num_kv_heads,
        scale_value=scale,
        inner_precise=inner_precise,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        softmax_lse_flag=0,
    )
    return output.unsqueeze(0)
