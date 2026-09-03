# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import sys
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.attention.backends import is_backend_compatible
from cosmos_framework.model.attention.npu_fusion_attention.functions import (
    _ascend_actual_seq_lengths,
    npu_fusion_attention,
)


@pytest.mark.L0
def test_ascend_actual_seq_lengths_converts_cumulative_offsets():
    offsets = torch.tensor([0, 2, 2, 7], dtype=torch.int32)

    assert _ascend_actual_seq_lengths(offsets) == [2, 2, 7]


@pytest.mark.L0
@pytest.mark.parametrize(
    ("offsets", "error"),
    [
        ([1, 3], "start with 0"),
        ([0], "at least one packed sequence"),
        ([0, 3, 2], "monotonically non-decreasing"),
    ],
)
def test_ascend_actual_seq_lengths_rejects_invalid_offsets(offsets: list[int], error: str):
    with pytest.raises(ValueError, match=error):
        _ascend_actual_seq_lengths(torch.tensor(offsets, dtype=torch.int32))


@pytest.mark.L0
def test_npu_fusion_attention_uses_one_packed_tnd_call(monkeypatch):
    calls: list[tuple[tuple[torch.Tensor, ...], dict]] = []

    def fake_npu_fusion_attention(*args, **kwargs):
        calls.append((args, kwargs))
        return (torch.zeros_like(args[0]),)

    fake_torch_npu = SimpleNamespace(npu_fusion_attention=fake_npu_fusion_attention)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    monkeypatch.setattr(
        "cosmos_framework.model.attention.npu_fusion_attention.checks.npu_fusion_attention_check",
        lambda **kwargs: True,
    )

    query = torch.randn(1, 5, 4, 3)
    key = torch.randn(1, 7, 2, 3)
    value = torch.randn(1, 7, 2, 3)
    output = npu_fusion_attention(
        query,
        key,
        value,
        cumulative_seqlen_Q=torch.tensor([0, 2, 5], dtype=torch.int32),
        cumulative_seqlen_KV=torch.tensor([0, 3, 7], dtype=torch.int32),
        max_seqlen_Q=3,
        max_seqlen_KV=4,
    )

    assert output.shape == query.shape
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert [tuple(tensor.shape) for tensor in args] == [(5, 4, 3), (7, 2, 3), (7, 2, 3)]
    assert kwargs["head_num"] == 4
    assert kwargs["input_layout"] == "TND"
    assert kwargs["actual_seq_qlen"] == [2, 5]
    assert kwargs["actual_seq_kvlen"] == [3, 7]
    assert kwargs["atten_mask"] is None
    assert kwargs["sparse_mode"] == 0


@pytest.mark.L0
def test_npu_fusion_attention_is_incompatible_with_lse_before_device_check():
    compatible = is_backend_compatible(
        backend="npu_fusion_attention",
        query_shape=torch.Size([1, 5, 4, 3]),
        key_shape=torch.Size([1, 7, 2, 3]),
        value_shape=torch.Size([1, 7, 2, 3]),
        dtype=torch.float32,
        device=torch.device("cpu"),
        requires_grad=False,
        is_causal=False,
        causal_type=None,
        is_varlen=True,
        return_lse=True,
        raise_error=False,
    )

    assert compatible is False
