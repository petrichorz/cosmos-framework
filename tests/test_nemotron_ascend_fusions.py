# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl import nemotron_3_dense_vl


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_ascend_fusion_env_is_ignored_for_cpu(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("COSMOS_ASCEND_FUSED_ROPE", value)
    assert not nemotron_3_dense_vl._ascend_fusion_enabled("COSMOS_ASCEND_FUSED_ROPE", torch.empty(1))


def test_rope_cpu_fallback_forward_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COSMOS_ASCEND_FUSED_ROPE", "1")
    q = torch.randn(2, 3, 8, requires_grad=True)
    k = torch.randn(2, 2, 8, requires_grad=True)
    cos = torch.randn(2, 8)
    sin = torch.randn(2, 8)

    q_out, k_out = nemotron_3_dense_vl.apply_rotary_pos_emb_partial(q, k, cos, sin)
    (q_out.sum() + k_out.sum()).backward()

    assert q.grad is not None
    assert k.grad is not None


def test_rope_fused_route_forward_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def rotary_mul(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mode: str) -> torch.Tensor:
        calls.append((x.shape, cos.shape, sin.shape, mode))
        return (x * cos) + (nemotron_3_dense_vl.rotate_half(x) * sin)

    monkeypatch.setattr(nemotron_3_dense_vl, "_ascend_fusion_enabled", lambda *_: True)
    monkeypatch.setattr(nemotron_3_dense_vl, "_get_torch_npu_api", lambda name: rotary_mul)
    q = torch.randn(2, 3, 8, requires_grad=True)
    k = torch.randn(2, 2, 8, requires_grad=True)
    cos = torch.randn(2, 8)
    sin = torch.randn(2, 8)

    q_out, k_out = nemotron_3_dense_vl.apply_rotary_pos_emb_partial(q, k, cos, sin)
    (q_out.sum() + k_out.sum()).backward()

    assert len(calls) == 2
    assert calls[0] == (torch.Size([1, 2, 3, 8]), torch.Size([1, 2, 1, 8]), torch.Size([1, 2, 1, 8]), "half")
    assert q.grad is not None
    assert k.grad is not None


def test_rope_falls_back_when_torch_npu_api_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nemotron_3_dense_vl, "_ascend_fusion_enabled", lambda *_: True)
    monkeypatch.setattr(nemotron_3_dense_vl, "_get_torch_npu_api", lambda name: None)
    q = torch.randn(2, 3, 8)
    k = torch.randn(2, 2, 8)
    cos = torch.randn(2, 8)
    sin = torch.randn(2, 8)

    actual_q, actual_k = nemotron_3_dense_vl.apply_rotary_pos_emb_partial(q, k, cos, sin)
    broadcast_cos = cos.unsqueeze(1)
    broadcast_sin = sin.unsqueeze(1)
    expected_q = q * broadcast_cos + nemotron_3_dense_vl.rotate_half(q) * broadcast_sin
    expected_k = k * broadcast_cos + nemotron_3_dense_vl.rotate_half(k) * broadcast_sin

    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)


def test_rmsnorm_cpu_fallback_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COSMOS_ASCEND_FUSED_RMSNORM", "1")
    norm = nemotron_3_dense_vl.Nemotron3DenseVLRMSNorm(8, eps=1e-5)
    hidden_states = torch.randn(2, 3, 8, requires_grad=True)

    output = norm(hidden_states)
    expected = hidden_states.float()
    expected = expected * torch.rsqrt(expected.pow(2).mean(-1, keepdim=True) + 1e-5)
    expected = (norm.weight.float() * expected).to(hidden_states.dtype)

    torch.testing.assert_close(output, expected)
    output.sum().backward()
    assert hidden_states.grad is not None
    assert norm.weight.grad is not None


def test_rmsnorm_falls_back_when_torch_npu_api_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nemotron_3_dense_vl, "_ascend_fusion_enabled", lambda *_: True)
    monkeypatch.setattr(nemotron_3_dense_vl, "_get_torch_npu_api", lambda name: None)
    norm = nemotron_3_dense_vl.Nemotron3DenseVLRMSNorm(8, eps=1e-5)
    hidden_states = torch.randn(2, 3, 8)

    output = norm(hidden_states)
    expected = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + 1e-5)
    expected = norm.weight * expected

    torch.testing.assert_close(output, expected)


def test_rmsnorm_fused_route_forward_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((x.shape, weight.shape, epsilon))
        rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon)
        return x * rstd * weight, rstd

    monkeypatch.setattr(nemotron_3_dense_vl, "_ascend_fusion_enabled", lambda *_: True)
    monkeypatch.setattr(nemotron_3_dense_vl, "_get_torch_npu_api", lambda name: rms_norm)
    norm = nemotron_3_dense_vl.Nemotron3DenseVLRMSNorm(8, eps=1e-5)
    hidden_states = torch.randn(2, 3, 8, requires_grad=True)

    output = norm(hidden_states)
    output.sum().backward()

    assert calls == [(torch.Size([2, 3, 8]), torch.Size([8]), 1e-5)]
    assert hidden_states.grad is not None
    assert norm.weight.grad is not None
