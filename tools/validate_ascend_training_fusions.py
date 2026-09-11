#!/usr/bin/env python3
"""Run small real-NPU forward/backward checks for the experimental training fusions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from cosmos_framework.model.generator.mot.teacher_forcing_attention import teacher_forcing_dense_attention
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.nemotron_3_dense_vl import (
    Nemotron3DenseVLRMSNorm,
    apply_rotary_pos_emb_partial,
)


def _errors(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = (reference.float() - candidate.float()).abs()
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "reference_abs_max": reference.float().abs().max().item(),
    }


def _leaf(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone().requires_grad_(True)


def check_rmsnorm(device: torch.device) -> dict[str, object]:
    torch.manual_seed(7)
    reference_module = Nemotron3DenseVLRMSNorm(2048).to(device)
    candidate_module = Nemotron3DenseVLRMSNorm(2048).to(device)
    candidate_module.load_state_dict(reference_module.state_dict())
    source = torch.randn(256, 2048, dtype=torch.bfloat16, device=device)
    gradient = torch.randn_like(source)

    os.environ["COSMOS_ASCEND_FUSED_RMSNORM"] = "0"
    reference_input = _leaf(source)
    reference_output = reference_module(reference_input)
    reference_output.backward(gradient)

    os.environ["COSMOS_ASCEND_FUSED_RMSNORM"] = "1"
    candidate_input = _leaf(source)
    candidate_output = candidate_module(candidate_input)
    candidate_output.backward(gradient)
    torch.npu.synchronize()
    return {
        "output": _errors(reference_output, candidate_output),
        "input_grad": _errors(reference_input.grad, candidate_input.grad),
        "weight_grad": _errors(reference_module.weight.grad, candidate_module.weight.grad),
    }


def check_rope(device: torch.device) -> dict[str, object]:
    torch.manual_seed(11)
    q_source = torch.randn(320, 16, 128, dtype=torch.bfloat16, device=device)
    k_source = torch.randn(320, 8, 128, dtype=torch.bfloat16, device=device)
    angles = torch.randn(320, 128, dtype=torch.float32, device=device)
    cos = angles.cos().to(torch.bfloat16)
    sin = angles.sin().to(torch.bfloat16)
    q_gradient = torch.randn_like(q_source)
    k_gradient = torch.randn_like(k_source)

    os.environ["COSMOS_ASCEND_FUSED_ROPE"] = "0"
    q_reference, k_reference = _leaf(q_source), _leaf(k_source)
    q_reference_output, k_reference_output = apply_rotary_pos_emb_partial(
        q_reference, k_reference, cos, sin, unsqueeze_dim=1
    )
    torch.autograd.backward((q_reference_output, k_reference_output), (q_gradient, k_gradient))

    os.environ["COSMOS_ASCEND_FUSED_ROPE"] = "1"
    q_candidate, k_candidate = _leaf(q_source), _leaf(k_source)
    q_candidate_output, k_candidate_output = apply_rotary_pos_emb_partial(
        q_candidate, k_candidate, cos, sin, unsqueeze_dim=1
    )
    torch.autograd.backward((q_candidate_output, k_candidate_output), (q_gradient, k_gradient))
    torch.npu.synchronize()
    return {
        "q_output": _errors(q_reference_output, q_candidate_output),
        "k_output": _errors(k_reference_output, k_candidate_output),
        "q_grad": _errors(q_reference.grad, q_candidate.grad),
        "k_grad": _errors(k_reference.grad, k_candidate.grad),
    }


def check_attention(device: torch.device) -> dict[str, object]:
    torch.manual_seed(13)
    q_source = torch.randn(64, 16, 128, dtype=torch.bfloat16, device=device)
    k_source = torch.randn(80, 8, 128, dtype=torch.bfloat16, device=device)
    v_source = torch.randn_like(k_source)
    allowed_mask = torch.rand(64, 80, device=device) > 0.35
    allowed_mask[:, 0] = True
    output_gradient = torch.randn_like(q_source)

    def run(enabled: bool):
        os.environ["COSMOS_ASCEND_FUSED_TEACHER_FORCING_ATTENTION"] = "1" if enabled else "0"
        q, k, v = _leaf(q_source), _leaf(k_source), _leaf(v_source)
        output = teacher_forcing_dense_attention(q, k, v, allowed_mask)
        output.backward(output_gradient)
        return output, q.grad, k.grad, v.grad

    reference = run(False)
    candidate = run(True)
    torch.npu.synchronize()
    return {
        name: _errors(reference[index], candidate[index])
        for index, name in enumerate(("output", "q_grad", "k_grad", "v_grad"))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("npu:0")
    results = {
        "device": torch.npu.get_device_name(0),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "rmsnorm": check_rmsnorm(device),
        "rope": check_rope(device),
        "attention": check_attention(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
