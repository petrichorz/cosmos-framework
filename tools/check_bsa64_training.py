"""Verify exact teacher-forcing visibility, ragged UND tails and GQA gradients."""

import argparse
import json
import time

import torch
import torch_npu

from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
    TeacherForcingGeometry,
    build_per_sample_teacher_forcing_gen_masks,
    build_teacher_forcing_layout,
)
from cosmos_framework.model.generator.mot.bsa64_attention import bsa64_per_sample_attention, build_bsa64_plans


def dense(q, k, v, blocked):
    return torch_npu.npu_fusion_attention(
        q[None],
        k[None],
        v[None],
        head_num=q.shape[1],
        input_layout="BSND",
        atten_mask=blocked,
        scale=128**-0.5,
        keep_prob=1.0,
        sparse_mode=1,
    )[0][0]


def check(und, heads, kv_heads, block_size, history, vision_shape=(5, 8, 8)):
    layout = build_teacher_forcing_layout(
        und_token_counts=[und],
        vision_token_shapes=[vision_shape],
        geometry=TeacherForcingGeometry((block_size,), (history,)),
    )
    allowed = build_per_sample_teacher_forcing_gen_masks(layout)[0]
    plan = build_bsa64_plans(layout, heads, "npu")[0]
    expanded = plan.mask[0, 0].cpu().bool().repeat_interleave(64, 0).repeat_interleave(64, 1)
    expanded = expanded[: plan.query_length, : plan.key_length]
    torch.testing.assert_close(expanded, allowed[:, plan.kv_indexes.cpu()], atol=0, rtol=0)
    blocked = allowed.logical_not().to("npu")
    inputs = [
        torch.randn(n, h, 128, device="npu", dtype=torch.bfloat16)
        for n, h in ((plan.query_length, heads), (plan.key_length, kv_heads), (plan.key_length, kv_heads))
    ]
    ref_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    new_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    torch.npu.synchronize()
    started = time.perf_counter()
    ref = dense(*ref_inputs, blocked)
    torch.npu.synchronize()
    dense_forward = time.perf_counter() - started
    started = time.perf_counter()
    out = bsa64_per_sample_attention(*new_inputs, (plan,))
    torch.npu.synchronize()
    bsa_forward = time.perf_counter() - started
    grad = torch.randn_like(ref)
    started = time.perf_counter()
    ref.backward(grad)
    torch.npu.synchronize()
    dense_backward = time.perf_counter() - started
    started = time.perf_counter()
    out.backward(grad)
    torch.npu.synchronize()
    bsa_backward = time.perf_counter() - started
    errors = {}
    for name, got, expected in [("out", out, ref)] + [
        (f"d{n}", a.grad, b.grad) for n, a, b in zip("QKV", new_inputs, ref_inputs)
    ]:
        errors[name] = {
            "max_absolute_error": (got.float() - expected.float()).abs().max().item(),
            "relative_l2_error": (
                (got.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
            ).item(),
        }
        torch.testing.assert_close(got, expected, atol=0.008, rtol=0.008)
    print(
        json.dumps(
            {
                "und": und,
                "heads": heads,
                "kv_heads": kv_heads,
                "block_size": block_size,
                "history": history,
                "density": plan.visible_density,
                "passed": True,
                "query_length": plan.query_length,
                "key_length": plan.key_length,
                "vision_shape": vision_shape,
                "single_call_unwarmed_seconds": {
                    "dense_forward": dense_forward,
                    "bsa_forward_including_reorder": bsa_forward,
                    "dense_backward": dense_backward,
                    "bsa_backward": bsa_backward,
                },
                "errors": errors,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(42)
    if args.large:
        check(67, 24, 8, 2, 8, vision_shape=(17, 32, 48))
    else:
        for und in (1, 17, 63, 64, 65, 129):
            for heads, kv_heads in ((4, 4), (24, 8)):
                for block_size, history in ((1, 1), (3, 64)):
                    check(und, heads, kv_heads, block_size, history)
