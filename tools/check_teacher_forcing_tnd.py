"""Compare exact masks, outputs and all Q/K/V gradients, including checkpoint."""

import argparse
import json

import torch
from torch.utils.checkpoint import checkpoint

from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
    TeacherForcingGeometry,
    build_dense_teacher_forcing_gen_mask,
    build_teacher_forcing_layout,
)
from cosmos_framework.model.generator.mot.teacher_forcing_attention import teacher_forcing_dense_attention
from cosmos_framework.model.generator.mot.teacher_forcing_tnd import build_tnd_plan, teacher_forcing_tnd_attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--long", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = args.device
    dtype = torch.bfloat16 if device == "npu" else torch.float64
    results = []
    geometries = [((1,), (15,))] if args.long else [((1, 3), (1, 2)), ((4, 2), (64, 1)), ((2, 4), (2, 64))]
    for bs, hist in geometries:
        layout = build_teacher_forcing_layout(
            und_token_counts=[67] if args.long else [3, 17],
            vision_token_shapes=[(96, 17, 23)] if args.long else [(6, 2, 3), (8, 2, 2)],
            geometry=TeacherForcingGeometry(bs, hist),
        )
        allowed = build_dense_teacher_forcing_gen_mask(layout)
        for cap in [131072] if args.long else [128, 131072]:
            plan = build_tnd_plan(layout, device=device, max_kv_tokens=cap)
            expanded = torch.zeros_like(allowed)
            all_q = []
            for chunk in plan.chunks:
                qi, ki = chunk.query_indexes.cpu(), chunk.kv_indexes.cpu()
                qs = ks = 0
                all_q.append(qi)
                for qe, ke in zip(chunk.query_ends, chunk.kv_ends):
                    expanded[qi[qs:qe, None], ki[None, ks:ke]] = True
                    qs, ks = qe, ke
            assert torch.equal(expanded, allowed)
            assert torch.equal(torch.cat(all_q), torch.arange(allowed.shape[0]))
            for hq, hkv in [(24, 8)] if args.long else [(4, 4), (24, 8)]:
                for ac in [False, True]:
                    inputs = [
                        torch.randn(n, h, 128, device=device, dtype=dtype, requires_grad=True)
                        for n, h in [(allowed.shape[0], hq), (allowed.shape[1], hkv), (allowed.shape[1], hkv)]
                    ]
                    q, k, v = inputs
                    if device == "npu":
                        ref = teacher_forcing_dense_attention(q, k, v, (~allowed).to(device))
                    else:
                        ref = torch.nn.functional.scaled_dot_product_attention(
                            q.transpose(0, 1),
                            k.repeat_interleave(hq // hkv, 1).transpose(0, 1),
                            v.repeat_interleave(hq // hkv, 1).transpose(0, 1),
                            attn_mask=allowed,
                        ).transpose(0, 1)

                    def run(q, k, v):
                        return teacher_forcing_tnd_attention(q, k, v, plan)

                    out = checkpoint(run, q, k, v, use_reentrant=False) if ac else run(q, k, v)
                    grad = torch.randn_like(out)
                    rg = torch.autograd.grad(ref, inputs, grad)
                    og = torch.autograd.grad(out, inputs, grad)
                    tol = 0.008 if device == "npu" else 1e-10
                    for name, actual, expected in zip(["out", "dq", "dk", "dv"], [out, *og], [ref, *rg]):
                        print(
                            "CHECK",
                            bs,
                            hist,
                            cap,
                            hq,
                            hkv,
                            ac,
                            name,
                            (actual - expected).abs().max().item(),
                            flush=True,
                        )
                        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
                    results.append(
                        {
                            "block": bs,
                            "history": hist,
                            "cap": cap,
                            "heads": [hq, hkv],
                            "checkpoint": ac,
                            "output_max_abs": (out - ref).abs().max().item(),
                            "grad_max_abs": max((a - b).abs().max().item() for a, b in zip(og, rg)),
                        }
                    )
    print(json.dumps({"device": device, "passed": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
