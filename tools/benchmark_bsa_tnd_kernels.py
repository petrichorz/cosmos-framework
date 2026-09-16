"""Same-shape BSA64/TND output/gradient checks and synchronized kernel timings."""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
    TeacherForcingGeometry,
    build_teacher_forcing_layout,
)
from cosmos_framework.model.generator.mot.bsa64_attention import bsa64_per_sample_attention, build_bsa64_plans
from cosmos_framework.model.generator.mot.teacher_forcing_tnd import build_tnd_plan, teacher_forcing_tnd_attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    report = []
    for case, (block, history) in enumerate([(1, 1), (1, 15), (1, 64), (4, 22)]):
        layout = build_teacher_forcing_layout(
            und_token_counts=[67],
            vision_token_shapes=[(96, 16, 24)],
            geometry=TeacherForcingGeometry((block,), (history,)),
        )
        nq = layout.gen_query_indexes.numel()
        nk = sum(layout.sample_lens)
        inputs = [
            torch.randn(n, h, 128, device="npu", dtype=torch.bfloat16, requires_grad=True)
            for n, h in [(nq, 24), (nk, 8), (nk, 8)]
        ]
        weight = torch.randn_like(inputs[0])
        bp = build_bsa64_plans(layout, 24, "npu")
        tp = build_tnd_plan(layout, device="npu")
        ref = bsa64_per_sample_attention(*inputs, bp)
        got = teacher_forcing_tnd_attention(*inputs, tp)
        rg = torch.autograd.grad(ref, inputs, weight)
        gg = torch.autograd.grad(got, inputs, weight)
        errors = {}
        for name, x, y in zip(["output", "dq", "dk", "dv"], [got, *gg], [ref, *rg], strict=True):
            errors[name] = {
                "max_abs": (x - y).abs().max().item(),
                "relative_l2": ((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-12)).item(),
            }
            torch.testing.assert_close(x, y, atol=0.008, rtol=0.008)
        record = {
            "shape": [96, 16, 24],
            "und": 67,
            "block": block,
            "history": history,
            "block_density": bp[0].visible_density,
            "errors": errors,
            "warmup": 3,
            "measured": 10,
            "results": {},
        }
        del x, y, ref, got, rg, gg, bp, tp
        for mode in ["bsa64", "tnd"] if case % 2 == 0 else ["tnd", "bsa64"]:
            gc.collect()
            torch.npu.empty_cache()
            torch.npu.synchronize()
            t = time.perf_counter()
            if mode == "bsa64":
                plan = build_bsa64_plans(layout, 24, "npu")
                call = bsa64_per_sample_attention
            else:
                plan = build_tnd_plan(layout, device="npu")
                call = teacher_forcing_tnd_attention
            torch.npu.synchronize()
            prep = time.perf_counter() - t
            fs, bs, alloc, reserved = [], [], [], []
            for step in range(13):
                torch.npu.synchronize()
                torch.npu.reset_peak_memory_stats()
                t = time.perf_counter()
                out = call(*inputs, plan)
                torch.npu.synchronize()
                mid = time.perf_counter()
                gradients = torch.autograd.grad(out, inputs, weight)
                torch.npu.synchronize()
                end = time.perf_counter()
                if step >= 3:
                    fs.append(mid - t)
                    bs.append(end - mid)
                    alloc.append(torch.npu.max_memory_allocated() / 2**30)
                    reserved.append(torch.npu.max_memory_reserved() / 2**30)
                del out, gradients
            record["results"][mode] = {
                "preparation_s": prep,
                "forward_s": statistics.mean(fs),
                "backward_s": statistics.mean(bs),
                "total_s": statistics.mean(fs) + statistics.mean(bs),
                "full_checkpoint_attention_estimate_s": 2 * statistics.mean(fs) + statistics.mean(bs),
                "allocated_gib": max(alloc),
                "reserved_gib": max(reserved),
                "forward_samples": fs,
                "backward_samples": bs,
            }
            del plan
        report.append(record)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        del inputs, weight


if __name__ == "__main__":
    main()
