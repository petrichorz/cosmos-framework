"""Isolated forward+backward timing including gather and FP32 gradient accumulation."""

import argparse
import gc
import json
import statistics
import time

import torch

from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
    TeacherForcingGeometry,
    build_dense_teacher_forcing_gen_mask,
    build_teacher_forcing_layout,
)
from cosmos_framework.model.generator.mot.teacher_forcing_attention import teacher_forcing_dense_attention
from cosmos_framework.model.generator.mot.teacher_forcing_tnd import build_tnd_plan, teacher_forcing_tnd_attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    layout = build_teacher_forcing_layout(
        und_token_counts=[67], vision_token_shapes=[(96, 17, 23)], geometry=TeacherForcingGeometry((1,), (15,))
    )
    inputs = [
        torch.randn(n, h, 128, device="npu", dtype=torch.bfloat16, requires_grad=True)
        for n, h in [(75072, 24), (75139, 8), (75139, 8)]
    ]
    upstream = torch.randn_like(inputs[0])
    report = {"shape": [96, 17, 23], "block_size": 1, "history": 15, "warmup": 3, "measured": 10, "results": {}}
    for mode in ["dense", "grouped_tnd"]:
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()
        prepare_start = time.perf_counter()
        if mode == "dense":
            metadata = (~build_dense_teacher_forcing_gen_mask(layout)).to("npu")
            call = teacher_forcing_dense_attention
        else:
            metadata = build_tnd_plan(layout, device="npu")
            call = teacher_forcing_tnd_attention
        torch.npu.synchronize()
        preparation = time.perf_counter() - prepare_start
        forward, backward, peaks, reserved = [], [], [], []
        for step in range(13):
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            start = time.perf_counter()
            out = call(*inputs, metadata)
            torch.npu.synchronize()
            split = time.perf_counter()
            gradients = torch.autograd.grad(out, inputs, upstream)
            torch.npu.synchronize()
            end = time.perf_counter()
            if step >= 3:
                forward.append(split - start)
                backward.append(end - split)
                peaks.append(torch.npu.max_memory_allocated() / 2**30)
                reserved.append(torch.npu.max_memory_reserved() / 2**30)
            del out, gradients
        report["results"][mode] = {
            "prepare_seconds": preparation,
            "forward_seconds": statistics.mean(forward),
            "backward_seconds": statistics.mean(backward),
            "total_seconds": statistics.mean(forward) + statistics.mean(backward),
            "peak_allocated_gib": max(peaks),
            "peak_reserved_gib": max(reserved),
            "forward_samples": forward,
            "backward_samples": backward,
        }
        del metadata
    from pathlib import Path

    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
