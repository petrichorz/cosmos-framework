"""Fixed-input history sweep; fresh process per geometry to isolate allocator caches."""

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def measure(args):
    import torch
    from torch.utils.checkpoint import checkpoint

    from cosmos_framework.data.generator.sequence_packing.teacher_forcing import (
        TeacherForcingGeometry,
        build_teacher_forcing_layout,
    )
    from cosmos_framework.model.generator.mot.teacher_forcing_tnd import (
        build_tnd_plan,
        teacher_forcing_tnd_attention,
    )

    torch.manual_seed(42)
    torch.set_num_threads(2)
    layout = build_teacher_forcing_layout(
        und_token_counts=[67],
        vision_token_shapes=[(96, 16, 24)],
        geometry=TeacherForcingGeometry((args.block,), (args.history,)),
    )
    plan = build_tnd_plan(layout, device="npu")
    inputs = [
        torch.randn(n, h, 128, device="npu", dtype=torch.bfloat16, requires_grad=True)
        for n, h in [(plan.num_queries, 24), (plan.num_keys, 8), (plan.num_keys, 8)]
    ]
    weight = torch.randn_like(inputs[0])

    def attention(q, k, v):
        return teacher_forcing_tnd_attention(q, k, v, plan)

    result = {
        "block": args.block,
        "history": args.history,
        "shape": [96, 16, 24],
        "chunks": len(plan.chunks),
        "expanded_kv_tokens": sum(c.kv_indexes.numel() for c in plan.chunks),
        "max_chunk_kv_tokens": max(c.kv_indexes.numel() for c in plan.chunks),
        "max_chunk_q_tokens": max(c.query_indexes.numel() for c in plan.chunks),
        "plan_index_mib": sum((c.kv_indexes.numel() + c.query_indexes.numel()) * 8 for c in plan.chunks) / 2**20,
        "warmup": 2,
        "measured": 5,
        "modes": {},
    }
    for mode in ("training", "checkpoint"):
        gc.collect()
        torch.npu.empty_cache()
        samples = []
        for step in range(7):
            torch.npu.synchronize()
            baseline = torch.npu.memory_allocated()
            torch.npu.reset_peak_memory_stats()
            start = time.perf_counter()
            out = attention(*inputs) if mode == "training" else checkpoint(attention, *inputs, use_reentrant=False)
            torch.npu.synchronize()
            forward_time = time.perf_counter() - start
            forward_peak = torch.npu.max_memory_allocated()
            forward_live = torch.npu.memory_allocated()
            gradients = torch.autograd.grad(out, inputs, weight)
            torch.npu.synchronize()
            total_time = time.perf_counter() - start
            if step >= 2:
                samples.append(
                    {
                        "baseline_gib": baseline / 2**30,
                        "forward_peak_gib": forward_peak / 2**30,
                        "forward_live_gib": forward_live / 2**30,
                        "total_peak_gib": torch.npu.max_memory_allocated() / 2**30,
                        "reserved_gib": torch.npu.max_memory_reserved() / 2**30,
                        "forward_s": forward_time,
                        "total_s": total_time,
                    }
                )
            del out, gradients
        result["modes"][mode] = {
            "samples": samples,
            "summary": {
                k: statistics.mean(s[k] for s in samples) if k.endswith("_s") else max(s[k] for s in samples)
                for k in samples[0]
            },
        }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "modes"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block", type=int)
    parser.add_argument("--history", type=int)
    args = parser.parse_args()
    if args.history is not None:
        measure(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for block in (1, 4):
        for history in (1, 2, 4, 8, 16, 32, 64, 96):
            path = args.output / f"block{block}_history{history}.json"
            subprocess.run(
                [sys.executable, __file__, "--block", str(block), "--history", str(history), "--output", str(path)],
                check=True,
                env=os.environ.copy(),
            )
            records.append(json.loads(path.read_text()))
            (args.output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
