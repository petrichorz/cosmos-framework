"""Opt-in synchronized iteration measurements for isolated experiments."""

import json
import os
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch
import torch.distributed as dist


class AscendBenchmark:
    def __init__(self):
        self.enabled = os.environ.get("COSMOS_ASCEND_BENCHMARK") == "1"
        self.start = None
        self.warmup = int(os.environ.get("COSMOS_ASCEND_BENCHMARK_WARMUP", "5"))
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.path = Path(os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "outputs")) / f"benchmark_rank{self.rank}.jsonl"

    def begin(self, iteration):
        if not self.enabled:
            return
        torch.npu.synchronize()
        if dist.is_initialized():
            dist.barrier()
        torch.npu.synchronize()
        if iteration >= self.warmup:
            torch.npu.reset_peak_memory_stats()
        self.start = time.perf_counter()

    def end(self, iteration, data_batch):
        if not self.enabled:
            return
        torch.npu.synchronize()
        elapsed = time.perf_counter() - self.start

        def describe(value, depth=0):
            if isinstance(value, torch.Tensor):
                item = {"shape": list(value.shape), "dtype": str(value.dtype)}
                if value.numel() <= 128:
                    item["value"] = value.detach().cpu().tolist()
                return item
            if isinstance(value, dict):
                return {str(k): describe(v, depth + 1) for k, v in value.items()}
            if is_dataclass(value):
                return {field.name: describe(getattr(value, field.name), depth + 1) for field in fields(value)}
            if isinstance(value, (list, tuple)):
                return [describe(x, depth + 1) for x in value]
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(type(value))

        item = {
            "iteration": iteration,
            "rank": self.rank,
            "seconds": elapsed,
            "recorded": iteration > self.warmup,
            "peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "peak_reserved_bytes": torch.npu.max_memory_reserved(),
            "input": describe(data_batch),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"BENCHMARK rank={self.rank} iter={iteration} seconds={elapsed:.6f}", flush=True)
