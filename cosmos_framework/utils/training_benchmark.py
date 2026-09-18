# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""轻量级训练迭代级性能基准：定位「数据加载等待 + rank 同步等待」。

与 ``cosmos_framework.utils.performance``（worker 内细粒度打点）互补，本模块
在训练主循环里记录每个 iteration 的 wall-clock 和各 phase 耗时，并在 iteration
结束时做一次 ``all_reduce(MAX)`` 得到「最慢 rank 的耗时」，从而算出每个 rank 的
同步等待时间（快 rank 等待慢 rank 的时间）。

启用方式（环境变量驱动，无需改动 config schema）：:

    export COSMOS_BENCHMARK_DIR=/path/to/benchmark_out

未设置该环境变量时，所有 API 都是 no-op，零开销。

输出：
- 每个 rank 写 ``benchmark_rank{rank:03d}.jsonl``，每行一个 iteration 记录。
- rank 0 调用 ``write_global_summary`` 后聚合出 ``iterations.csv`` / ``summary.json``。
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist

from cosmos_framework.utils import distributed, log

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _benchmark_dir() -> Path | None:
    output = os.environ.get("COSMOS_BENCHMARK_DIR")
    return Path(output) if output else None


def benchmark_enabled() -> bool:
    return _benchmark_dir() is not None


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (1.0 - (index - lower)) + ordered[upper] * (index - lower)


def _summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


class TrainingBenchmark:
    """按 iteration 记录 phase 耗时 + 跨 rank 计算同步等待的轻量基准。

    Usage（在 trainer 训练主循环里）::

        benchmark = TrainingBenchmark(...)
        for iteration in ...:
            benchmark.begin_iteration()
            with benchmark.phase("dataloader_train"):
                data = next(dataloader)
            with benchmark.phase("training_step"):
                ...
            benchmark.finish_iteration(iteration)
        benchmark.close()
        benchmark.write_global_summary()   # 仅 rank 0
    """

    def __init__(self, output_dir: str | Path | None = None, warmup_iterations: int = 2) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else _benchmark_dir()
        self.enabled = self.output_dir is not None
        self.warmup_iterations = warmup_iterations
        self._file = None
        self._records: list[dict[str, Any]] = []
        self._phases: dict[str, float] = defaultdict(float)
        self._iteration_started = 0.0
        self.rank = 0
        self.world_size = 1

        if not self.enabled:
            return

        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._file = (self.output_dir / f"benchmark_rank{self.rank:03d}.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        log.info(
            f"Training benchmark enabled: rank={self.rank}, world_size={self.world_size}, "
            f"output={self.output_dir}",
            rank0_only=False,
        )

    def begin_iteration(self) -> None:
        if not self.enabled:
            return
        self._phases.clear()
        self._iteration_started = time.perf_counter()

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self._phases[name] += time.perf_counter() - started

    @staticmethod
    def _sync_device() -> torch.device:
        """返回 all_reduce 同步用的加速器设备（NPU > CUDA > CPU）。"""
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_available():
            return torch.device("npu", npu.current_device())
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def finish_iteration(self, iteration: int) -> None:
        if not self.enabled:
            return
        local_seconds = time.perf_counter() - self._iteration_started

        # 跨 rank 取最慢 rank 的 iteration 耗时（所有 rank 都要等最慢的）。
        sync_seconds = 0.0
        max_seconds = local_seconds
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            # all_reduce 的 tensor 必须放在当前后端支持的设备上（NPU/HCCL 下 CPU tensor 会报
            # "No backend type associated with device type cpu"），且 dtype 必须用 float32
            # （HCCL 不支持 float64/kDouble）。秒级 wall-clock 用 float32 精度足够。
            tensor = torch.tensor([local_seconds], dtype=torch.float32, device=self._sync_device())
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            max_seconds = float(tensor[0].item())
            sync_seconds = max_seconds - local_seconds

        record: dict[str, Any] = {
            "timestamp_ns": time.time_ns(),
            "rank": self.rank,
            "iteration": int(iteration),
            "iteration_seconds": local_seconds,
            "max_rank_iteration_seconds": max_seconds,
            "sync_wait_seconds": sync_seconds,
        }
        record.update({f"phase/{name}_seconds": value for name, value in self._phases.items()})
        assert self._file is not None
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        self._records.append(record)

    def close(self) -> None:
        if not self.enabled:
            return
        if self._file is not None:
            self._file.close()
            self._file = None

    def write_global_summary(self) -> None:
        """Rank 0 聚合各 rank 的 JSONL，产出 iterations.csv 和 summary.json。"""
        if not self.enabled or self.rank != 0 or self.output_dir is None:
            return

        by_iteration: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for path in sorted(self.output_dir.glob("benchmark_rank[0-9][0-9][0-9].jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                record = json.loads(line)
                by_iteration[int(record["iteration"])].append(record)

        rows: list[dict[str, Any]] = []
        for iteration, records in sorted(by_iteration.items()):
            phase_keys = sorted({k for r in records for k in r if k.startswith("phase/")})
            row: dict[str, Any] = {
                "iteration": iteration,
                "ranks": len(records),
                "iteration_seconds_max_rank": max(r["iteration_seconds"] for r in records),
                "sync_wait_seconds_max_rank": max(r.get("sync_wait_seconds", 0.0) for r in records),
                "sync_wait_seconds_sum": sum(r.get("sync_wait_seconds", 0.0) for r in records),
            }
            for key in phase_keys:
                row[f"{key}_max_rank"] = max(float(r.get(key, 0.0)) for r in records)
                row[f"{key}_mean_rank"] = statistics.fmean(float(r.get(key, 0.0)) for r in records)
            rows.append(row)

        fieldnames = sorted({k for row in rows for k in row})
        with (self.output_dir / "iterations.csv").open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        effective_warmup = min(self.warmup_iterations, max(0, len(rows) - 1))
        steady = rows[effective_warmup:]
        metrics: dict[str, Any] = {}
        for key in fieldnames:
            values = [float(row[key]) for row in steady if isinstance(row.get(key), (int, float))]
            if values:
                metrics[key] = _summarize(values)
        summary = {
            "world_size": self.world_size,
            "iterations": len(rows),
            "warmup_iterations_excluded": effective_warmup,
            "metrics": metrics,
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        log.info(f"Training benchmark summary written to {self.output_dir}")
