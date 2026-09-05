# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Low-overhead, per-iteration metrics for distributed training baselines."""

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

import psutil
import torch
import torch.distributed as dist

from cosmos_framework.utils import distributed, log


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().sum().item())
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return sum(_number(item) for item in value)
    return default


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


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
    """Collect CPU phase timing, throughput and memory without retaining tensors."""

    def __init__(self, config: Any, job_path: str, dataloader: Any, start_iteration: int = 0) -> None:
        self.enabled = bool(config.enabled)
        self.rank = distributed.get_rank()
        self.warmup_iterations = int(config.warmup_iterations)
        self.num_epochs = int(config.num_epochs)
        self.dataset_name = os.environ.get("BENCHMARK_DATASET_NAME", "unknown")
        self.output_dir = Path(job_path) / str(config.output_subdir)
        self._file = None
        self._records: list[dict[str, Any]] = []
        self._phases: dict[str, float] = defaultdict(float)
        self._iteration_started = 0.0
        self._batch_metrics: dict[str, float | int] = {}
        self.cumulative_global_samples = 0
        self.epoch_samples = int(len(dataloader))
        if self.epoch_samples <= 0:
            raise ValueError(f"Benchmarking requires a positive dataloader length, got {self.epoch_samples}")
        self.target_global_samples = self.epoch_samples * self.num_epochs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._file = (self.output_dir / f"rank_{self.rank:03d}.jsonl").open("w", encoding="utf-8", buffering=1)
        log.info(
            f"Training benchmark enabled: epoch_samples={self.epoch_samples}, epochs={self.num_epochs}, "
            f"target_global_samples={self.target_global_samples}, output={self.output_dir}",
            rank0_only=False,
        )

    def begin_iteration(self) -> None:
        if not self.enabled:
            return
        self._phases.clear()
        self._batch_metrics.clear()
        self._iteration_started = time.perf_counter()
        backend = self._accelerator_backend()
        if backend is None:
            return
        try:
            backend.reset_peak_memory_stats()
        except Exception:
            pass

    @staticmethod
    def _accelerator_backend() -> Any | None:
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_available():
            return npu
        if torch.cuda.is_available():
            return torch.cuda
        return None

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

    def record_batch(self, batch: dict[str, Any]) -> None:
        if not self.enabled:
            return
        videos = batch.get("video", [])
        local_samples = int(_number(batch.get("_packing_num_samples"), len(videos)))
        local_frames = int(_number(batch.get("num_frames")))
        batch_metrics = {
            "local_samples": local_samples,
            "local_frames": local_frames,
            "local_tokens": int(_number(batch.get("_packing_num_tokens"))),
            "worker_batch_seconds": _number(batch.get("_worker_batch_time")),
            "worker_io_seconds": _number(batch.get("_worker_io_time")),
            "worker_aug_seconds": _number(batch.get("_worker_aug_time")),
        }
        for key, value in batch_metrics.items():
            self._batch_metrics[key] = self._batch_metrics.get(key, 0) + value
        microbatches = int(self._batch_metrics.get("microbatches", 0))
        previous_efficiency = float(self._batch_metrics.get("packing_efficiency", 0.0))
        self._batch_metrics["packing_efficiency"] = (
            previous_efficiency * microbatches + _number(batch.get("_packing_efficiency"))
        ) / (microbatches + 1)
        self._batch_metrics["microbatches"] = microbatches + 1
        step_times = batch.get("_worker_aug_step_times", {})
        if isinstance(step_times, dict):
            for key, value in step_times.items():
                metric = f"worker_step/{key}_seconds"
                self._batch_metrics[metric] = self._batch_metrics.get(metric, 0.0) + _number(value)

    @staticmethod
    def _memory_metrics() -> dict[str, int | str]:
        process = psutil.Process(os.getpid())
        info = process.memory_info()
        children_rss = 0
        children_fds = 0
        for child in process.children(recursive=True):
            try:
                children_rss += child.memory_info().rss
                if hasattr(child, "num_fds"):
                    children_fds += child.num_fds()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        result: dict[str, int | str] = {
            "process_rss_bytes": info.rss,
            "process_vms_bytes": info.vms,
            "process_fds": process.num_fds() if hasattr(process, "num_fds") else -1,
            "children_rss_bytes": children_rss,
            "children_fds": children_fds,
        }
        backend = TrainingBenchmark._accelerator_backend()
        if backend is None:
            result["accelerator_backend"] = "none"
            return result
        result["accelerator_backend"] = "npu" if backend is getattr(torch, "npu", None) else "cuda"
        try:
            result.update(
                accelerator_allocated_bytes=backend.memory_allocated(),
                accelerator_reserved_bytes=backend.memory_reserved(),
                accelerator_max_allocated_bytes=backend.max_memory_allocated(),
                accelerator_max_reserved_bytes=backend.max_memory_reserved(),
            )
        except Exception as error:
            result["accelerator_memory_error"] = str(error)
        return result

    def finish_iteration(self, iteration: int) -> bool:
        """Write one record and return a rank-synchronized logical-epoch stop decision."""
        if not self.enabled:
            return False
        local_samples = int(self._batch_metrics.get("local_samples", 0))
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_available():
            device = torch.device("npu", npu.current_device())
        elif torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
        count = torch.tensor(local_samples, dtype=torch.int64, device=device)
        sync_started = time.perf_counter()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        self._phases["benchmark_count_sync"] += time.perf_counter() - sync_started
        global_samples = int(count.item())
        self.cumulative_global_samples += global_samples

        record: dict[str, Any] = {
            "timestamp_ns": time.time_ns(),
            "dataset_name": self.dataset_name,
            "rank": self.rank,
            "iteration": int(iteration),
            "iteration_seconds": time.perf_counter() - self._iteration_started,
            "global_samples": global_samples,
            "cumulative_global_samples": self.cumulative_global_samples,
            "target_global_samples": self.target_global_samples,
            "epoch_progress": self.cumulative_global_samples / self.epoch_samples,
        }
        record.update(self._batch_metrics)
        record.update({f"phase/{key}_seconds": value for key, value in self._phases.items()})
        record.update(self._memory_metrics())
        assert self._file is not None
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        self._records.append(record)
        return self.cumulative_global_samples >= self.target_global_samples

    def close(self) -> None:
        if not self.enabled:
            return
        if self._file is not None:
            self._file.close()
            self._file = None
        effective_warmup = min(self.warmup_iterations, max(0, len(self._records) - 1))
        steady = self._records[effective_warmup:]
        numeric: dict[str, list[float]] = defaultdict(list)
        for record in steady:
            for key, value in record.items():
                if isinstance(value, (int, float)) and key not in {"timestamp_ns", "rank", "iteration"}:
                    numeric[key].append(float(value))
        summary = {
            "rank": self.rank,
            "epoch_samples": self.epoch_samples,
            "dataset_name": self.dataset_name,
            "target_global_samples": self.target_global_samples,
            "consumed_global_samples": self.cumulative_global_samples,
            "sample_overshoot": max(0, self.cumulative_global_samples - self.target_global_samples),
            "iterations": len(self._records),
            "warmup_iterations_excluded": effective_warmup,
            "metrics": {key: _summarize(values) for key, values in sorted(numeric.items())},
        }
        (self.output_dir / f"rank_{self.rank:03d}_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def write_global_summary(self) -> None:
        """Rank 0 aggregates rank JSONL files after every rank has closed them."""
        if not self.enabled or self.rank != 0:
            return
        by_iteration: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for path in sorted(self.output_dir.glob("rank_[0-9][0-9][0-9].jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    record = json.loads(line)
                    by_iteration[int(record["iteration"])].append(record)
        iteration_rows: list[dict[str, Any]] = []
        for iteration, records in sorted(by_iteration.items()):
            row: dict[str, Any] = {
                "iteration": iteration,
                "ranks": len(records),
                "global_samples": records[0]["global_samples"],
                "global_frames": sum(record.get("local_frames", 0) for record in records),
                "global_tokens": sum(record.get("local_tokens", 0) for record in records),
                "iteration_seconds_max_rank": max(record["iteration_seconds"] for record in records),
                "packing_efficiency_mean_rank": statistics.fmean(
                    float(record.get("packing_efficiency", 0.0)) for record in records
                ),
                "microbatches_max_rank": max(record.get("microbatches", 0) for record in records),
                "process_rss_bytes_sum": sum(record.get("process_rss_bytes", 0) for record in records),
                "children_rss_bytes_sum": sum(record.get("children_rss_bytes", 0) for record in records),
                "process_and_children_rss_bytes_sum": sum(
                    record.get("process_rss_bytes", 0) + record.get("children_rss_bytes", 0) for record in records
                ),
                "process_fds_sum": sum(record.get("process_fds", 0) for record in records),
                "children_fds_sum": sum(record.get("children_fds", 0) for record in records),
                "accelerator_allocated_bytes_sum": sum(
                    record.get("accelerator_allocated_bytes", 0) for record in records
                ),
                "accelerator_reserved_bytes_sum": sum(
                    record.get("accelerator_reserved_bytes", 0) for record in records
                ),
                "accelerator_max_allocated_bytes_max_rank": max(
                    record.get("accelerator_max_allocated_bytes", 0) for record in records
                ),
                "accelerator_max_reserved_bytes_max_rank": max(
                    record.get("accelerator_max_reserved_bytes", 0) for record in records
                ),
            }
            duration = row["iteration_seconds_max_rank"]
            row["samples_per_second"] = row["global_samples"] / duration if duration else 0.0
            row["frames_per_second"] = row["global_frames"] / duration if duration else 0.0
            row["tokens_per_second"] = row["global_tokens"] / duration if duration else 0.0
            for key in records[0]:
                if key.startswith(("phase/", "worker_step/")) or key in {
                    "worker_batch_seconds",
                    "worker_io_seconds",
                    "worker_aug_seconds",
                }:
                    row[f"{key}_max_rank"] = max(float(record.get(key, 0.0)) for record in records)
            iteration_rows.append(row)
        csv_path = self.output_dir / "iterations.csv"
        fieldnames = sorted({key for row in iteration_rows for key in row})
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(iteration_rows)
        effective_warmup = min(self.warmup_iterations, max(0, len(iteration_rows) - 1))
        steady = iteration_rows[effective_warmup:]
        summary_metrics: dict[str, Any] = {}
        for key in fieldnames:
            values = [float(row[key]) for row in steady if isinstance(row.get(key), (int, float))]
            if values:
                summary_metrics[key] = _summarize(values)
        summary = {
            "dataset_name": self.dataset_name,
            "epoch_samples": self.epoch_samples,
            "target_global_samples": self.target_global_samples,
            "consumed_global_samples": self.cumulative_global_samples,
            "sample_overshoot": max(0, self.cumulative_global_samples - self.target_global_samples),
            "iterations": len(iteration_rows),
            "warmup_iterations_excluded": effective_warmup,
            "world_size": dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1,
            "metrics": summary_metrics,
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
